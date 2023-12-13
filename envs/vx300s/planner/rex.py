from functools import partial
import tqdm
from typing import List, Tuple, Dict, Optional, Callable, Union
import networkx as nx
import jax
import jumpy

import numpy as onp
import jax.numpy as jnp
import jumpy.numpy as jp
import rex.jumpy as rjp

import jax.debug as jdb

from brax import base, math
from flax import struct

import rex.open_colors as oc
from rex.proto import log_pb2
from rex.base import StepState, GraphState, Empty, InputState
from rex.node import Node
from rex.compiled import CompiledGraph
import rex.supergraph

import supergraph as sg

from envs.vx300s.planner.cost import CostParams, box_pushing_cost
from envs.vx300s.planner.cem import CEMParams, cem_rex
from envs.vx300s.env import PlannerOutput, get_next_jpos
from envs.vx300s.env import Controller
from envs.vx300s.brax.world import World, ArmActuator, BoxSensor, ArmSensor, BraxOutput, State as WorldState, Params as WorldParams
from envs.vx300s.planner.brax import print_sys_info

import envs.vx300s as vx300s


@struct.dataclass
class State:
    pipeline_state: base.State
    q_des: jnp.ndarray


@struct.dataclass
class RexCEMParams:
    """See https://arxiv.org/pdf/1907.03613.pdf for details on CEM"""
    dt: jp.float32
    cem_params: CEMParams
    cost_params: CostParams


@struct.dataclass
class RexCEMState:
    goalpos: jp.ndarray
    goalyaw: jp.ndarray
    prev_plans: InputState


class Cost(Node):
    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
        """Default output of the root."""
        return Empty()

    def step(self, step_state: StepState) -> Tuple[StepState, Empty]:
        """Default step of the root."""
        return step_state, Empty()


@struct.dataclass
class CostState:
    cost_orn: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cost_down: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cost_align: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cost_height: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    alpha: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cost_force: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cost_near: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cost_dist: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    cum_cost: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.))
    timestep: jp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.int32(0))


@struct.dataclass
class WrappedWorldState:
    world_state: WorldState
    cost_state: CostState


@struct.dataclass
class WrappedWorldParams:
    world_params: WorldParams
    cost_params: CostParams


class WorldWrapper:
    def __init__(self, node: World):
        self._node = node

    # Redirect all calls to the planner unless they are explicitly defined here
    def __getattr__(self, item):
        return getattr(self._node, item)

    def step(self, step_state: StepState) -> Tuple[StepState, BraxOutput]:
        # print("DummyWorld.step")
        ss = step_state

        # Prepare world step_state
        ss_world = step_state.replace(params=step_state.params.world_params, state=step_state.state.world_state)
        new_ss_world, output = self._node.step(ss_world)
        new_world_state = new_ss_world.state

        # Calculate cost
        cp = step_state.params.cost_params
        cs = step_state.state.cost_state
        new_cs = self._cost(new_world_state, cp, cs)
        new_ss = ss.replace(state=ss.state.replace(world_state=new_world_state, cost_state=new_cs))
        return new_ss, output

    def _cost(self, world_state: WorldState, cost_params: CostParams, cost_state: CostState) -> CostState:
        """Calculate cost"""
        time_step = cost_state.timestep
        pipeline_state = world_state.pipeline_state
        sys = self._node.sys

        # Get indices
        ee_arm_idx = sys.link_names.index("ee_link")
        box_idx = sys.link_names.index("box")
        goal_idx = sys.link_names.index("goal")

        x_i = pipeline_state.x.vmap().do(
            base.Transform.create(pos=sys.link.inertia.transform.pos)
        )

        boxpos = x_i.pos[box_idx]
        eepos = x_i.pos[ee_arm_idx]
        eeorn = self._node._convert_wxyz_to_xyzw(x_i.rot[ee_arm_idx])  # require xyzw convention
        goalpos = x_i.pos[goal_idx][:2]
        force = pipeline_state.qf_constraint
        discounted_cost, info = box_pushing_cost(cost_params, boxpos, eepos, goalpos, eeorn, force, action=None, time_step=time_step)

        # Update cost state
        next_timestep = time_step + 1
        cum_cost = cost_state.cum_cost + discounted_cost
        next_cost_state = CostState(cost_orn=info["cost_orn"],
                                    cost_down=info["cost_down"],
                                    cost_align=info["cost_align"],
                                    cost_height=info["cost_height"],
                                    alpha=info["alpha"],
                                    cost_force=info["cost_force"],
                                    cost_near=info["cost_near"],
                                    cost_dist=info["cost_dist"],
                                    cum_cost=cum_cost,
                                    timestep=next_timestep)
        return next_cost_state


class RexCEMPlanner(Node):
    def __init__(self, *args,
                 graph_path: str,
                 nodes: Dict[str, Node],
                 mj_path: str,
                 supergraph_mode: str = "MCS",
                 episode_idx: int = -1,
                 graph_idx: int = -1,
                 num_cost_est: int = 14,  # 14,
                 num_cost_mpc: int = 20,  # 20,
                 randomize_eps: bool = True,
                 use_estimator: bool = True,
                 z_fixed: float = 0.051,
                 pipeline: str = "generalized",
                 u_max: float = 0.35 * 3.14,
                 dt: float = 0.15,
                 dt_substeps: float = 0.015,
                 horizon: int = 4,
                 num_samples: int = 300,
                 max_iter: int = 1,
                 sampling_smoothing: float = 0.,
                 evolution_smoothing: float = 0.1,
                 elite_portion: float = 0.1,
                 progress_bar: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_idx = episode_idx
        self._graph_idx = graph_idx
        self._num_cost_est = num_cost_est
        self._num_cost_mpc = num_cost_mpc
        self._randomize_eps = randomize_eps
        self._use_estimator = use_estimator
        self._z_fixed = z_fixed
        self._horizon = horizon
        self._u_max = u_max * jp.ones((6,), dtype=jp.float32) if isinstance(u_max, float) else jp.array(u_max, dtype=jp.float32)
        assert self._u_max.shape == (6,)
        self._dt = dt
        self._dt_substeps = dt_substeps
        self._num_samples = num_samples
        self._max_iter = max_iter
        self._sampling_smoothing = sampling_smoothing
        self._evolution_smoothing = evolution_smoothing
        self._elite_portion = elite_portion

        # Load proto experiment record
        record_pre = log_pb2.ExperimentRecord()
        with open(graph_path, "rb") as f:
            record_pre.ParseFromString(f.read())

        # Construct graph
        self._G = rex.supergraph.create_graph(record_pre.episode[self._episode_idx])
        self._Gs = make_partitions(self._G.copy(as_view=False), dt_horizon=dt*horizon, progress_bar=progress_bar)
        self._Gs_est, self._Gs_mpc = split_partitions_into_estimator_mpc_partitions(self._Gs,
                                                                                    num_cost_est=self._num_cost_est,
                                                                                    num_cost_mpc=self._num_cost_mpc)

        # Construct supergraph
        S_est, timings_est = make_supergraph_and_timings(list(self._Gs_est.values()), root="cost",
                                                         num_cost=self._num_cost_est,
                                                         supergraph_mode=supergraph_mode, backtrack=30,
                                                         progress_bar=progress_bar)
        S_mpc, timings_mpc = make_supergraph_and_timings(list(self._Gs_mpc.values()), root="cost",
                                                         num_cost=self._num_cost_mpc,
                                                         supergraph_mode=supergraph_mode, backtrack=30,
                                                         progress_bar=progress_bar)

        # Get window of planner armsensor sufficiently large and select first jpos after boxsensor measurement.
        self.window_prev_plans = get_planner_window(self._Gs_est)

        # Initialize nodes
        cost = Cost(name="cost", rate=1)  # dummy rate
        planner = self

        # Initialize controller
        args, kwargs, _ = super(nodes["controller"].__class__, nodes["controller"]).__getstate__()
        controller = Controller(*args, **kwargs)
        controller.inputs = nodes["controller"].inputs

        # Initialize actuator
        args, kwargs, _ = super(nodes["armactuator"].__class__, nodes["armactuator"]).__getstate__()
        armactuator = ArmActuator(*args, **kwargs)
        armactuator.inputs = nodes["armactuator"].inputs

        # Initialize world
        args, kwargs, _ = super(nodes["world"].__class__, nodes["world"]).__getstate__()
        world = World(*args, xml_path=mj_path, dt_brax=dt_substeps, backend=pipeline, **kwargs)
        world: Node = WorldWrapper(world)
        self._world = world
        cost.connect(world, blocking=True)
        world.connect(armactuator, window=1, blocking=False)

        # Define graph nodes
        graph_nodes = {"supervisor": nodes["supervisor"], "planner": planner, "controller": controller,
                       "armactuator": armactuator, "world": world}

        # Define compiled graphs
        skip = ["planner", "supervisor", "armsensor", "boxsensor"]
        self.graph_est = CompiledGraph(graph_nodes, root=cost, S=S_est, default_timings=timings_est, skip=skip)
        self.graph_mpc = CompiledGraph(graph_nodes, root=cost, S=S_mpc, default_timings=timings_mpc, skip=skip)

        # if False:
        # _ = vx300s.show_computation_graph(self._G, root=None, xmax=2.0, draw_pruned=False)
        # _ = vx300s.show_computation_graph(self._Gs["planner_4"], root=None, xmax=None, draw_pruned=True)
        # _ = vx300s.show_computation_graph(self._Gs_est["planner_4"], root=None, xmax=None, draw_pruned=True)
        # _ = vx300s.show_computation_graph(self._Gs_mpc["planner_4"], root=None, xmax=None, draw_pruned=True)
        # import matplotlib.pyplot as plt
        # plt.show()
        # print("PLOTTING RexCEMPlanner")

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> RexCEMParams:
        cost_params = CostParams.default()
        cem_params = CEMParams(u_min=-self._u_max * jp.ones((6,), dtype=jp.float32),
                               u_max=self._u_max * jp.ones((6,), dtype=jp.float32),
                               sampling_smoothing=self._sampling_smoothing,
                               evolution_smoothing=self._evolution_smoothing)
        params = RexCEMParams(cem_params=cem_params,
                              cost_params=cost_params,
                              dt=self._dt)
        return params

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> RexCEMState:
        goalpos = graph_state.nodes["supervisor"].state.goalpos
        goalyaw = graph_state.nodes["supervisor"].state.goalyaw

        # Initialize previous plans
        plan = self.default_output(rng, graph_state)
        seq = 0 * jp.arange(-self.window_prev_plans, 0, dtype=jp.int32) - 1
        ts_sent = 0 * jp.arange(-self.window_prev_plans, 0, dtype=jp.float32)
        ts_recv = 0 * jp.arange(-self.window_prev_plans, 0, dtype=jp.float32)
        _msgs = [plan] * self.window_prev_plans
        prev_plans = InputState.from_outputs(seq, ts_sent, ts_recv, _msgs)
        return RexCEMState(prev_plans=prev_plans, goalpos=goalpos, goalyaw=goalyaw)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> PlannerOutput:
        p_sup = graph_state.nodes["supervisor"].params
        p_plan = graph_state.nodes["planner"].params
        jpos = p_sup.home_jpos
        jvel = jp.zeros((self._horizon, jpos.shape[0]), dtype=jp.float32)
        timestamps = p_plan.dt * jp.arange(0, self._horizon + 1, dtype=jp.float32)
        return PlannerOutput(jpos=jpos, jvel=jvel, timestamps=timestamps)

    def step(self, step_state: StepState):
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Determine first armsensor.jpos reading before boxsensor reading
        ts_boxpos = step_state.inputs["boxsensor"][-1].ts_sent
        ts_jpos = step_state.inputs["armsensor"].ts_sent
        idx_all = jp.where(ts_jpos <= ts_boxpos, jp.arange(0, ts_jpos.shape[0]), jp.zeros(ts_jpos.shape, dtype=jp.int32))
        idx = jp.argmax(idx_all)
        jpos_now = jp.take(step_state.inputs["armsensor"].data.jpos, idx, axis=0)
        goalpos = state.goalpos
        boxyaw_now = jp.array([step_state.inputs["boxsensor"][-1].data.wrapped_yaw])
        boxpos_now = step_state.inputs["boxsensor"][-1].data.boxpos

        # Fixe boxpos_now[z] to be self._z_fixed
        boxpos_now = boxpos_now.at[2].set(self._z_fixed)

        # Pre-initialize world step_state
        qpos = jp.concatenate([boxpos_now, boxyaw_now, goalpos, jpos_now, jnp.array([0])])  # added [0] for additional ee joint.
        qd = jp.zeros((qpos.shape[0],), dtype=jnp.float32)
        pipeline_state = self._world._pipeline.init(self._world.sys, qpos, qd)
        rng, rng_wss = jumpy.random.split(rng)
        wss = StepState(rng=rng_wss, state=WorldState(pipeline_state=pipeline_state), params=WorldParams())

        # Initialize estimator graph
        rng, rng_est = jumpy.random.split(rng)
        init_gs_est = self.graph_est.init(rng=rng_est, step_states={"world": wss}, order=("supervisor", "world", "planner"), starting_eps=jp.as_int32(self._graph_idx))

        # Infer future timestamp, which is the start of the next plan
        dt_future = self.graph_est.nodes["planner"].output.delay + self.graph_est.nodes["planner"].get_connected_input("controller").delay
        ts_future = step_state.ts + dt_future
        timestamps = params.dt * jp.arange(0, self._horizon + 1, dtype=jp.float32) + ts_future

        # Prepare buffer for planner with previous plans
        prev_plans = step_state.state.prev_plans.data
        init_plan = get_init_plan(step_state.state.prev_plans[-1].data, timestamps)
        # Shift timestamps of init_plan and prev_plans with step_state.ts
        shifted_init_plan = init_plan.replace(timestamps=init_plan.timestamps - step_state.ts)
        shifted_prev_plans = prev_plans.replace(timestamps=prev_plans.timestamps - step_state.ts)
        # Extend shifted_prev_plans with an initial plan at the beginning --> is going to be the new plan (seq=0).
        planner_buffer = jax.tree_util.tree_map(lambda x, y: jp.concatenate([x[None], y], axis=0), shifted_init_plan, shifted_prev_plans)
        init_gs_est = init_gs_est.replace_buffer({"planner": planner_buffer})  # Update buffer with previous plans

        # Update wrapped world step_state
        new_wss = init_gs_est.nodes["world"].replace(params=WrappedWorldParams(world_params=WorldParams(),
                                                                               cost_params=step_state.params.cost_params),
                                                     state=WrappedWorldState(world_state=WorldState(pipeline_state=pipeline_state),
                                                                             cost_state=CostState(cum_cost=jp.array(0.), timestep=jp.int32(0)))
                                                     )
        init_gs_est = init_gs_est.replace_nodes({"world": new_wss})

        # Run estimator graph
        _scan_est_fn = partial(_scan_graph_fn, self.graph_est, self._use_estimator)
        final_gs_est, _ = jumpy.lax.scan(_scan_est_fn, init_gs_est, jp.arange(0, self.graph_est.max_runs()))

        # Initialize MPC world state
        rng, rng_wss = jumpy.random.split(rng)
        wss = StepState(rng=rng_wss,
                        state=final_gs_est.nodes["world"].state.world_state,
                        params=final_gs_est.nodes["world"].params.world_params
                        )
        init_gs_mpc = final_gs_est.replace_nodes({"world": wss})

        # Initialize estimator graph
        rng, rng_mpc = jumpy.random.split(rng)
        init_gs_mpc = self.graph_mpc.init(rng=rng_mpc, step_states=init_gs_mpc.nodes, order=("supervisor", "world", "planner"),
                                          starting_eps=jp.as_int32(jp.as_int32(self._graph_idx)))
        # init_gs_mpc = self._jit_graph_mpc_init(rng=rng_mpc, step_states=init_gs_mpc.nodes, order=("supervisor", "world", "planner"), starting_eps=jp.as_int32(2))
        init_gs_mpc = init_gs_mpc.replace_nodes({"world": final_gs_est.nodes["world"]})  # Replace with wrapped world step_state
        init_gs_mpc = init_gs_mpc.replace_buffer({"planner": planner_buffer})  # Update buffer with previous plans

        # Initialize MPC
        def _mpc_objective_fn(graph: CompiledGraph, eps_idx, controls, _init_gs_mpc):
            # Replace eps
            if self._randomize_eps:
                eps = eps_idx % graph.max_eps(_init_gs_mpc)  # Makes sure eps is within bounds
                _init_gs_mpc = _init_gs_mpc.replace_eps(eps)

            # Add controls to planner_buffer.jvel[0] --> this is the to-be-optimized plan.
            next_jvel = jumpy.index_update(planner_buffer.jvel, jp.array(0), controls)
            next_p = _init_gs_mpc.buffer["planner"].replace(jvel=next_jvel)
            next_gs = _init_gs_mpc.replace_buffer({"planner": next_p})

            # Run mpc graph
            _scan_mpc_fn = partial(_scan_graph_fn, graph, True)
            _final_gs_mpc, _ = jumpy.lax.scan(_scan_mpc_fn, next_gs, jp.arange(0, graph.max_runs()))

            # Compute cost (subtract initial cost from forward estimation graph
            cum_cost = _final_gs_mpc.nodes["world"].state.cost_state.cum_cost - _init_gs_mpc.nodes["world"].state.cost_state.cum_cost
            return cum_cost

        # Prepare objective function for CEM
        mpc_objective_fn = partial(_mpc_objective_fn, self.graph_mpc)

        def _vmap_mpc_objective_fn(eps_idx, controls, _init_gs_mpc):
            _vmap_mpc_objective_fn = jax.vmap(mpc_objective_fn, in_axes=(0, None, None), out_axes=0)
            eps_idx = jp.arange(2, 3)  # + eps_idx % 1
            cum_cost = _vmap_mpc_objective_fn(eps_idx, controls, _init_gs_mpc)
            return cum_cost.mean()

        # Run CEM
        objective_fn = _vmap_mpc_objective_fn if self._randomize_eps else mpc_objective_fn
        rng, rng_cem = jumpy.random.split(rng)
        cem_params = params.cem_params
        cem_hyperparams = dict(sampling_smoothing=cem_params.sampling_smoothing,
                               evolution_smoothing=cem_params.evolution_smoothing,
                               elite_portion=self._elite_portion, num_samples=self._num_samples, max_iter=self._max_iter)
        mean, obj = cem_rex(objective_fn, init_gs_mpc, shifted_init_plan.jvel, cem_params.u_min, cem_params.u_max, rng_cem, hyperparams=cem_hyperparams)

        # Update plan
        next_plan = init_plan.replace(jvel=mean)

        # Update prev_plans
        next_prev_plans = step_state.state.prev_plans.push(seq=step_state.seq, ts_sent=step_state.ts, ts_recv=ts_future, data=next_plan)

        # Update step_state
        next_step_state = step_state.replace(rng=rng, state=step_state.state.replace(prev_plans=next_prev_plans))
        return next_step_state, next_plan

    def _convert_wxyz_to_xyzw(self, quat: jp.ndarray):
        """Convert quaternion (w,x,y,z) -> (x,y,z,w)"""
        return jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype="float32")


def _scan_graph_fn(graph, run_graph, carry, x):
    if run_graph:
        next_carry = graph.run(carry)
    else:
        next_carry = carry
    y = x
    return next_carry, y


def get_init_plan(last_plan: PlannerOutput, timestamps: jp.ndarray) -> PlannerOutput:
    get_next_jpos_vmap = jumpy.vmap(get_next_jpos, include=(False, True))
    jpos_timestamps = get_next_jpos_vmap(last_plan, timestamps)
    dt = timestamps[1:] - timestamps[:-1]
    jvel_timestamps = (jpos_timestamps[1:] - jpos_timestamps[:-1]) / dt[:, None]
    return PlannerOutput(jpos=jpos_timestamps[0], jvel=jvel_timestamps, timestamps=timestamps)


def calibrate_graph(G: nx.DiGraph, ts_offset: float = 0., relabel_nodes: bool = False) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """
    Calibrates a directed graph by adjusting sequence numbers and time steps of its nodes
    and optionally relabels the nodes.

    Args:
        G (nx.DiGraph): The directed graph to be calibrated.
        ts_offset (float): The time step offset to be applied.
        relabel_nodes (bool, optional): If True, relabels the nodes of the graph. Defaults to False.

    Returns:
        nx.DiGraph: The calibrated (and optionally relabeled) graph.
    """
    seq_offset = {}
    last_seq = {}
    topo = list(nx.topological_sort(G))
    G_relabel = G.copy(as_view=False)
    relabel_map = {}

    for n in topo:
        kind = G_relabel.nodes[n]["kind"]
        ts = G_relabel.nodes[n]["ts_step"] - ts_offset
        if kind not in seq_offset and ts >= 0:
            seq_offset[kind] = G_relabel.nodes[n]["seq"]
        else:
            last_seq[kind] = G_relabel.nodes[n]["seq"]+1

    # If no nodes is found with ts >= 0, then we use the seq-1 of the most recent node (i.e. the most recent is -1)
    not_in_seq_offset = {k: v for k, v in last_seq.items() if k not in seq_offset.keys()}
    seq_offset.update(not_in_seq_offset)

    # Offset data
    for n in topo:
        kind = G_relabel.nodes[n]["kind"]
        G_relabel.nodes[n]["seq"] -= seq_offset[kind]
        G_relabel.nodes[n]["ts_step"] -= ts_offset
        G_relabel.nodes[n]["ts_sent"] -= ts_offset
        G_relabel.nodes[n]["position"] = (G_relabel.nodes[n]["position"][0] - ts_offset, G_relabel.nodes[n]["position"][1])

        if relabel_nodes:
            new_label = f"{kind}_{G_relabel.nodes[n]['seq']}"
            relabel_map[n] = new_label
        else:
            relabel_map[n] = n

        for u, _, edata in G_relabel.out_edges(n, data=True):
            edata["ts_recv"] -= ts_offset
            edata["ts_sent"] -= ts_offset
            edata["seq"] -= seq_offset[kind]

    if relabel_nodes:
        G_relabel = nx.relabel_nodes(G_relabel, relabel_map, copy=False)
    return G_relabel, relabel_map


def constrained_topological_sort(graph, constraint_fn=None, edge_constraint_fn=None, start=None, in_degree=None,
                                 return_delta=False):
    """
    Perform a constrained topological sort on a DAG.
    This function filters nodes based on the provided constraint function while performing the sort.
    """
    if constraint_fn is None:
        constraint_fn = lambda *args, **kwargs: True

    if edge_constraint_fn is None:
        edge_constraint_fn = lambda *args, **kwargs: True

    # Initialize the list for the sorted order
    order = []

    # Determine the in-degree for each node
    if in_degree is None:
        in_degree = {u: 0 for u in graph.nodes()}
        for u, v in graph.edges():
            in_degree[v] += 1

    # Determine the initial set of nodes to start traversals from
    if start is None:
        # Use a set to manage the nodes with zero in-degree
        zero_in_degree = {u for u in graph.nodes() if in_degree[u] == 0}
    else:
        zero_in_degree = set(start)
    start = zero_in_degree.copy() if start is None else start

    delta_in_degree = {}
    while zero_in_degree:
        # Pop a node from the zero in-degree set
        u = zero_in_degree.pop()

        # Check if the node satisfies the constraint
        if constraint_fn(graph, u):
            order.append(u)

            # Decrease the in-degree of successors and add to zero in-degree set if needed
            for v in graph.successors(u):
                if edge_constraint_fn(graph, u, v) and v not in start:
                    delta_in_degree[v] = delta_in_degree.get(v, 0) - 1
                    # in_degree[v] -= 1
                    if in_degree[v] + delta_in_degree[v] == 0:
                        zero_in_degree.add(v)

    if return_delta:
        return order, delta_in_degree
    else:
        return order


def make_partitions(G: nx.DiGraph, dt_horizon: float, verbose: bool = False, progress_bar: bool = False):
    """
    This function partitions a directed graph (DiGraph) into estimator and model predictive control (MPC) partitions
    based on the nodes' roles and interactions within a robotic system's computational graph.
    It prunes the graph, sorts nodes by kind and sequence, and creates partitions for MPC planning nodes
    considering their connections to sensors, actuators, and controllers.

    Arguments:
    G (nx.DiGraph): The directed graph representing the computational network of the robotic system.
        It is modified in-place.
    dt_horizon (float): The time horizon for the MPC planning. It defines the time frame within which the
        planner operates.
    verbose (bool, optional): If True, enables verbose output during the partitioning process. Default is False.
    progress_bar (bool, optional): If True, displays a progress bar for the partitioning process. Default is False.

    Returns:
    dict: A dictionary where keys are planner node names and values are dictionaries containing the
    subgraphs for estimator ('est') and MPC ('mpc') partitions, as well as the original ('partition') and calibrated ('calibrated') subgraphs.
    """

    # Base graph has all pruned graphs removed
    G = rex.supergraph.prune_graph(G, copy=False)

    # Split nodes by kind and sort by sequence
    ndict: Dict[str, Dict[str, Dict]] = {}
    for n, data in G.nodes(data=True):
        if data["kind"] not in ndict:
            ndict[data["kind"]] = {}
        ndict[data["kind"]][n] = data

    # Sort nodes by sequence in list
    # nlist_sorted: Dict[str, List[Tuple[str, Dict]]] = {}
    nlist_sorted: Dict[str, List[str]] = {}
    for kind, nodes in ndict.items():
        # nlist_sorted[kind] = [(k, v) for k, v in sorted(nodes.items(), key=lambda item: item[1]["seq"])]
        nlist_sorted[kind] = [k for k, v in sorted(nodes.items(), key=lambda item: item[1]["seq"])]

    # Determine the in-degree for each node
    in_degree = {u: 0 for u in G.nodes()}
    for u, v, edata in G.edges(data=True):
        if not edata["pruned"]:
            in_degree[v] += 1

    # Determine window of planner
    win_planner = G.nodes[nlist_sorted["controller"][-1]]["inputs"]["planner"]["window"]

    # Step-by-step process for partitioning each MPC planner node:
    # 1. Iterate through each MPC planner node in the sorted list.
    # 2. For each planner node, follow the graph edges to identify key sensor, actuator, and controller nodes connected to it.
    # 3. Starting from the planner node, trace back to find the nearest boxsensor node, then to the world node connected to this boxsensor.
    # 4. Identify the nearest world node connected to an armsensor node.
    # 5. Determine the actuator and controller nodes associated with the world nodes.
    # 6. For the controller nodes, establish the planning horizon based on the time steps and the given dt_horizon.
    # 7. Collect all controller nodes within this planning horizon.
    # 8. Apply node constraints and perform a constrained topological sort to obtain the initial partition.
    # 9. Split this partition into two: one before the horizon (pre-horizon) and one within the horizon.
    # 10. In the partition, redirect edges as necessary to maintain graph consistency.
    # 11. Store the partitions in the Gs dictionary, keyed by the planner node name.
    # 12. If verbose, print the details of the partitioned nodes and edges for each planner node.
    # 13. Optionally, visualize the computational graph for debugging or presentation purposes.
    Gs = {}
    desc = "Finding MPC partitions"
    pbar = tqdm.tqdm(total=len(nlist_sorted["planner"]), desc=desc, disable=not progress_bar)
    for j, n_plan in enumerate(nlist_sorted["planner"]):
        # Update progressbar
        skipped = j + 1 - len(Gs)
        pbar.set_postfix_str(f"skipped {skipped}/{j+1}")
        pbar.update(1)

        n_box, n_arm, n_world_box, n_world_arm, n_world, n_first_controller, n_controller, n_actuator, n_first_plan = None, None, None, None, None, None, None, None, None
        # Find the previous boxsensor that provided its measurement to the planner
        for u, _ in G.in_edges(n_plan):
            if G.nodes[u]["kind"] == "boxsensor":
                n_box = u
                break
        if n_box is None:
            print(f"Skipping planner node {n_plan} because no boxsensor node found") if verbose else None
            continue

        # Find the previous world node that provided its state to the boxsensor
        for u, _ in G.in_edges(n_box):
            if G.nodes[u]["kind"] == "world":
                n_world_box = u
                break
        if n_world_box is None:
            print(f"Skipping planner node {n_plan} because no world node for the boxsensor found") if verbose else None
            continue

        # Find the nearest world node that provided a measurement to the armsensor
        n_world_arm = n_world_box
        while True:
            n_prev_world = None
            for _, v, edata in G.out_edges(n_world_arm, data=True):
                if G.nodes[v]["kind"] == "armsensor":
                    n_arm = v
                    break
                elif edata["stateful"]:
                    n_prev_world = v
            if n_arm is not None or n_prev_world is None:
                break
            else:
                n_world_arm = n_prev_world
        if n_arm is None:
            print(f"Skipping planner node {n_plan} because no armsensor node found") if verbose else None
            continue

        # Select the world node that comes right after the world_arm node, because that's the node we need to initialize with the arm and box state measurements
        for _, v, edata in G.out_edges(n_world_arm, data=True):
            if edata["stateful"]:
                n_world = v
                break
        if n_world is None:
            print(f"Skipping planner node {n_plan} because no world node found") if verbose else None
            continue

        # Find the previous actuator, controller, and prev_plan nodes
        for u, _ in G.in_edges(n_world):
            if G.nodes[u]["kind"] == "armactuator":
                n_actuator = u
                break
        if n_actuator is None:
            print(f"Skipping planner node {n_plan} because no armactuator node found") if verbose else None
            continue

        for u, _ in G.in_edges(n_actuator):
            if G.nodes[u]["kind"] == "controller":
                n_first_controller = u
                break
        if n_first_controller is None:
            print(f"Skipping planner node {n_plan} because no first controller node found") if verbose else None
            continue

        # Determine previous & first planner node (i.e. before n_plan)
        n_prev_plans = []
        for u, _ in G.in_edges(n_first_controller):
            if G.nodes[u]["kind"] == "planner":
                n_prev_plans.append(u)
                # n_first_plan = u
        if len(n_prev_plans) < win_planner:
            print(f"Skipping planner node {n_plan} because not enough prev_plan nodes found") if verbose else None
            continue

        # Add all plan nodes between n_plan and n_first_plan to n_prev_plans
        n_first_plan = min(n_prev_plans, key=lambda x: G.nodes[x]["seq"])
        if n_first_plan is None:
            print(f"Skipping planner node {n_plan} because no prev_plan node found") if verbose else None
            continue

        n_next_plan = n_first_plan
        while n_next_plan != n_plan:
            for _, v, edata in G.out_edges(n_next_plan, data=True):
                if edata["stateful"]:
                    n_prev_plans.append(v)
                    n_next_plan = v
                    break
        n_prev_plans = list(set(n_prev_plans))

        # Find n_first_controller plan horizon
        n_horizon = []
        for _, v in G.out_edges(n_plan):
            if G.nodes[v]["kind"] == "controller":
                n_horizon.append(v)
        if len(n_horizon) == 0:
            print(f"Skipping planner node {n_plan} because no downstream controller node found") if verbose else None
            continue

        # Find the controller in n_horizon based on the minimum ts_step (don't use numpy)
        n_controller = n_horizon[0]
        for v in n_horizon:
            if G.nodes[v]["ts_step"] < G.nodes[n_controller]["ts_step"]:
                n_controller = v

        # Determine the start and end time steps for the planner
        ts_first_controller = G.nodes[n_first_controller]["ts_step"]
        ts_start = G.nodes[n_plan]["ts_step"]
        ts_controller = G.nodes[n_controller]["ts_step"]
        # ts_future = ts_controller - ts_start
        ts_end = ts_controller + dt_horizon

        # Find all controller nodes that are in the horizon of this planner
        n_all_controllers = [n_first_controller]
        while True:
            next_controller = None
            for _, v, edata in G.out_edges(n_all_controllers[-1], data=True):
                if edata["stateful"]:
                    next_controller = v
                    break

            # If no next controller found, then we're at the end of the graph
            if next_controller is None:
                break

            # If the next controller is outside the horizon, then we're at the end of the horizon
            if G.nodes[next_controller]["ts_step"] <= ts_end:
                n_all_controllers.append(next_controller)
            else:
                break
        if next_controller is None:
            print(f"Skipping planner node {n_plan} because no controller outside horizon found") if verbose else None
            continue

        def _node_constraint_fn(_G, _u):
            if _G.nodes[_u]["kind"] in ["armactuator", "planner", "world"]:
                return True
            elif _G.nodes[_u]["kind"] == "controller" and ts_first_controller <= _G.nodes[_u]["ts_step"] <= ts_end:
                return True
            else:
                return False

        # Correct for the delta_in_degree
        start_partition = n_prev_plans + n_all_controllers + [n_plan, n_actuator, n_world]
        partition = constrained_topological_sort(G, start=start_partition, in_degree=in_degree,
                                                 constraint_fn=_node_constraint_fn, return_delta=False,
                                                 edge_constraint_fn=None)

        # Split partition into two parts: the part before the horizon, and the part in the horizon
        G_part = G.subgraph(partition).copy()
        n_pre_horizon, n_horizon = [], []
        for n in n_all_controllers:
            if G.nodes[n]["ts_step"] < ts_controller:
                n_pre_horizon.append(n)
            else:
                n_horizon.append(n)

        # Redirect edges to first planner node (may not always be connected if window=1)
        edata = G_part.edges[n_plan, n_controller]
        for n in n_horizon:
            G_part.add_edge(n_plan, n, **edata)
        assert len(G_part.out_edges(n_plan)) == len(n_horizon), "Not all edges redirected"

        # Update planner node attributes
        for n in n_prev_plans + [n_plan]:
            G_part.nodes[n]["inputs"] = {}

        # Store partitioned graph
        Gs[n_plan] = G_part

        print(f"Partitioned planner node {n_plan}: {G_part.number_of_nodes()} nodes, {G_part.number_of_edges()} edges") if verbose else None
        if False:
            # for n in Gs[n_plan]["mpc"].nodes():
            #     Gs[n_plan]["part"].nodes[n]["pruned"] = True
            #     Gs[n_plan]["part"].nodes[n]["edgecolor"] = ecolor.mpc
            #     Gs[n_plan]["part"].nodes[n]["facecolor"] = fcolor.mpc
            # for n in Gs[n_plan]["est"].nodes():
            #     Gs[n_plan]["part"].nodes[n]["pruned"] = True
            #     Gs[n_plan]["part"].nodes[n]["edgecolor"] = ecolor.est
            #     Gs[n_plan]["part"].nodes[n]["facecolor"] = fcolor.est
            # _ = vx300s.show_computation_graph(G_relabel, root=None, xmax=None, draw_pruned=True)

            # Key nodes used to identify the partition
            nodes = [n_plan, n_box, n_arm, n_world_box, n_world_arm, n_world, n_actuator, n_first_controller, n_first_plan]
            for n in nodes:
                G.nodes[n]["pruned"] = True
                G.nodes[n]["edgecolor"] = ecolor.selected
                G.nodes[n]["facecolor"] = fcolor.selected
            _ = vx300s.show_computation_graph(G_part, root=None, xmax=2.0, draw_pruned=True)
    return Gs


def cost_data(ts: float, seq: int, seq_world: int, order=1):
    ndata = {
        "seq": seq,
        "ts_step": ts,
        "ts_sent": ts,
        "pruned": False,
        "super": False,
        "stateful": True,
        "inputs": {"world": {"input_name": "world", 'window': 1}},
        "order": order,  # could be wrong
        "position": (ts, order),
        "color": "yellow",
        "kind": "cost",
        "edgecolor": oc.ewheel["yellow"],
        "facecolor": oc.fwheel["yellow"],
        "alpha": 1.0,
    }
    edata = {'kind': 'world',
             'output': 'world',
             'window': 1,
             'seq': seq_world,
             'ts_sent': ts,
             'ts_recv': ts,
             'stateful': False,
             'pruned': False,
             'color': '#212529',
             'linestyle': '-',
             'alpha': 1.0
             }
    return ndata, edata


def connect_cost(G, n_world, seq):
    ts_world, seq_world = G.nodes[n_world]["ts_step"], G.nodes[n_world]["seq"]
    ndata, edata = cost_data(ts=ts_world, seq=seq, seq_world=seq_world)
    G.add_node(f"cost_{seq}", **ndata)
    G.add_edge(n_world, f"cost_{seq}", **edata)
    if seq > 0:
        edata_state = edata.copy()
        edata_state.update({'kind': 'cost', "stateful": True, "seq": seq - 1})
        G.add_edge(f"cost_{seq - 1}", f"cost_{seq}", **edata_state)


def partition_list(base_list, T):
    """
    Partitions a list into T equal consecutive parts.
    If len(base_list)=N is not perfectly divisible by T, distributes the deficit equally
    over the partitions, preferring to add them to the first ones.

    Args:
    base_list (int): The  list.
    T (int): The number of partitions.

    Returns:
    List[List[int]]: A list of partitions.
    """
    N = len(base_list)

    # Calculate the base size of each partition and the number of leftovers
    partition_size, leftovers = divmod(N, T)

    # Initialize partitions
    partitions = []
    start_index = 0

    for i in range(T):
        # Determine the end index of the current partition
        # Add an extra element to the later partitions if there are leftovers
        end_index = start_index + partition_size + (1 if i < leftovers else 0)

        # Slice the partition and add it to the partitions list
        partitions.append(base_list[start_index:end_index])

        # Update the start index for the next partition
        start_index = end_index

    return partitions


def split_partitions_into_estimator_mpc_partitions(Gs: Dict[str, nx.DiGraph], num_cost_est=9, num_cost_mpc=11):
    Gs_est = {}
    Gs_mpc = {}
    # 1. Separate the partitioned graph into estimator and MPC subgraphs.
    # 2. Recalibrate sequence numbers and time steps for all nodes in the partition relative to the current planner node.
    # 3. Relabel nodes in the partition for clarity and consistency.
    # 4. Identify and process world nodes for connecting cost nodes.
    # 5. Integrate cost nodes into the graph, with appropriate node and edge data.
    for n_plan, G in Gs.items():
        ts_start = G.nodes[n_plan]["ts_step"]
        G, mapping = calibrate_graph(G, ts_offset=ts_start, relabel_nodes=False)

        # Split into estimator and MPC partitions
        n_plan_relabeled = mapping[n_plan]
        desc = {n_plan_relabeled}.union(nx.descendants(G, n_plan_relabeled))
        n_mpc = desc.union(nx.ancestors(G, n_plan_relabeled))  # includes previous planner nodes
        n_est = set(G.nodes).difference(desc)  # includes previous planner nodes
        G_mpc = G.subgraph(n_mpc).copy()
        G_est = G.subgraph(n_est).copy()

        # Make sure all mpc controller nodes have window connected edges
        n_controllers = [e[1] for e in G_mpc.out_edges(n_plan_relabeled)]
        e_controllers = {u: edata for n_ctrl in n_controllers for u, v, edata in G_mpc.in_edges(n_ctrl, data=True) if u not in n_controllers}
        for n_ctrl in n_controllers:
            G_mpc.add_edges_from([(u, n_ctrl, edata) for u, edata in e_controllers.items()])

        # Calibrate the estimator and MPC partitions
        # G_mpc = calibrate_graph(G_mpc, ts_offset=ts_start, relabel_nodes=True)
        # G_est = calibrate_graph(G_est, ts_offset=ts_start, relabel_nodes=True)
        Gs_est[n_plan] = G_est
        Gs_mpc[n_plan] = G_mpc

        # Add cost nodes
        for (num_cost, G_split) in zip((num_cost_mpc, num_cost_est), (G_mpc, G_est)):
            # sort world_nodes by seq
            n_worlds = [n for n in G_split.nodes if G_split.nodes[n]["kind"] == "world"]
            n_worlds = sorted(n_worlds, key=lambda n: G_split.nodes[n]["seq"])

            worlds_partitions = partition_list(list(reversed(n_worlds)), num_cost)
            try:
                n_connect_to_cost = [l[0] for l in reversed(worlds_partitions)]
            except IndexError:
                print("SELECT smaller num_cost_mpc, num_cost_est.(")
                raise IndexError

            # Add cost nodes
            for i, n_world in enumerate(n_connect_to_cost):
                connect_cost(G_split, n_world, i)

    return Gs_est, Gs_mpc


def make_supergraph_and_timings(Gs_raw: List[nx.DiGraph], root: str, num_cost: int, supergraph_mode: str = "MCS", backtrack: int = 30, progress_bar: bool = False)\
        -> Tuple[nx.DiGraph, rex.supergraph.Timings]:
    # Get all graphs
    Gs = []
    for i, G in enumerate(Gs_raw):
        # Trace root node (not pruned yet)
        G_traced = rex.supergraph.trace_root(G, root=root, seq=-1)

        # Prune unused nodes (not in computation graph of traced root)
        G_traced_pruned = rex.supergraph.prune_graph(G_traced)
        Gs.append(G_traced_pruned)

    if supergraph_mode == "MCS":
        # Define initial supergraph
        S_init, _ = sg.as_supergraph(Gs[0], leaf_kind=root, sort=[f"{root}_0"])

        # Run evaluation
        S, S_init_to_S, Gs_monomorphism = sg.grow_supergraph(
            Gs,
            S_init,
            root,
            combination_mode="linear",
            backtrack=backtrack,
            progress_fn=None,
            progress_bar=progress_bar,
            validate=False,

        )
    elif supergraph_mode == "topological":
        from supergraph.evaluate import baselines_S

        S, _ = baselines_S(Gs, root)
        S_init_to_S = {n: n for n in S.nodes()}
        Gs_monomorphism = sg.evaluate_supergraph(Gs, S, progress_bar=progress_bar)
    elif supergraph_mode == "generational":
        from supergraph.evaluate import baselines_S

        _, S = baselines_S(Gs, root)
        S_init_to_S = {n: n for n in S.nodes()}
        Gs_monomorphism = sg.evaluate_supergraph(Gs, S, progress_bar=progress_bar)
    else:
        raise ValueError(f"Unknown supergraph mode '{supergraph_mode}'.")

    # Get timings
    timings = []
    for i, (G, G_monomorphism) in enumerate(zip(Gs_raw, Gs_monomorphism)):
        t = rex.supergraph.get_timings(S, G, G_monomorphism, num_root_steps=num_cost, root="cost")
        timings.append(t)

    timings = jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *timings)
    return S, timings


def get_planner_window(Gs_est: Dict[str, nx.DiGraph]) -> int:
    num_planners = []
    for n_plan, G in Gs_est.items():
        num = len([n for n in G.nodes if G.nodes[n]["kind"] == "planner"])
        num_planners.append(num)
    window_prev_plans = max(num_planners)
    return window_prev_plans
