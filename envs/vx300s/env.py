from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
import jax
import rex.jumpy as rjp
from flax import struct
from flax.core import FrozenDict

from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, WARN
from rex.graph import BaseGraph
from rex.base import StepState, GraphState, RexStepReturn, RexResetReturn
from rex.env import BaseEnv
from rex.node import Node
from rex.agent import Agent as BaseAgent
from rex.spaces import Box


@struct.dataclass
class SupervisorParams:
    # Other parameters
    goalpos_dist: jp.float32 = struct.field(pytree_node=True, default_factory=lambda: jp.float32(0.25))
    home_boxpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0.35, 0.0, 0.051], jp.float32))
    home_boxyaw: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0.0], jp.float32))
    home_jpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0., 0., 0., 0*-3.14/4, 3.1415 / 2, 0.], jp.float32))
    # Observation parameters
    min_jpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([-3.14159, -1.8500, -1.7628, -3.14159, -1.8675, -3.14159], jp.float32))
    max_jpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([3.14159, 1.2566, 1.6057, 3.14159, 2.2340, 3.14159], jp.float32))
    min_boxpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([-2.0, -2.0, 0.], jp.float32))
    max_boxpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([2.0, 2.0, 0.5], jp.float32))
    min_boxorn: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([-3.14, -3.14, -3.14], jp.float32))
    max_boxorn: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([3.14, 3.14, 3.14], jp.float32))
    # Action parameters
    # horizon: jp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.array(4, jp.int32))
    # max_jvel: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: 0.35*3.14159*jp.ones((6,), dtype=jp.float32))
    # dt: jp.float32 = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.15, jp.float32))
    # dt_brax: jp.float32 = struct.field(pytree_node=True, default_factory=lambda: jp.array(0.015, jp.float32))
    # pipeline: str = struct.field(pytree_node=False, default_factory=lambda: "generalized")


@struct.dataclass
class PlannerOutput:
    jpos: jp.ndarray  # (6,) --> jpos
    jvel: jp.ndarray  # (horizon, 6) --> [jvel[0], jvel[1], ...]
    timestamps: jp.ndarray  # (horizon+1,) --> jpos(timestamps[0])=jpos, jpos(timestamps[1])=jpos+jvel[0], ...


@struct.dataclass
class SupervisorState:
    goalpos: jp.ndarray
    goalyaw: jp.ndarray


@struct.dataclass
class SupervisorOutput:
    goalpos: jp.ndarray
    goalyaw: jp.ndarray


@struct.dataclass
class ActuatorOutput:
    jpos: jp.ndarray


@struct.dataclass
class ArmOutput:
    jpos: jp.ndarray
    eepos: jp.ndarray
    eeorn: jp.ndarray

    @property
    def orn_to_3x3(self):
        q = self.eeorn
        d = jp.dot(q, q)
        x, y, z, w = q
        s = 2 / d
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        return jp.array([
            jp.array([1 - (yy + zz), xy - wz, xz + wy]),
            jp.array([xy + wz, 1 - (xx + zz), yz - wx]),
            jp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
        ])


@struct.dataclass
class BoxOutput:
    boxpos: jp.ndarray
    boxorn: jp.ndarray

    @property
    def wrapped_yaw(self):
        """Get the wrapped yaw from boxorn (quaternion xyzw)"""
        boxorn = self.boxorn
        boxyaw = jp.arctan2(2 * (boxorn[3] * boxorn[2] + boxorn[0] * boxorn[1]), 1 - 2 * (boxorn[1] ** 2 + boxorn[2] ** 2))
        return boxyaw

    @property
    def orn_to_3x3(self):
        q = self.boxorn
        d = jp.dot(q, q)
        x, y, z, w = q
        s = 2 / d
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        return jp.array([
            jp.array([1 - (yy + zz), xy - wz, xz + wy]),
            jp.array([xy + wz, 1 - (xx + zz), yz - wx]),
            jp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
        ])


def get_wrapped_yaw(boxorn):
    """Get the wrapped yaw from boxorn (quaternion xyzw)"""
    boxyaw = jp.arctan2(2 * (boxorn[3] * boxorn[2] + boxorn[0] * boxorn[1]), 1 - 2 * (boxorn[1] ** 2 + boxorn[2] ** 2))
    return boxyaw


def get_next_jpos(plan, ts):
    """Interpolate next joint position"""
    jpos = plan.jpos
    jvel = plan.jvel
    timestamps = plan.timestamps
    next_jpos = jpos

    # Loop body function
    def loop_body(idx, _next_jpos):
        last_timestamp = timestamps[idx - 1]
        current_timestamp = timestamps[idx]
        current_delta = jvel[idx - 1]

        # Calculate the interpolation factor alpha
        alpha = (ts - last_timestamp) / (current_timestamp - last_timestamp + jp.float32(1e-6))
        alpha = jp.clip(alpha, jp.float32(0), jp.float32(1))

        # Determine interpolated delta
        interpolated_delta = alpha * current_delta * (current_timestamp - last_timestamp)

        # Update _next_jpos
        _next_jpos = _next_jpos + interpolated_delta
        return _next_jpos

    # Perform loop
    next_jpos = jumpy.lax.fori_loop(1, len(timestamps), loop_body, next_jpos)
    return next_jpos


def update_global_plan(timestamps_global, jvel_global, timestamps, jvel):
    idx = jp.argmax(timestamps_global > timestamps[0])
    timestamps_global = rjp.dynamic_update_slice(timestamps_global, timestamps, (idx,))
    jvel_global = rjp.dynamic_update_slice(jvel_global, jvel, (idx, 0))
    return timestamps_global, jvel_global


# @checkify.checkify
def get_global_plan(plan_history: PlannerOutput, debug: bool = False):
    num_plans = plan_history.jvel.shape[0]
    horizon = plan_history.jvel.shape[1]
    other_dims = plan_history.jvel.shape[2:]

    # Initialize global plan
    jpos_global = plan_history.jpos[0]
    jvel_global = jp.zeros((num_plans * horizon + num_plans - 1,) + other_dims, dtype=jp.float32)
    timestamps_global = jp.amax(plan_history.timestamps[-1]) + 1e-6*jp.arange(1, num_plans*horizon+num_plans+1, dtype=jp.float32)

    # Update global plan
    for i in range(num_plans):
        timestamps_global, jvel_global = update_global_plan(timestamps_global, jvel_global, plan_history.timestamps[i], plan_history.jvel[i])

    # Return global plan
    plan_global = PlannerOutput(jpos_global, jvel_global, timestamps_global)

    if debug:
        for (check_jpos, check_ts) in zip(plan_history.jpos[1:], plan_history.timestamps[1:, 0]):
            check_jpos_global = get_next_jpos(plan_global, check_ts)
            equal = jp.all(jax.numpy.isclose(check_jpos_global, check_jpos))
            jax.debug.print("EQUAL?={equal} {check_jpos_global} vs {check_jpos}", equal=equal, check_jpos_global=check_jpos_global, check_jpos=check_jpos)
            # checkify.check(equal, "NOT EQUAL! {check_jpos_global} vs {check_jpos}", check_jpos_global=check_jpos_global, check_jpos=check_jpos)  # convenient but effectful API
    return plan_global


class Controller(Node):
    def __init__(self, planner, *args, **kwargs):
        self._planner = planner
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        planner_output = self._planner.default_output(rng, graph_state)
        return ActuatorOutput(jpos=planner_output.jpos)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Get global plan
        plan_global = get_global_plan(inputs["planner"].data, debug=False)

        # Prepare output
        ts = step_state.ts

        # Get next joint position
        next_jpos = get_next_jpos(plan_global, ts)
        actuator_output = ActuatorOutput(jpos=next_jpos)
        return new_step_state, actuator_output


class Supervisor(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> SupervisorParams:
        """Default params of the root."""
        return SupervisorParams()

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> SupervisorState:
        """Default state of the root."""
        # Place goal on semi-circle around home position with radius goalpos_dist
        params: SupervisorParams = graph_state.nodes["supervisor"].params
        rng_planner, rng_goal = jax.random.split(rng, num=2)
        goalyaw = jp.array([jumpy.random.uniform(rng_goal, low=jp.float32(0), high=jp.pi/2)])
        angle = jumpy.random.uniform(rng_goal, low=jp.float32(0), high=jp.pi)
        dx = jp.sin(angle)*params.goalpos_dist
        dy = jp.cos(angle)*params.goalpos_dist
        goalpos = params.home_boxpos[:2] + jp.array([dx, dy])
        # goalpos = params.home_boxpos[:2] + jp.array([0.05, 0.2])  # todo: goal hardcoded here.
        # goalpos = params.home_boxpos[:2] + jp.array([-0.10, 0.45])  # todo: goal hardcoded here.
        return SupervisorState(goalpos=goalpos, goalyaw=goalyaw)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> SupervisorOutput:
        """Default output of the root."""
        state = graph_state.nodes["supervisor"].state
        return SupervisorOutput(goalpos=state.goalpos, goalyaw=state.goalyaw)


# todo: add to BaseGraph?
# def init_graph_state(rng: jp.ndarray = None,
#                      graph_state: GraphState = None,
#                      starting_step: jp.int32 = jp.int32(0),
#                      starting_eps: jp.int32 = jp.int32(0),
#                      randomize_eps: bool = True)


class Vx300sEnv(BaseEnv):
    supervisor_cls = Supervisor
    controller_cls = Controller

    def __init__(
            self,
            graph: BaseGraph,
            max_steps: int = 100,
            name: str = "vx300s-v0"
    ):
        super().__init__(graph, max_steps, name=name)

        # Required for step and reset functions
        assert "world" in self.graph.nodes, "Environment requires a world node."
        self.world = self.graph.nodes["world"]
        self.supervisor = self.graph.root
        self.nodes = {node.name: node for _, node in self.graph.nodes.items() if node.name != self.world.name}
        self.nodes_world_and_supervisor = self.graph.nodes_and_root

        from envs.vx300s.cem_brax import box_pushing_cost
        self._cost_fn = box_pushing_cost

    # todo: automate _get_graph_state -> find order of reset iteratively.
    def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
        """Get the graph state."""
        # Prepare new graph state
        rng, rng_eps = jumpy.random.split(rng, num=2)
        starting_step = jp.int32(0)
        starting_eps = jumpy.random.choice(rng, self.graph.max_eps(), shape=()) if graph_state is None else graph_state.eps
        new_nodes = dict()
        graph_state = GraphState(nodes=new_nodes)

        # For every node, prepare the initial stepstate
        rng, rng_planner, rng_world = jumpy.random.split(rng, num=3)

        # Get new step_state
        def get_step_state(node: Node, _rng: jp.ndarray, _graph_state) -> StepState:
            """Get new step_state for a node."""
            rng_params, rng_state, rng_step = jumpy.random.split(_rng, num=3)
            params = node.default_params(rng_params, _graph_state)
            # Already add params here, as the state may depend on them
            new_nodes[node.name] = StepState(rng=rng_step, params=params, state=None, inputs=None)
            state = node.default_state(rng_state, _graph_state)
            return new_nodes[node.name].replace(state=state)

        # Step_state root & world (root must be reset before world, as the world may copy some params from the root)
        new_nodes[self.supervisor.name] = get_step_state(self.supervisor, rng_planner, graph_state)
        new_nodes[self.world.name] = get_step_state(self.world, rng_world, graph_state)

        # Get new step_state for other nodes in arbitrary order
        rng, *rngs = jumpy.random.split(rng, num=len(self.nodes) + 1)
        for (name, n), rng_n in zip(self.nodes.items(), rngs):
            # Replace step state in graph state
            new_nodes[name] = get_step_state(n, rng_n, graph_state)

        rng, *rngs = jumpy.random.split(rng, num=len(self.graph.nodes_and_root) + 1)
        for (name, n), rng_n in zip(self.graph.nodes_and_root.items(), rngs):
            new_nodes[name] = new_nodes[name].replace(inputs=n.default_inputs(rng_n, graph_state))
        return GraphState(eps=starting_eps, step=starting_step, nodes=FrozenDict(new_nodes))

    def reset(self, rng: jp.ndarray, graph_state: GraphState = None):
        """Reset environment."""
        new_graph_state = self._get_graph_state(rng, graph_state)

        # Reset nodes
        rng, *rngs = jumpy.random.split(rng, num=len(self.nodes_world_and_supervisor) + 1)
        # todo: we reset with step state once more here.
        [n.reset(rng_reset, new_graph_state) for (n, rng_reset) in zip(self.nodes_world_and_supervisor.values(), rngs)]

        # Reset environment to get initial step_state (runs up-until the first step)
        graph_state, step_state = self.graph.reset(new_graph_state)

        # Get cost
        cost, info = self._get_cost(graph_state)
        return graph_state, step_state, info

    def step(self, graph_state: GraphState, action):
        """Perform step transition in environment."""
        # Update step_state (if necessary)
        step_state = self.supervisor.get_step_state(graph_state)

        # Apply step and receive next step_state
        output = SupervisorOutput(goalpos=step_state.state.goalpos, goalyaw=step_state.state.goalyaw)
        graph_state, step_state = self.graph.step(graph_state, step_state, output)

        # Get cost
        cost, info = self._get_cost(graph_state)

        # Determine done flag
        terminated = self._is_terminal(graph_state)
        truncated = graph_state.step >= self.max_steps

        return graph_state, step_state, -cost, terminated, truncated, info

    def _is_terminal(self, graph_state: GraphState) -> bool:
        return False

    def _get_cost(self, graph_state: GraphState):
        step_state = self.supervisor.get_step_state(graph_state)
        # Calculate cost
        boxpos = step_state.inputs["boxsensor"][-1].data.boxpos
        eepos = step_state.inputs["armsensor"][-1].data.eepos
        goalpos = step_state.state.goalpos
        eeorn = step_state.inputs["armsensor"][-1].data.eeorn
        cp = graph_state.nodes["planner"].params.cost_params
        cost, info = self._cost_fn(cp, boxpos, eepos, goalpos, eeorn)
        return cost, info

    # def _get_cost(self, step_state: StepState):
    #     boxpos = step_state.inputs["boxsensor"][-1].data.boxpos
    #     eepos = step_state.inputs["armsensor"][-1].data.eepos
    #     goalpos = step_state.state.goalpos
    #
    #     rot_mat = step_state.inputs["armsensor"][-1].data.orn_to_3x3
    #     ee_to_goal = goalpos - eepos[:2]
    #     box_to_goal = goalpos - boxpos[:2]
    #
    #     # if dot(ee_yaxis (in global), ee_to_goal (in global))==0 --> ee_yaxis = perpedicular to box
    #     # ee_yaxis is parallel to gripper_bar axis
    #     norm_ee_to_goal = ee_to_goal / jp.linalg.safe_norm(ee_to_goal)
    #     cost_orn = jp.abs(jp.dot(rot_mat[:2, 1], norm_ee_to_goal))
    #
    #     # ee_xaxis points in -z if ee is oriented downward
    #     # norm_box_to_goal = box_to_goal / math.safe_norm(box_to_goal)
    #     # target_ee_xaxis = jnp.concatenate([norm_box_to_goal, jnp.array([-5.0])])  # making this more negative forces the ee to be pointing downward
    #     # norm_target_ee_xaxis = target_ee_xaxis / math.safe_norm(target_ee_xaxis)
    #     # cost_down = (1-jnp.dot(rot_mat[:3, 0], norm_target_ee_xaxis))  # Here, the dot is 1 if ee_xaxis == target_ee_axis
    #     cost_down = jp.abs(rot_mat[:2, 0]).sum()
    #
    #     # Distances in xy-plane
    #     box_to_ee = (eepos - boxpos)[:2]
    #     box_to_goal = (goalpos - boxpos[:2])
    #     dist_box_to_ee = jp.linalg.safe_norm(box_to_ee)
    #     dist_box_to_goal = jp.linalg.safe_norm(box_to_goal)
    #     cost_align = (jp.sum(box_to_ee * box_to_goal) / (dist_box_to_ee * dist_box_to_goal) + 1)
    #
    #     # Force cost
    #     cost_force = 0.
    #
    #     cost_z = jp.abs(eepos[2] - 0.02)
    #     cost_near = jp.abs(jp.linalg.safe_norm((boxpos - eepos)[:2]) - 0.06)
    #     cost_dist = jp.linalg.safe_norm(boxpos[:2] - goalpos)
    #     cost_ctrl = 0.
    #
    #     cm = cost_dist
    #
    #     # Weight all costs
    #     alpha = 1 / (1 + 2.0 * jp.abs(cost_down + cost_orn))
    #     cost_orn = 3.0 * cost_orn
    #     cost_down = 3.0 * cost_down
    #     cost_force = 0.2 * cost_force
    #     cost_align = 2.0 * cost_align
    #     cost_z = 1.0 * cost_z * alpha
    #     cost_near = 10.0 * cost_near * alpha
    #     cost_dist = 20.0 * cost_dist * alpha
    #     cost_ctrl = 0.1 * cost_ctrl
    #
    #     total_cost = cost_ctrl + cost_z + cost_near + cost_dist + cost_orn + cost_down + cost_align + cost_force
    #     info = {"cm": cm*100, "cost": total_cost, "cost_orn": cost_orn, "cost_force": cost_force, "cost_down": cost_down,
    #             "cost_align": cost_align, "cost_z": cost_z, "cost_near": cost_near, "cost_dist": cost_dist,
    #             "cost_ctrl": cost_ctrl, "alpha": alpha}
    #     return total_cost, info

