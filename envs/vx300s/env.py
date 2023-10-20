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
class PlannerParams:
    # Other parameters
    goalpos_dist: jp.float32 = struct.field(pytree_node=True, default_factory=lambda: jp.float32(0.15))
    boxpos_home: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0.45, 0.0], jp.float32))
    home_jpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.zeros((6,), jp.float32))
    # Observation parameters
    min_jpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([-3.14159, -1.8500, -1.7628, -3.14159, -1.8675, -3.14159], jp.float32))
    max_jpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([3.14159, 1.2566, 1.6057, 3.14159, 2.2340, 3.14159], jp.float32))
    min_boxpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([-2.0, -2.0, 0.], jp.float32))
    max_boxpos: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([2.0, 2.0, 0.5], jp.float32))
    min_boxorn: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array( [-3.14, -3.14, -3.14], jp.float32))
    max_boxorn: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array( [3.14, 3.14, 3.14], jp.float32))
    # Action parameters
    horizon: jp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.array(5, jp.int32))
    max_jvel: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: 0.2*jp.array([3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159], jp.float32))


@struct.dataclass
class PlannerOutput:
    jpos: jp.ndarray  # (6,) --> jpos
    jvel: jp.ndarray  # (horizon, 6) --> [jvel[0], jvel[1], ...]
    timestamps: jp.ndarray  # (horizon+1,) --> jpos(timestamps[0])=jpos, jpos(timestamps[1])=jpos+jvel[0], ...


@struct.dataclass
class PlannerState:
    goalpos: jp.ndarray
    last_plan: PlannerOutput = struct.field(pytree_node=True, default_factory=lambda: None)


@struct.dataclass
class ActuatorOutput:
    jpos: jp.ndarray


@struct.dataclass
class ArmOutput:
    jpos: jp.ndarray
    eepos: jp.ndarray
    eeorn: jp.ndarray


@struct.dataclass
class BoxOutput:
    boxpos: jp.ndarray
    boxorn: jp.ndarray


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


class Planner(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> PlannerParams:
        """Default params of the root."""
        return PlannerParams()

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> PlannerState:
        """Default state of the root."""
        # Place goal on semi-circle around home position with radius goalpos_dist
        params: PlannerParams = graph_state.nodes["planner"].params
        rng_planner, rng_goal = jax.random.split(rng, num=2)
        angle = jumpy.random.uniform(rng_goal, low=jp.float32(0), high=jp.pi)
        dx = jp.sin(angle)*params.goalpos_dist
        dy = jp.cos(angle)*params.goalpos_dist
        goalpos = params.boxpos_home + jp.array([dx, dy])
        return PlannerState(last_plan=self.default_output(rng, graph_state), goalpos=goalpos)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> PlannerOutput:
        """Default output of the root."""
        params = graph_state.nodes["planner"].params
        jpos = jp.zeros(params.home_jpos.shape, dtype=jp.float32)
        jvel =  jp.zeros((params.horizon, params.home_jpos.shape[0]), dtype=jp.float32)
        timestamps = (1/self.rate) * jp.arange(0, params.horizon+1, dtype=jp.float32)
        return PlannerOutput(jpos=jpos, jvel=jvel, timestamps=timestamps)

    # def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
    #     """Reset the root."""
    #     rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
    #     params = self.default_params(rng_params, graph_state)
    #     state = self.default_state(rng_state, graph_state)
    #     inputs = self.default_inputs(rng_inputs, graph_state)
    #     return StepState(rng=rng_step, params=params, state=state, inputs=inputs)


class Vx300sEnv(BaseEnv):
    planner_cls: Planner = Planner
    controller_cls: Controller = Controller

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
        self.planner = self.graph.root
        self.nodes = {node.name: node for _, node in self.graph.nodes.items() if node.name != self.world.name}
        self.nodes_world_and_planner = self.graph.nodes_and_root

    # def observation_space(self, params: PlannerParams = None):
    #     """Observation space of the environment."""
    #     if params is not None:
    #         self.log("reloading", "Current implementation does not support custom parametrized observation spaces.")
    #     params = self.planner.default_params(jumpy.random.PRNGKey(0))
    #     inputs = {u.input_name: u for u in self.planner.inputs}
    #
    #     # Prepare
    #     win_armsensor = inputs["armsensor"].window
    #     low = jp.array(params.min_jpos.tolist() * win_armsensor + params.min_boxpos.tolist() + params.min_boxorn.tolist(), dtype=jp.float32)
    #     high = jp.array(params.max_jpos.tolist() * win_armsensor + params.max_boxpos.tolist() + params.max_boxorn.tolist(), dtype=jp.float32)
    #     return Box(low=low, high=high, dtype=jp.float32)
    #
    # def action_space(self, params: PlannerParams = None):
    #     """Action space of the environment."""
    #     if params is not None:
    #         self.log("reloading", "Current implementation does not support custom parametrized action spaces.")
    #     params = self.planner.default_params(jumpy.random.PRNGKey(0))
    #     high = jp.array([[params.max_jvel]]*params.horizon, dtype=jp.float32)
    #     return Box(low=-high, high=high, shape=high.shape, dtype=jp.float32)

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
        new_nodes[self.planner.name] = get_step_state(self.planner, rng_planner, graph_state)
        new_nodes[self.world.name] = get_step_state(self.world, rng_world, graph_state)

        # Get new step_state for other nodes in arbitrary order
        rng, *rngs = jumpy.random.split(rng, num=len(self.nodes) + 1)
        for (name, n), rng_n in zip(self.nodes.items(), rngs):
            # Replace step state in graph state
            new_nodes[name] = get_step_state(n, rng_n, graph_state)

        # Reset nodes
        rng, *rngs = jumpy.random.split(rng, num=len(self.nodes_world_and_planner) + 1)
        [n.reset(rng_reset, graph_state) for (n, rng_reset) in zip(self.nodes_world_and_planner.values(), rngs)]
        return GraphState(eps=starting_eps, step=starting_step, nodes=FrozenDict(new_nodes))

    def reset(self, rng: jp.ndarray, graph_state: GraphState = None):
        """Reset environment."""
        new_graph_state = self._get_graph_state(rng, graph_state)

        # Reset environment to get initial step_state (runs up-until the first step)
        graph_state, step_state = self.graph.reset(new_graph_state)

        # Get observation
        cost, info = self._get_cost(step_state)
        return graph_state, step_state, info

    def step(self, graph_state: GraphState, plan: PlannerOutput):
        """Perform step transition in environment."""
        # Update step_state (if necessary)
        step_state = self.planner.get_step_state(graph_state)
        new_step_state = step_state.replace(state=step_state.state.replace(last_plan=plan))

        # Apply step and receive next step_state
        graph_state, step_state = self.graph.step(graph_state, new_step_state, plan)

        # Calculate distance
        cost, info = self._get_cost(step_state)

        # Determine done flag
        terminated = self._is_terminal(graph_state)
        truncated = graph_state.step >= self.max_steps

        return graph_state, step_state, -cost, terminated, truncated, info

    def _is_terminal(self, graph_state: GraphState) -> bool:
        return False

    def _get_cost(self, step_state: StepState):
        boxpos = step_state.inputs["boxsensor"][-1].data.boxpos
        eepos = step_state.inputs["armsensor"][-1].data.eepos
        goalpos = step_state.state.goalpos
        cost_z = 1.0 * jumpy.numpy.linalg.safe_norm(eepos[2] - 0.075)
        cost_near = 0.4 * jumpy.numpy.linalg.safe_norm((boxpos - eepos)[:2])
        cost_dist = 4.0 * jumpy.numpy.linalg.safe_norm(boxpos[:2] - goalpos)
        cost = cost_z + cost_near + cost_dist
        info = {"cost_z": cost_z, "cost_near": cost_near, "cost_dist": cost_dist, "cost": cost}
        return cost, info

