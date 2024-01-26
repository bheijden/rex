from typing import Any, Dict, Tuple, Union
from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, WARN
from rex.graph import BaseGraph
from rex.base import StepState, GraphState, RexStepReturn, RexResetReturn
from rex.env import BaseEnv
from rex.node import Node
from rex.spaces import Box
from rex.utils import timer


@struct.dataclass
class SupervisorParams:
    # Other parameters
    goalpos_dist: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.25)
    home_boxpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([0.32, -0.15, 0.051], jnp.float32))
    home_boxyaw: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([0.0], jnp.float32))
    # home_boxpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([0.35, 0.0, 0.051], jnp.float32))
    # home_jpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([0., 0., 0., 0*-3.14/4, 3.1415 / 2, 0.], jnp.float32))
    home_jpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([-0.723, 0.396, 0.136, 0, 1.04, 0.88], jnp.float32))
    # Observation parameters
    min_jpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([-3.14159, -1.8500, -1.7628, -3.14159, -1.8675, -3.14159], jnp.float32))
    max_jpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([3.14159, 1.2566, 1.6057, 3.14159, 2.2340, 3.14159], jnp.float32))
    min_boxpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([-2.0, -2.0, 0.], jnp.float32))
    max_boxpos: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([2.0, 2.0, 0.5], jnp.float32))
    min_boxorn: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([-3.14, -3.14, -3.14], jnp.float32))
    max_boxorn: jax.typing.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: jnp.array([3.14, 3.14, 3.14], jnp.float32))


@struct.dataclass
class PlannerOutput:
    jpos: jax.typing.ArrayLike  # (6,) --> jpos
    jvel: jax.typing.ArrayLike  # (horizon, 6) --> [jvel[0], jvel[1], ...]
    timestamps: jax.typing.ArrayLike  # (horizon+1,) --> jpos(timestamps[0])=jpos, jpos(timestamps[1])=jpos+jvel[0], ...


@struct.dataclass
class SupervisorState:
    goalpos: jax.typing.ArrayLike
    goalyaw: jax.typing.ArrayLike


@struct.dataclass
class SupervisorOutput:
    goalpos: jax.typing.ArrayLike
    goalyaw: jax.typing.ArrayLike


@struct.dataclass
class ActuatorOutput:
    jpos: jax.typing.ArrayLike


@struct.dataclass
class ArmOutput:
    jpos: jax.typing.ArrayLike
    eepos: jax.typing.ArrayLike
    eeorn: jax.typing.ArrayLike

    @property
    def orn_to_3x3(self):
        """Get the rotation matrix from eeorn (quaternion xyzw)"""
        return self.static_orn_to_3x3(self.eeorn)

    @staticmethod
    def static_orn_to_3x3(orn):
        """Get the rotation matrix from orn (quaternion xyzw)"""
        q = orn
        d = jnp.dot(q, q)
        x, y, z, w = q
        s = 2 / d
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        return jnp.array([
            jnp.array([1 - (yy + zz), xy - wz, xz + wy]),
            jnp.array([xy + wz, 1 - (xx + zz), yz - wx]),
            jnp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
        ])


@struct.dataclass
class BoxOutput:
    boxpos: jax.typing.ArrayLike
    boxorn: jax.typing.ArrayLike

    @property
    def orn_to_3x3(self):
        """Get the rotation matrix from boxorn (quaternion xyzw)"""
        return self.static_orn_to_3x3(self.boxorn)

    @property
    def wrapped_yaw(self):
        """Get the wrapped yaw from boxorn (quaternion xyzw)"""
        return self.static_wrapped_yaw(self.boxorn)

    @staticmethod
    def static_orn_to_3x3(orn):
        """Get the rotation matrix from orn (quaternion xyzw)"""
        q = orn
        d = jnp.dot(q, q)
        x, y, z, w = q
        s = 2 / d
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        return jnp.array([
            jnp.array([1 - (yy + zz), xy - wz, xz + wy]),
            jnp.array([xy + wz, 1 - (xx + zz), yz - wx]),
            jnp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
        ])

    @staticmethod
    def static_wrapped_yaw(boxorn):
        """Get the wrapped yaw from boxorn (quaternion xyzw)"""
        rot = BoxOutput.static_orn_to_3x3(boxorn)
        # Remove z axis from rotation matrix
        axis_idx = 2  # Upward pointing axis of robot base
        z_idx = jnp.argmax(jnp.abs(rot[axis_idx, :]), axis=0)  # Take absolute value, if axis points downward.
        # Calculate angle
        tmp = rot[[i for i in range(2) if i !=axis_idx], :]
        rot_red = jnp.zeros((2, 2), dtype=jnp.float32)
        rot_red = rot_red + tmp[:, [0, 1]]*(z_idx == 2)
        rot_red = rot_red + tmp[:, [1, 2]]*(z_idx == 0)
        rot_red = rot_red + tmp[:, [0, 2]]*(z_idx == 1)
        s = jnp.sign(jnp.take(rot[axis_idx], z_idx))
        c1 = (s > 0) * (z_idx == 1)
        c2 = (s < 0) * (z_idx != 1)
        c = jnp.logical_or(c1, c2)
        rot_red = (1-c) * rot_red + c * rot_red @ jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
        th_cos = rot_red[0, 0]
        th_sin = rot_red[1, 0]
        th = jnp.arctan2(th_sin, th_cos)
        yaw = th % (jnp.pi / 2)
        return yaw



        # rot = self.orn_to_3x3
        # # Remove z axis from rotation matrix
        # axis_idx = 2  # Upward pointing axis of robot base
        # z_idx = jnp.argmax(jnp.abs(rot[axis_idx, :]), axis=0)  # Take absolute value, if axis points downward.
        # # Calculate angle
        # tmp = rot[[i for i in range(2) if i !=axis_idx], :]
        # rot_red = jnp.zeros((2, 2), dtype=jnp.float32)
        # rot_red = rot_red + tmp[:, [0, 1]]*(z_idx == 2)
        # rot_red = rot_red + tmp[:, [1, 2]]*(z_idx == 0)
        # rot_red = rot_red + tmp[:, [0, 2]]*(z_idx == 1)
        # s = jnp.sign(jnp.take(rot[axis_idx], z_idx))
        # c1 = (s > 0) * (z_idx == 1)
        # c2 = (s < 0) * (z_idx != 1)
        # c = jnp.logical_or(c1, c2)
        # rot_red = (1-c) * rot_red + c * rot_red @ jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
        # th_cos = rot_red[0, 0]
        # th_sin = rot_red[1, 0]
        # th = jnp.arctan2(th_sin, th_cos)
        # yaw = th % (jnp.pi / 2)
        # return yaw

    # @property
    # def orn_to_3x3(self):
    #     q = self.boxorn
    #     d = jnp.dot(q, q)
    #     x, y, z, w = q
    #     s = 2 / d
    #     xs, ys, zs = x * s, y * s, z * s
    #     wx, wy, wz = w * xs, w * ys, w * zs
    #     xx, xy, xz = x * xs, x * ys, x * zs
    #     yy, yz, zz = y * ys, y * zs, z * zs
    #
    #     return jnp.array([
    #         jnp.array([1 - (yy + zz), xy - wz, xz + wy]),
    #         jnp.array([xy + wz, 1 - (xx + zz), yz - wx]),
    #         jnp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
    #     ])


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
        alpha = (ts - last_timestamp) / (current_timestamp - last_timestamp + 1e-6)
        alpha = jnp.clip(alpha, 0, 1)

        # Determine interpolated delta
        interpolated_delta = alpha * current_delta * (current_timestamp - last_timestamp)

        # Update _next_jpos
        _next_jpos = _next_jpos + interpolated_delta
        return _next_jpos

    # Perform loop
    next_jpos = jax.lax.fori_loop(1, len(timestamps), loop_body, next_jpos)
    return next_jpos


def update_global_plan(timestamps_global, jvel_global, timestamps, jvel):
    idx = jnp.argmax(timestamps_global > timestamps[0])
    timestamps_global = jax.lax.dynamic_update_slice(timestamps_global, timestamps, jnp.array((idx,), dtype=jnp.int32))
    jvel_global = jax.lax.dynamic_update_slice(jvel_global, jvel, jnp.array((idx, 0), jnp.int32))
    return timestamps_global, jvel_global


# @checkify.checkify
def get_global_plan(plan_history: PlannerOutput, debug: bool = False):
    num_plans = plan_history.jvel.shape[0]
    horizon = plan_history.jvel.shape[1]
    other_dims = plan_history.jvel.shape[2:]

    # Initialize global plan
    jpos_global = plan_history.jpos[0]
    jvel_global = jnp.zeros((num_plans * horizon + num_plans - 1,) + other_dims, dtype=jnp.float32)
    timestamps_global = jnp.amax(plan_history.timestamps[-1]) + 1e-6*jnp.arange(1, num_plans*horizon+num_plans+1, dtype=jnp.float32)

    # Update global plan
    for i in range(num_plans):
        timestamps_global, jvel_global = update_global_plan(timestamps_global, jvel_global, plan_history.timestamps[i], plan_history.jvel[i])

    # Return global plan
    plan_global = PlannerOutput(jpos_global, jvel_global, timestamps_global)

    if debug:
        for (check_jpos, check_ts) in zip(plan_history.jpos[1:], plan_history.timestamps[1:, 0]):
            check_jpos_global = get_next_jpos(plan_global, check_ts)
            equal = jnp.all(jax.numpy.isclose(check_jpos_global, check_jpos))
            jax.debug.print("EQUAL?={equal} {check_jpos_global} vs {check_jpos}", equal=equal, check_jpos_global=check_jpos_global, check_jpos=check_jpos)
            # checkify.check(equal, "NOT EQUAL! {check_jpos_global} vs {check_jpos}", check_jpos_global=check_jpos_global, check_jpos=check_jpos)  # convenient but effectful API
    return plan_global


def get_init_plan(last_plan: PlannerOutput, timestamps: jax.typing.ArrayLike) -> PlannerOutput:
    get_next_jpos_vmap = jax.vmap(get_next_jpos, in_axes=(None, 0))
    jpos_timestamps = get_next_jpos_vmap(last_plan, timestamps)
    dt = timestamps[1:] - timestamps[:-1]
    jvel_timestamps = (jpos_timestamps[1:] - jpos_timestamps[:-1]) / dt[:, None]
    return PlannerOutput(jpos=jpos_timestamps[0], jvel=jvel_timestamps, timestamps=timestamps)


class Controller(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        planner = None
        for i in self.inputs:
            if i.input_name == "planner":
                planner = i.output.node
                break
        assert planner is not None, "No planner found!"
        planner_output = planner.default_output(rng, graph_state)
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


class Supervisor(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_params(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> SupervisorParams:
        """Default params of the root."""
        return SupervisorParams()

    def default_state(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> SupervisorState:
        """Default state of the root."""
        # Place goal on semi-circle around home position with radius goalpos_dist
        params: SupervisorParams = graph_state.nodes["supervisor"].params
        rng_planner, rng_goal = jax.random.split(rng, num=2)
        goalyaw = jnp.array([jax.random.uniform(rng_goal, minval=0, maxval=jnp.pi/2)])
        angle = jax.random.uniform(rng_goal, minval=0, maxval=jnp.pi)
        dx = jnp.sin(angle)*params.goalpos_dist
        dy = jnp.cos(angle)*params.goalpos_dist
        # goalpos = params.home_boxpos[:2] + jnp.array([dx, dy])
        goalpos = params.home_boxpos[:2] + jnp.array([0.1, 0.4])  # todo: goal hardcoded here.
        # goalpos = params.home_boxpos[:2] + jnp.array([-0.10, 0.45])  # todo: goal hardcoded here.
        return SupervisorState(goalpos=goalpos, goalyaw=goalyaw)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> SupervisorOutput:
        """Default output of the root."""
        state = graph_state.nodes["supervisor"].state
        return SupervisorOutput(goalpos=state.goalpos, goalyaw=state.goalyaw)

    def step(self, step_state: StepState) -> Tuple[StepState, SupervisorOutput]:
        state = step_state.state
        output = SupervisorOutput(goalpos=state.goalpos, goalyaw=state.goalyaw)
        return step_state, output


class Vx300sEnv(BaseEnv):
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

        from envs.vx300s.planner import box_pushing_cost
        self._cost_fn = box_pushing_cost

    def reset(self, rng: jax.random.KeyArray, graph_state: GraphState = None):
        """Reset environment."""
        new_graph_state = self.graph.init(rng, order=("supervisor", "world"))

        # Reset nodes
        rng, *rngs = jax.random.split(rng, num=len(self.nodes_world_and_supervisor) + 1)
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
        truncated = step_state.seq >= self.max_steps

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

