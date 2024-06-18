from typing import Dict, Tuple, Union, List, Any
from math import ceil
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict

try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.io import mjcf

    BRAX_INSTALLED = True
except ModuleNotFoundError:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False

from rexv2.graph import Graph
from rexv2 import base
from rexv2 import rl
from rexv2.base import GraphState, StepState
from rexv2.node import BaseNode


@struct.dataclass
class OdeParams(base.Base):
    """Pendulum state definition"""

    max_speed: Union[float, jax.typing.ArrayLike]
    J: Union[float, jax.typing.ArrayLike]
    mass: Union[float, jax.typing.ArrayLike]
    length: Union[float, jax.typing.ArrayLike]
    b: Union[float, jax.typing.ArrayLike]
    K: Union[float, jax.typing.ArrayLike]
    R: Union[float, jax.typing.ArrayLike]
    c: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class OdeState(base.Base):
    """Pendulum state definition"""

    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class BraxParams(base.Base):
    """Pendulum param definition"""

    max_speed: Union[float, jax.typing.ArrayLike]
    friction_loss: Union[float, jax.typing.ArrayLike]
    sys: gen_pipeline.System


@struct.dataclass
class BraxState(base.Base):
    """Pendulum state definition"""

    pipeline_state: gen_pipeline.State


@struct.dataclass
class WorldOutput(base.Base):
    """World output definition"""

    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorOutput(base.Base):
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class ActuatorOutput(base.Base):
    """Pendulum actuator output"""

    action: jax.typing.ArrayLike  # Torque to apply to the pendulum


class OdeWorld(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 100, **kwargs):
        super().__init__(*args, **kwargs)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeParams:
        """Default params of the node."""
        return OdeParams(
            max_speed=40.0,
            J=0.00019745720783248544,  # 0.000159931461600856,
            mass=0.053909555077552795,  # 0.0508581731919534,
            length=0.0471346490085125,  # 0.0415233722862552,
            b=1.3641421901411377e-05,  # 1.43298488358436e-05,
            K=0.046251337975263596,  # 0.0333391179016334,
            R=8.3718843460083,  # 7.73125142447252,
            c=0.0006091465475037694,  # 0.000975041213361349,
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeState:
        """Default state of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        init_th = jax.random.uniform(rng, shape=(), minval=-onp.pi, maxval=onp.pi)
        init_thdot = jax.random.uniform(rng, shape=(), minval=-2.0, maxval=2.0)
        return OdeState(th=init_th, thdot=init_thdot)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        world_state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return WorldOutput(th=world_state.th, thdot=world_state.thdot)

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        u = inputs["actuator"].data.action[-1][0]
        x = jnp.array([state.th, state.thdot])
        next_x = x

        # Calculate next state
        for _ in range(self.substeps):
            next_x = self._runge_kutta4(self._ode_disk_pendulum, self.dt_substeps, params, next_x, u)

        # Update state
        next_th, next_thdot = next_x
        next_thdot = jnp.clip(next_thdot, -params.max_speed, params.max_speed)
        new_state = state.replace(th=next_th, thdot=next_thdot)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        output = WorldOutput(th=next_th, thdot=next_thdot)
        # print(f"{self.name.ljust(14)} | x: {x} | u: {u} -> next_x: {next_x}")
        return new_step_state, output

    @staticmethod
    def _runge_kutta4(ode, dt, params, x, u):
        k1 = ode(params, x, u)
        k2 = ode(params, x + 0.5 * dt * k1, u)
        k3 = ode(params, x + 0.5 * dt * k2, u)
        k4 = ode(params, x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _ode_disk_pendulum(params: OdeParams, x, u):
        g, J, m, l, b, K, R, c = 9.81, params.J, params.mass, params.length, params.b, params.K, params.R, params.c
        activation = jnp.sign(x[1])
        ddx = (u * K / R + m * g * l * jnp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * activation) / J
        return jnp.array([x[1], ddx])


class BraxWorld(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 100, **kwargs):
        super().__init__(*args, **kwargs)
        assert BRAX_INSTALLED, "Brax not installed. Install it with `pip install brax`"
        self.sys = mjcf.loads(DISK_PENDULUM_XML)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxParams:
        """Default params of the node."""
        # Realistic parameters for the disk pendulum
        damping = 0.00015877
        armature = 6.4940527e-06
        gear = 0.00428677
        mass_weight = 0.05076142
        radius_weight = 0.05121992
        offset = 0.04161447
        friction_loss = 0.00097525

        # Appropriately replace parameters for the disk pendulum
        itransform = self.sys.link.inertia.transform.replace(pos=jnp.array([[0.0, offset, 0.0]]))
        i = self.sys.link.inertia.i.at[0, 0, 0].set(
            0.5 * mass_weight * radius_weight**2
        )  # inertia of cylinder in local frame.
        inertia = self.sys.link.inertia.replace(transform=itransform, mass=jnp.array([mass_weight]), i=i)
        link = self.sys.link.replace(inertia=inertia)
        actuator = self.sys.actuator.replace(gear=jnp.array([gear]))
        dof = self.sys.dof.replace(armature=jnp.array([armature]), damping=jnp.array([damping]))
        new_sys = self.sys.replace(link=link, actuator=actuator, dof=dof, dt=self.dt_substeps)
        return BraxParams(max_speed=40.0, friction_loss=friction_loss, sys=new_sys)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxState:
        """Default state of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        graph_state = graph_state or GraphState()

        # Sample initial state
        init_th = jax.random.uniform(rng, shape=(), minval=-onp.pi, maxval=onp.pi)
        init_thdot = jax.random.uniform(rng, shape=(), minval=-2.0, maxval=2.0)

        # Set the initial state of the disk pendulum
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        qpos = params.sys.init_q.at[0].set(init_th)
        qvel = jnp.array([init_thdot])
        pipeline_state = gen_pipeline.init(params.sys, qpos, qvel)
        return BraxState(pipeline_state=pipeline_state)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return WorldOutput(th=state.pipeline_state.q[0], thdot=state.pipeline_state.qd[0])

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        action = inputs["actuator"].data.action[-1]

        # Brax does not have static friction implemented
        thdot = state.pipeline_state.qd[0]
        activation = jnp.sign(thdot)
        friction = params.friction_loss * activation / params.sys.actuator.gear[0]
        action_friction = action - friction

        # Run the pipeline for the number of substeps
        def f(state, _):
            return (
                gen_pipeline.step(self.sys, state, action_friction),
                None,
            )

        new_pipeline_state = jax.lax.scan(f, state.pipeline_state, (), self.substeps)[0]

        # Update state
        new_state = state.replace(pipeline_state=new_pipeline_state)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        output = WorldOutput(th=new_pipeline_state.q[0], thdot=new_pipeline_state.qd[0])

        return new_step_state, output


class Sensor(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        th = jnp.pi
        thdot = 0.0
        return SensorOutput(th=th, thdot=thdot)

    def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
        """Step the node."""
        world = step_state.inputs["world"][-1].data

        # Prepare output
        output = SensorOutput(th=world.th, thdot=world.thdot)

        # Update state (NOOP)
        new_step_state = step_state

        return new_step_state, output


class Actuator(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        # key = "controller" if "controller" in inputs else "agent"
        controller_output = next(iter(inputs.values()))[-1].data
        output = ActuatorOutput(action=controller_output.action)
        return new_step_state, output


class RandomAgent(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action = jax.random.uniform(rng, shape=(1,), minval=-2.0, maxval=2.0)
        return ActuatorOutput(action=action)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Prepare output
        rng, rng_net = jax.random.split(step_state.rng)
        action = jax.random.uniform(rng_net, shape=(1,), minval=-2.0, maxval=2.0)
        output = ActuatorOutput(action=action)

        # Update state
        new_step_state = step_state.replace(rng=rng)

        return new_step_state, output


DISK_PENDULUM_XML = """
<mujoco model="disk_pendulum">
    <compiler inertiafromgeom="auto" angle="radian" coordinate="local" eulerseq="xyz" autolimits="true"/>
    <option gravity="0 0 -9.81" timestep="0.01" iterations="10"/>
    <custom>
        <numeric data="10" name="constraint_ang_damping"/> <!-- positional & spring -->
        <numeric data="1" name="spring_inertia_scale"/>  <!-- positional & spring -->
        <numeric data="0" name="ang_damping"/>  <!-- positional & spring -->
        <numeric data="0" name="spring_mass_scale"/>  <!-- positional & spring -->
        <numeric data="0.5" name="joint_scale_pos"/> <!-- positional -->
        <numeric data="0.1" name="joint_scale_ang"/> <!-- positional -->
        <numeric data="3000" name="constraint_stiffness"/>  <!-- spring -->
        <numeric data="10000" name="constraint_limit_stiffness"/>  <!-- spring -->
        <numeric data="50" name="constraint_vel_damping"/>  <!-- spring -->
        <numeric data="10" name="solver_maxls"/>  <!-- generalized -->
    </custom>

    <asset>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>

    <default>
        <geom contype="0" friction="1 0.1 0.1" material="geom"/>
    </default>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom name="table" type="plane" pos="0 0.0 -0.1" size="1 1 0.1" contype="8" conaffinity="11" condim="3"/>
        <body name="disk" pos="0.0 0.0 0.0" euler="1.5708 0.0 0.0">
            <joint name="hinge_joint" type="hinge" axis="0 0 1" range="-180 180" armature="0.00022993" damping="0.0001" limited="false"/>
            <geom name="disk_geom" type="cylinder" size="0.06 0.001" contype="0" conaffinity="0" condim="3" mass="0.0"/>
            <geom name="mass_geom" type="cylinder" size="0.02 0.005" contype="0" conaffinity="0"  condim="3" rgba="0.04 0.04 0.04 1"
                  pos="0.0 0.04 0." mass="0.05085817"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="hinge_joint" ctrllimited="false" ctrlrange="-3.0 3.0"  gear="0.01"/>
    </actuator>
</mujoco>
"""


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath

    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


def render(rollout: Union[List[GraphState], GraphState]):
    """Render the rollout as an HTML file."""
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    from brax.io import html

    if isinstance(rollout, list):
        rollout = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *rollout)

    # Extract rollout data
    th_rollout = rollout.inputs["agent"]["sensor"][:, -1].data.th
    thdot_rollout = rollout.inputs["agent"]["sensor"][:, -1].data.thdot

    # Determine fps
    dt = rollout.ts["agent"][-1] / rollout.ts["agent"].shape[-1]

    # Initialize system
    sys = mjcf.loads(DISK_PENDULUM_XML)
    sys = sys.replace(dt=dt)

    def _set_pipeline_state(th, thdot):
        qpos = sys.init_q.at[0].set(th)
        qvel = jnp.array([thdot])
        pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    pipeline_state_rollout = jax.vmap(_set_pipeline_state)(th_rollout, thdot_rollout)
    pipeline_state_lst = []
    for i in range(th_rollout.shape[0]):
        pipeline_state_i = jax.tree_util.tree_map(lambda x: x[i], pipeline_state_rollout)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json


class Environment(rl.Environment):
    def __len__(self, graph: Graph, step_states: Dict[str, base.StepState] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        super().__init__(graph, step_states, only_init, starting_eps, randomize_eps, order)

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return rl.Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        low = jnp.array([-2])
        high = jnp.array([2])
        return rl.Box(low, high, shape=low.shape, dtype=float)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = self.get_step_state(graph_state)
        all_data = [i.data for i in ss.inputs.values()] + [ss.state]
        all_fdata = []
        for data in all_data:
            # Vectorize data
            vdata = jax.tree_util.tree_map(lambda x: jnp.array(x).reshape(-1), data)
            if isinstance(vdata, SensorOutput):
                vdata = {"sin_th": jnp.sin(vdata.th), "cos_th": jnp.cos(vdata.th), "thdot": vdata.thdot}
            # Flatten pytree
            fdata, _ = jax.tree_util.tree_flatten(vdata)
            # Add to all_fdata
            all_fdata += fdata
        # Concatenate all_fdata
        cdata = jnp.concatenate(all_fdata)
        return cdata

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ss = self.get_step_state(graph_state)
        return ss.seq >= self.graph.max_steps

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ss = self.get_step_state(graph_state)
        terminated = False
        return terminated  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        ss = self.get_step_state(graph_state)
        data = ss.inputs["sensor"][-1].data
        th = self._angle_normalize(data.th)
        thdot = data.thdot
        u = action[0]
        cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2
        # cost = th ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return -cost

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        return {}

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> ActuatorOutput:
        return ActuatorOutput(action=action)

    def update_step_state(self, graph_state: base.GraphState, action: jax.Array = None) -> Tuple[base.GraphState, base.StepState]:
        """Override this method if you want to update the step state."""
        step_state = self.get_step_state(graph_state)
        return graph_state, step_state

    def _angle_normalize(self, th: jax.typing.ArrayLike):
        th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        return th_norm


@struct.dataclass
class GymnaxPendulum(base.Base):
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    g: float = 10.0  # gravity
    m: float = 1.0  # mass
    l: float = 1.0  # length


@struct.dataclass
class DiskPendulum(OdeParams):
    max_torque: float


@struct.dataclass
class TestState(base.Base):
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    last_u: jnp.ndarray  # Only needed for rendering
    time: int


class TestEnvironment(rl.Environment):
    def __init__(self):
        super().__init__(None, None, True, None, None, None)
        self.max_steps_in_episode = 200

    @staticmethod
    def _angle_normalize(th: jax.typing.ArrayLike):
        """Normalize the angle - radians."""
        return ((th + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    @property
    def max_steps(self) -> Union[int, jax.typing.ArrayLike]:
        return self.max_steps_in_episode  # todo: CHANGE BACK

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        ss = self.get_step_state(graph_state)
        max_speed = ss.params.max_speed
        high = jnp.array([1.0, 1.0, max_speed], dtype=float)
        low = -high
        return rl.Box(low, high, shape=low.shape, dtype=low.dtype)

    def get_step_state(self, graph_state: base.GraphState, name: str = None) -> base.StepState:
        return graph_state.nodes["env"]

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        ss = self.get_step_state(graph_state)
        max_torque = ss.params.max_torque
        low = jnp.array([-max_torque])
        high = jnp.array([max_torque])
        return rl.Box(low, high, shape=low.shape, dtype=float)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = self.get_step_state(graph_state)
        params = ss.params
        state = ss.state
        obs = jnp.array([jnp.cos(state.theta), jnp.sin(state.theta), state.theta_dot], dtype=float)
        return obs

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ss = self.get_step_state(graph_state)
        params = ss.params
        state = ss.state
        return state.time >= self.max_steps

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        return False  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        ss = self.get_step_state(graph_state)
        params = ss.params
        state = ss.state
        # Clip action
        u = action[0]  # reduce to scalar
        u = jnp.clip(u, -params.max_torque, params.max_torque)
        theta = self._angle_normalize(state.theta)
        cost = theta ** 2 + 0.1 * (state.theta_dot / (1 + 10 * abs(theta))) ** 2 + 0.01 * u ** 2
        # cost = theta ** 2 + 0.1 * state.theta_dot**2 + 0.001 * (u**2)
        return -cost

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        return {}

    def dynamics(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        raise NotImplementedError("Dynamics method must be implemented.")

    def step(self, graph_state: base.GraphState, action: jax.Array) -> rl.StepReturn:
        """
        Step the environment.
        Can be overridden to provide custom step behavior.

        :param graph_state: The current graph state.
        :param action: The action to take.
        :return: Tuple of (graph_state, observation, reward, terminated, truncated, info)
        """
        # Get reward
        reward = self.get_reward(graph_state, action)
        # Step the environment
        gs_post = self.dynamics(graph_state, action)
        # Get termination conditions
        truncated = self.get_truncated(gs_post)
        terminated = self.get_terminated(gs_post)
        # Get observation
        obs = self.get_observation(gs_post)
        # Get info
        info = self.get_info(gs_post, action)
        return gs_post, obs, reward, terminated, truncated, info


class TestGymnaxPendulum(TestEnvironment):

    def reset(self, rng: jax.Array = None) -> rl.ResetReturn:
        """
        Reset the environment.
        Can be overridden to provide custom reset behavior.

        :param rng: Random number generator. Used to initialize a new graph state.
        :return: Tuple of (graph_state, observation, info)
        """
        # Get params
        params = GymnaxPendulum(
            # max_steps_in_episode=self.max_steps_in_episode
        )

        # Get state
        rng, rng_state = jax.random.split(rng)
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(rng_state, shape=(2,), minval=-high, maxval=high)
        state = TestState(
            theta=state[0], theta_dot=state[1], last_u=jnp.array([0.0]), time=0
        )

        # Initialize Gs
        ss = base.StepState(rng=rng, state=state, params=params)
        gs = base.GraphState(nodes=FrozenDict({"env": ss}))
        obs = self.get_observation(gs)
        info = self.get_info(gs)
        return gs, obs, info

    def dynamics(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        ss = self.get_step_state(graph_state)
        params = ss.params
        state = ss.state
        u = action[0]  # reduce to scalar
        u = jnp.clip(u, -params.max_torque, params.max_torque)
        newthdot = state.theta_dot + (
                (
                        3 * params.g / (2 * params.l) * jnp.sin(state.theta)
                        + 3.0 / (params.m * params.l ** 2) * u
                )
                * params.dt
        )
        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * params.dt
        # Update state dict and evaluate termination conditions
        state = TestState(
            theta=newth,
            theta_dot=newthdot,
            last_u=action,  # shape: (1,)
            time=state.time + 1,
        )
        ss_post = base.StepState(rng=ss.rng, state=state, params=params)
        gs_post = graph_state.replace_nodes(nodes={"env": ss_post})
        return gs_post


class TestDiskPendulum(TestEnvironment):

    @staticmethod
    def _angle_normalize(th: jax.typing.ArrayLike):
        th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        return th_norm

    @staticmethod
    def _runge_kutta4(ode, dt, params, x, u):
        k1 = ode(params, x, u)
        k2 = ode(params, x + 0.5 * dt * k1, u)
        k3 = ode(params, x + 0.5 * dt * k2, u)
        k4 = ode(params, x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _ode_disk_pendulum(params: OdeParams, x, u):
        g, J, m, l, b, K, R, c = 9.81, params.J, params.mass, params.length, params.b, params.K, params.R, params.c
        activation = jnp.sign(x[1])
        ddx = (u * K / R + m * g * l * jnp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * activation) / J
        return jnp.array([x[1], ddx])

    def reset(self, rng: jax.Array = None) -> rl.ResetReturn:
        """
        Reset the environment.
        Can be overridden to provide custom reset behavior.

        :param rng: Random number generator. Used to initialize a new graph state.
        :return: Tuple of (graph_state, observation, info)
        """
        # Get params
        params = DiskPendulum(
            max_speed=40.0,
            max_torque=2.0,
            # max_steps_in_episode=self.max_steps_in_episode,
            J=0.00019745720783248544,  # 0.000159931461600856,
            mass=0.053909555077552795,  # 0.0508581731919534,
            length=0.0471346490085125,  # 0.0415233722862552,
            b=1.3641421901411377e-05,  # 1.43298488358436e-05,
            K=0.046251337975263596,  # 0.0333391179016334,
            R=8.3718843460083,  # 7.73125142447252,
            c=0.0006091465475037694,  # 0.000975041213361349,
        )

        # Get state
        rng, rng_state = jax.random.split(rng)
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(rng_state, shape=(2,), minval=-high, maxval=high)
        state = TestState(
            theta=state[0], theta_dot=state[1], last_u=jnp.array([0.0]), time=0
        )

        # Initialize Gs
        ss = base.StepState(rng=rng, state=state, params=params)
        gs = base.GraphState(nodes=FrozenDict({"env": ss}))
        obs = self.get_observation(gs)
        info = self.get_info(gs)
        return gs, obs, info

    def dynamics(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        # Calculate substeps and world dt
        _substeps = 3
        _dt_world = 1/90

        ss = self.get_step_state(graph_state)
        params = ss.params
        state = ss.state
        u = action[0]  # reduce to scalar
        u = jnp.clip(u, -params.max_torque, params.max_torque)

        x = jnp.array([state.theta, state.theta_dot])
        next_x = x
        for _ in range(_substeps):
            next_x = self._runge_kutta4(self._ode_disk_pendulum, _dt_world, params, next_x, u)

            # Update state
            # next_th, next_thdot = next_x
            # next_thdot = jnp.clip(next_thdot, -params.max_speed, params.max_speed)

        # Update state dict and evaluate termination conditions
        next_th, next_thdot = next_x
        next_thdot = jnp.clip(next_thdot, -params.max_speed, params.max_speed)
        state = TestState(
            theta=next_th,
            theta_dot=next_thdot,
            last_u=action,  # shape: (1,)
            time=state.time + 1,
        )
        ss_post = base.StepState(rng=ss.rng, state=state, params=params)
        gs_post = graph_state.replace_nodes(nodes={"env": ss_post})
        return gs_post
