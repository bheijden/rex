from typing import Dict, Tuple, Union, List, Any
from math import ceil
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict

try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.spring import pipeline as spring_pipeline
    from brax.positional import pipeline as pos_pipeline
    from brax.base import System, State
    from brax.io import mjcf
    Systems = Union[gen_pipeline.System, spring_pipeline.System, pos_pipeline.System]
    Pipelines = Union[gen_pipeline.State, spring_pipeline.State, pos_pipeline.State]
    BRAX_INSTALLED = True
except ModuleNotFoundError:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False

from rex.graph import Graph
from rex import base
from rex import rl
from rex.jax_utils import tree_dynamic_slice
from rex.ppo import Policy
from rex.base import GraphState, StepState
from rex.node import BaseNode


@struct.dataclass
class OdeParams(base.Base):
    """Pendulum ode param definition"""
    max_speed: Union[float, jax.typing.ArrayLike]
    J: Union[float, jax.typing.ArrayLike]
    mass: Union[float, jax.typing.ArrayLike]
    length: Union[float, jax.typing.ArrayLike]
    b: Union[float, jax.typing.ArrayLike]
    K: Union[float, jax.typing.ArrayLike]
    R: Union[float, jax.typing.ArrayLike]
    c: Union[float, jax.typing.ArrayLike]
    dt_substeps_min: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False)
    dt: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False)

    @property
    def substeps(self) -> int:
        substeps = ceil(self.dt / self.dt_substeps_min)
        return int(substeps)

    @property
    def dt_substeps(self) -> float:
        substeps = self.substeps
        dt_substeps = self.dt / substeps
        return dt_substeps

    def step(self, substeps: int, dt_substeps: jax.typing.ArrayLike, x: "OdeState", us: jax.typing.ArrayLike) -> Tuple["OdeState", "OdeState"]:
        """Step the pendulum ode."""

        def _scan_fn(_x, _u):
            next_x = self._runge_kutta4(dt_substeps, _x, _u)
            # Clip velocity
            clip_thdot = jnp.clip(next_x.thdot, -self.max_speed, self.max_speed)
            next_x = next_x.replace(thdot=clip_thdot)
            return next_x, next_x

        x_final, x_substeps = jax.lax.scan(_scan_fn, x, us, length=substeps)
        return x_final, x_substeps

    def _runge_kutta4(self, dt: jax.typing.ArrayLike, x: "OdeState", u: jax.typing.ArrayLike) -> "OdeState":
        k1 = self.ode(x, u)
        k2 = self.ode(x + k1 * dt * 0.5, u)
        k3 = self.ode(x + k2 * dt * 0.5, u)
        k4 = self.ode(x + k3 * dt, u)
        return x + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    def ode(self, x: "OdeState", u: jax.typing.ArrayLike) -> "OdeState":
        """dx function for the pendulum ode"""
        # Downward := [pi, 0], Upward := [0, 0]
        g, J, m, l, b, K, R, c = 9.81, self.J, self.mass, self.length, self.b, self.K, self.R, self.c
        th, thdot = x.th, x.thdot
        activation = jnp.sign(thdot)
        ddx = (u * K / R + m * g * l * jnp.sin(th) - b * thdot - thdot * K * K / R - c * activation) / J
        return OdeState(th=thdot, thdot=ddx, loss_task=0.0)  # No derivative for loss_task


@struct.dataclass
class OdeState(base.Base):
    """Pendulum state definition"""
    loss_task: Union[float, jax.typing.ArrayLike]
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class BraxParams(base.Base):
    max_speed: Union[float, jax.typing.ArrayLike]
    damping: Union[float, jax.typing.ArrayLike]
    armature: Union[float, jax.typing.ArrayLike]
    gear: Union[float, jax.typing.ArrayLike]
    mass_weight: Union[float, jax.typing.ArrayLike]
    radius_weight: Union[float, jax.typing.ArrayLike]
    offset: Union[float, jax.typing.ArrayLike]
    friction_loss: Union[float, jax.typing.ArrayLike]
    backend: str = struct.field(pytree_node=False)
    base_sys: Systems = struct.field(pytree_node=False)
    dt: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False)

    @property
    def substeps(self) -> int:
        dt_substeps_per_backend = {
            "generalized": 1 / 100,
            "spring": 1 / 100,
            "positional": 1 / 100
        }[self.backend]
        substeps = ceil(self.dt / dt_substeps_per_backend)
        return int(substeps)

    @property
    def dt_substeps(self) -> float:
        substeps = self.substeps
        dt_substeps = self.dt / substeps
        return dt_substeps

    @property
    def pipeline(self) -> Pipelines:
        return {
            "generalized": gen_pipeline,
            "spring": spring_pipeline,
            "positional": pos_pipeline
        }[self.backend]

    @property
    def sys(self) -> Systems:
        # Appropriately replace parameters for the disk pendulum
        itransform = self.base_sys.link.inertia.transform.replace(pos=jnp.array([[0.0, self.offset, 0.0]]))
        i = self.base_sys.link.inertia.i.at[0, 0, 0].set(
            0.5 * self.mass_weight * self.radius_weight**2
        )  # inertia of cylinder in local frame.
        inertia = self.base_sys.link.inertia.replace(transform=itransform, mass=jnp.array([self.mass_weight]), i=i)
        link = self.base_sys.link.replace(inertia=inertia)
        actuator = self.base_sys.actuator.replace(gear=jnp.array([self.gear]))
        dof = self.base_sys.dof.replace(armature=jnp.array([self.armature]), damping=jnp.array([self.damping]))
        opt = self.base_sys.opt.replace(timestep=self.dt_substeps)
        new_sys = self.base_sys.replace(link=link, actuator=actuator, dof=dof, opt=opt)
        return new_sys

    def step(self, substeps: int, dt_substeps: jax.typing.ArrayLike, x: Pipelines, us: jax.typing.ArrayLike) -> Tuple[Pipelines, Pipelines]:
        """Step the pendulum ode."""
        # Appropriately replace timestep for the disk pendulum
        sys = self.sys.replace(opt=self.sys.opt.replace(timestep=dt_substeps))

        def _scan_fn(_x, _u):
            # Add friction loss
            thdot = x.qd[0]
            activation = jnp.sign(thdot)
            friction = self.friction_loss * activation / sys.actuator.gear[0]
            _u_friction = _u - friction
            # Step
            next_x = gen_pipeline.step(sys, _x, jnp.array(_u_friction)[None])
            # Clip velocity
            next_x = next_x.replace(qd=jnp.clip(next_x.qd, -self.max_speed, self.max_speed))
            return next_x, next_x

        x_final, x_substeps = jax.lax.scan(_scan_fn, x, us, length=substeps)
        return x_final, x_substeps


@struct.dataclass
class BraxState(base.Base):
    """Pendulum state definition"""
    loss_task: Union[float, jax.typing.ArrayLike]
    pipeline_state: Pipelines

    @property
    def th(self):
        return self.pipeline_state.q[..., 0]

    @property
    def thdot(self):
        return self.pipeline_state.qd[..., 0]


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
    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeParams:
        """Default params of the node."""
        return OdeParams(
            max_speed=40.0,  # Clip angular velocity to this value
            J=0.000159931461600856,  # 0.000159931461600856,
            mass=0.0508581731919534,  # 0.0508581731919534,
            length=0.0415233722862552,  # 0.0415233722862552,
            b=1.43298488e-05,  # 1.43298488358436e-05,
            K=0.03333912,  # 0.0333391179016334,
            R=7.73125142,  # 7.73125142447252,
            c=0.000975041213361349,  # 0.000975041213361349,
            # Backend parameters
            dt_substeps_min=1 / 100,  # Minimum substep size for ode integration
            dt=1 / self.rate,  # Time step per .step() call
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeState:
        """Default state of the node."""
        graph_state = graph_state or GraphState()

        # Try to grab state from graph_state
        state = graph_state.state.get("agent", None)
        init_th = state.init_th if state is not None else jnp.pi
        init_thdot = state.init_thdot if state is not None else 0.
        return OdeState(th=init_th, thdot=init_thdot, loss_task=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        world_state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return WorldOutput(th=world_state.th, thdot=world_state.thdot)

    def init_delays(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> Dict[str, Union[float, jax.typing.ArrayLike]]:
        graph_state = graph_state or GraphState()
        params = graph_state.params.get("actuator")
        delays = {}
        if hasattr(params, "actuator_delay"):
            delays["actuator"] = params.actuator_delay
        return delays

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Apply dynamics
        u = inputs["actuator"].data.action[-1][0]
        us = jnp.array([u] * params.substeps)
        new_state = params.step(params.substeps, params.dt_substeps, state, us)[0]
        next_th, next_thdot = new_state.th, new_state.thdot
        output = WorldOutput(th=next_th, thdot=next_thdot)  # Prepare output

        # Calculate cost (penalize angle error, angular velocity and input voltage)
        norm_next_th = self._angle_normalize(next_th)
        loss_task = state.loss_task + norm_next_th ** 2 + 0.1 * (
                    next_thdot / (1 + 10 * abs(norm_next_th))) ** 2 + 0.01 * u ** 2

        # Update state
        new_state = new_state.replace(loss_task=loss_task)
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output

    @staticmethod
    def _angle_normalize(th: jax.typing.ArrayLike):
        th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        return th_norm


class BraxWorld(BaseNode):
    def __init__(self, *args, backend: str = "generalized", **kwargs):
        assert BRAX_INSTALLED, "Brax not installed. Install it with `pip install brax`"
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.sys = mjcf.loads(DISK_PENDULUM_XML)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxParams:
        """Default params of the node."""
        return BraxParams(
            # Realistic parameters for the disk pendulum
            max_speed=40.0,
            damping=0.00015877,
            armature=6.4940527e-06,
            gear=0.00428677,
            mass_weight=0.05076142,
            radius_weight=0.05121992,
            offset=0.04161447,
            friction_loss=0.00097525,
            # Backend parameters
            dt=1/self.rate,
            base_sys=self.sys,
            backend=self.backend,
        )
        # # matlab nlgreyest params
        # J = 0.000159931461600856
        # M = 0.0508581731919534
        # offset = 0.0415233722862552
        # radius_weight = 0.0508581731919534
        # b = 1.43298488e-05
        # K = 0.03333912
        # R = 7.73125142
        #
        # damping = (b + K**2 / R)
        # armature = (J - 0.5*radius_weight**2*M - M*offset**2)
        # gear = (K/R)
        # mass_weight = M
        # radius_weight = radius_weight
        # offset = offset
        # friction_loss = 0.000975041213361349
        #
        # # Realistic parameters for the disk pendulum
        # # damping = 0.00015877
        # # armature = 6.4940527e-06
        # # gear = 0.00428677
        # # mass_weight = 0.05076142
        # # radius_weight = 0.05121992
        # # offset = 0.04161447
        # # friction_loss = 0.00097525
        #
        # # Appropriately replace parameters for the disk pendulum
        # itransform = self.sys.link.inertia.transform.replace(pos=jnp.array([[0.0, offset, 0.0]]))
        # i = self.sys.link.inertia.i.at[0, 0, 0].set(
        #     0.5 * mass_weight * radius_weight**2
        # )  # inertia of cylinder in local frame.
        # inertia = self.sys.link.inertia.replace(transform=itransform, mass=jnp.array([mass_weight]), i=i)
        # link = self.sys.link.replace(inertia=inertia)
        # actuator = self.sys.actuator.replace(gear=jnp.array([gear]))
        # dof = self.sys.dof.replace(armature=jnp.array([armature]), damping=jnp.array([damping]))
        # opt = self.sys.opt.replace(timestep=self.dt_substeps)
        # new_sys = self.sys.replace(link=link, actuator=actuator, dof=dof, opt=opt)
        # return BraxParams(max_speed=40.0, friction_loss=friction_loss, sys=new_sys)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxState:
        """Default state of the node."""
        graph_state = graph_state or GraphState()

        # Try to grab state from graph_state
        state = graph_state.state.get("agent", None)
        init_th = state.init_th if state is not None else jnp.pi
        init_thdot = state.init_thdot if state is not None else 0.

        # Set the initial state of the disk pendulum
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sys = params.sys
        q = sys.init_q.at[0].set(init_th)
        qd = jnp.array([init_thdot])
        pipeline_state = params.pipeline.init(sys, q, qd)
        return BraxState(pipeline_state=pipeline_state, loss_task=0.0)

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

        # Apply dynamics
        u = inputs["actuator"].data.action[-1][0]
        us = jnp.array([u] * params.substeps)
        x = state.pipeline_state
        next_x = params.step(params.substeps, params.dt_substeps, x, us)[0]
        new_state = state.replace(pipeline_state=next_x)
        next_th, next_thdot = new_state.th, new_state.thdot
        output = WorldOutput(th=next_th, thdot=next_thdot)  # Prepare output

        # Calculate cost (penalize angle error, angular velocity and input voltage)
        norm_next_th = self._angle_normalize(next_th)
        loss_task = state.loss_task + norm_next_th ** 2 + 0.1 * (
                    next_thdot / (1 + 10 * abs(norm_next_th))) ** 2 + 0.01 * u ** 2

        # Update state
        new_state = new_state.replace(loss_task=loss_task)
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output

    @staticmethod
    def _angle_normalize(th: jax.typing.ArrayLike):
        th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        return th_norm


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


@struct.dataclass
class SensorParams(base.Base):
    sensor_delay: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorState:
    loss_th: Union[float, jax.typing.ArrayLike]
    loss_thdot: Union[float, jax.typing.ArrayLike]


class SimSensor(BaseNode):
    def __init__(self, *args, outputs: SensorOutput = None, **kwargs):
        """Initialize Sensor for system identification.

        Args:
        outputs: Recorded sensor Outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorParams:
        """Default params of the node."""
        sensor_delay = 0.05
        return SensorParams(sensor_delay=sensor_delay)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorState:
        """Default state of the node."""
        return SensorState(loss_th=0.0, loss_thdot=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        th = jnp.pi
        thdot = 0.0
        return SensorOutput(th=th, thdot=thdot)

    def init_delays(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> Dict[str, Union[float, jax.typing.ArrayLike]]:
        """Initialize trainable communication delays.

        **Note** These only include trainable delays that were specified while connecting the nodes.

        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default output.
        :return: Trainable delays (e.g., {input_name: delay}). Can be an incomplete dictionary.
                 Entries for non-trainable delays or non-existent connections are ignored.
        """
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        delays = {"world": params.sensor_delay}
        return delays

    def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
        # Determine output
        data = step_state.inputs["world"][-1].data
        output = SensorOutput(th=data.th, thdot=data.thdot)

        # Calculate loss
        if self._outputs is not None:
            output_rec = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            th_rec, thdot_rec = output_rec.th, output_rec.thdot
            state = step_state.state
            loss_th = state.loss_th + (jnp.sin(output.th) - jnp.sin(th_rec)) ** 2 + (jnp.cos(output.th) - jnp.cos(th_rec)) ** 2
            loss_thdot = state.loss_thdot + (output.thdot - thdot_rec) ** 2
            new_state = state.replace(loss_th=loss_th, loss_thdot=loss_thdot)
        else:
            new_state = step_state.state

        # Update step_state
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output


class Actuator(BaseNode):
    # todo: add host_callback code & stop routine example
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""
        # Prepare output
        output = step_state.inputs["agent"][-1].data
        output = ActuatorOutput(action=output.action)

        # Update state
        new_step_state = step_state
        return new_step_state, output

    def stop(self, timeout: float = None) -> bool:
        """Stopping routine that is called after the episode is done."""
        return True

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        """Starts the node in the state specified by graph_state."""
        return True


@struct.dataclass
class ActuatorParams(base.Base):
    actuator_delay: Union[float, jax.typing.ArrayLike]


class SimActuator(BaseNode):
    def __init__(self, *args, outputs: ActuatorOutput = None, **kwargs):
        """Initialize Actuator for system identification.

        Args:
        outputs: Recorded actuator Outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorParams:
        """Default params of the node."""
        actuator_delay = 0.05
        return ActuatorParams(actuator_delay=actuator_delay)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        # Get action from dataset if available, else use the one provided by the agent
        if self._outputs is not None:  # Use the recorded action (for system identification)
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
        else:  # Feedthrough the agent's action (for normal operation, e.g., training)
            output = step_state.inputs["agent"][-1].data
            output = ActuatorOutput(action=output.action)
        new_step_state = step_state
        return new_step_state, output


@struct.dataclass
class AgentParams(base.Base):
    # Policy
    policy: Policy
    # Observations
    num_act: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Action history length
    num_obs: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Observation history length
    # Action
    max_torque: Union[float, jax.typing.ArrayLike]
    # Initial state
    init_method: str = struct.field(pytree_node=False)  # "random", "parametrized"
    parametrized: jax.typing.ArrayLike
    max_th: Union[float, jax.typing.ArrayLike]
    max_thdot: Union[float, jax.typing.ArrayLike]
    # Train
    gamma: Union[float, jax.typing.ArrayLike]
    tmax: Union[float, jax.typing.ArrayLike]

    @staticmethod
    def process_inputs(inputs: FrozenDict[str, base.InputState]) -> jax.Array:
        th, thdot = inputs["sensor"][-1].data.th, inputs["sensor"][-1].data.thdot
        obs = jnp.array([jnp.cos(th), jnp.sin(th), thdot])
        return obs

    @staticmethod
    def get_observation(step_state: StepState) -> jax.Array:
        # Unpack StepState
        inputs, state = step_state.inputs, step_state.state

        # Convert inputs to single observation
        single_obs = AgentParams.process_inputs(inputs)

        # Concatenate with previous observations
        obs = jnp.concatenate([single_obs, state.history_obs.flatten(), state.history_act.flatten()])
        return obs

    @staticmethod
    def update_state(step_state: StepState, action: jax.Array) -> "AgentState":
        # Unpack StepState
        state, params, inputs = step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        single_obs = AgentParams.process_inputs(inputs)

        # Update obs history
        if params.num_obs > 0:
            history_obs = jnp.roll(state.history_obs, shift=1, axis=0)
            history_obs = history_obs.at[0].set(single_obs)
        else:
            history_obs = state.history_obs

        # Update act history
        if params.num_act > 0:
            history_act = jnp.roll(state.history_act, shift=1, axis=0)
            history_act = history_act.at[0].set(action)
        else:
            history_act = state.history_act

        new_state = state.replace(history_obs=history_obs, history_act=history_act)
        return new_state

    @staticmethod
    def to_output(action: jax.Array) -> ActuatorOutput:
        return ActuatorOutput(action=action)


@struct.dataclass
class AgentState(base.Base):
    history_act: jax.typing.ArrayLike
    history_obs: jax.typing.ArrayLike
    init_th: Union[float, jax.typing.ArrayLike]
    init_thdot: Union[float, jax.typing.ArrayLike]


class Agent(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> AgentParams:
        return AgentParams(
            policy=None,  # Policy must be set by the user
            num_act=4,
            num_obs=4,
            max_torque=2.0,
            init_method="parametrized",
            parametrized=jnp.array([jnp.pi, 0.0]),
            max_th=jnp.pi,
            max_thdot=9.0,
            gamma=0.99,
            tmax=3.0,
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> AgentState:
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        history_act = jnp.zeros((params.num_act, 1), dtype=jnp.float32)  # [torque]
        history_obs = jnp.zeros((params.num_obs, 3), dtype=jnp.float32)  # [cos(th), sin(th), thdot]

        # Set the initial state of the pendulum
        if params.init_method == "parametrized":
            init_th, init_thdot = params.parametrized
        elif params.init_method == "random":
            rng = rng if rng is not None else jax.random.PRNGKey(0)
            rngs = jax.random.split(rng, num=2)
            init_th = jax.random.uniform(rngs[0], shape=(), minval=-params.max_th, maxval=params.max_th)
            init_thdot = jax.random.uniform(rngs[1], shape=(), minval=-params.max_thdot, maxval=params.max_thdot)
        else:
            raise ValueError(f"Invalid init_method: {params.init_method}")
        return AgentState(history_act=history_act, history_obs=history_obs, init_th=init_th, init_thdot=init_thdot)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        rng = jax.random.PRNGKey(0) if rng is None else rng
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        action = jax.random.uniform(rng, shape=(1,), minval=-params.max_torque, maxval=params.max_torque)
        return ActuatorOutput(action=action)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""
        # Unpack StepState
        rng, params = step_state.rng, step_state.params

        # Prepare output
        rng, rng_net = jax.random.split(rng)
        if params.policy is not None:
            obs = AgentParams.get_observation(step_state)
            action = params.policy.get_action(obs, rng=None)  # Supply rng for stochastic policies
        else:
            action = jax.random.uniform(rng_net, shape=(1,), minval=-params.max_torque, maxval=params.max_torque)
        output = AgentParams.to_output(action)

        # Update step_state (observation and action history)
        new_state = params.update_state(step_state, action)
        new_step_state = step_state.replace(rng=rng, state=new_state)
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

DISK_PENDULUM_VISUAL_XML = """
<mujoco model="disk_pendulum">
    <compiler inertiafromgeom="auto" angle="radian" coordinate="local" eulerseq="xyz" autolimits="true" meshdir="/home/r2ci/rex/envs/pendulum/assets"/>
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
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true" specular="1" shininess="1.0"/>
        <material name="disk" reflectance="1.0" specular="1" shininess="1"/>
    </asset>

    <default>
        <geom contype="0" friction="1 0.1 0.1" material="disk"/>
    </default>

    <worldbody>
        <light cutoff="45" diffuse="1 1 1" dir="0 -1 -1" directional="true" exponent="1" pos="0. 1.0 1.0" specular="1 1 1"/>
        <geom name="floor" type="plane" conaffinity="0" condim="3" pos="0 0 -0.095" rgba="0.79 0.79 0.79 1.0" size="1 1 1" />
        <geom name="table" type="box" size="0.15 0.25 0.01" contype="0" conaffinity="0" condim="3" rgba="0.65 0.41 0.199 1.0" mass="0.0" pos="0. 0.0 -0.1"/>
        <geom name="light_geom" type="cylinder" size="0.003 0.001" contype="0" conaffinity="0" condim="3" rgba="0.2 0.2 1.0 1" mass="0.0" pos="-0.06 0. 0.06" euler="1.5708 0.0 0.0"/>
        <geom name="box_geom" type="box" size="0.08 0.12 0.08" contype="0" conaffinity="0" condim="3" rgba="0.8 0.8 0.8 1" mass="0.0" pos="0. -0.122 0."/>
        <geom name="corner1_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="-0.075 -0.12 -0.075" euler="1.5708 0.0 0.0"/>
        <geom name="corner2_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="0.075 -0.12 0.075" euler="1.5708 0.0 0.0"/>
        <geom name="corner3_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="-0.075 -0.12 0.075" euler="1.5708 0.0 0.0"/>
        <geom name="corner4_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="0.075 -0.12 -0.075" euler="1.5708 0.0 0.0"/>
        <body name="disk" pos="0.0 0.0 0.0" euler="1.5708 0.0 0.0">
            <joint name="hinge_joint" type="hinge" axis="0 0 1" range="-180 180" armature="0.00022993" damping="0.0001" limited="false"/>
            <geom name="hinge_geom" type="cylinder" size="0.014 0.007" contype="0" conaffinity="0" condim="3" rgba="0.6 0.6 0.6 1" mass="0.0"/>
            <geom name="screw_top_geom" type="cylinder" size="0.003 0.002" contype="0" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" mass="0.0" pos="0.0 -0.005 -0.007"/>
            <geom name="screw_right_geom" type="cylinder" size="0.003 0.002" contype="0" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" mass="0.0" pos="0.005 0.004 -0.007"/>
            <geom name="screw_left_geom" type="cylinder" size="0.003 0.002" contype="0" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" mass="0.0" pos="-0.005 0.004 -0.007"/>
            <geom name="disk_geom" type="cylinder" size="0.06 0.001" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0"/>
            <geom name="mass_geom" type="cylinder" size="0.02 0.005" contype="0" conaffinity="0"  condim="3" rgba="0.5 0.08 0.08 1" pos="0.0 0.04 0." mass="0.05085817"/>
            <geom name="hole_geom" type="cylinder" size="0.002 0.002" contype="0" conaffinity="0"  condim="3" rgba="0.8 0.8 0.8 1" pos="0.0 -0.04 0." mass="0.0"/>      
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


def render(rollout: Union[BraxState, OdeState], dt: Union[float, jax.typing.ArrayLike] = 0.02, xml_string: str = DISK_PENDULUM_VISUAL_XML):
    """Render the rollout as an HTML file."""
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    from brax.io import html

    # Initialize system
    sys = mjcf.loads(xml_string)
    sys = sys.replace(opt=sys.opt.replace(timestep=dt))

    def _set_pipeline_state(th, thdot):
        qpos = sys.init_q.at[0].set(th)
        qvel = jnp.array([thdot])
        pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    pipeline_state_rollout = jax.vmap(_set_pipeline_state)(rollout.th, rollout.thdot)
    pipeline_state_lst = []
    for i in range(rollout.th.shape[0]):
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
