from typing import Any, Dict, Tuple, Union
import jax
import jax.numpy as jnp
from math import ceil
from flax import struct
from flax.core import FrozenDict
from rexv2.base import StepState, GraphState, Empty, TrainableDist, Base
from rexv2.node import BaseNode
from rexv2.jax_utils import tree_dynamic_slice

from envs.pendulum.base import ActuatorOutput, WorldState, SensorOutput, WorldParams, SensorParams, ActuatorParams


@struct.dataclass
class OdeParams(WorldParams):
    """Pendulum ode param definition"""
    max_speed: Union[float, jax.typing.ArrayLike]
    J: Union[float, jax.typing.ArrayLike]
    mass: Union[float, jax.typing.ArrayLike]
    length: Union[float, jax.typing.ArrayLike]
    b: Union[float, jax.typing.ArrayLike]
    K: Union[float, jax.typing.ArrayLike]
    R: Union[float, jax.typing.ArrayLike]
    c: Union[float, jax.typing.ArrayLike]

    def step(self, substeps: int, dt_substeps: Union[float, jax.typing.ArrayLike], x: jax.typing.ArrayLike, us: jax.typing.ArrayLike) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
        """Step the pendulum ode."""

        def _scan_fn(_x, _u):
            next_x = self._runge_kutta4(dt_substeps, _x, _u)
            return next_x, next_x

        x_final, x_substeps = jax.lax.scan(_scan_fn, x, us, length=substeps)
        return x_final, x_substeps

    def _runge_kutta4(self, dt, x, u):
        k1 = self.ode(x, u)
        k2 = self.ode(x + 0.5 * dt * k1, u)
        k3 = self.ode(x + 0.5 * dt * k2, u)
        k4 = self.ode(x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def ode(self, x: jax.typing.ArrayLike, u: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
        """dx function for the pendulum ode"""
        # Downward := [pi, 0], Upward := [0, 0]
        g, J, m, l, b, K, R, c = 9.81, self.J, self.mass, self.length, self.b, self.K, self.R, self.c
        activation = jnp.sign(x[1])
        ddx = (u * K / R + m * g * l * jnp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * activation) / J
        return jnp.array([x[1], ddx])


@struct.dataclass
class OdeState(WorldState):
    """Pendulum ode state definition"""
    last_ts: Union[int, jax.typing.ArrayLike]
    loss_th: Union[float, jax.typing.ArrayLike]
    loss_thdot: Union[float, jax.typing.ArrayLike]
    loss_ts: Union[float, jax.typing.ArrayLike]
    loss_task: Union[float, jax.typing.ArrayLike]


class World(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 100, **kwargs):
        super().__init__(*args, **kwargs)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeParams:
        """Default params of the node."""
        # Try to grab params from graph_state
        graph_state = graph_state or GraphState()
        actuator = self.inputs["actuator"].output_node
        actuator_delay = graph_state.params.get("actuator", actuator.init_params(rng, graph_state)).actuator_delay
        return OdeParams(
            actuator_delay=actuator_delay,
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
        graph_state = graph_state or GraphState()
        # Try to grab state from graph_state
        state = graph_state.state.get("supervisor", None)
        th = state.init_th if state is not None else jnp.pi
        thdot = state.init_thdot if state is not None else 0.
        return OdeState(th=th, thdot=thdot,
                        last_ts=0., loss_th=0.0, loss_thdot=0.0, loss_ts=0.0, loss_task=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldState:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return WorldState(th=state.th, thdot=state.thdot)

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        if isinstance(inputs["actuator"].delay_dist, TrainableDist):
            inputs["actuator"] = inputs["actuator"].replace(delay_dist=params.actuator_delay)
        return FrozenDict(inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, WorldState]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get state estimate
        data: ActuatorOutput = step_state.inputs["actuator"][-1].data
        ts = step_state.ts if data.state_estimate is None else data.state_estimate.ts
        mean = state if data.state_estimate is None else data.state_estimate.mean

        # Only update loss if actuator.seq is new (i.e. only penalize state mismatch at the start of a new action)
        alpha = state.last_ts < ts
        loss_th = state.loss_th + alpha * (jnp.sin(mean.th) - jnp.sin(state.th)) ** 2 + alpha * (jnp.cos(mean.th) - jnp.cos(state.th)) ** 2
        loss_thdot = state.loss_thdot + alpha * (mean.thdot - state.thdot) ** 2
        loss_ts = state.loss_ts + alpha * (ts - step_state.ts) ** 2
        new_last_ts = ts

        # Apply dynamics
        u = inputs["actuator"].data.action[-1][0]
        x = jnp.array([state.th, state.thdot])
        next_x = x
        for _ in range(self.substeps):
            next_x = self._runge_kutta4(self._ode_disk_pendulum, self.dt_substeps, params, next_x, u)
        next_th, next_thdot = next_x
        next_thdot = jnp.clip(next_thdot, -params.max_speed, params.max_speed)
        output = WorldState(th=next_th, thdot=next_thdot)  # Prepare output

        # Calculate cost (penalize angle error, angular velocity and input voltage)
        norm_next_th = self._angle_normalize(next_th)
        loss_task = state.loss_task + norm_next_th ** 2 + 0.1 * (next_thdot / (1 + 10 * abs(norm_next_th))) ** 2 + 0.01 * u ** 2
        # loss_task = state.loss_task +  norm_next_th ** 2 + 0.01 * next_thdot ** 2 + 0.001 * (u ** 2)

        # Update state
        new_state = state.replace(th=next_th, thdot=next_thdot, last_ts=new_last_ts, loss_th=loss_th, loss_thdot=loss_thdot, loss_ts=loss_ts, loss_task=loss_task)
        new_step_state = step_state.replace(state=new_state)

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

    @staticmethod
    def _angle_normalize(th: jax.typing.ArrayLike):
        th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        return th_norm


@struct.dataclass
class SensorState:
    loss_th: Union[float, jax.typing.ArrayLike]
    loss_thdot: Union[float, jax.typing.ArrayLike]


class Sensor(BaseNode):
    def __init__(self, *args, outputs: SensorOutput = None, **kwargs):
        """Initialize Sensor for system identification.

        Args:
        images: Recorded sensor Outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorParams:
        """Default params of the node."""
        sensor_delay = TrainableDist.create(delay=0., min=0.0, max=0.05)
        return SensorParams(sensor_delay=sensor_delay)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorState:
        """Default state of the node."""
        return SensorState(loss_th=0.0, loss_thdot=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        state = graph_state.state.get("world", None)
        th = state.th if state is not None else jnp.pi
        thdot = state.thdot if state is not None else 0.
        # Account for sensor delay
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        ts = -1. / self.rate - sensor_delay
        return SensorOutput(th=th, thdot=thdot, ts=ts)

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        if isinstance(inputs["world"].delay_dist, TrainableDist):
            inputs["world"] = inputs["world"].replace(delay_dist=params.sensor_delay)
        return FrozenDict(inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
        """Step the node."""
        rng = step_state.rng

        # Adjust ts_start (i.e. step_state.ts) to reflect the timestamp of the world state that generated the sensor output
        ts = step_state.ts - step_state.params.sensor_delay.mean()  # Should be equal to world_interp.ts_sent[-1]?
        data = step_state.inputs["world"][-1].data

        # Determine output
        if self._outputs is not None:
            recorded_output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            output = recorded_output.replace(ts=ts)
        else:
            # rng, rng_noise = jax.random.split(rng)
            # th = data.th + jax.random.normal(rng_noise) * 0.02  # todo: IS HARDCODED
            output = SensorOutput(th=data.th, thdot=data.thdot, ts=ts)

        # Calculate loss
        state = step_state.state
        th_world = data.th
        thdot_world = data.thdot
        loss_th = state.loss_th + (jnp.sin(output.th) - jnp.sin(th_world)) ** 2 + (jnp.cos(output.th) - jnp.cos(th_world)) ** 2
        loss_thdot = state.loss_thdot + (output.thdot - thdot_world) ** 2

        # Update state
        new_state = state.replace(loss_th=loss_th, loss_thdot=loss_thdot)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        return new_step_state, output


class Actuator(BaseNode):
    def __init__(self, *args, outputs: Union[ActuatorOutput, ActuatorOutput] = None, **kwargs):
        """Initialize Actuator for system identification.

        Args:
        images: Recorded actuator Outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorParams:
        actuator_delay = TrainableDist.create(delay=0., min=0.0, max=0.05)
        return ActuatorParams(actuator_delay=actuator_delay)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        assert "controller" in self.inputs, "No controller connected to actuator"
        output = self.inputs["controller"].output_node.init_output(rng, graph_state)
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Grab ctrl output
        assert len(self.inputs) > 0, "No controller connected to actuator"
        ctrl_output = inputs["controller"][-1].data
        state_estimate = ctrl_output.state_estimate

        # Get action from dataset or use passed through.
        if self._outputs is not None:
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            action = output.action
        else:
            action = ctrl_output.action

        # jax.debug.print(f"{self.name:<10} | " + "seq={seq}| controller.seq={seq_controller} | action={action}",
        #                 seq=step_state.seq, seq_controller=inputs["controller"][-1].seq, action=action[0])
        # jax.debug.print(f"{self.name:<10} | " + "seq={seq} | ts={ts} | state_estimate.ts={ts_estimate}",
        #                 seq=step_state.seq, ts=step_state.ts, ts_estimate=state_estimate.ts if state_estimate else None)

        # Prepare output
        output = ActuatorOutput(action=action, state_estimate=state_estimate)

        # Update state
        new_step_state = step_state

        return new_step_state, output
