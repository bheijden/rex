from math import ceil
from typing import Union, Tuple, Dict
import jax
from jax import numpy as jnp
from flax import struct

from rex import base
from rex.base import GraphState, StepState
from rex.node import BaseWorld


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
        k1 = self._ode(x, u)
        k2 = self._ode(x + k1 * dt * 0.5, u)
        k3 = self._ode(x + k2 * dt * 0.5, u)
        k4 = self._ode(x + k3 * dt, u)
        return x + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    def _ode(self, x: "OdeState", u: jax.typing.ArrayLike) -> "OdeState":
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
class OdeOutput(base.Base):
    """World output definition"""
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


class OdeWorld(BaseWorld):  # We inherit from BaseWorld for convenience, but you can inherit from BaseNode if you want
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

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        world_state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return OdeOutput(th=world_state.th, thdot=world_state.thdot)

    def init_delays(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> Dict[str, Union[float, jax.typing.ArrayLike]]:
        graph_state = graph_state or GraphState()
        params = graph_state.params.get("actuator")
        delays = {}
        if hasattr(params, "actuator_delay"):
            delays["actuator"] = params.actuator_delay
        return delays

    def step(self, step_state: StepState) -> Tuple[StepState, OdeOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Apply dynamics
        u = inputs["actuator"].data.action[-1][0]  # [-1] to get the latest action, [0] reduces the dimension to scalar
        us = jnp.array([u] * params.substeps)
        new_state = params.step(params.substeps, params.dt_substeps, state, us)[0]
        next_th, next_thdot = new_state.th, new_state.thdot
        output = OdeOutput(th=next_th, thdot=next_thdot)  # Prepare output

        # Calculate cost (penalize angle error, angular velocity and input voltage)
        norm_next_th = next_th - 2 * jnp.pi * jnp.floor((next_th + jnp.pi) / (2 * jnp.pi))
        loss_task = state.loss_task + norm_next_th ** 2 + 0.1 * (
                    next_thdot / (1 + 10 * abs(norm_next_th))) ** 2 + 0.01 * u ** 2

        # Update state
        new_state = new_state.replace(loss_task=loss_task)
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output
