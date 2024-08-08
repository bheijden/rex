from typing import Union, Tuple, Any, Callable

import jax
from flax import struct
from jax import numpy as jnp

from envs.crazyflie.ode import WorldParams, MoCapOutput
from envs.pendulum.ukf import UKFState, UKFParams
from rexv2.base import GraphState, StepState, Base
from rexv2.node import BaseNode


# @struct.dataclass
# class UKFCrazyflie(WorldParams):
#     def make_Fx(self, substeps: int, ts: Union[float, jax.typing.ArrayLike], dt: Union[float, jax.typing.ArrayLike], ts_us: Union[float, jax.typing.ArrayLike], us: Any) -> Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]:
#         def _Fx(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
#             """State transition function."""
#             # Interpolate control inputs
#             dt_substeps = dt / substeps
#             ts_substeps = ts + jnp.arange(0, substeps) * dt_substeps
#             us_substeps = jnp.interp(ts_substeps, ts_us, us)
#             # Run dynamics
#             x, _x_substeps = self.step(substeps, dt_substeps, x, us_substeps)
#             return x
#         return _Fx
#
#     def make_Qx(self, dt: Union[float, jax.typing.ArrayLike], std_acc: Union[float, jax.typing.ArrayLike]) -> Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]:
#         def _Qx(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
#             """State noise function."""
#             # Add process noise
#             B = jnp.array([0.5 * dt ** 2, dt])  # _dt is different for forward and update steps
#             # Outer product of B with B
#             return jnp.outer(B, B) * std_acc**2
#         return _Qx
#
#     def make_Gx(self) -> Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]:
#         def _Gx(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
#             """Measurement function."""
#             th = x[0][None]  # Add dimension
#             return jnp.concatenate([jnp.cos(th), jnp.sin(th)], axis=0)
#         return _Gx
#
#     def make_Rx(self, std_th: Union[float, jax.typing.ArrayLike]) -> Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]:
#         def _Rx(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
#             """Measurement noise function."""
#             th, thdot = x
#             Rx = std_th**2 * jnp.array([[jnp.sin(th) ** 2, -jnp.cos(th) * jnp.sin(th)],
#                                        [-jnp.cos(th) * jnp.sin(th), jnp.cos(th) ** 2]])
#             return Rx
#         return _Rx
#

@struct.dataclass
class EstimatorParams(Base):
    ukf: UKFParams
    ode: WorldParams
    dt_future: Union[float, jax.typing.ArrayLike]  # Time to predict into the future [0, inf]
    # std_acc: Union[float, jax.typing.ArrayLike]  # Standard deviation of acceleration noise
    # std_th: Union[float, jax.typing.ArrayLike]  # Standard deviation of angle noise
    # std_init: Union[float, jax.typing.ArrayLike]  # Standard deviation of initial state
    substeps_update: int = struct.field(pytree_node=False)
    substeps_predict: int = struct.field(pytree_node=False)


@struct.dataclass
class EstimatorState(Base):
    ts: Union[float, jax.typing.ArrayLike]  # ts of prior
    prior: UKFState
    loss_th: Union[float, jax.typing.ArrayLike]  # Running reconstruction loss of the estimated pendulum angle

    def to_output(self) -> "EstimatorOutput":
        raise NotImplementedError("Not implemented yet")
        mean = WorldState(th=self.prior.mu[0], thdot=self.prior.mu[1])
        return EstimatorOutput(ts=self.ts, mean=mean, cov=self.prior.sigma)


@struct.dataclass
class EstimatorOutput(Base):
    ts: Union[float, jax.typing.ArrayLike]  # ts of estimate
    mean: MoCapOutput
    cov: jax.Array  # Covariance of the estimate: outer(MoCapOutput, MoCapOutput)


class Estimator(BaseNode):
    def __init__(self, *args, use_pred: bool = True, use_ukf: bool = True, **kwargs):
        self.use_pred = use_pred
        self.use_ukf = use_ukf
        super().__init__(*args, **kwargs)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> EstimatorParams:
        """Default params of the root."""
        # ode = UKFOde(
        #     None,
        #     max_speed=40.0,
        #     J=0.00019745720783248544,
        #     mass=0.053909555077552795,
        #     length=0.0471346490085125,
        #     b=1.3641421901411377e-05,
        #     K=0.046251337975263596,
        #     R=8.3718843460083,
        #     c=0.0006091465475037694,
        # )
        ode = None
        return EstimatorParams(ukf=UKFParams(alpha=None, beta=None, kappa=None),  # Use default values
                               ode=ode,
                               substeps_update=2,  # todo: set appropriately
                               substeps_predict=4,  # todo: set appropriately
                               # Note: dt_future measures latency from estimator to control application on the world
                               # This means that: dt_future ~= (world.inputs["actuator"].phase - estimator.phase)
                               dt_future=0.027,
                               # std_acc=3.154,
                               # std_th=0.43,
                               # std_init=0.5,
                            )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> EstimatorState:
        """Default state of the root."""
        # Try to grab state from graph_state
        graph_state = graph_state or GraphState()
        state = graph_state.state.get("supervisor", None)
        prior = None
        # th = state.init_th if state is not None else jnp.pi
        # thdot = state.init_thdot if state is not None else 0.
        # std_init = graph_state.params.get(self.name, self.init_params(rng, graph_state)).std_init
        # prior = UKFState(mu=jnp.array([th, thdot]),
        #                  sigma=jnp.eye(2) * std_init**2)
        return EstimatorState(ts=0.0, prior=prior, loss_th=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> EstimatorOutput:
        """Default output of the root."""
        # output = graph_state.state.get(self.name, self.init_state(rng, graph_state)).to_output()
        output = self.inputs["mocap"].output_node.init_output(rng, graph_state)
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, EstimatorOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Prepare actions (to be used in smith-predictor)
        actions = inputs["agent"].data
        ts_actions = inputs["agent"].data.state_estimate.ts

        # Filter finite difference of the pendulum angle
        mocap = inputs["mocap"][-1].data
        return step_state, mocap

        # th = mocap.th[None]
        # trig_th = jnp.concatenate([jnp.cos(th), jnp.sin(th)], axis=0)
        #
        # # Predict to ts_meas
        # ts_prev = state.ts
        # ts_meas = mocap.ts
        # dt_upd = jnp.clip(ts_meas - ts_prev, 0.0, None)
        # Fx_upd = params.ode.make_Fx(params.substeps_update, ts_prev, dt_upd, ts_actions, actions)
        # Qx_upd = params.ode.make_Qx(dt_upd, params.std_acc)
        # Gx = params.ode.make_Gx()
        # Rx = params.ode.make_Rx(params.std_th)
        #
        # def _update(_state: EstimatorState) -> EstimatorState:
        #     """Only update if new measurement (i.e. ts_prev < ts_meas)."""
        #     _posterior = params.ukf.predict_and_update(Fx_upd, Qx_upd, Gx, Rx, state.prior, trig_th)
        #     # Calculate loss_th
        #     _th_est = _posterior.mu[0]
        #     loss_th = _state.loss_th + 10*(jnp.sin(_th_est) - jnp.sin(th[0])) ** 2 + (jnp.cos(_th_est) - jnp.cos(th[0])) ** 2
        #     _new_state = EstimatorState(ts=ts_meas, prior=_posterior, loss_th=loss_th)
        #     return _new_state
        #
        # def _no_update(_state: EstimatorState) -> EstimatorState:
        #     """No update if no new measurement (i.e. ts_prev >= ts_meas)."""
        #     return _state
        #
        # new_state = jax.lax.cond(ts_prev < ts_meas, _update, _no_update, state)
        # posterior = new_state.prior
        #
        # # Update step_state
        # new_step_state = step_state.replace(state=new_state)
        #
        # # Note: dt_future measures latency from estimator to control application on the world
        # # This means that: dt_future ~= (world.inputs["actuator"].phase - estimator.phase)
        # ts_fut = step_state.ts + params.dt_future
        # dt_fut = jnp.clip(ts_fut - ts_meas, 0.0, None)
        #
        # # Forward predict to ts_future
        # Fx_fut = params.ode.make_Fx(params.substeps_predict, new_state.ts, dt_fut, ts_actions, actions)
        # Qx_fut = params.ode.make_Qx(dt_fut, params.std_acc)
        # ukf_fut = params.ukf.predict(Fx_fut, Qx_fut, posterior) if self.use_pred else posterior
        # est_fut = EstimatorState(ts=ts_fut, prior=ukf_fut, loss_th=None)
        #
        # # To output
        # output = est_fut.to_output()
        #
        # # Overwrite output with mocap if not using camera
        # if not self.use_ukf:
        #     mean = output.mean.replace(th=mocap.th, thdot=mocap.thdot)
        #     output = output.replace(mean=mean, cov=jnp.eye(2) * params.std_th**2)
        # return new_step_state, output
