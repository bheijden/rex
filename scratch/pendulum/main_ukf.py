"""https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb"""
from typing import Union, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


from rexv2.base import Base
from envs.pendulum.ukf import UKFState, UKFParams
from envs.pendulum.estimator import UKFOde


@struct.dataclass
class WorldState(Base):
    ts: jax.typing.ArrayLike
    th: jax.typing.ArrayLike
    thdot: jax.typing.ArrayLike


@struct.dataclass
class Measurement(Base):
    ts: jax.typing.ArrayLike
    th: jax.typing.ArrayLike
    trig_th: jax.typing.ArrayLike


@struct.dataclass
class Action(Base):
    ts: jax.typing.ArrayLike
    action: jax.typing.ArrayLike


@struct.dataclass
class Estimator:
    ts: Union[float, jax.typing.ArrayLike]  # timestamp of the state
    state: UKFState


if __name__ == "__main__":
    # Only update if new measurement seq number.
    # Keep estimate w.r.t latest measurement
    # Forward predict estimate to ts_actuator with action history
    # State P_t_meas, x_t_meas
    # Output P_t_meas+dt, x_t_meas+dt where t_meas+dt = ts_next_actuator
    # Keep history of control inputs with expected timestamps when they affect the system

    from envs.pendulum.ode import OdeParams, World

    ODE = World._ode_disk_pendulum
    RK4 = World._runge_kutta4
    RNG = jax.random.PRNGKey(0)
    # World parameters
    INIT_WORLD = WorldState(ts=0.0, th=0.5, thdot=0.0)
    PARAMS = OdeParams(
        actuator_delay=None,
        max_speed=40.0,
        J=0.00019745720783248544,  # 0.000159931461600856,
        mass=0.053909555077552795,  # 0.0508581731919534,
        length=0.0471346490085125,  # 0.0415233722862552,
        b=1.3641421901411377e-05,  # 1.43298488358436e-05,
        K=0.046251337975263596,  # 0.0333391179016334,
        R=8.3718843460083,  # 7.73125142447252,
        c=0.0006091465475037694,  # 0.000975041213361349,
    )
    VAR_Y = 0.1  # True measurement noise
    VAR_ACC = 10.0  # True acceleration noise variance
    # Time constants
    T = 10.0
    DT_WORLD = 0.01
    DT_CAM = 0.02
    DT_UKF = 0.02
    DT_CTRL = 0.04
    SUBSTEPS = int(jnp.ceil(DT_CTRL / DT_WORLD).astype(int))  # Overwrite DT_WORLD to match a multiple of DT_CTRL
    DT_WORLD = DT_CTRL / SUBSTEPS
    # UKF parameters
    UKF_PARAMS = UKFParams()
    SUBSTEPS_UPDATE = 1
    SUBSTEPS_PRED = 4
    USE_TRIG = True
    DT_FUTURE = DT_CTRL*5
    INIT_UKF = UKFState(mu=jnp.array([0.0, 0.0]),  # Initial mean
                        sigma=jnp.array([[0.5, 0.04],
                                         [0.04, 0.5]]))  # Initial covariance
    INIT_EST = Estimator(ts=0.0, state=INIT_UKF)
    UKF_ODE = UKFOde(**PARAMS.__dict__)

    # Compute world timestamps
    ts_world = jnp.arange(0.0, T, DT_WORLD)
    ts_ctrl = jnp.arange(0.0, T, DT_CTRL)
    ts_ukf = jnp.arange(0.0, T, DT_UKF)
    ts_cam = jnp.arange(0.0, T, DT_CAM)

    # Prepare actions
    # u = -1*jnp.sin(ts_ctrl*2.0*jnp.pi/0.2)  # Sinusoidal action
    # Random actions that are persistent for 0.5 seconds
    u = jnp.repeat(4*jax.random.uniform(RNG, (int(10/0.5)+1,))-2, int(0.5/DT_CTRL)+1)[:len(ts_ctrl)]
    actions = Action(ts=ts_ctrl, action=u)

    # Apply actions
    def _scan_world(carry, y):
        rng, ts_prev, th, thdot = carry[0], carry[1].ts, carry[1].th, carry[1].thdot
        ts_next, action = y.ts, y.action

        # Compute
        rng_next, rng_noise = jax.random.split(rng)

        # Compute
        dt = jnp.clip(ts_next - ts_prev, 0.0, None)  # Clip to avoid negative time
        dt_substeps = dt / SUBSTEPS
        ts = []
        xs = []
        ts_now = ts_prev
        x = jnp.array([th, thdot])
        u = jnp.array(action)
        for _ in range(SUBSTEPS):
            # Innovate
            x = RK4(ODE, dt_substeps, PARAMS, x, u)
            # Add process noise
            x = x + jax.random.normal(rng_noise, x.shape) * jnp.sqrt(VAR_ACC) * jnp.array([0.5 * dt_substeps ** 2, dt_substeps])
            # Update time
            ts_now = ts_now + dt_substeps
            # Append
            ts.append(ts_now)
            xs.append(x)
        ts, xs = jnp.array(ts), jnp.array(xs)
        world = WorldState(ts=ts, th=xs[:, 0], thdot=xs[:, 1])
        return (rng_next, world[-1]), world

    RNG, rng_scan = jax.random.split(RNG)
    _, state = jax.lax.scan(_scan_world, (rng_scan, INIT_WORLD), actions)
    state = state.reshape(-1)

    # Create measurements
    RNG, rng_noise = jax.random.split(RNG)
    y = (jnp.interp(ts_cam, state.ts, state.th) + jax.random.normal(rng_noise, ts_cam.shape) * jnp.sqrt(VAR_Y))[:, None]
    trig_th = jnp.concatenate([jnp.cos(y), jnp.sin(y)], axis=1)
    measurements = Measurement(ts=ts_cam, th=y, trig_th=trig_th)

    # Apply UKF
    def _scan_ukf(est: Estimator, meas: Measurement):
        # Determine time interval
        ts_prev = est.ts
        ts_meas = meas.ts
        ts_future = ts_meas + DT_FUTURE
        dt = jnp.clip(ts_meas - ts_prev, 0.0, None)

        # Predict to ts_meas
        assert USE_TRIG, "Only trigonometric measurements are supported"
        Gx = UKF_ODE.make_Gx()
        Rx = UKF_ODE.make_Rx(jnp.sqrt(VAR_Y))
        Fx_update = UKF_ODE.make_Fx(SUBSTEPS_UPDATE, ts_prev, dt, actions.ts, actions.action)
        Qx_update = UKF_ODE.make_Qx(dt, jnp.sqrt(VAR_ACC))
        y = meas.trig_th if USE_TRIG else meas.th
        ukf_upd = UKF_PARAMS.predict_and_update(Fx_update, Qx_update, Gx, Rx, est.state, y)
        est_upd = Estimator(ts=ts_meas, state=ukf_upd)

        # Forward predict to ts_future
        Fx_future = UKF_ODE.make_Fx(SUBSTEPS_PRED, est_upd.ts, DT_FUTURE, actions.ts, actions.action)
        Qx_future = UKF_ODE.make_Qx(DT_FUTURE, jnp.sqrt(VAR_ACC))
        ukf_fut = UKF_PARAMS.predict(Fx_future, Qx_future, ukf_upd)
        est_fut = Estimator(ts=ts_future, state=ukf_fut)

        return est_upd, (est_upd, est_fut)

    # TEST API
    _scan_ukf(INIT_EST, measurements[0])

    # Filter all
    _, (est_upd, est_fut) = jax.lax.scan(_scan_ukf, INIT_EST, measurements)

    # Calculate errors
    th_upd_interp = jnp.interp(est_upd.ts, state.ts, state.th)
    thdot_upd_interp = jnp.interp(est_upd.ts, state.ts, state.thdot)
    th_fut_interp = jnp.interp(est_fut.ts, state.ts, state.th)
    thdot_fut_interp = jnp.interp(est_fut.ts, state.ts, state.thdot)
    err_upd = jnp.sqrt(jnp.mean((est_upd.state.mu[:, 0] - th_upd_interp)**2))
    err_fut = jnp.sqrt(jnp.mean((est_fut.state.mu[:, 0] - th_fut_interp)**2))
    print(f"Error upd: {err_upd}, Error fut: {err_fut}")

    # Get std
    std_vfn = jax.vmap(lambda x: jnp.diag(jnp.sqrt(x)))
    std_upd = std_vfn(est_upd.state.sigma)
    std_fut = std_vfn(est_fut.state.sigma)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(state.ts, state.th, label="th", color='r')
    axes[0].plot(est_upd.ts, est_upd.state.mu[:, 0], label="upd", color='g')
    axes[0].fill_between(est_upd.ts, est_upd.state.mu[:, 0] - std_upd[:, 0], est_upd.state.mu[:, 0] + std_upd[:, 0], alpha=0.5, color='g')
    axes[0].plot(est_fut.ts, est_fut.state.mu[:, 0], label="fut", color='b')
    axes[0].fill_between(est_fut.ts, est_fut.state.mu[:, 0] - std_fut[:, 0], est_fut.state.mu[:, 0] + std_fut[:, 0], alpha=0.5, color='b')
    axes[0].plot(measurements.ts, measurements.th, 'o', label="y", markersize=2)
    axes[0].legend()

    axes[1].plot(state.ts, state.thdot, label="thdot", color='r')
    axes[1].plot(est_upd.ts, est_upd.state.mu[:, 1], label="upd", color='g')
    axes[1].fill_between(est_upd.ts, est_upd.state.mu[:, 1] - std_upd[:, 1], est_upd.state.mu[:, 1] + std_upd[:, 1], alpha=0.5, color='g')
    axes[1].plot(est_fut.ts, est_fut.state.mu[:, 1], label="fut", color='b')
    axes[1].fill_between(est_fut.ts, est_fut.state.mu[:, 1] - std_fut[:, 1], est_fut.state.mu[:, 1] + std_fut[:, 1], alpha=0.5, color='b')
    axes[1].legend()
    axes[2].plot(actions.ts, actions.action, label="action", color='r')
    axes[2].legend()
    plt.show()
