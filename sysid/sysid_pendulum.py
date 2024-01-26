from functools import partial
from typing import Union
import time
import os
import multiprocessing

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()-4
)
import dill as pickle
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax._src.interpreters import batching
import numpy as onp
import jax.random as rnd
from jax.tree_util import tree_map

# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)

import optimistix as optx
import equinox as eqx
import jaxopt
import scipy.io

import rex.utils as utils
from rex.utils import timer, make_put_output_on_device
import rex.open_colors as oc
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
from rex.proto import log_pb2
from rex.node import Node
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, SYNC, ASYNC, REAL_TIME, FAST_AS_POSSIBLE, FREQUENCY, PHASE, SIMULATED, WALL_CLOCK
from rex.distributions import Gaussian, GMM
from rex.base import GraphState, StepState

utils.set_log_level(WARN)

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]

import experiments as exp
import envs.pendulum as pend

from math import ceil
import os
import tqdm

import brax
from brax.io import html
from brax.io import mjcf
from brax.generalized import pipeline as gen_pipeline
from brax.positional import pipeline as pos_pipeline
from brax.spring import pipeline as spr_pipeline
from brax import base, math
from flax import struct

import mujoco
from mujoco import mjx

from rex.utils import timer


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath
    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


@struct.dataclass
class Params:
    c: Union[float, jax.typing.ArrayLike]
    d: Union[float, jax.typing.ArrayLike]
    sys: Union[gen_pipeline.State, mjx._src.types.Model]


@struct.dataclass
class ODEParams:
    """Pendulum ode param definition"""
    dt: Union[float, jax.typing.ArrayLike]
    J: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.000159931461600856)
    mass: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.0508581731919534)
    length: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.0415233722862552)
    b: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 1.43298488358436e-05)
    K: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.0333391179016334)
    R: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 7.73125142447252)
    c: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.000975041213361349)
    d: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 165.417960777425)


def sigmoid(x):
    # For very large negative values of x, the sigmoid is very close to 0.
    # For very large positive values of x, the sigmoid is very close to 1.
    # We can use these properties to avoid computing the exponential in these cases.
    safe_x = jnp.clip(x, -35, 35)
    # large_positive = x > 35
    # large_negative = x < -35
    # safe_x = jnp.where(large_positive, 0, jnp.where(large_negative, 0, x))

    pos = 1.0 / (1.0 + jnp.exp(-safe_x))
    return pos


def runge_kutta4(ode, dt, params, x, u):
    k1 = ode(params, x, u)
    k2 = ode(params, x + 0.5 * dt * k1, u)
    k3 = ode(params, x + 0.5 * dt * k2, u)
    k4 = ode(params, x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def ode_pendulum(params: ODEParams, x, u):
    g, J, m, l, b, K, R, c, d = 9.81, params.J, params.mass, params.length, params.b, params.K, params.R, params.c, params.d

    ddx = (u * K / R + m * g * l * jnp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * (2 * sigmoid(d * x[1]) - 1)) / J

    return jnp.array([x[1], ddx])


from sysid import cem, utils, lsq, pendulum

if __name__ == "__main__":
    # Initialize backend
    # backend = pendulum.BraxBackend()
    backend = pendulum.ODEBackend()
    base_params = backend.init_backend(dt_sysid=0.01)
    residual = partial(pendulum.residual, backend)


    # Get time
    dt = base_params.sys.dt
    dt_sysid = base_params.dt_sysid
    substeps = base_params.substeps

    # Get real data
    mat = scipy.io.loadmat("/home/r2ci/Downloads/data.mat")
    mat_actions = mat["data"][0][0][5][0, 0]
    mat_state = mat["data"][0][0][2][0, 0]
    mat_T = jnp.arange(mat_actions.shape[0]) * (1/150)
    interp_T = jnp.arange(int(mat_T[-1]/dt)) * dt
    interp_actions = jnp.interp(interp_T, mat_T, mat_actions[:, 0])[None, :]
    interp_jpos = jnp.interp(interp_T, mat_T, mat_state[:, 0])[None, :]
    interp_jvel = jnp.interp(interp_T, mat_T, mat_state[:, 1])[None, :]
    interp_init_jpos = interp_jpos[:, 0]
    interp_init_jvel = interp_jvel[:, 0]
    T = interp_T[None, :]
    jpos = interp_jpos[:, :]
    jvel = interp_jvel[:, :]
    actions = interp_actions[:, :]

    # Define params
    pre_s = pendulum.State(init_jpos=jpos[:, 0], init_jvel=jvel[:, 0], pipeline_state=None)
    actions = pendulum.Action(voltage=actions[:, :-1])  # Exclude last action since we don't have the next state for it
    init_y_ys = pendulum.Output(jpos=jpos[:, :], jvel=jvel[:, :])
    args = (base_params, pre_s, actions, init_y_ys)

    # Initial params
    init_params = pendulum.Params.default()
    u_min = jax.tree_util.tree_map(lambda x: x * 1.1, init_params)
    u_max = jax.tree_util.tree_map(lambda x: x * 0.9, init_params)
    init_params = jax.tree_util.tree_map(lambda x: 1.1*x, init_params)

    # CEM
    # todo: I am not seeing any improvement in the loss --> START HERE!
    solver = cem.CEMSolver.init(u_min=u_min, u_max=u_max, evolution_smoothing=0.1, elite_portion=0.1, num_samples=100)
    _cem = partial(cem.cem, residual, solver, max_steps=100, verbose=True)
    with timer("cem", log_level=100):
        sol = jax.jit(_cem)(init_params, args)
    print(sol)

    # Least squares
    # todo: uses forward or reverse mode autodiff?
    solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-7, norm=optx.rms_norm, verbose=frozenset({"step", "accepted", "step_size", "loss"}))
    solver = optx.BestSoFarRootFinder(solver)
    lsq_residual = utils.exp_transformed(residual, where=(True, False), pre=True)
    _lsq = partial(lsq.least_squares, lsq_residual, solver, max_steps=100, throw=False)
    log_init_params = jax.tree_map(jnp.log, init_params)
    with timer("lsq", log_level=100):
        sol = eqx.filter_jit(_lsq)(log_init_params, args)
    print(jax.tree_util.tree_map(jnp.exp, sol.value))
    print(sol.stats)
    exit()

    print("loading system")
    m_mjx = mjx.device_put(mujoco.MjModel.from_xml_path('/home/r2ci/rex/envs/pendulum/assets/disk_pendulum.xml'))
    m_brax = mjcf.load('/home/r2ci/rex/envs/pendulum/assets/disk_pendulum.xml')
    m_ODE = ODEParams(dt=m_brax.dt)
    # p = Params(c=0., d=1, sys=m_brax)
    # p = Params(c=0., d=1, sys=m_brax)
    print(f"degrees of freedom: {m_brax.qd_size()}")

    # Determine collision pairs
    print("\nCOLLISIONS")
    from brax.geometry.contact import _geom_pairs

    for (geom_i, geom_j) in _geom_pairs(m_brax):
        # print(geom_i.link_idx, geom_j.link_idx)
        name_i = m_brax.link_names[geom_i.link_idx[0]] if geom_i.link_idx is not None else "world"
        name_j = m_brax.link_names[geom_j.link_idx[0]] if geom_j.link_idx is not None else "world"
        print(f"collision pair: {name_i} --> {name_j}")

    # Actuators
    print("\nACTUATOR SIZE")
    print(f"actuator size: {m_brax.act_size()}")
    q_id = m_brax.actuator.q_id[:1]

    # DOFS
    print("\nDEGREES OF FREEDOM SIZE")
    print(f"degrees of freedom: {m_brax.qd_size()}")

    # Select pipeline
    pipeline = gen_pipeline

    # Get sampling time (0.8s horizon needed)
    horizon = 119
    total_steps = ceil(horizon / m_brax.dt)
    dt = m_brax.dt
    horizon = total_steps * m_brax.dt
    print(f"\nTIME")
    print(f"dt: {m_brax.dt}, total_steps: {total_steps}, horizon: {horizon}")

    def init_mjx_sys(params: Params, damping: float = 1e-4, armature: float = -1.e9, gear: float = 1e-2, mass_weight: float = 5e-2,
                     radius_weight: float = 2e-2, offset: float = 4e-2, friction_loss: float = 0.000975, d: float = 0.1,
                     link_invweight: float = 1., dof_invweight: float = 1.):
        _m = m_mjx
        body_invweight0 = link_invweight * _m.body_invweight0  # todo: check if nonzero gradient w.r.t. link_invweight
        dof_M0 = dof_invweight * _m.dof_M0  # todo: check if nonzero gradient w.r.t. dof_invweight
        dof_invweight0 = dof_invweight * _m.dof_invweight0  # todo: check if nonzero gradient w.r.t. dof_invweight

        # Set parameters
        actuator_gear = _m.actuator_gear.at[0, 0].set(gear)
        dof_armature = _m.dof_armature.at[0].set(jnp.abs(armature))
        dof_damping = _m.dof_damping.at[0].set(damping)
        dof_frictionloss = _m.dof_frictionloss.at[0].set(friction_loss)
        body_mass = _m.body_mass.at[1].set(mass_weight)
        body_subtreemass = _m.body_subtreemass.at[:2].set(mass_weight)
        body_inertia = _m.body_inertia.at[1, 0].set(0.5 * mass_weight * radius_weight ** 2)
        body_ipos = _m.body_ipos.at[1, 1].set(offset)


        new_m = _m.replace(actuator_gear=actuator_gear, dof_armature=dof_armature, dof_damping=dof_damping,
                           dof_frictionloss=dof_frictionloss, body_mass=body_mass, body_subtreemass=body_subtreemass,
                           body_inertia=body_inertia, body_ipos=body_ipos, body_invweight0=body_invweight0,
                           dof_M0=dof_M0, dof_invweight0=dof_invweight0)
        return new_m

    def init_brax_sys(params: Params, damping: float = 1e-4, armature: float = -1.e9, gear: float = 1e-2, mass_weight: float = 5e-2,
                      radius_weight: float = 2e-2, offset: float = 4e-2, friction_loss: float = 0.000975, d: float = 0.1,
                      link_invweight: float = 1., dof_invweight: float = 1.):
        # brax
        # dof.damping --> restorative force back to zero velocity
        # dof.armature --> models the inertia of a rotor (moving part of a motor)
        # actuator.gear --> actuator gain
        # link.transform.pos
        # link.inertia.transform.pos[y=1] --> center of mass (mj.body_ipos)
        # link.inertia.i[0, 0] --> Ixx (mj.body_inertia)
        # link.inertia.mass
        # link.invweight --> mj.body_invweight0[:, 0]  # todo: check if nonzero gradient w.r.t. link_invweight
        # dof.invweight --> mj.dof_invweight0   # todo: check if nonzero gradient w.r.t. dof_invweight
        # Set parameters
        _m = m_brax
        itransform = _m.link.inertia.transform.replace(pos=jnp.array([[0., offset, 0.]]))
        i = _m.link.inertia.i.at[0, 0, 0].set(0.5 * mass_weight * radius_weight ** 2)  # inertia of cylinder in local frame.
        inertia = _m.link.inertia.replace(transform=itransform, mass=jnp.array([mass_weight]), i=i)
        link = _m.link.replace(inertia=inertia, invweight=_m.link.invweight * link_invweight)
        actuator = _m.actuator.replace(gear=jnp.array([gear]))
        dof = _m.dof.replace(armature=jnp.array([jnp.abs(armature)]), damping=jnp.array([damping]),
                             invweight=_m.dof.invweight * dof_invweight)
        new_m = _m.replace(link=link, actuator=actuator, dof=dof)
        return new_m

    def init_ode_sys(params: Params, damping: float = 1e-4, armature: float = -1.e9, gear: float = 1e-2, mass_weight: float = 5e-2,
                     radius_weight: float = 2e-2, offset: float = 4e-2, friction_loss: float = 0.000975041213361349, d: float = 165.417960777425,
                     link_invweight: float = 1., dof_invweight: float = 1.):
        J = armature + 0.5*radius_weight**2*mass_weight + mass_weight*offset**2
        mass = mass_weight
        length = offset
        R = 7.73125142
        K = gear * R
        b = damping - K**2 / R
        c = friction_loss
        d = d
        m = ODEParams(dt=m_ODE.dt, J=J, mass=mass, length=length, b=b, K=K, R=R, c=c, d=d)
        return m

    def init_sys(params: Params, damping: float = 1e-4, armature: float = -1.e9, gear: float = 1e-2, mass_weight: float = 5e-2,
                 radius_weight: float = 2e-2, offset: float = 4e-2, friction_loss: float = 0.000975041213361349, d: float = 165.417960777425,
                 link_invweight: float = 1., dof_invweight: float = 1.):
        # Make all non negative
        damping = jnp.exp(damping)
        armature = jnp.exp(armature)
        gear = jnp.exp(gear)
        mass_weight = jnp.exp(mass_weight)
        radius_weight = jnp.exp(radius_weight)
        offset = jnp.exp(offset)
        friction_loss = jnp.exp(friction_loss)
        d = jnp.exp(d)
        link_invweight = jnp.exp(link_invweight)
        dof_invweight = jnp.exp(dof_invweight)
        if isinstance(params.sys, brax.base.System):
            m = init_brax_sys(params, damping, armature, gear, mass_weight, radius_weight, offset, friction_loss, d, link_invweight, dof_invweight)
        elif isinstance(params.sys, ODEParams):
            m = init_ode_sys(params, damping, armature, gear, mass_weight, radius_weight, offset, friction_loss, d, link_invweight, dof_invweight)
        else:
            m = init_mjx_sys(params, damping, armature, gear, mass_weight, radius_weight, offset, friction_loss, d, link_invweight, dof_invweight)
        return Params(sys=m, c=friction_loss, d=d)

    def init_brax_pipeline(params: Params, init_jpos: jax.typing.ArrayLike, init_jvel: jax.typing.ArrayLike):
        _m = params.sys
        # Set state.
        qpos = _m.init_q.at[0].set(init_jpos)
        qvel = jnp.array([init_jvel])
        pipeline_state = pipeline.init(_m, qpos, qvel)
        return pipeline_state

    def init_mjx_pipeline(params: Params, init_jpos: jax.typing.ArrayLike, init_jvel: jax.typing.ArrayLike):
        _m = params.sys
        # Set state.
        d = mjx.make_data(_m)
        qpos = d.qpos.at[0].set(init_jpos)
        qvel = d.qvel.at[0].set(init_jvel)
        d = d.replace(qpos=qpos, qvel=qvel)
        d = mjx.forward(_m, d)
        return d

    def init_ode_pipeline(params: Params, init_jpos: jax.typing.ArrayLike, init_jvel: jax.typing.ArrayLike):
        _m = params.sys
        # Set state.
        pipeline_state = jnp.array([init_jpos, init_jvel])
        return pipeline_state

    def init_pipeline(params: Params, init_jpos: jax.typing.ArrayLike, init_jvel: jax.typing.ArrayLike):
        if isinstance(params.sys, brax.base.System):
            return init_brax_pipeline(params, init_jpos, init_jvel)
        elif isinstance(params.sys, ODEParams):
            return init_ode_pipeline(params, init_jpos, init_jvel)
        else:
            return init_mjx_pipeline(params, init_jpos, init_jvel)

    def brax_step(params: Params, pipeline_state, a):
        jvel = pipeline_state.qd[0]
        activation = 2*sigmoid(params.d * jvel) - 1
        # activation = jnp.sign(jvel)
        friction = params.c * activation / params.sys.actuator.gear[0]
        a_friction = a - friction
        pipeline_state = pipeline.step(params.sys, pipeline_state, a_friction)
        y = dict(jpos=pipeline_state.q, jvel=pipeline_state.qd)
        return pipeline_state, y

    def mjx_step(params: Params, pipeline_state, a):
        pipeline_state = pipeline_state.replace(ctrl=a)
        pipeline_state = mjx.step(params.sys, pipeline_state)
        y = dict(jpos=pipeline_state.qpos, jvel=pipeline_state.qvel)
        return pipeline_state, y

    def ode_step(params: Params, pipeline_state, a):
        _m = params.sys
        new_pipeline_state = runge_kutta4(ode_pendulum, _m.dt, _m, pipeline_state, a[0])
        y = dict(jpos=new_pipeline_state[0][None], jvel=new_pipeline_state[1][None])
        return new_pipeline_state, y

    def step(params: Params, pipeline_state, a):
        if isinstance(params.sys, brax.base.System):
            pipeline_state, y = brax_step(params, pipeline_state, a)
        elif isinstance(params.sys, ODEParams):
            pipeline_state, y = ode_step(params, pipeline_state, a)
        else:
            pipeline_state, y = mjx_step(params, pipeline_state, a)
        return pipeline_state, y

    def rollout_fn(params: Params, init_jpos, init_jvel, actions):
        pipeline_state = init_pipeline(params, init_jpos, init_jvel)
        _step = partial(step, params)

        def fn(carry, a):
            pipeline_state = carry
            pipeline_state, y = _step(pipeline_state, a)
            return pipeline_state, y

        carry, y = jax.lax.scan(fn, pipeline_state, actions)

        # is_brax = isinstance(params.sys, brax.base.System)
        # output = dict(jpos=y.q, jvel=y.qd) if is_brax else dict(jpos=y.qpos, jvel=y.qvel)

        return y, y

    vmap_rollout = jax.vmap(rollout_fn, in_axes=(None, 0, 0, 0), out_axes=0)
    pmap_vmap_rollout = jax.pmap(vmap_rollout, axis_name="batch", in_axes=(None, 0, 0, 0), out_axes=0, devices=jax.devices("cpu"))
    jit_vmap_rollout = jax.jit(vmap_rollout, device=cpu_device)

    # Initialize true system
    J = 0.000159931461600856
    M = 0.0508581731919534
    offset = 0.0415233722862552
    radius_weight = 0.0508581731919534
    b = 1.43298488e-05
    K = 0.03333912
    R = 7.73125142
    params_true = {
        "damping": (b + K**2 / R),
        "armature": (J - 0.5*radius_weight**2*M - M*offset**2),
        "gear": K/R,  # 0.03333912/7.73
        "mass_weight": M,
        "radius_weight": radius_weight,
        "offset": offset,
        "friction_loss": 0.000975041213361349,
        "d": 165.417960777425,
        "link_invweight": 1.,
        "dof_invweight": 1.
    }
    log_params_true = jax.tree_map(lambda x: jnp.log(x), params_true)
    print("[params_true]: ", params_true)
    p_true = init_sys(p, **log_params_true)

    # Get train data
    # num_eps = 10
    # init_jpos = jnp.array(rnd.uniform(rnd.PRNGKey(0), shape=(num_eps,), minval=-jnp.pi, maxval=jnp.pi))
    # init_jvel = jnp.array(rnd.uniform(rnd.PRNGKey(1), shape=(num_eps,), minval=-8., maxval=8.))
    # actions = jnp.array(rnd.uniform(rnd.PRNGKey(1), shape=(num_eps, total_steps, 1), minval=-2., maxval=2.))
    # with timer("jit[data]", log_level=100):
    #     rollout, data = jit_vmap_rollout(m_true, init_jpos, init_jvel, actions)

    # Get real data
    mat = scipy.io.loadmat("/home/r2ci/Downloads/data.mat")
    mat_actions = mat["data"][0][0][5][0, 0]
    mat_state = mat["data"][0][0][2][0, 0]
    mat_T = jnp.arange(mat_actions.shape[0]) * (1/150)
    interp_T = jnp.arange(int(mat_T[-1]/dt/total_steps) * total_steps) * dt
    interp_actions = jnp.interp(interp_T, mat_T, mat_actions[:, 0])[None, :, None]
    interp_jpos = jnp.interp(interp_T, mat_T, mat_state[:, 0])[None, :]
    interp_jvel = jnp.interp(interp_T, mat_T, mat_state[:, 1])[None, :]
    interp_init_jpos = interp_jpos[:, 0]
    interp_init_jvel = interp_jvel[:, 0]
    T = interp_T.reshape(-1, total_steps)[:, :]
    jpos = interp_jpos.reshape(-1, total_steps)[:, :]
    jvel = interp_jvel.reshape(-1, total_steps)[:, :]
    actions = interp_actions.reshape(-1, total_steps, 1)[:, :]
    # num_cpus = len(jax.devices("cpu"))
    # repeat = 80*num_cpus  # Repeats jpos, jvel, actions in axis=0 direction
    # T = jnp.repeat(T, repeat, axis=0).reshape(num_cpus, -1, total_steps)
    # jpos = jnp.repeat(jpos, repeat, axis=0).reshape(num_cpus, -1, total_steps)
    # jvel = jnp.repeat(jvel, repeat, axis=0).reshape(num_cpus, -1, total_steps)
    # actions = jnp.repeat(actions, repeat, axis=0).reshape(num_cpus, -1, total_steps, 1)
    # data = {"jpos": jpos[:, :, :, None], "jvel": jvel[:, :, :, None]}
    # init_jpos = jpos[:, :, 0]
    # init_jvel = jvel[:, :, 0]
    # with timer("pmap[rollout(m_true)]", log_level=100):
    #     rollout, data_true = pmap_vmap_rollout(m_true, init_jpos, init_jvel, actions)
    # with timer("pmap_eval[rollout(m_true)]", log_level=100):
    #     for i in range(10):
    #         rollout, data_true = pmap_vmap_rollout(m_true, init_jpos, init_jvel, actions)
    #         data_true["jpos"].block_until_ready()

    num_eps = T.shape[0]
    data = {"jpos": jpos[:, :, None], "jvel": jvel[:, :, None]}
    init_jpos = jpos[:, 0]
    init_jvel = jvel[:, 0]
    with timer("jit[rollout(p_true)]", log_level=100):
        rollout, data_true = jit_vmap_rollout(p_true, init_jpos, init_jvel, actions)
    with timer("eval[rollout(p_true)]", log_level=100):
        for i in range(10):
            _ = jit_vmap_rollout(p_true, init_jpos, init_jvel, actions)

    # interp_rollout, interp_data = jit_vmap_rollout(m_true, interp_init_jpos, interp_init_jvel, interp_actions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interp_T, interp_jpos[0])
    ax.plot(interp_T, interp_jvel[0])
    ax.plot(T.T, data_true["jpos"][:, :, 0].T)
    ax.plot(T.T, data_true["jvel"][:, :, 0].T)
    # ax.plot(T.T, actions[:, :, 0].T)
    # plt.show()

    def residual_fn(params, *args):
        _params = params.copy()
        # _params = jax.tree_util.tree_map(lambda x: jnp.abs(x), _params)
        # _params.update(params)
        p_init = init_sys(p, **_params)
        rollout_pred, data_pred = rollout_fn(p_init, interp_init_jpos[0], interp_init_jvel[0], interp_actions[0])
        res_jpos = interp_jpos[0, 1:] - data_pred["jpos"][:-1, 0]
        res_jvel = interp_jvel[0, 1:] - data_pred["jvel"][:-1, 0]
        res = jnp.concatenate([res_jpos, res_jvel], axis=0)
        return res

    jit_vmap_residual_fn = jax.jit(jax.vmap(residual_fn, in_axes=(0,), out_axes=0))

    def loss(params):
        _params = params.copy()
        _params.update(params)
        p_init = init_sys(p, **params)
        rollout_pred, data_pred = jit_vmap_rollout(p_init, init_jpos, init_jvel, actions)
        loss_jpos = jnp.square(data["jpos"][:, 1:] - data_pred["jpos"][:, :-1]).mean()
        loss_jvel = jnp.square(data["jvel"][:, 1:] - data_pred["jvel"][:, :-1]).mean()
        loss = loss_jpos + 0.01*loss_jvel
        return loss

    loss_grad = jax.value_and_grad(loss)

    def optimize(params, lr=1e-3, steps=100):

        def gradient_descent(params, _):
            loss, grad = loss_grad(params)
            new_params = jax.tree_util.tree_map(lambda x, dx: x - lr * dx, params, grad)
            return new_params, loss

        params_final, loss_hist = jax.lax.scan(gradient_descent, params, jnp.arange(steps))
        return params_final, loss_hist

    jit_optimize = jax.jit(optimize, static_argnums=(1, 2), device=cpu_device)

    # Initialize
    params_init = {"damping": params_true["damping"]*1.0,
                   "armature": params_true["armature"]*1.0,
                   "gear": params_true["gear"]*1.0,
                   "mass_weight": params_true["mass_weight"]*1.0,
                   "radius_weight": params_true["radius_weight"]*1.0,
                   "offset": params_true["offset"]*1.0,
                   "friction_loss": params_true["friction_loss"]*1.0,
                   # "d": params_true["d"],
                   }
    log_params_init = jax.tree_map(lambda x: jnp.log(x), params_init)
    p_init = init_sys(p, **log_params_init)
    log_params_pred = log_params_init.copy()
    with timer("jit[rollout(p_init)]", log_level=100):
        _, data_init = jit_vmap_rollout(p_init, init_jpos, init_jvel, actions)

    # Vmap initial conditions
    solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-7, norm=optx.rms_norm, verbose=frozenset({"step", "accepted", "step_size", "loss"}))
    solver = optx.BestSoFarRootFinder(solver)

    def least_squares(log_params_init):
        sol = optx.least_squares(residual_fn, solver, log_params_init, max_steps=100, throw=False)
        new_sol = optx.Solution(sol.value, None, sol.aux, sol.stats, None)
        return new_sol

    jit_vmap_lsq = jax.jit(jax.vmap(least_squares, in_axes=(0,), out_axes=0), device=cpu_device)
    pmap_lsq = jax.pmap(least_squares, in_axes=(0,), out_axes=0, devices=jax.devices("cpu"))

    # Initialize initial conditions
    def get_init_log_params(rng, params_init):
        params = jax.tree_map(lambda x: x * rnd.uniform(rng, minval=1.5, maxval=2), params_init)
        log_params = jax.tree_map(lambda x: jnp.log(x), params)
        return log_params

    num_copies = len(jax.devices("cpu"))
    vmap_get_init_log_params = jax.vmap(get_init_log_params, in_axes=(0, None), out_axes=0)
    rngs = rnd.split(rnd.PRNGKey(0), num_copies)
    all_log_params_init = vmap_get_init_log_params(rngs, params_init)
    with timer("jit[jit_vmap_lsq]", log_level=100):
        sols = pmap_lsq(all_log_params_init)
        sols.stats["num_steps"].block_until_ready()
    all_log_params_final = sols.value
    all_params_final = jax.tree_map(lambda x: jnp.exp(x), all_log_params_final)
    print(sols.stats)
    print("[all_params_final] ", all_params_final)

    with timer("jit[optx.least_squares]", log_level=100):
        sol = optx.least_squares(residual_fn, solver, log_params_pred, max_steps=100, throw=False)
    log_params_final = sol.value
    params_final = jax.tree_map(lambda x: jnp.exp(x), log_params_final)
    print(optx.RESULTS[sol.result])
    print(sol.stats)
    print("[params_final] ", params_final)

    # Gauss Newton
    # gn = jaxopt.GaussNewton(residual_fn, tol=1e-7, maxiter=100, verbose=True)
    # with timer("jit[gn.run]", log_level=100):
    #     log_params_final = gn.run(log_params_pred).params
    # params_final = jax.tree_map(lambda x: jnp.exp(x), log_params_final)
    # loss, grad = loss_grad(log_params_final)
    # print("[params_final] loss: ", loss, "grad: ", grad)
    # print("[params_final] ", params_final)

    # msg_params = {k: f"{v:5f} --> {params_pred[k] :5f} ({params_true[k]:5f})" for k, v in params_init.items()}
    # print("START: ", msg_params)
    # for i in range(10):
    #     t = timer("jit[optimize]", log_level=0)
    #     with t:
    #         params_pred, loss_hist = jit_optimize(params_pred, lr=1e-6, steps=10)
    #     duration = t.duration
    #     msg_params = {k: f"{v:5f} --> {params_pred[k]:5f} ({params_true[k]:5f})" for k, v in params_init.items()}
    #     msg_loss = f"loss: {loss_hist[0]:5f} --> {loss_hist[-1]:5f}"
    #     msg_time = f"time: {duration:2f}"
    #     print(msg_time, msg_loss, msg_params)

    # data_final = jit_vmap_rollout(init_sys(p, **log_params_final), init_jpos, init_jvel, actions)[1]
    from rex.jax_utils import tree_take
    data_final = jit_vmap_rollout(init_sys(p, **tree_take(all_log_params_final, 1)), init_jpos, init_jvel, actions)[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interp_T, interp_jpos[0])
    ax.plot(interp_T, interp_jvel[0])
    ax.plot(T.T, data_init["jpos"][:, :, 0].T)
    ax.plot(T.T, data_init["jvel"][:, :, 0].T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interp_T, interp_jpos[0])
    ax.plot(interp_T, interp_jvel[0])
    ax.plot(T.T, data_final["jpos"][:, :, 0].T)
    ax.plot(T.T, data_final["jvel"][:, :, 0].T)
    # ax.plot(T.T, actions[:, :, 0].T)
    plt.show()

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 20))
    # t = jnp.vstack([jnp.arange(total_steps) * m.dt]*num_eps)
    # axes[0, 0].plot(t.T, data["jpos"][:, :, 0].T)
    # axes[1, 0].plot(t.T, data["jpos"][:, :, 0].T - data_final["jpos"][:, :, 0].T)
    # axes[0, 0].set_title("jpos")
    # axes[0, 0].set_title("error[jpos]")
    # axes[0, 1].plot(t.T, data["jvel"][:, :, 0].T)
    # axes[1, 1].plot(t.T, data["jvel"][:, :, 0].T - data_final["jvel"][:, :, 0].T)
    # axes[0, 1].set_title("jvel")
    #
    # plt.show()

    # # Initialize (NO CONTROL)
    # jax.config.update("jax_debug_nans", True)
    # jit_env_reset = jax.jit(env_reset, device=cpu_device)
    # jit_env_step = jax.jit(pipeline.step, device=cpu_device)
    # rng = jax.random.PRNGKey(seed=1)
    # with timer("jit[reset]", log_level=100):
    #     m_exp, pipeline_state = jit_env_reset(m, 0.)
    # with timer("eval[reset]", log_level=100):
    #     _ = jit_env_reset(m, 0.)
    # # Step
    # with timer("jit[step]", log_level=100):
    #     _ = jit_env_step(m_exp, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(1))
    # with timer("eval[step]", log_level=100):
    #     _ = jit_env_step(m_exp, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(1))
    #
    #
    # rollout = [pipeline_state]

