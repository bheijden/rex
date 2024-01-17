from typing import Union
import time
import os

import scipy.io

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import dill as pickle
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as onp
import jax.random as rnd
from jax.tree_util import tree_map

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
# cpu_device = jax.devices('gpu')[0]
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

from rex.utils import timer


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath
    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


if __name__ == "__main__":
    print("loading system")
    m = mjcf.load('/home/r2ci/rex/envs/pendulum/assets/disk_pendulum.xml')
    print(f"degrees of freedom: {m.qd_size()}")

    # Determine collision pairs
    print("\nCOLLISIONS")
    from brax.geometry.contact import _geom_pairs

    for (geom_i, geom_j) in _geom_pairs(m):
        # print(geom_i.link_idx, geom_j.link_idx)
        name_i = m.link_names[geom_i.link_idx[0]] if geom_i.link_idx is not None else "world"
        name_j = m.link_names[geom_j.link_idx[0]] if geom_j.link_idx is not None else "world"
        print(f"collision pair: {name_i} --> {name_j}")

    # Actuators
    print("\nACTUATOR SIZE")
    print(f"actuator size: {m.act_size()}")
    q_id = m.actuator.q_id[:1]

    # DOFS
    print("\nDEGREES OF FREEDOM SIZE")
    print(f"degrees of freedom: {m.qd_size()}")

    # Select pipeline
    pipeline = gen_pipeline

    # Overwrite xml values
    # m = m.replace(dt=0.015, solver_maxls=10)  # generalized
    # m = m.replace(dt=0.01,
    #               baumgarte_erp=0.1,  # default=0.1
    #               spring_inertia_scale=1.0, # default=0.0
    #               spring_mass_scale=0.0, # default=0.0
    #               vel_damping=0.0,  # default=0.0
    #               ang_damping=-0.05,  # default=0.0
    #               joint_scale_pos=0.5,  # default=0.5
    #               joint_scale_ang=0.1,  # default=0.2
    #              )

    # Get sampling time (0.8s horizon needed)
    horizon = 5.0
    total_steps = ceil(horizon / m.dt)
    horizon = total_steps * m.dt
    print(f"\nTIME")
    print(f"dt: {m.dt}, total_steps: {total_steps}, horizon: {horizon}")

    # Print relevant parameters
    parameters_dict = {
        pos_pipeline: [
            'dt',
            'joint_scale_pos',
            'joint_scale_ang',
            'collide_scale',
            'ang_damping',  # shared with `brax.physics.spring`
            'vel_damping',  # shared with `brax.physics.spring`
            'baumgarte_erp',  # shared with `brax.physics.spring`
            'spring_mass_scale',  # shared with `brax.physics.spring`
            'spring_inertia_scale',  # shared with `brax.physics.spring`
            'constraint_ang_damping',  # shared with `brax.physics.spring`
            'elasticity',  # shared with `brax.physics.spring`
        ],
        spr_pipeline: [
            'dt',
            'constraint_stiffness',
            'constraint_limit_stiffness',
            'constraint_vel_damping',
            'ang_damping',  # shared with `brax.physics.positional`
            'vel_damping',  # shared with `brax.physics.positional`
            'baumgarte_erp',  # shared with `brax.physics.positional`
            'spring_mass_scale',  # shared with `brax.physics.positional`
            'spring_inertia_scale',  # shared with `brax.physics.positional`
            'constraint_ang_damping',  # shared with `brax.physics.positional`
            'elasticity',  # shared with `brax.physics.positional`
        ],
        gen_pipeline: [
            'dt',
            'matrix_inv_iterations',
            'solver_iterations',
            'solver_maxls',
        ]
        # The 'convex' parameter is not included due to its unknown usage.
    }

    print(f"\nPARAMETERS: {pipeline.__name__}")
    for p in parameters_dict[pipeline]:
        try:
            print(f"{p}: {m.__getattribute__(p)}")
            continue
        except AttributeError:
            pass
        try:
            print(f"{p}: {m.link.__getattribute__(p)}")
            continue
        except AttributeError:
            pass
        try:
            print(f"{p}: {m.geoms[0].__getattribute__(p)}")
            continue
        except AttributeError:
            pass

    # dof.damping --> restorative force back to zero velocity
    # dof.armature --> models the inertia of a rotor (moving part of a motor)
    # actuator.gear --> actuator gain
    # link.transform.pos
    # link.inertia.transform.pos[y=1] --> center of mass (mj.body_ipos)
    # link.inertia.i[0, 0] --> Ixx (mj.body_inertia)
    # link.inertia.mass
    # link.invweight --> mj.body_invweight0[:, 0]  # todo: check if nonzero gradient w.r.t. link_invweight
    # dof.invweight --> mj.dof_invweight0   # todo: check if nonzero gradient w.r.t. dof_invweight

    # todo: Debug
    # qpos = m.init_q.at[0].set(1.57)
    # qvel = jnp.array([0])
    # pipeline_state = pipeline.init(m, qpos, qvel)


    def init_sys(_m, damping: float = 1e-4, armature: float = -1.e9, gear: float = 1e-2, mass_weight: float = 5e-2,
                 radius_weight: float = 2e-2, offset: float = 4e-2, link_invweight: float = 1., dof_invweight: float = 1.):
        # Set parameters
        itransform = _m.link.inertia.transform.replace(pos=jnp.array([[0., offset, 0.]]))
        i = _m.link.inertia.i.at[0, 0, 0].set(0.5 * mass_weight * radius_weight ** 2)  # inertia of cylinder in local frame.
        inertia = _m.link.inertia.replace(transform=itransform, mass=jnp.array([mass_weight]), i=i)
        link = _m.link.replace(inertia=inertia, invweight=_m.link.invweight*link_invweight)
        actuator = _m.actuator.replace(gear=jnp.array([gear]))
        dof = _m.dof.replace(armature=jnp.array([jnp.exp(armature)]), damping=jnp.array([damping]), invweight=_m.dof.invweight*dof_invweight)
        new_m = _m.replace(link=link, actuator=actuator, dof=dof)
        return new_m


    def init_pipeline(_m, init_jpos: jax.typing.ArrayLike, init_jvel: jax.typing.ArrayLike):
        # Set state.
        qpos = _m.init_q.at[0].set(init_jpos)
        qvel = jnp.array([init_jvel])
        pipeline_state = pipeline.init(_m, qpos, qvel)
        return pipeline_state

    def rollout(_m, init_jpos, init_jvel, actions):
        # Initialize pipeline
        pipeline_state = init_pipeline(_m, init_jpos, init_jvel)

        def step(carry, a):
            pipeline_state = carry
            pipeline_state = pipeline.step(_m, pipeline_state, a)
            return pipeline_state, pipeline_state

        carry, y = jax.lax.scan(step, pipeline_state, actions)

        output = dict(jpos=y.q, jvel=y.qd)

        return y, output

    jit_vmap_rollout = jax.jit(jax.vmap(rollout, in_axes=(None, 0, 0, 0), out_axes=0))

    # Initialize true system
    # constrain to non-negative.
    M = 0.05085817
    b = 1.43298488e-05
    K = 0.03333912
    R = 7.73125142
    params_true = {
        "damping": 1*(b + K**2 / R),
        "armature": jnp.log(0.00015993 - 0.5*0.02**2*M - M*0.04**2),
        "gear": K/R,  # 0.03333912/7.73
        "mass_weight": M,
        "radius_weight": 2e-2,
        "offset": 4e-2,
        "link_invweight": 1.,
        "dof_invweight": 1.
    }
    m_true = init_sys(m, **params_true)
    # pipeline.init(m_true, jnp.array([0.]), jnp.array([0.]))

    # Get train data
    # num_eps = 10
    # init_jpos = jnp.array(rnd.uniform(rnd.PRNGKey(0), shape=(num_eps,), minval=-jnp.pi, maxval=jnp.pi))
    # init_jvel = jnp.array(rnd.uniform(rnd.PRNGKey(1), shape=(num_eps,), minval=-8., maxval=8.))
    # actions = jnp.array(rnd.uniform(rnd.PRNGKey(1), shape=(num_eps, total_steps, m.act_size()), minval=-2., maxval=2.))
    # with timer("jit[data]", log_level=100):
    #     rollout, data = jit_vmap_rollout(m_true, init_jpos, init_jvel, actions)

    # Get real data
    mat = scipy.io.loadmat("/home/r2ci/Downloads/data.mat")
    mat_actions = mat["data"][0][0][5][0, 0]
    mat_state = mat["data"][0][0][2][0, 0]
    mat_T = jnp.arange(mat_actions.shape[0]) * (1/150)
    interp_T = jnp.arange(int(mat_T[-1]/m.dt/total_steps) * total_steps) * m.dt
    interp_actions = jnp.interp(interp_T, mat_T, mat_actions[:, 0])[None, :, None]
    interp_jpos = jnp.interp(interp_T, mat_T, mat_state[:, 0])[None, :]
    interp_jvel = jnp.interp(interp_T, mat_T, mat_state[:, 1])[None, :]
    interp_init_jpos = interp_jpos[:, 0]
    interp_init_jvel = interp_jvel[:, 0]
    num_eps = 100
    T = interp_T.reshape(-1, total_steps)[:num_eps, :200]
    jpos = interp_jpos.reshape(-1, total_steps)[:num_eps, :200]
    jvel = interp_jvel.reshape(-1, total_steps)[:num_eps, :200]
    actions = interp_actions.reshape(-1, total_steps, m.act_size())[:num_eps, :200]
    num_eps = T.shape[0]
    data = {"jpos": jpos[:, :, None], "jvel": jvel[:, :, None]}
    init_jpos = jpos[:, 0]
    init_jvel = jvel[:, 0]
    with timer("jit[real_data]", log_level=100):
        rollout, data_true = jit_vmap_rollout(m_true, init_jpos, init_jvel, actions)

    # interp_rollout, interp_data = jit_vmap_rollout(m_true, interp_init_jpos, interp_init_jvel, interp_actions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interp_T, interp_jpos[0])
    ax.plot(interp_T, interp_jvel[0])
    ax.plot(T.T, data_true["jpos"][:, :, 0].T)
    ax.plot(T.T, data_true["jvel"][:, :, 0].T)
    # ax.plot(T.T, actions[:, :, 0].T)
    # plt.show()


    def loss(params):
        _params = params.copy()
        _params.update(params)
        m_init = init_sys(m, **params)
        rollout_pred, data_pred = jit_vmap_rollout(m_init, init_jpos, init_jvel, actions)
        loss_jpos = jnp.square(data["jpos"][:, 1:] - data_pred["jpos"][:, :-1]).mean()
        loss_jvel = jnp.square(data["jvel"][:, 1:] - data_pred["jvel"][:, :-1]).mean()
        loss = loss_jpos + 0.0*loss_jvel
        return loss

    loss_grad = jax.value_and_grad(loss)

    def optimize(params, lr=1e-3, steps=100):

        def gradient_descent(params, _):
            loss, grad = loss_grad(params)
            new_params = jax.tree_util.tree_map(lambda x, dx: x - lr * dx, params, grad)
            return new_params, loss

        params_final, loss_hist = jax.lax.scan(gradient_descent, params, jnp.arange(steps))
        return params_final, loss_hist

    jit_optimize = jax.jit(optimize, static_argnums=(1, 2))

    # Initialize
    params_init = {"damping": params_true["damping"],
                   # "armature": params_true["armature"],
                   "gear": params_true["gear"],
                   "mass_weight": params_true["mass_weight"],
                   "radius_weight": params_true["radius_weight"],
                   }

    # Optimize
    params_pred = params_init.copy()

    with timer("jit[loss_grad]", log_level=100):
        loss, grad = loss_grad(params_pred)
        print("loss: ", loss, "grad: ", grad)

    msg_params = {k: f"{v} --> {params_pred[k]} ({params_true[k]})" for k, v in params_init.items()}
    print("START: ", msg_params)
    for i in range(10):
        with timer("jit[optimize]", log_level=100 if i == 0 else 0):
            params_pred, loss_hist = jit_optimize(params_pred, lr=1e-6, steps=100)
            msg_params = {k: f"{v} --> {params_pred[k]} ({params_true[k]})" for k, v in params_init.items()}
            msg_loss = f"loss: {loss_hist[0]} --> {loss_hist[-1]}"
            print(msg_params, msg_loss)

    params_final = params_true.copy()
    params_final.update(params_pred)
    params_start = params_true.copy()
    params_start.update(params_init)
    data_final = jit_vmap_rollout(init_sys(m, **params_final), init_jpos, init_jvel, actions)[1]
    # data_start = jit_vmap_rollout(init_sys(m, **params_start), init_jpos, init_jvel, actions)[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interp_T, interp_jpos[0])
    ax.plot(interp_T, interp_jvel[0])
    ax.plot(T.T, data_final["jpos"][:, :, 0].T)
    ax.plot(T.T, data_final["jvel"][:, :, 0].T)
    # ax.plot(T.T, actions[:, :, 0].T)
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 20))
    t = jnp.vstack([jnp.arange(total_steps) * m.dt]*num_eps)
    axes[0, 0].plot(t.T, data["jpos"][:, :, 0].T)
    axes[1, 0].plot(t.T, data["jpos"][:, :, 0].T - data_final["jpos"][:, :, 0].T)
    axes[0, 0].set_title("jpos")
    axes[0, 0].set_title("error[jpos]")
    axes[0, 1].plot(t.T, data["jvel"][:, :, 0].T)
    axes[1, 1].plot(t.T, data["jvel"][:, :, 0].T - data_final["jvel"][:, :, 0].T)
    axes[0, 1].set_title("jvel")

    plt.show()

    # Initialize (NO CONTROL)
    jax.config.update("jax_debug_nans", True)
    jit_env_reset = jax.jit(env_reset, device=cpu_device)
    jit_env_step = jax.jit(pipeline.step, device=cpu_device)
    rng = jax.random.PRNGKey(seed=1)
    with timer("jit[reset]", log_level=100):
        m_exp, pipeline_state = jit_env_reset(m, 0.)
    with timer("eval[reset]", log_level=100):
        _ = jit_env_reset(m, 0.)
    # Step
    with timer("jit[step]", log_level=100):
        _ = jit_env_step(m_exp, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(m.act_size()))
    with timer("eval[step]", log_level=100):
        _ = jit_env_step(m_exp, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(m.act_size()))


    rollout = [pipeline_state]

