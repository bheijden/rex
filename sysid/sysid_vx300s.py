from functools import partial
from typing import Union, Tuple
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
import numpy as onp
from jax._src.interpreters import batching
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
from rex.jax_utils import tree_extend
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
from brax.base import Transform, System
from brax.io import mjcf
from brax.generalized import pipeline as gen_pipeline
from brax.positional import pipeline as pos_pipeline
from brax.spring import pipeline as spr_pipeline
from brax import base, math
from flax import struct

import mujoco
from mujoco import mjx

import envs.vx300s as vx300s
from rex.utils import timer


CSCHEME = {"planner": "indigo", "controller": "violet", "armactuator": "grape", "armsensor": "pink"}
CSCHEME.update({"cm": "blue", "cost": "red", "cost_orn": "pink", "cost_down": "grape", "cost_align": "violet",
                "cost_height": "indigo", "cost_near": "blue", "cost_dist": "cyan", "simulation": "blue"})
ECOLOR, FCOLOR = oc.cscheme_fn(CSCHEME)

# Import the required modules
from sysid import utils, vx300s, lsq, cem

if __name__ == "__main__":
    backend = vx300s.BraxBackend()
    base_params = backend.init_backend(dt_sysid=0.09)
    residual = partial(vx300s.residual, backend)

    # Get time
    dt = base_params.sys.dt
    dt_sysid = base_params.dt_sysid
    substeps = base_params.substeps

    # Prepare data
    LOG_DIR = f"/home/r2ci/rex/paper/logs/2023-12-12-1636_real_2ndcalib_rex_randomeps_MCS_recorded_VarHz_3iter_vx300s"
    data = vx300s.load_or_gen_data(cache_dir=".", log_dir=LOG_DIR, dt_sysid=dt_sysid, xml_path=vx300s.BraxBackend.xml_path)
    armactuator = vx300s.Action(jpos=data["jpos_target"], jvel=jnp.zeros_like(data["jpos_target"]))
    armsensor = data["armsensor"]
    boxsensor = data["boxsensor"]
    armactuator, armsensor, boxsensor = jax.tree_util.tree_map(lambda x: x[:2, :100], (armactuator, armsensor, boxsensor))
    yaw = jax.vmap(jax.vmap(boxsensor.static_wrapped_yaw))(boxsensor.boxorn)
    init_boxpos = boxsensor.boxpos[:, 0]
    init_boxyaw = yaw[:, 0]
    init_goalpos = data["goalpos"][:2]
    init_jpos = armsensor.jpos[:, 0]
    init_jvel = jnp.zeros_like(armsensor.jpos[:, 0])

    # Initial params
    # base_params = vx300s.Params.default(dt_sysid=dt_sysid, substeps=substeps, sys=m_brax)
    pre_s = vx300s.State(pipeline_state=None, init_boxpos=init_boxpos, init_boxyaw=init_boxyaw, init_goalpos=init_goalpos, init_jpos=init_jpos, init_jvel=init_jvel)
    actions = jax.tree_util.tree_map(lambda x: x[:, :-1], armactuator)  # exclude last step, because we don't have the label for the last step
    init_y_ys = vx300s.Output(arm_output=armsensor, box_output=boxsensor)
    args = (base_params, pre_s, actions, init_y_ys)

    # Initial params
    init_params = vx300s.Params.default()
    init_params = eqx.filter(init_params, lambda x: sum(x.shape) <= 1)

    # CEM
    u_min = jax.tree_util.tree_map(lambda x: x * 0.5, init_params)
    u_max = jax.tree_util.tree_map(lambda x: x * 1.5, init_params)
    solver = cem.CEMSolver.init(u_min=u_min, u_max=u_max, evolution_smoothing=0.1, elite_portion=0.1, num_samples=100)
    _cem = partial(cem.cem, residual, solver, max_steps=100, verbose=True)
    with timer("cem", log_level=100):
        sol = jax.jit(_cem)(init_params, args)

    # Least squares
    # todo: add regularization prior
    solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-7, norm=optx.rms_norm, verbose=frozenset({"step", "accepted", "step_size", "loss"}))
    solver = optx.BestSoFarRootFinder(solver)
    _lsq = partial(lsq.least_squares, residual, solver, max_steps=100, throw=False)
    with timer("least squares", log_level=100):
        sol = eqx.filter_jit(_lsq)(init_params, args)
    print(sol.value)
    print(sol.stats)

    # Rollout
    init_s = jax.vmap(backend.init_pipeline, in_axes=(None, 0))(base_params, pre_s)  # Initial state
    final_s, ys = jax.vmap(backend.rollout, in_axes=(None, 0, 0))(base_params, init_s, actions)

    # todo: first, learn gains for each joint together with the box

    # Plot jpos
    timestamps = data["timestamps"]
    jpos = data["jpos"]
    jpos_target = data["jpos_target"]
    jpos_pred = ys.arm_output.jpos
    jpos_err_abs = data["jpos_err_abs"]
    jpos_err_pred_abs = jnp.abs(jpos - jpos_pred)
    fig_jpos, axes_jpos = plt.subplots(3, 6, figsize=(24, 12))
    for joint_idx in range(6):
        ax_jpos = axes_jpos[0, joint_idx]
        ax_err = axes_jpos[1, joint_idx]
        ax_err_pred = axes_jpos[2, joint_idx]

        joint_labels = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

        # Set axis labels
        ax_jpos.set_title(f"{joint_labels[joint_idx]}")
        ax_err.set_ylim([0, 0.3])
        ax_err.set_xlabel("time (s)")

        # Plot controller output
        ax_jpos.plot(timestamps[:, :].T, jpos_target[:, :, joint_idx].T, label=f"armactuator", color=ECOLOR["armactuator"])
        ax_jpos.plot(timestamps[:, :].T, jpos_pred[:, :, joint_idx].T, label=f"simulation", color=ECOLOR["simulation"])
        ax_err.plot(timestamps[:, :].T, jpos_err_abs[:, :, joint_idx].T, label=f"armactuator", color=ECOLOR["armactuator"])

        # Plot armsensor output
        ax_jpos.plot(timestamps[:, :].T, jpos[:, :, joint_idx].T, label=f"armsensor", color=ECOLOR["armsensor"])

        #
        ax_err_pred.plot(timestamps[:, :].T, jpos_err_pred_abs[:, :, joint_idx].T, label=f"simulation", color=ECOLOR["simulation"])

    axes_jpos[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    axes_jpos[0, 0].set_ylabel("joint position (rad)")
    axes_jpos[1, 0].set_ylabel("error (rad)")


