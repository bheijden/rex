from typing import Dict, Union, Callable, Any
import dill as pickle
import tqdm
import os
import multiprocessing
import itertools
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import distrax

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
import envs.crazyflie.systems as csys
from envs.crazyflie.ode import plot_data

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu'))
    CPU_DEVICE = next(CPU_DEVICES)
    RNG = jax.random.PRNGKey(0)
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    # ORDER = ["mocap", "world", "pid", "agent", "estimator", "supervisor"]
    # CSCHEME = {"world": "gray", "mocap": "grape", "estimator": "violet", "agent": "lime", "pid": "green", "actuator": "indigo", "supervisor": "gray"}
    # MOCK = "ode"  # "ode", "copilot, or "real" else todo: "real"
    # CENTER = onp.array([0.0, 0.0, 1.75])
    # RADIUS = 1.25
    EPS_INDEX = -1
    SKIP_SEC = 3.0
    # PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
    # AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
    RECORD_FILE = f"{LOG_DIR}/data_evaluate_0.60A_10s.pkl"
    # RECORD_FILE = f"{LOG_DIR}/data_first_runs/data_evaluate_0.5A_6s_nocrash.pkl"
    # RECORD_FILE = f"{LOG_DIR}/data_first_runs/data_evaluate_0.5A_6s_nocrash.pkl"
    # RECORD_FILE = f"{LOG_DIR}/data_first_runs/data_evaluate_0.90A_6s_new_nocrash.pkl"

    # OPen rollout file
    with open(RECORD_FILE, "rb") as f:
        record = pickle.load(f)

    # Select first episode
    record = record[EPS_INDEX]

    CENTER = record.nodes["supervisor"].params.center
    RADIUS = record.nodes["supervisor"].params.fixed_radius

    # Plot
    mocap = record.nodes["mocap"].steps
    mocap_ia = jax.vmap(mocap.output.static_in_agent_frame, in_axes=(0, None))(mocap.output, CENTER)
    estimator = record.nodes["estimator"].steps
    agent = record.nodes["agent"].steps
    pid = record.nodes["pid"].steps

    fig, axes = plot_data(output={"att": mocap.output.att,
                                  "pos": mocap.output.pos,
                                  "pos_ia": mocap_ia.pos,
                                  "vel_ia": mocap_ia.vel,
                                  "pwm_ref": pid.output.pwm_ref,
                                  "phi_ref": pid.output.phi_ref,
                                  "theta_ref": pid.output.theta_ref,
                                  "z_ref": pid.output.z_ref},
                          ts={"att": mocap.ts_start,
                              "pos": mocap.ts_start,
                              "pos_ia": mocap.ts_start,
                              "vel_ia": mocap.ts_start,
                              "pwm_ref": pid.ts_end,
                              "phi_ref": pid.ts_end,
                              "theta_ref": pid.ts_end,
                              "z_ref": pid.ts_end},
                          # ts_max=3.0,
                          )
    plt.show()