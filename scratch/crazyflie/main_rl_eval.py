import dill as pickle
import tqdm
import functools
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
JAX_USE_CACHE = False
# if JAX_USE_CACHE:
    # os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

# Cache settings
if JAX_USE_CACHE:
    # More info: https://github.com/google/jax/pull/22271/files?short_path=71526fb#diff-71526fb9807ead876cbde1c3c88a868e56d49888023dd561e6705d403ab026c0
    jax.config.update("jax_compilation_cache_dir", "./cache-rl")
    # -1: disable the size restriction and prevent overrides.
    # 0: Leave at default (0) to allow for overrides.
    #    The override will typically ensure that the minimum size is optimal for the file system being used for the cache.
    # > 0: the actual minimum size desired; no overrides.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # A computation will only be written to the persistent cache if the compilation time is longer than the specified value.
    # It is defaulted to 1.0 second.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_explain_cache_misses", False)  # True --> results in error

import supergraph
import rex
from rex import base, jax_utils as jutils, constants
from rex.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rex.utils import timer
import rex.utils as rutils
from rex.jax_utils import same_structure
from rex import artificial
import envs.crazyflie.systems as psys
import envs.crazyflie.ppo as ppo_config
import rex.rl as rl
from envs.crazyflie.ode import metrics

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    # Make sysid nodes
    jnp.set_printoptions(precision=4, suppress=True)
    onp.set_printoptions(precision=4, suppress=True)
    SEED = 0
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    TS_MAX = 9.0
    WORLD_RATE = 100.
    STD_TH = 0.02  # Overwrite std_th in estimator and camera --> None to keep default
    SUPERVISOR = "agent"
    # Input files
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    RECORD_FILE = f"{LOG_DIR}/data_rl.pkl"  # todo:  change
    PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"  # todo:  change
    ROLLOUT_FILE = f"{LOG_DIR}/rollout.pkl"
    AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
    # Output files
    METRICS_FILE = f"{LOG_DIR}/metrics.pkl"
    HTML_FILE = f"{LOG_DIR}/rollout.html"
    SAVE_FILE = True
    # Evaluation settings
    POSITION = onp.array([0.0, 0.0, 1.75])
    CENTER = onp.array([0.0, 0.0, 1.75])
    RADIUS = 1.25
    SKIP_SEC = 3.0

    # Get rollout
    if False:
        # Open rollout file
        with open(ROLLOUT_FILE, "rb") as f:
            rollout = pickle.load(f)
        gs = rollout.next_gs
        model = gs.params["agent"].model
        # todo: not doing anything with

    # Seed
    rng = jax.random.PRNGKey(SEED)

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    nodes = psys.simulated_system(record, world_rate=WORLD_RATE)

    # Get graph
    graphs_real = record.to_graph()
    graphs_real = graphs_real.filter(nodes)  # Filter nodes

    # Generate computation graph
    rng, rng_graph = jax.random.split(rng)
    graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes, rng_graph)

    # Create graph
    graph = rex.graph.Graph(nodes, nodes[SUPERVISOR], graphs_aug, supergraph=Supergraph.MCS)

    # Get initial graph state
    rng, rng_init = jax.random.split(rng)
    gs_init = graph.init(rng_init, order=("agent", "pid"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        raise NotImplementedError("Make sure they match dtype of dtypes returned by .step. Else recompilation....")
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
    else:
        params = gs_init.params.unfreeze()
        print(f"Params not found at {PARAMS_FILE}")

    # Modify supervisor params
    params["agent"] = params["agent"].replace(
        init_cf="fixed",
        fixed_position=POSITION,
        init_path="fixed",
        fixed_radius=RADIUS,
        center=CENTER,
    )

    # Load trained params (if file exists)
    with open(AGENT_FILE, "rb") as f:
        agent_params = pickle.load(f)
    print(f"Agent params loaded from {AGENT_FILE}")
    params["agent"] = gs_init.params["agent"].replace(**agent_params.__dict__)

    # Evaluate
    from envs.crazyflie.path_following import Environment
    from envs.crazyflie.ode import env_rollout, render, save
    env_eval = Environment(graph, params=params, order=("agent", "pid"), randomize_eps=False)  # No randomization.
    rng, rng_rollout = jax.random.split(rng)
    res = env_rollout(env_eval, rng_rollout)
    gs = res.next_gs

    if True:
        # Plot results
        from envs.crazyflie.ode import plot_data
        pid = gs.inputs["world"]["pid"][:, -1]
        fig, axes = plot_data(output={"att": gs.state["world"].att,
                                      "pos": gs.state["world"].pos,
                                      "pwm": pid.data.pwm_ref,
                                      "pwm_ref": pid.data.pwm_ref,
                                      "phi_ref": pid.data.phi_ref,
                                      "theta_ref": pid.data.theta_ref,
                                      "z_ref": pid.data.z_ref},
                              ts={"att": gs.ts["world"],
                                  "pos": gs.ts["world"],
                                  "pwm": pid.ts_recv,
                                  "pwm_ref": pid.ts_recv,
                                  "phi_ref": pid.ts_recv,
                                  "theta_ref": pid.ts_recv,
                                  "z_ref": pid.ts_recv},
                              # ts_max=3.0,
                              )

    if False:
        json_rollout = render(gs.ts["world"], gs.state["world"].pos, gs.state["world"].att, gs.state["world"].radius, gs.state["world"].center)
        save(HTML_FILE, json_rollout)
        print(f"Render saved to {HTML_FILE}")

    plt.show()







