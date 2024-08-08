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
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
from rexv2 import artificial
import envs.crazyflie.systems as psys
import rexv2.rl as rl

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=5, suppress=True)
    SEED = 0
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    EPS_IDX = -1
    WORLD_RATE = 100.
    SUPERVISOR = "pid"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    RECORD_FILE = f"{LOG_DIR}/data_sysid.pkl"
    PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"

    # Seed experiment
    rng = jax.random.PRNGKey(SEED)

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Gather outputs
    outputs_sysid = {name: n.steps.output for name, n in record.nodes.items()}

    # Create nodes
    nodes_sysid = psys.simulated_system(record, outputs=outputs_sysid, world_rate=WORLD_RATE)

    # Generate computation graph
    graphs_real = record.to_graph()
    rng, rng_art = jax.random.split(rng)
    graphs_aug = rexv2.artificial.augment_graphs(graphs_real, nodes_sysid, rng_art)

    # Create compiled graph
    graph_sysid = rexv2.graph.Graph(nodes_sysid, nodes_sysid[SUPERVISOR], graphs_aug, supergraph=SUPERGRAPH, skip=["supervisor"])

    # Visualize graph
    if True:
        MAX_X = 1.0
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(graph_sysid.Gs):
            if i > 1:
                break  # Only plot first two episodes
            supergraph.plot_graph(G, max_x=MAX_X, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()
