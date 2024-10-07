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
    WORLD_RATE = 50.
    MAX_ANGLE = onp.pi / 6
    POSITION = onp.array([0.0, 0.0, 1.0])
    CENTER = onp.array([0.0, 0.0, 1.0])
    RADII = onp.array([1.0, 0.75, 0.5])
    ACTION_DIM = 2
    NO_DELAY = True  # todo: change
    # MAPPING = ["t eta_ref", "phi_ref"]
    SUPERVISOR = "agent"
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    EXP_DIR = f"{LOG_DIR}/20240816_path_following_inclined_landing_experiments_sim"  # todo:  change
    # Input files
    RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"  # todo:  change
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"  # todo:  change
    # Output files
    delay_str = "nodelay" if NO_DELAY else "delay"
    AGENT_FILE = f"{EXP_DIR}/{delay_str}_agent_params.pkl"
    # FIG_FILE = f"{EXP_DIR}/sim_{delay_str}_fig.png"
    ROLLOUT_FILE = f"{EXP_DIR}/{delay_str}_sim_radii_rollout.pkl"
    # HTML_FILE = f"{EXP_DIR}/sim_{delay_str}_rollout.html"
    SAVE_FILE = True

    # Seed
    rng = jax.random.PRNGKey(SEED)

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    if NO_DELAY:
        nodes = psys.nodelay_simulated_system(record, world_rate=WORLD_RATE)
        graphs_real = artificial.generate_graphs(nodes, 15)
        graphs_aug = graphs_real
    else:
        nodes = psys.simulated_system(record, world_rate=WORLD_RATE)
        # Get graph
        graphs_real = record.to_graph()
        graphs_real = graphs_real.filter(nodes)  # Filter nodes

        # Generate computation graph
        rng, rng_graph = jax.random.split(rng)
        graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes, rng_graph)
        graphs_aug = graphs_aug.filter(nodes)  # Filter nodes

    # Create graph
    graph = rex.graph.Graph(nodes, nodes[SUPERVISOR], graphs_aug, supergraph=Supergraph.MCS)

    # Visualize the graph
    if False:
        Gs = graph.Gs
        # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(Gs):
            if i > 1:
                break
            supergraph.plot_graph(G, max_x=1.0, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()

    # Get initial graph state
    rng, rng_init = jax.random.split(rng)
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs_init = graph.init(rng_init, order=("agent", "pid"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
        if "agent" in params:
            params.pop("agent")
    else:
        raise NotImplementedError("Params file not found!")

    # Load trained params (if file exists)
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "rb") as f:
            agent_params = pickle.load(f)
        print(f"Agent params loaded from {AGENT_FILE}")
        params["agent"] = gs_init.params["agent"].replace(**agent_params.__dict__)
    else:
        # print(f"Agent params not found at {AGENT_FILE}")
        raise FileNotFoundError(f"Agent params not found at {AGENT_FILE}")

    # Add agent params
    params["agent"] = params["agent"].replace(
        init_cf="fixed",
        fixed_position=POSITION,
        init_path="fixed",
        center=CENTER,
        # phi_max=MAX_ANGLE,
        # theta_max=MAX_ANGLE,
        # action_dim=ACTION_DIM,
        use_noise=True,
        use_dr=False,
    )

    def rollout_radius(_params, rng, _radius):
        # Replace radius
        _params = _params.copy()
        _params["agent"] = _params["agent"].replace(fixed_radius=_radius)
        _gs = graph.init(rng, params=_params, order=("agent", "pid"))
        _gs_rollout = graph.rollout(_gs, carry_only=False)
        return _gs_rollout.replace(timings_eps=None, buffer=None)

    # Rollout
    rng, rng_rollout = jax.random.split(rng)
    rngs_rollout = jax.random.split(rng_rollout, num=len(RADII))
    rollout_radius_jv = jax.jit(jax.vmap(rollout_radius, in_axes=(None, 0, 0)))
    rollout_radius_jv = rollout_radius_jv.lower(params, rngs_rollout, RADII).compile()
    res_rollout = rollout_radius_jv(params, rngs_rollout, RADII)

    pos = res_rollout.state["world"].pos

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, r in enumerate(RADII):
        axes[i].add_artist(plt.Circle(CENTER[:2], r, fill=False, color="r", linestyle="--"))
        axes[i].plot(pos[i, :, 0], pos[i, :, 1], label=f"r={r}")
        # Plot circle
        axes[i].set_aspect("equal")
        axes[i].set(
            xlabel="x [m]",
            ylabel="y [m]",
            title=f"Radius: {r}",
            xlim=(-1.5, 1.5),
            ylim=(-1.5, 1.5),
        )
        axes[i].set_title(f"Radius: {r}")
        axes[i].legend()
    plt.show()

    # Save rollout
    if SAVE_FILE:
        with open(ROLLOUT_FILE, "wb") as f:
            pickle.dump(res_rollout, f)
        print(f"Rollout saved to {ROLLOUT_FILE}")
    exit()
