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
import rex.rl as rl

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
    MAX_STEPS = 100
    WORLD_RATE = 100.
    SUPERVISOR = "pid"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    EXP_DIR = f"{LOG_DIR}/20240813_142721_no_zref_eval_sysid_retry_eval_redo_sysid_correct_z"
    # Input files
    RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"
    # Output files
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"
    LOG_STATE_FILE = f"{EXP_DIR}/sysid_log_state.pkl"
    SOL_STATE_FILE = f"{EXP_DIR}/sysid_sol_state.pkl"
    ELAPSED_FILE = f"{EXP_DIR}/sysid_elapsed.pkl"
    GS_EVAL_FILE = f"{EXP_DIR}/sysid_gs.pkl"
    GS_EVAL_NODELAY_FILE = f"{EXP_DIR}/sysid_gs_nodelay.pkl"
    FIG_FILE = f"{EXP_DIR}/sysid_fig"
    SAVE_FILES = True

    # Seed experiment
    rng = jax.random.PRNGKey(SEED)

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Gather outputs
    outputs_sysid = {name: n.steps.output for name, n in record.nodes.items()}

    # Create nodes
    nodes = psys.simulated_system(record, outputs=outputs_sysid, world_rate=WORLD_RATE)

    # Generate computation graph
    graphs_real = record.to_graph()
    rng, rng_art = jax.random.split(rng)
    graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes, rng_art)
    graphs_aug = graphs_aug.filter(nodes)

    # Create compiled graph
    graph = rex.graph.Graph(nodes, nodes[SUPERVISOR], graphs_aug, supergraph=SUPERGRAPH)

    # Visualize graph
    if False:
        MAX_X = 1.0
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(graph.Gs):
            if i > 1:
                break  # Only plot first two episodes
            supergraph.plot_graph(G, max_x=MAX_X, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()

    # Make sure params are correctly set.
    params_sup = nodes["agent"].init_params().replace(
        init_cf="fixed",
        fixed_position=outputs_sysid["mocap"].pos[0, 0],
        use_noise=False,
        use_dr=False,
    )

    # Get initial state
    rng, rng_init = jax.random.split(rng)
    gs = graph.init(rng_init, params={"agent": params_sup}, order=("agent", "pid"))

    # System identification
    import envs.crazyflie.tasks as tasks
    figs = []
    task = tasks.create_sysid_task(graph, gs).replace(max_steps=MAX_STEPS)
    # Evaluate initial
    rng, rng_eval_init = jax.random.split(rng, num=2)
    gs_eval_init = task.evaluate(gs.params, rng_eval_init, -1, order=("agent", "pid"))
    figs += task.plot(gs_eval_init, identifier="init")
    init_loss = task.loss(gs.params)
    print(f"Initial loss: {init_loss}")
    # Jit, lower, precompile
    t_solve_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
    with t_solve_jit:
        task_solve = jax.jit(task.solve)
        with timer("lower", log_level=100):
            task_solve = task_solve.lower(gs)
        with timer("compile", log_level=100):
            task_solve = task_solve.compile()
    # Solve
    t_solve = timer("solve", log_level=100)
    with t_solve:
        sol_state, opt_params, log_state = task_solve(gs)
    params_sysid = task.to_extended_params(gs, opt_params)
    # Store timings
    elapsed_sysid = dict(solve=t_solve.duration, solve_jit=t_solve_jit.duration)
    # Evaluate
    rng, rng_eval = jax.random.split(rng, num=2)
    gs_eval = task.evaluate(params_sysid, rng_eval, -1, order=("agent", "pid"))
    figs += task.plot(gs_eval, identifier="opt")
    # Evaluate (no delays)
    params_nodelay = eqx.tree_at(lambda x: x["pid"].actuator_delay.alpha, params_sysid, 0.0)
    params_nodelay = eqx.tree_at(lambda x: x["pid"].sensor_delay.alpha, params_nodelay, 0.0)
    params_nodelay = eqx.tree_at(lambda x: x["mocap"].sensor_delay.alpha, params_nodelay, 0.0)
    rng, rng_eval = jax.random.split(rng, num=2)
    gs_eval_nodelay = task.evaluate(params_nodelay, rng_eval, -1, order=("agent", "pid"))
    figs += task.plot(gs_eval, identifier="opt (no delay)")
    # Reduce size
    gs_eval_nodelay = gs_eval_nodelay.replace(buffer=None, timings_eps=None)
    gs_eval = gs_eval.replace(buffer=None, timings_eps=None)
    plt.show()
    # Save
    if SAVE_FILES:
        # Save params
        # todo: note that this will also save the Agent settings (use_noise=False, use_dr=False, etc...)
        # todo: make sure to correct in main_rl, main_real, etc...
        with open(PARAMS_FILE, "wb") as f:
            pickle.dump(params_sysid, f)
        print(f"Sysid params saved to {PARAMS_FILE}")
        # Save data_sysid used for sysid
        with open(f"{EXP_DIR}/sysid_data.pkl", "wb") as f:
            pickle.dump(record, f)
        print(f"Data_sysid saved to {EXP_DIR}/sysid_data.pkl")
        # Save log_state
        with open(LOG_STATE_FILE, "wb") as f:
            pickle.dump(log_state, f)
        print(f"Log_state saved to {LOG_STATE_FILE}")
        # Save sol_state
        with open(SOL_STATE_FILE, "wb") as f:
            pickle.dump(sol_state, f)
        print(f"Sol_state saved to{SOL_STATE_FILE}")
        # Save
        with open(ELAPSED_FILE, "wb") as f:
            pickle.dump(elapsed_sysid, f)
        print(f"Elapsed_sysid saved to {ELAPSED_FILE}")
        # Save gs_eval
        with open(GS_EVAL_FILE, "wb") as f:
            pickle.dump(gs_eval, f)
        print(f"gs_eval saved to {GS_EVAL_FILE}")
        # Save gs_eval
        with open(GS_EVAL_NODELAY_FILE, "wb") as f:
            pickle.dump(gs_eval_nodelay5, f)
        print(f"gs_eval saved to {GS_EVAL_NODELAY_FILE}")
        # Save figs with suptitle
        for fig in figs:
            suptitle = fig._suptitle.get_text() if fig._suptitle else "Untitled"
            fig.savefig(f"{FIG_FILE}_{suptitle}.png")
            print(f"Saved: {FIG_FILE}_{suptitle}.png")