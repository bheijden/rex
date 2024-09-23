import dill as pickle
import tqdm
import functools
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
from rexv2 import artificial
import envs.pendulum.systems as psys

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
    WORLD_RATE = 100
    MAX_ANGLE = onp.pi / 6
    POSITION = onp.array([0.0, 0.0, 1.0])
    CENTER = onp.array([0.0, 0.0, 1.0])
    NUM_ROLLOUTS = [2**i for i in range(0, 15)]
    MAX_STEPS = 200
    ACTION_DIM = 2
    # MAPPING = ["t eta_ref", "phi_ref"]
    SUPERVISOR = "controller"
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    EXP_DIR = f"{LOG_DIR}/20240710_141737_brax_norandomization_longerstack_v4_dark"  # todo:  change
    # Input files
    RECORD_FILE = f"{LOG_DIR}/data_control.pkl" # todo:  change
    # RECORD_FILE = f"{EXP_DIR}/data_sysid.pkl"  # todo:  change
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"  # todo:  change
    CONTROLLER_FILE = f"{EXP_DIR}/ctrl_controllers.pkl"  # todo: change to brax
    # Output files
    STD_TH_RL = 0.02  # Overwrite std_th in estimator and camera --> None to keep default
    STATS_FILE = f"{EXP_DIR}/rollout_stats.pkl"
    SAVE_FILE = True

    # Seed
    rng = jax.random.PRNGKey(SEED)

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    nodes = psys.simulated_system(record, world_rate=WORLD_RATE, use_cam=True, use_brax=True)

    # Get graph
    graphs_real = record.to_graph()

    if True:  # Exclude sensor if using camera
        sensor = nodes.pop("sensor")
        [v.disconnect() for k, v in list(sensor.inputs.items())]
        [v.disconnect() for k, v in list(sensor.outputs.items())]
        graphs_real.vertices.pop("sensor")
        graphs_real.edges.pop(("sensor", "estimator"))

    # Generate computation graph
    rng, rng_graph = jax.random.split(rng)
    graphs_aug = rexv2.artificial.augment_graphs(graphs_real, nodes, rng_graph)

    # Create graph
    graph = rexv2.graph.Graph(nodes, nodes[SUPERVISOR], graphs_aug, supergraph=Supergraph.MCS)

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

    # # Get initial graph state
    # rng, rng_init = jax.random.split(rng)
    # with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
    #     gs_init = graph.init(rng_init, order=("supervisor", "actuator"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
        if "controller" in params:
            params.pop("controller")
    else:
        raise NotImplementedError("Params file not found!")

    params["estimator"] = params["estimator"].replace(std_th=STD_TH_RL)
    params["camera"] = params["camera"].replace(std_th=STD_TH_RL)
    print(f"[RL] Overwriting std_th to {STD_TH_RL}")

    # Load trained params (if file exists)
    if os.path.exists(CONTROLLER_FILE):
        with open(CONTROLLER_FILE, "rb") as f:
            controllers = pickle.load(f)
        print(f"Controllers loaded from {CONTROLLER_FILE}")

        # Prepare controllers
        tree_flat, treedef = jax.tree_util.tree_flatten(controllers)
        num_controllers = tree_flat[0].shape[0]
        controllers_lst = [jax.tree_util.tree_unflatten(treedef, [c[i] for c in tree_flat]) for i in range(num_controllers)]
        params["controller"] = controllers_lst[0]  # use the first controller
    else:
        # print(f"Agent params not found at {AGENT_FILE}")
        raise FileNotFoundError(f"Agent params not found at {AGENT_FILE}")

    def rollout_fn(rng):
        # Initialize graph state
        _gs = graph.init(rng, params=params, order=("supervisor", "actuator"))
        _gs_rollout = graph.rollout(_gs, carry_only=False, max_steps=MAX_STEPS)
        return _gs_rollout.state["world"].th

    # Rollout
    stats_rollouts = {}
    for num_rollouts in NUM_ROLLOUTS:
        rng, rng_rollout = jax.random.split(rng)
        rngs_rollout = jax.random.split(rng_rollout, num=num_rollouts)
        t_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
        t_lower = timer("lower", log_level=100)
        t_compile = timer("compile", log_level=100)
        with t_jit:
            rollout_fn_jv = jax.jit(jax.vmap(rollout_fn))
            with t_lower:
                rollout_fn_jv = rollout_fn_jv.lower(rngs_rollout)
            with t_compile:
                rollout_fn_jv = rollout_fn_jv.compile()
        t_run = timer("run", log_level=100)
        with t_run:
            final_states = rollout_fn_jv(rngs_rollout).block_until_ready()
        print(f"Final states: {final_states.mean():.2f} ± {final_states.std():.2f}")
        stats = {
            "num_rollouts": num_rollouts,
            "num_steps": MAX_STEPS,
            "fps": (num_rollouts * MAX_STEPS) / t_run.duration,
            "jit": t_jit.duration,
            "lower": t_lower.duration,
            "compile": t_compile.duration,
            "run": t_run.duration,
        }
        stats_str = ", ".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        print(f"Stats: {stats_str}")
        stats_rollouts[num_rollouts] = stats
        # Save stats
        if SAVE_FILE:
            with open(STATS_FILE, "wb") as f:
                pickle.dump(stats_rollouts, f)
            print(f"Stats saved to {STATS_FILE}")
