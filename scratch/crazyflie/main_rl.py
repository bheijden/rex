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
import envs.crazyflie.ppo as ppo_config
import rexv2.rl as rl

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
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    RECORD_FILE = f"{LOG_DIR}/data_rl.pkl"
    PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
    METRICS_FILE = f"{LOG_DIR}/metrics.pkl"
    AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
    ROLLOUT_FILE = f"{LOG_DIR}/rollout.pkl"
    HTML_FILE = f"{LOG_DIR}/rollout.html"
    SAVE_FILE = True

    # Seed
    rng = jax.random.PRNGKey(SEED)

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    nodes = psys.simulated_system(record, world_rate=WORLD_RATE)

    # Get graph
    graphs_real = record.to_graph()

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

    # Get initial graph state
    rng, rng_init = jax.random.split(rng)
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs_init = graph.init(rng_init, order=("supervisor", "pid"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        raise NotImplementedError("Make sure they match dtype of dtypes returned by .step. Else recompilation....")
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
    else:
        params = gs_init.params.unfreeze()
        print(f"Params not found at {PARAMS_FILE}")

    # Make environment
    from envs.crazyflie.env import Environment
    env = Environment(graph, params=params, order=("supervisor", "pid"), randomize_eps=False)  # No randomization.

    # Test env API
    # obs_space = env.observation_space(gs_init)
    # act_space = env.action_space(gs_init)
    # _ = env.get_observation(gs_init)
    # _ = env.get_truncated(gs_init)
    # _ = env.get_terminated(gs_init)
    # _ = env.get_reward(gs_init, jnp.zeros(act_space.low.shape))
    # _ = env.get_info(gs_init, jnp.zeros(act_space.low.shape))
    # _ = env.get_output(gs_init, jnp.zeros(act_space.low.shape))
    # _ = env.update_graph_state_pre_step(gs_init, jnp.zeros(act_space.low.shape))
    # gs, obs, info = jax.jit(env.reset)()

    config = ppo_config.path_following#.replace(FIXED_INIT=True, VERBOSE=True, NUM_ENVS=16,)  # todo: Set to True...
    train = functools.partial(rexv2.ppo.train, env)
    rng, rng_train = jax.random.split(rng)
    with timer("train | jit"):
        train = jax.jit(train).lower(config, rng_train).compile()
    with timer("train | run"):
        res = train(config, rng_train)
    print("Training done!")

    # Get agent params
    agent_params = params["agent"].replace(act_scaling=res["act_scaling"],
                                           obs_scaling=res["norm_obs"],
                                           model=res["runner_state"][0].params["params"],
                                           hidden_activation=config.HIDDEN_ACTIVATION,
                                           stochastic=False)

    # Save params
    if SAVE_FILE:
        with open(AGENT_FILE, "wb") as f:
            pickle.dump(agent_params, f)
        print(f"Agent params saved to {AGENT_FILE}")
    else:
        print(f"Agent params not saved!")

    # Get agent params
    params_eval = params.copy()
    params_eval["agent"] = agent_params

    # Evaluate
    from envs.crazyflie.ode import env_rollout, render, save
    env_eval = Environment(graph, params=params_eval, order=("supervisor", "pid"), randomize_eps=False)  # No randomization.
    rng, rng_rollout = jax.random.split(rng)
    res = env_rollout(env_eval, rng_rollout)
    print("Rollout done!")
    gs_rollout = res.next_gs
    json_rollout = render(gs_rollout.ts["world"], gs_rollout.state["world"].pos, gs_rollout.state["world"].att, gs_rollout.state["world"].radius, gs_rollout.state["world"].center)
    save(HTML_FILE, json_rollout)
    print(f"Render saved to {HTML_FILE}")

    # Save rollout
    if SAVE_FILE:
        res_red = res.replace(next_gs=res.next_gs.replace(buffer=None, timings_eps=None))
        with open(ROLLOUT_FILE, "wb") as f:
            pickle.dump(res_red, f)
        print(f"Rollout saved to {ROLLOUT_FILE}")
    else:
        print(f"Rollout not saved!")

    # Plot results
    from envs.crazyflie.ode import plot_data
    pid = gs_rollout.inputs["world"]["pid"][:, -1]
    fig, axes = plot_data(output={"att": gs_rollout.state["world"].att,
                                  "pos": gs_rollout.state["world"].pos,
                                  "pwm": pid.data.pwm_ref,
                                  "pwm_ref": pid.data.pwm_ref,
                                  "phi_ref": pid.data.phi_ref,
                                  "theta_ref": pid.data.theta_ref,
                                  "z_ref": pid.data.z_ref},
                          ts={"att": gs_rollout.ts["world"],
                              "pos": gs_rollout.ts["world"],
                              "pwm": pid.ts_recv,
                              "pwm_ref": pid.ts_recv,
                              "phi_ref": pid.ts_recv,
                              "theta_ref": pid.ts_recv,
                              "z_ref": pid.ts_recv},
                          # ts_max=3.0,
                          )
    plt.show()

