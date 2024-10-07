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
    ACTION_DIM = 2
    NO_DELAY = False  # todo: change
    # MAPPING = ["t eta_ref", "phi_ref"]
    SUPERVISOR = "agent"
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    EXP_DIR = f"{LOG_DIR}/20240813_142721_no_zref_eval_sysid_retry"  # todo:  change
    # Input files
    RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"  # todo:  change
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"  # todo:  change
    # Output files
    delay_str = "nodelay" if NO_DELAY else "delay"
    FIG_FILE = f"{EXP_DIR}/{delay_str}_fig.png"
    METRICS_FILE = f"{EXP_DIR}/{delay_str}_metrics.pkl"
    AGENT_FILE = f"{EXP_DIR}/{delay_str}_agent_params.pkl"
    ROLLOUT_FILE = f"{EXP_DIR}/{delay_str}_rollout.pkl"
    HTML_FILE = f"{EXP_DIR}/{delay_str}_rollout.html"
    ELAPSED_FILE = f"{EXP_DIR}/{delay_str}_elapsed.pkl"
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
        params = gs_init.params.unfreeze()
        print(f"Params not found at {PARAMS_FILE}")
        if "agent" in params:
            params.pop("agent")

    # Add agent params
    rng, rng_agent = jax.random.split(rng)
    params["agent"] = nodes["agent"].init_params(rng_agent)
    params["agent"] = params["agent"].replace(
        init_cf="random",
        init_path="random",
        phi_max=MAX_ANGLE,
        theta_max=MAX_ANGLE,
        action_dim=ACTION_DIM,
        use_noise=True,
        use_dr=True,
    )

    # Make environment
    # todo: make sure use_noise, use_dr are True (i.e. set in the init_state instead of params).
    from envs.crazyflie.path_following import Environment, ppo_config
    env = Environment(graph, params=params, order=("agent", "pid"), randomize_eps=False)  # No randomization.

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

    config = ppo_config.replace(FIXED_INIT=True, VERBOSE=True, TOTAL_TIMESTEPS=10e6, UPDATE_EPOCHS=16, NUM_MINIBATCHES=8, NUM_STEPS=64)
    train = functools.partial(rex.ppo.train, env)
    rng, rng_train = jax.random.split(rng)
    t_jit = timer("train | jit")
    with t_jit:
        train = jax.jit(train).lower(config, rng_train).compile()
    t_train = timer("train | run")
    with t_train:
        res = train(config, rng_train)
    print("Training done!")

    t_elapsed = {"train": t_train.duration, "train_jit": t_jit.duration}

    if SAVE_FILE:
        with open(ELAPSED_FILE, "wb") as f:
            pickle.dump(t_elapsed, f)
        print(f"Elapsed time saved to {ELAPSED_FILE}")
    # exit()
    # Get agent params
    agent_params = params["agent"].replace(act_scaling=res.act_scaling,
                                           obs_scaling=res.obs_scaling,
                                           model=res.policy.model,
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
    env_eval = Environment(graph, params=params_eval, order=("agent", "pid"), randomize_eps=False)  # No randomization.
    rng, rng_rollout = jax.random.split(rng)
    res = env_rollout(env_eval, rng_rollout)
    print("Rollout done!")

    gs_rollout = res.next_gs
    json_rollout = render(gs_rollout.ts["world"], gs_rollout.state["world"].pos, gs_rollout.state["world"].att, gs_rollout.state["agent"].radius, gs_rollout.state["agent"].center)
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

    # Save
    fig.savefig(FIG_FILE)
    print(f"Fig saved to {FIG_FILE}")
