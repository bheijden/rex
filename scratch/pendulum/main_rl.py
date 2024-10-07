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
import envs.pendulum.systems as psys
import envs.pendulum.ppo as ppo_config
import rex.rl as rl

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    # Make sysid nodes
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    TS_MAX = 5.0
    WORLD_RATE = 100.
    USE_CAM = True
    USE_BRAX = True  # Use brax for simulation # todo: change to brax
    STD_TH = 0.02  # Overwrite std_th in estimator and camera --> None to keep default
    INCL_COVARIANCE = False
    SUPERVISOR = "controller"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    RECORD_FILE = f"{LOG_DIR}/data_control.pkl"
    PARAMS_FILE = f"{LOG_DIR}/sysid_params_brax.pkl"  # todo: change to brax
    METRICS_FILE = f"{LOG_DIR}/metrics_brax.pkl"  # todo: change to brax
    # CTRL_FILE = f"{LOG_DIR}/controller_trained_params_cov.pkl"
    CTRL_FILE = f"{LOG_DIR}/controller_params_brax.pkl"  # todo: change to brax
    SAVE_FILE = False
    # ORDER = ["camera", "sensor", "actuator", "controller", "estimator", "supervisor"]
    # CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
    #            "actuator": "green", "supervisor": "indigo"}
    # RATES = dict(sensor=30, camera=30, estimator=30, controller=30, actuator=30, supervisor=10)
    # DELAYS_SIM = psys.load_distribution(f"{LOG_DIR}/dists.pkl")  # get_default_distributions()
    # DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0

    # Load params
    with open(PARAMS_FILE, "rb") as f:
        params: Dict[str, Any] = pickle.load(f)
    print(f"Params loaded from {PARAMS_FILE}")

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    nodes = psys.simulated_system(record, world_rate=WORLD_RATE, use_cam=USE_CAM, use_brax=USE_BRAX)

    # Set initialization method
    nodes["supervisor"].set_init_method("random")

    # Get graph
    graphs_real = record.to_graph()

    # Exclude sensor if using camera
    if USE_CAM:
        sensor = nodes.pop("sensor")
        [v.disconnect() for k, v in list(sensor.inputs.items())]
        [v.disconnect() for k, v in list(sensor.outputs.items())]
        graphs_real.vertices.pop("sensor")
        graphs_real.edges.pop(("sensor", "estimator"))

    # Generate computation graph
    graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes, RNG)

    # Create simulation nodes
    # nodes = make_pendulum_system_nodes(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER)
    # Generate computation graphs
    # graphs_gen = artificial.generate_graphs(nodes, ts_max=TS_MAX, num_episodes=1)

    # Create graph
    graph = rex.graph.Graph(nodes, nodes[SUPERVISOR], graphs_aug, supergraph=SUPERGRAPH)

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
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs = graph.init(RNG, order=("supervisor", "actuator"))

    # Modify params if necessary
    params_env = params.copy()
    # params_env["supervisor"] = params_env["supervisor"].replace(tmax=5.0)
    if STD_TH is not None:
        params_env["estimator"] = params_env["estimator"].replace(std_th=STD_TH)
        params_env["camera"] = params_env["camera"].replace(std_th=STD_TH)
        print(f"Overwriting std_th to {STD_TH}")
    params_env["controller"] = params_env["controller"].replace(incl_covariance=INCL_COVARIANCE)
    eqx.tree_pprint(params_env, short_arrays=False)

    # Jit functions & warmup
    if False:
        for name, node in graph.nodes.items():
            cpu = next(CPU_DEVICES)
            print(f"Jitting {name} on {cpu}")
            node.step = jax.jit(node.step, device=cpu)

        # Warmup & get initial graph state
        # import logging
        # logging.getLogger("jax").setLevel(logging.INFO)

        for name, node in graph.nodes.items():
            ss = gs.step_state[name]
            default_o = node.init_output(RNG, gs)
            with timer(f"warmup[{name}]", log_level=LogLevel.SILENT):
                with jax.log_compiles():
                    ss, o = node.async_step(ss)
            _ = same_structure(default_o, o, tag=name, raise_on_mismatch=False)  # todo: Turn on raise_on_mismatch
            _ = same_structure(gs.step_state[name], ss, tag=name, raise_on_mismatch=False)
            with timer(f"eval[{name}]", log_level=LogLevel.WARN, repeat=10):
                for _ in range(10):
                    ss, o = node.async_step(ss)

    from envs.pendulum.env import Environment
    env = Environment(graph, params=params_env, order=("supervisor", "actuator"), randomize_eps=True)

    # Evaluate in simulation
    if False:
        # Load trained params
        with open(CTRL_FILE, "rb") as f:
            ctrl_params = pickle.load(f)
        print(f"Controller params loaded from {CTRL_FILE}")
        # with open(CTRL_FILE, "wb") as f:
        #     ctrl_params = ctrl_params.replace(incl_covariance=False)
        #     pickle.dump(ctrl_params, f)
        params_eval = params_env.copy()
        params_eval["controller"] = ctrl_params

        # Evaluate the controller
        env_eval = Environment(graph, params=params_eval, order=("supervisor", "actuator"), randomize_eps=True)

        # Make rollout function
        rollout_fn = functools.partial(rl.rollout, env_eval, ctrl_params.get_action, int(3.0*nodes["controller"].rate))
        jv_rollout_fn = jax.jit(jax.vmap(rollout_fn))
        rngs_eval = jax.random.split(RNG, 10)
        print("cache-hit?")
        with timer("jv_rollout_fn"):
            transitions = jv_rollout_fn(rngs_eval)

        cum_reward = transitions.reward.sum(axis=-1)
        idx_min = cum_reward.argmin()
        print(f"cum_reward: {cum_reward.mean():.2f} +/- {cum_reward.std():.2f} | max: {cum_reward.max():.2f} | min: {cum_reward.min():.2f}")

        gs = transitions.gs
        ts_world = gs.inputs["camera"]["world"].ts_sent[idx_min, ..., -1]
        ts_estimator = gs.inputs["controller"]["estimator"].data.ts[idx_min, :, -1]
        ts_camera = gs.state["camera"].tsn_1[idx_min]

        th_world = gs.inputs["camera"]["world"].data.th[idx_min, ..., -1]
        th_camera = jnp.unwrap(gs.state["camera"].thn_1[idx_min])
        th_estimator = gs.inputs["controller"]["estimator"].data.mean.th[idx_min, :, 0]

        thdot_world = gs.inputs["camera"]["world"].data.thdot[idx_min, ..., -1]
        thdot_estimator = gs.inputs["controller"]["estimator"].data.mean.thdot[idx_min, :, 0]

        # Get std
        std_vfn = jax.vmap(lambda x: jnp.diag(jnp.sqrt(x)))
        std_est = std_vfn(gs.inputs["controller"]["estimator"].data.cov[idx_min, :, 0])

        is_upright = jnp.cos(gs.inputs["camera"]["world"].data.th[..., -1]) > 0.95
        is_static = jnp.abs(gs.inputs["camera"]["world"].data.thdot[..., -1]) < 2.0
        is_valid = jnp.logical_and(is_upright, is_static)
        success_rate = is_valid.sum() / is_valid.size
        print(f"success_rate: {100*success_rate:.2f}%")

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        axes[0].plot(ts_world, th_world, label="world", color="blue")
        # axes[0].plot(ts_world, is_valid[idx_min], label="world", color="blue")
        axes[0].plot(ts_camera, th_camera, label="camera", color="green")
        axes[0].plot(ts_estimator, th_estimator, label="estimator", color='r')
        axes[0].fill_between(ts_estimator, th_estimator - std_est[:, 0], th_estimator + std_est[:, 0], alpha=0.5, color='r')
        axes[0].set(ylabel="th")
        axes[0].legend()
        axes[1].plot(ts_world, thdot_world, label="world", color="blue")
        # axes[1].plot(ts_world, is_valid[idx_min], label="world", color="blue")
        axes[1].plot(ts_estimator, thdot_estimator, label="estimator", color='r')
        axes[1].fill_between(ts_estimator, thdot_estimator - std_est[:, 1], thdot_estimator + std_est[:, 1], alpha=0.5, color='r')
        axes[1].set(ylabel="thdot", ylim=[-30, 30])
        axes[1].legend()
        fig.suptitle(f"(th, thdot)")
        plt.show()

    # Test env API
    # obs_space = env.observation_space(gs)
    # act_space = env.action_space(gs)
    # _ = env.get_observation(gs)
    # _ = env.get_truncated(gs)
    # _ = env.get_terminated(gs)
    # _ = env.get_reward(gs, jnp.zeros(act_space.low.shape))
    # _ = env.get_info(gs, jnp.zeros(act_space.low.shape))
    # _ = env.get_output(gs, jnp.zeros(act_space.low.shape))
    # _ = env.update_graph_state_pre_step(gs, jnp.zeros(act_space.low.shape))
    # _ = env.reset()

    config = ppo_config.sweep_pmv2r1zf.replace(TOTAL_TIMESTEPS=5e6, VERBOSE=not JAX_USE_CACHE, EVAL_FREQ=20)

    train = functools.partial(rex.ppo.train, env)
    print("cache-hit?")
    train = jax.jit(train)
    with timer("train"):
        res = train(config, jax.random.PRNGKey(2))
    print("Training done!")

    # Initialize agent params
    model_params = res.policy.model
    ctrl_params = params_env["controller"].replace(act_scaling=res.act_scaling, obs_scaling=res.obs_scaling,
                                                   model=model_params, hidden_activation=config.HIDDEN_ACTIVATION,
                                                   stochastic=False)
    ctrl_params_onp = jax.tree_util.tree_map(lambda x: onp.array(x), ctrl_params)

    # Save agent params
    if SAVE_FILE:
        with open(CTRL_FILE, "wb") as f:
            pickle.dump(ctrl_params_onp, f)
        print(f"Controller params saved to {CTRL_FILE}")

