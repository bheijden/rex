import dill as pickle
import tqdm
import functools
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
import datetime

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
import envs.pendulum.ppo as ppo_config
import rexv2.rl as rl

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    jnp.set_printoptions(precision=5, suppress=True)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])

    # General settings
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    EXP_DIR = f"{LOG_DIR}/20240710_141737_brax"
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"  # todo: change to brax
    RECORD_FILE = f"{EXP_DIR}/ctrl_data.pkl"
    SEED = 0
    WORLD_RATE = 100.
    STD_TH = 0.02  # Overwrite std_th in estimator and camera --> None to keep default
    NUM_ACT = 2
    NUM_OBS = 2
    INCL_THDOT = False
    USE_BRAX = True
    USE_PRED = True
    USE_UKF = False
    USE_CAM = True
    TS_MAX = 5.0
    TOTAL_TIMESTEPS = 1_500_000
    EVAL_FREQ = 50  # Evaluate every 50 steps
    NUM_POLICIES = 5
    DIST_FILE = f"{LOG_DIR}/dists.pkl"
    RATES = dict(sensor=50, camera=50, estimator=50, controller=50, actuator=50, supervisor=10, world=100)
    CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
               "actuator": "green", "supervisor": "indigo"}

    # Initialize RNG
    rng = jax.random.PRNGKey(SEED)

    # Load params
    with open(PARAMS_FILE, "rb") as f:
        params: Dict[str, Any] = pickle.load(f)
    print(f"Params loaded from {PARAMS_FILE}")

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    nodes = psys.simulated_system(record, world_rate=WORLD_RATE, use_cam=USE_CAM, use_brax=USE_BRAX, use_ukf=USE_UKF)

    # Set supervisor init method
    nodes["supervisor"].set_init_method("random")  # Set initialization method

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
    rng, rng_aug = jax.random.split(rng)
    graphs_aug = rexv2.artificial.augment_graphs(graphs_real, nodes, rng_aug)

    # Create graph
    graph = rexv2.graph.Graph(nodes, nodes["controller"], graphs_aug)

    # Visualize the graph
    if False:
        Gs = graph.Gs
        # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(Gs):
            supergraph.plot_graph(G, max_x=1.0, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()

    with open(PARAMS_FILE, "rb") as f:
        params_sysid = pickle.load(f)
    params_train = params_sysid.copy()
    if STD_TH is not None:
        params_train["estimator"] = params_train["estimator"].replace(std_th=STD_TH)
        params_train["camera"] = params_train["camera"].replace(std_th=STD_TH)
        print(f"Overwriting std_th to {STD_TH}")
    params_train["controller"] = params_train["controller"].replace(incl_covariance=False, incl_thdot=INCL_THDOT,
                                                                    num_act=NUM_ACT, num_obs=NUM_OBS)

    # Initialize the graph
    rng, rng_init = jax.random.split(rng)
    gs = graph.init(rng_init, params=params_train, order=("supervisor", "actuator"))

    # Create environment
    from envs.pendulum.env import Environment
    env = Environment(graph, params=params_train, order=("supervisor", "actuator"), randomize_eps=False)

    obs_space = env.observation_space(gs)
    print(f"obs_space | shape={obs_space.shape}")
    # exit()

    # Create train function
    import rexv2.ppo as ppo

    ppo_config = ppo_config.sweep_pmv2r1zf.replace(TOTAL_TIMESTEPS=TOTAL_TIMESTEPS, EVAL_FREQ=EVAL_FREQ)
    train = functools.partial(ppo.train, env)
    train_v = jax.vmap(train, in_axes=(None, 0))
    train_vjit = jax.jit(train_v)
    rng, rng_ppo = jax.random.split(rng, num=2)
    rngs_policies = jax.random.split(rng_ppo, NUM_POLICIES)
    # Jit, lower, precompile
    t_train_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
    with t_train_jit:
        with timer("lower", log_level=100):
            train_vjit = train_vjit.lower(ppo_config, rngs_policies)
        with timer("compile", log_level=100):
            train_vjit = train_vjit.compile()
    # Train
    t_train = timer("train", log_level=100)
    with t_train:
        ppo_out = train_vjit(ppo_config, rngs_policies)
    # Store timings
    elapsed_ctrl = dict(solve=t_train.duration, solve_jit=t_train_jit.duration)
    # Extract policies
    model_params = ppo_out.policy.model
    act_scaling = ppo_out.act_scaling
    obs_scaling = ppo_out.obs_scaling
    controllers = params_train["controller"].replace(act_scaling=act_scaling, obs_scaling=obs_scaling, model=model_params,
                                                     hidden_activation=ppo_config.HIDDEN_ACTIVATION, stochastic=False)
    controllers = jax.tree_util.tree_map(lambda x: onp.array(x), controllers)
