import time
import dill as pickle
import functools
from typing import Dict, Any
import os
import sys

sys.path.insert(0, '/home/r2ci/supergraph')
sys.path.insert(0, '/home/r2ci/rex')

JAX_USE_CACHE = False
# if JAX_USE_CACHE:
#     os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
# import multiprocessing
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     multiprocessing.cpu_count() - 4
# )
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
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

import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import envs.pendulum.systems as psys
import envs.pendulum.ppo as pendulum_ppo
import rexv2.rl as rl
import rexv2.ppo as ppo

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


import wandb
os.environ["WANDB_SILENT"] = "true"


if __name__ == "__main__":
    # poetry run wandb sweep --project supergraph ./train_sweep.yaml
    MODE = "online"  # NOTE! Change to "online" to enable wandb
    PROJECT = "rex"
    GROUP = "pendulum-rl"
    JOB_TYPE = "sweep"
    if MODE in ["disabled", "offline"]:
        print(f"IMPORTANT! Running in {MODE} mode.")
    print("using cache" if JAX_USE_CACHE else "not using cache")

    # Set-up wandb
    wandb.setup()
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    default_config = pendulum_ppo.default_config.__dict__.copy()
    default_config["VERBOSE"] = not JAX_USE_CACHE
    default_config["JAX_USE_CACHE"] = JAX_USE_CACHE
    default_config["SEED"] = 0
    default_config["NUM_SEEDS"] = 4
    default_config["INCL_COVARIANCE"] = True
    default_config["USE_CAM"] = True
    default_config["TS_MAX_EVAL"] = 5.0
    default_config["WORLD_RATE"] = 100.
    default_config["RECORD_FILE"] = f"{LOG_DIR}/pendulum_data_control.pkl"
    default_config["PARAMS_FILE"] = f"{LOG_DIR}/sysid_params.pkl"
    # Load params
    with open(default_config["PARAMS_FILE"], "rb") as f:
        params: Dict[str, Any] = pickle.load(f)
    print(f"Params loaded from {default_config['PARAMS_FILE']}")
    default_config["STD_TH"] = float(params["estimator"].std_th)
    default_config["STD_ACC"] = float(params["estimator"].std_acc)
    default_config["STD_INIT"] = float(params["estimator"].std_init)
    print(default_config)
    run = wandb.init(
        mode=MODE,
        project=PROJECT,
        group=GROUP,
        job_type=JOB_TYPE,
        allow_val_change=True,
        config=default_config
    )
    print(f"After init")
    config = wandb.config

    # Make sysid nodes
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)

    # Load params
    with open(config.PARAMS_FILE, "rb") as f:
        params: Dict[str, Any] = pickle.load(f)
    print(f"Params loaded from {config.PARAMS_FILE}")

    # Load record
    with open(config.RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)
    print(f"Record loaded from {config.RECORD_FILE}")

    # Create nodes
    nodes = psys.simulated_system(record, world_rate=config.WORLD_RATE, use_cam=config.USE_CAM)

    # Set initialization method
    nodes["supervisor"].set_init_method("random")

    # Get graph
    graphs_real = record.to_graph()

    # Exclude sensor if using camera
    if config.USE_CAM:
        sensor = nodes.pop("sensor")
        [v.disconnect() for k, v in list(sensor.inputs.items())]
        [v.disconnect() for k, v in list(sensor.outputs.items())]
        graphs_real.vertices.pop("sensor")
        graphs_real.edges.pop(("sensor", "estimator"))

    # Generate computation graph
    graphs_aug = rexv2.artificial.augment_graphs(graphs_real, nodes, RNG)

    # Create graph
    graph = rexv2.graph.Graph(nodes, nodes["controller"], graphs_aug)

    # Get initial graph state
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs = graph.init(RNG, order=("supervisor", "actuator"))

    # Modify params if necessary
    params_env = params.copy()
    params_env["estimator"] = params_env["estimator"].replace(std_th=config.STD_TH, std_acc=config.STD_ACC, std_init=config.STD_INIT)
    params_env["camera"] = params_env["camera"].replace(std_th=config.STD_TH)
    params_env["controller"] = params_env["controller"].replace(incl_covariance=config.INCL_COVARIANCE)
    eqx.tree_pprint(params_env, short_arrays=False)

    # Create environment
    from envs.pendulum.env import Environment
    env = Environment(graph, params=params_env, order=("supervisor", "actuator"), randomize_eps=True)

    # Convert config to PPO config
    ppo_config = pendulum_ppo.PendulumConfig(
        **{k: v for k, v in config.items() if k in ppo.Config.__dict__}
    )

    # Create train function
    train = functools.partial(ppo.train, env)
    train_v = jax.vmap(train, in_axes=(None, 0))
    train_vjit = jax.jit(train_v)

    # Train
    seeds = jax.random.split(jax.random.PRNGKey(config.SEED), config.NUM_SEEDS)
    print("cache-hit?") if JAX_USE_CACHE else print("Not caching")
    with timer("train"):
        out = train_vjit(ppo_config, seeds)
    print("Training done!")

    metrics = out["metrics"]
    approxkl = metrics["train/mean_approxkl"]
    approxkl = approxkl.mean(axis=0)
    # Return values should be reshaped, but make sure it is divisible by NUM_SEEDS
    # return_values = metrics["returned_episode_returns"][metrics["returned_episode"]]
    # num_elements = (len(return_values) // config.NUM_SEEDS) * config.NUM_SEEDS
    # return_values = return_values[:num_elements].reshape(config.NUM_SEEDS, -1)  # Calculate the number of elements that can be divided evenly by num_seeds
    eval_return_values = metrics["eval/mean_returns"].mean(axis=0)
    eval_total_steps = metrics["train/total_steps"].mean(axis=0)
    eval_return_std = metrics["eval/std_returns"].mean(axis=0)
    eval_success_rate = metrics["eval/success_rate"].mean(axis=0)

    # Log evaluation metrics
    for i in range(len(eval_return_values)):
        m = {
            "train/total_steps": eval_total_steps[i],
            "train/mean_approxkl": approxkl[i],
            "eval/mean_returns": eval_return_values[i],
            "eval/std_returns": eval_return_std[i],
            "eval/success_rate": eval_success_rate[i],
        }
        run.log(m)
        time.sleep(0.05)

    # Log final metrics
    m = {
        "final/mean_returns": eval_return_values[-1],
        "final/total_steps": eval_total_steps[-1],
        "final/success_rate": eval_success_rate[-1],
    }
    run.log(m)
    print(f"{m}")

    # Initialize agent params
    res = jax.tree_util.tree_map(lambda x: x[0], out)
    model_params = res["runner_state"][0].params["params"]
    act_scaling = res["act_scaling"]
    obs_scaling = res["norm_obs"]
    ctrl_params = params_env["controller"].replace(act_scaling=act_scaling, obs_scaling=obs_scaling, model=model_params,
                                                   hidden_activation=config.HIDDEN_ACTIVATION, stochastic=False)
    ctrl_params_onp = jax.tree_util.tree_map(lambda x: onp.array(x), ctrl_params)

    # Save control params
    ctrl_file = "controller_params.pkl"
    path_to_ctrl_file = f"/tmp/{ctrl_file}"
    with open(path_to_ctrl_file, "wb") as f:
        pickle.dump(ctrl_params_onp, f)
    print(f"Controller params saved to {path_to_ctrl_file}")

    # Save ppo config
    ppo_config_file = "ppo_config.pkl"
    path_to_ppo_config_file = f"/tmp/{ppo_config_file}"
    with open(path_to_ppo_config_file, "wb") as f:
        pickle.dump(ppo_config, f)
    print(f"PPO config saved to {path_to_ppo_config_file}")

    # Save rollout
    wandb.save(path_to_ctrl_file)
    wandb.save(path_to_ppo_config_file)
    print("Files logged to wandb!")

    # Finish wandb
    wandb.finish()

