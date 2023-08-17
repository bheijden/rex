# HACK: https://github.com/DLR-RM/stable-baselines3/pull/780
import sys
import gymnasium

sys.modules["gym"] = gymnasium

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import wandb
os.environ["WANDB_SILENT"] = "true"

# Add /home/r2ci/rex/paper to PYTHONPATH
sys.path.append("/home/r2ci/rex")
sys.path.append("/home/r2ci/supergraph")
sys.path.append("/home/r2ci/sbx")

import datetime
import yaml
from stable_baselines3.common.vec_env import VecMonitor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import matplotlib

matplotlib.use("TkAgg")

import sbx
import paper
from rex.env import BaseEnv
import rex.constants as rc
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper


if __name__ == "__main__":
    wandb.setup()

    run = wandb.init(
        mode="online",
        project="supergraph",
        # group=f"train-{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}",
        group=f"test-train-sweep",
        job_type="train-pendulum",
        config={
            # Logging
            "save_model": True,
            "model_dir": "./",
            # Environment
            "log_dir": "/home/r2ci/rex/paper/logs",
            "env": "async-sbx-eps10-disc-pendulum-envs.pendulum.ode.world-wall-clock-real-time-frequency-latest-2023-08-14-1110",
            # "env": "deterministic-sbx-eps1-disc-pendulum-envs.pendulum.ode.world-simulated-clock-fast-as-possible-frequency-latest-2023-08-14-1109",
            # Training
            "seed": 0,
            "time_budget": 180,
            "steps": 50_000,
            "eval_freq": 10_000,
            "n_eval_episodes": 50,
            # SAC Hyperparameters
            "model": "SAC",
            "num_envs": 4,  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            "gamma": 0.9427860014779296,  # [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
            "learning_rate": 0.0016222232059594057,  # [1e-5 to 1]
            "batch_size": 2048,  # [16, 32, 64, 128, 256, 512, 1024, 2048]
            "buffer_size": int(1e4),  # [1e4, 1e5, 1e6]
            "learning_starts": 0,  # [0, 1000, 10000]
            # "train_freq": num_envs,  # Antonin [1, 4, 8, 16, 32, 64, 128, 256, 512]
            # "gradient_steps": num_envs,  # Antonin [1, 4, 8, 16, 32, 64, 128, 256, 512]
            "tau": 0.08,  # [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]
            "ent_coef": "auto",
            "qf_learning_rate": 0.06341856428465494,  # Antonin
            "log_std_init": -3,  # [-4, 1]
            "net_arch": "small",  # ["small", "medium"]
            # PPO Hyperparameters
            # "model": "PPO",
            # "num_envs": 4,  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            # "batch_size": 64,  # [8, 16, 32, 64, 128, 256, 512]
            # "n_steps": 1024,  # [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            # "gamma": 0.99,  # [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
            # "learning_rate": 3e-3,  # [1e-4 to 1e-1]
            # "lr_schedule": "constant",
            # "ent_coef": 0.000001,  # [0.00000001 to 0.1]
            # "clip_range": 0.2,  # [0.1, 0.2, 0.3, 0.4]
            # "n_epochs": 10,  # [1, 5, 10, 20]
            # "gae_lambda": 0.9,  # [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
            # "max_grad_norm": 0.5,  # [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
            # "vf_coef": 0.5,  # [0.0 to 1.0]
            # "net_arch": "small",  # ["small", "medium"]
            # "ortho_init": True,  # [True, False]
            # "activation_fn": "relu",  # ["tanh", "relu"]
        }
    )
    config = wandb.config

    # Environment
    LOG_DIR = f"{config.log_dir}/{config.env}"
    DIST_FILE = f"{LOG_DIR}/distributions.pkl"
    CENV_FILE = f"{LOG_DIR}/" + next(filter(lambda x: x.endswith("compiled.pkl"), os.listdir(LOG_DIR)))
    cenv = BaseEnv.load(CENV_FILE)  # Load the environment
    cenv_eval = BaseEnv.load(CENV_FILE)  # Load the environment

    # Wrap model
    cenv = AutoResetWrapper(cenv)  # Wrap into auto reset wrapper
    cenv = VecGymWrapper(cenv, num_envs=config.num_envs)  # Wrap into vectorized environment
    cenv = VecMonitor(cenv)  # Wrap into vectorized monitor
    cenv.jit()  # Jit

    # log env params
    with open(f"{LOG_DIR}/params.yaml", "r") as f:
        env_params = yaml.load(f, Loader=paper.SafeLoader)
    env_params["dist_file"] = DIST_FILE
    env_params["size"] = cenv.graph.S.size()
    wandb.log({f"env/{k}": v for k, v in env_params.items()})

    # Training
    model_params = paper.HYPERPARAMETERS_FN[config.model](config)
    cmodel = getattr(sbx, config.model)(policy="MlpPolicy", env=cenv, seed=config.seed, verbose=1, **model_params)

    # Learn
    from wandb.integration.sb3 import WandbCallback

    budget_cb = paper.StopTrainingOnTimeBudget(time_budget=config.time_budget, verbose=1)
    eval_cb = paper.WandbCallback(eval_env=cenv_eval, n_eval_episodes=config.n_eval_episodes,
                                  eval_freq=max(config.eval_freq // config.num_envs, 1),
                                  verbose=1)
    cmodel.learn(total_timesteps=config.steps, progress_bar=True, callback=[eval_cb, budget_cb])

    if config.save_model:
        cmodel.save(f"{config.model_dir}/model.zip")
        wandb.save(f"{config.model_dir}/model.zip")

    # Finish wandb
    run.finish()