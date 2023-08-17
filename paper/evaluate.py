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

import time
import datetime
import yaml
from stable_baselines3.common.vec_env import VecMonitor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import matplotlib

matplotlib.use("TkAgg")

import jax
import numpy as onp
import sbx
import stable_baselines3 as sb3
import experiments as exp
import paper
import rex.constants as rc
from rex.proto import log_pb2
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import envs.pendulum as pend


if __name__ == "__main__":
    wandb.setup()
    GROUP = f"train-real-evaluate-{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
    real_env = None
    for j in range(5):
        for JOB_TYPE in ["async", "deterministic"]:
            print(f"Running {JOB_TYPE}, seed {j}")
            ENV = {"async": "real-0.5Q-20hz-async-sbx-eps10-disc-pendulum-envs.pendulum.real.world-wall-clock-real-time-frequency-latest-2023-08-16-1201",
                   "deterministic": "ode-0.0Q-20hz-det-sbx-eps1-disc-pendulum-envs.pendulum.ode.world-simulated-clock-fast-as-possible-frequency-latest-2023-08-16-1152"
                   }[JOB_TYPE]
            run = wandb.init(
                mode="disabled",
                project="supergraph",
                group=GROUP,
                job_type=JOB_TYPE,
                config={
                    # Logging
                    "save_model": True,
                    "model_dir": ".",
                    "log_freq": 10,
                    "eval_freq": 3_000,
                    "n_eval_episodes": 10,
                    # Evaluate supergraph
                    "eval_supergraph": True,
                    "n_parallel_rollouts": 1000,
                    # Environment (real)
                    "eval_real": True,
                    "dist_file": "real-0.5Q-20hz-async-sbx-eps10-disc-pendulum-envs.pendulum.real.world-wall-clock-real-time-frequency-latest-2023-08-16-1154.pkl",
                    "env_type": "real",
                    "advance": False,
                    "blocking": False,
                    "jitter": rc.LATEST,
                    "scheduling": rc.FREQUENCY,
                    "real_time_factor": rc.REAL_TIME,
                    "clock": rc.WALL_CLOCK,
                    "use_delays": True,
                    # Environment (train)
                    "log_dir": "/home/r2ci/rex/paper/logs",
                    "supergraph_mode": "MCS",
                    "env": ENV,
                    # "env": "real-0.5Q-20hz-async-sbx-eps10-disc-pendulum-envs.pendulum.real.world-wall-clock-real-time-frequency-latest-2023-08-16-1201",
                    # "env": "ode-0.0Q-20hz-det-sbx-eps1-disc-pendulum-envs.pendulum.ode.world-simulated-clock-fast-as-possible-frequency-latest-2023-08-16-1152",
                    # Training
                    "seed": j,
                    "time_budget": 180,
                    "steps": 50_000,
                    # SAC Hyperparameters
                    "model": "SAC",
                    "num_envs": 4,  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                    "gamma": 0.9427860014779296,  # [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
                    "learning_rate": 0.0016222232059594057,  # [1e-5 to 1]
                    "batch_size": 2048,  # [16, 32, 64, 128, 256, 512, 1024, 2048]
                    "buffer_size": int(1e4),  # [1e4, 1e5, 1e6]
                    "learning_starts": 0,  # [0, 1000, 10000]
                    "tau": 0.08,  # [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]
                    "ent_coef": "auto",
                    "qf_learning_rate": 0.06341856428465494,  # Antonin
                    "log_std_init": -3,  # [-4, 1]
                    "net_arch": "small",  # ["small", "medium"]
                }
            )
            config = wandb.config

            # log env params
            LOG_DIR = f"{config.log_dir}/{config.env}"
            with open(f"{LOG_DIR}/params.yaml", "r") as f:
                params = yaml.load(f, Loader=paper.SafeLoader)
            # DIST_FILE = f"{LOG_DIR}/distributions.pkl"
            # params["dist_file"] = config.dist_file
            # DIST_FILE = params["DIST_FILE"]
            wandb.log({f"env/{k}": v for k, v in params.items()})

            # Load distributions
            delays_sim = exp.load_distributions(config.dist_file)

            # Load proto record
            record = log_pb2.ExperimentRecord()
            with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
                record.ParseFromString(f.read())

            # Train environment
            ode_env = exp.make_env(delays_sim, lambda x: 0., params["RATES"], win_action=params["WIN_ACTION"], win_obs=params["WIN_OBS"],
                                   env_fn=pend.ode.build_pendulum, env_cls=pend.env.PendulumEnv, name="disc-pendulum",
                                   eval_env=True, max_steps=params["MAX_STEPS"])
            cenv = exp.make_compiled_env(ode_env, record.episode, max_steps=ode_env.max_steps, eval_env=False, nodes_from="env",
                                         supergraph_mode=config.supergraph_mode, progress_bar=True)  # Compile env
            cenv_eval = exp.make_compiled_env(ode_env, record.episode, max_steps=ode_env.max_steps, eval_env=True, nodes_from="env",
                                              supergraph_mode=config.supergraph_mode, progress_bar=True)  # Compile env

            # Wrap model
            cenv = AutoResetWrapper(cenv)  # Wrap into auto reset wrapper
            cenv = VecGymWrapper(cenv, num_envs=config.num_envs)  # Wrap into vectorized environment
            cenv = VecMonitor(cenv)  # Wrap into vectorized monitor
            cenv.jit()  # Jit

            # Training
            model_params = paper.HYPERPARAMETERS_FN[config.model](config)
            cmodel = getattr(sbx, config.model)(policy="MlpPolicy", env=cenv, seed=config.seed, verbose=1, **model_params)

            # Learn
            budget_cb = paper.StopTrainingOnTimeBudget(time_budget=config.time_budget, verbose=1)
            eval_cb = paper.WandbCallback(eval_env=cenv_eval, n_eval_episodes=config.n_eval_episodes,
                                          log_freq=max(config.log_freq // config.num_envs, 1),
                                          eval_freq=max(config.eval_freq // config.num_envs, 1),
                                          model_dir=config.model_dir, save_model=config.save_model,
                                          verbose=1)
            cmodel.learn(total_timesteps=config.steps, progress_bar=True, callback=[eval_cb, budget_cb])

            # Save model
            cmodel.save(f"{config.model_dir}/model.zip")
            if config.save_model:
                wandb.save(f"{config.model_dir}/model.zip")
                wandb.save(f"{config.model_dir}/model_best.zip")

            # Reload best model
            model_best = getattr(sbx, config.model).load(f"{config.model_dir}/model_best.zip", verbose=1)

            # Evaluation real environment
            if config.eval_real:
                if real_env is None:
                    print("Building real environment...")
                    # Real environment
                    delay_fn = lambda d: float(d.quantile(params["QUANTILE"]))  # todo: this is slow (takes 3 seconds).
                    env_fn = {"real": pend.real.build_pendulum, "ode": pend.ode.build_pendulum}[config.env_type]
                    real_env = exp.make_env(delays_sim, delay_fn, params["RATES"], blocking=config.blocking, advance=config.advance,
                                            win_action=params["WIN_ACTION"], win_obs=params["WIN_OBS"],
                                            scheduling=config.scheduling, jitter=config.jitter,
                                            env_fn=env_fn, env_cls=pend.env.PendulumEnv, name="disc-pendulum", eval_env=True,
                                            clock=config.clock, real_time_factor=config.real_time_factor,
                                            max_steps=params["MAX_STEPS"], use_delays=config.use_delays)
                    real_env = GymWrapper(real_env)
                else:
                    print("Using existing real environment...")

                # Evaluate environment
                policy = exp.make_policy(model_best, constant_action=None)
                record_pre, episode_rewards = exp.eval_env(real_env, policy, n_eval_episodes=config.n_eval_episodes, verbose=True, seed=config.seed, return_rewards=True)

                # Log
                for r in episode_rewards:
                    wandb.log({"real/ep_rew": r})
                metrics = {"ep_rew_mean": onp.mean(episode_rewards), "ep_rew_std": onp.std(episode_rewards)}
                print(metrics)
                wandb.log({f"final/real/{k}": v for k, v in metrics.items()})

            # Evaluation supergraph
            if config.eval_supergraph:
                cenv_top = exp.make_compiled_env(ode_env, record.episode, max_steps=ode_env.max_steps, eval_env=True,
                                                  nodes_from="env", supergraph_mode="topological", progress_bar=True)  # Compile env
                cenv_gen = exp.make_compiled_env(ode_env, record.episode, max_steps=ode_env.max_steps, eval_env=True,
                                                 nodes_from="env", supergraph_mode="generational", progress_bar=True)
                supergraph_type = {"topological": cenv_top, "generational": cenv_gen, "mcs": cenv_eval}
                for mode, cenv_sup in supergraph_type.items():
                    timings = cenv_sup.graph._default_timings
                    num_nodes_run = 0
                    num_nodes = 0
                    for slots in timings:
                        for s, t in slots.items():
                            num_nodes += t["run"].sum()
                            num_nodes_run += (t["run"] == t["run"]).sum()
                    efficiency = 100 * num_nodes / num_nodes_run
                    size = cenv_sup.graph.S.number_of_nodes()

                    # Perform rollouts
                    rw = paper.RolloutWrapper(cenv_sup, model_best)

                    # Time to jit
                    seed = jax.random.PRNGKey(0)
                    rng = jax.random.split(seed, num=config.n_parallel_rollouts)
                    start = time.time()
                    obs, action, reward, next_obs, done, cum_return = rw.batch_rollout(rng, model_best.policy.actor_state)
                    end = time.time()
                    duration = end - start
                    fps = (config.n_parallel_rollouts * cenv_sup.max_steps) / duration
                    return_mean, return_std = cum_return.mean(), cum_return.std()
                    metrics = {"supergraph_type": mode, "time_to_jit": duration, "efficiency_percentage": efficiency, "size": size}
                    print(f"{mode} | time_to_jit | {metrics}")
                    wandb.log({f"supergraph/{k}": v for k, v in metrics.items()})

                    # Time to run
                    for i in range(1, 10):
                        seed = jax.random.PRNGKey(i)
                        rng = jax.random.split(seed, num=config.n_parallel_rollouts)
                        start = time.time()
                        obs, action, reward, next_obs, done, cum_return = rw.batch_rollout(rng, model_best.policy.actor_state)
                        end = time.time()
                        duration = end - start
                        fps = (config.n_parallel_rollouts * cenv_sup.max_steps) / duration
                        return_mean, return_std = cum_return.mean(), cum_return.std()
                        metrics = {"supergraph_type": mode, "fps": fps, "duration": duration, "ep_rew_mean": return_mean, "ep_rew_std": return_std}
                        print(f"{mode} | speed | {metrics}")
                        wandb.log({f"speed/{k}": v for k, v in metrics.items()})

            # Finish wandb
            run.finish()
