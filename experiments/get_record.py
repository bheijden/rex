# HACK: https://github.com/DLR-RM/stable-baselines3/pull/780
import sys
import gymnasium
sys.modules["gym"] = gymnasium

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tempfile
import datetime
from typing import Dict, List, Tuple, Union, Callable, Any, Type
from types import ModuleType
import dill as pickle
import time
import jax
import numpy as onp
from stable_baselines3.common.vec_env import VecMonitor

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

import sbx
import stable_baselines3 as sb3
import time
import rex
from rex.supergraph import create_graph
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
	WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import experiments as exp
import envs.pendulum as pend


if __name__ == "__main__":
	# Environment
	ENV = "disc_pendulum"  # "disc_pendulum"
	DIST_FILE = "21eps_pretrained_sbx_sac_gmms_2comps.pkl"
	JITTER = LATEST
	SCHEDULING = FREQUENCY
	MAX_STEPS = 5*20
	START_STEPS = 0
	WIN_ACTION = 2
	WIN_OBS = 3
	BLOCKING = False
	ADVANCE = False
	ENV_FN = pend.ode.build_pendulum  #  dpend.ode.build_double_pendulum  # pend.ode.build_pendulum
	ENV_CLS = pend.env.PendulumEnv  # dpend.env.DoublePendulumEnv  # pend.env.PendulumEnv
	CLOCK = SIMULATED
	RTF = FAST_AS_POSSIBLE
	RATES = dict(world=100, agent=20, actuator=20, sensor=20, render=20)
	USE_DELAYS = True
	# DELAY_FN = lambda d: 0.0
	DELAY_FN = lambda d: d.quantile(0.5)*int(USE_DELAYS)  # todo: this is slow (takes 3 seconds).

	# Logging
	NAME = f"supergraph_optimized_{ENV}"
	LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
	MUST_LOG = False
	SHOW_PLOTS = False
	RECORD_SETTINGS = {"agent": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
	                   "world": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
	                   "actuator": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
	                   "sensor": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
	                   "render": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False)}

	# Load models. PPO=[64, 64], SAC=[256, 256]
	MODEL_CLS = sbx.SAC  # sbx.SAC  sb3.SAC
	MODEL_MODULE = pend.models
	MODEL_PRELOAD = None # "sbx_ppo_pendulum"  # sbx_sac_pendulum
	# MODEL_PRELOAD = "/home/r2ci/rex/logs/sim_dp_sbx_2023-02-01-1510/model.zip"  # todo: very good sbx model.

	# Training
	CONTINUE = False
	SEED = 0
	NUM_ENVS = 4
	SAVE_FREQ = 40_000
	NSTEPS = 40_000
	NUM_EVAL_PRE = 1
	NUM_EVAL_POST = 20
	HYPERPARAMS = {
		"gamma": 0.9427860014779296,  # [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
        "learning_rate": 0.0016222232059594057,  # [1e-5 to 1]
        "batch_size": 2048,  # [16, 32, 64, 128, 256, 512, 1024, 2048]
        "buffer_size": int(1e4),  # [1e4, 1e5, 1e6]
        "learning_starts": 0,  # [0, 1000, 10000]
		"tau": 0.08,  # [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]
		"ent_coef": "auto",
		"qf_learning_rate": 0.06341856428465494,
		"policy_kwargs": dict(log_std_init=-3, net_arch=[64, 64], use_sde=False),
	}

	# Load distributions
	delays_sim = exp.load_distributions(DIST_FILE, module=pend.dists)

	# Prepare environment
	env = exp.make_env(delays_sim, DELAY_FN, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
	                   scheduling=SCHEDULING, jitter=JITTER,
	                   env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True, clock=CLOCK, real_time_factor=RTF,
	                   max_steps=MAX_STEPS + START_STEPS, use_delays=USE_DELAYS)
	gym_env = GymWrapper(env)

	# Load model
	model: MODEL_CLS = exp.load_model(MODEL_PRELOAD, MODEL_CLS, env=gym_env, seed=SEED, module=MODEL_MODULE)
	sys_model = exp.SysIdPolicy(rate=RATES["agent"], duration=3.0, min=-2.0, max=2.0, seed=0, model=model, use_ros=False)
	policy = exp.make_policy(sys_model)
	# policy = exp.make_policy(model)

	# Evaluate model
	record_pre = exp.eval_env(gym_env, policy, n_eval_episodes=NUM_EVAL_PRE, verbose=True, seed=SEED, record_settings=RECORD_SETTINGS)

	# Compile env
	cenv = exp.make_compiled_env(env, record_pre.episode, max_steps=MAX_STEPS, eval_env=False)

	# Plot
	G = create_graph(record_pre.episode[-1])
	if SHOW_PLOTS or MUST_LOG:
		fig_cg, _ = exp.show_computation_graph(G, cenv.graph.S, root="agent", plot_type="computation")
		fig_com, _ = exp.show_communication(record_pre.episode[-1])
		fig_grp, _ = exp.show_grouped(record_pre.episode[-1].node[-1], "state")

	# Only show
	if SHOW_PLOTS:
		plt.show()

	if MUST_LOG:
		os.mkdir(LOG_DIR)
		# Save plots
		fig_cg.savefig(LOG_DIR + "/computation_graph.png")
		fig_com.savefig(LOG_DIR + "/communication.png")
		fig_grp.savefig(LOG_DIR + "/grouped_agent_sensor.png")
		# Save envs
		env.unwrapped.save(f"{LOG_DIR}/{env.unwrapped.name}.pkl")
		cenv.unwrapped.save(f"{LOG_DIR}/{cenv.unwrapped.name}.pkl")
		# Save pre-train record to file
		with open(LOG_DIR + "/record_pre.pb", "wb") as f:
			f.write(record_pre.SerializeToString())

		from stable_baselines3.common.callbacks import CheckpointCallback

		checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ // NUM_ENVS, save_path=LOG_DIR, name_prefix="checkpoint")
	else:
		LOG_DIR = None
		checkpoint_callback = None

	# Wrap model
	cenv = AutoResetWrapper(cenv)  # Wrap into auto reset wrapper
	cenv = VecGymWrapper(cenv, num_envs=NUM_ENVS)  # Wrap into vectorized environment
	cenv = VecMonitor(cenv)  # Wrap into vectorized monitor
	cenv.jit()  # Jit

	# Initialize model
	if CONTINUE:
		model_path = f"{LOG_DIR}/model.zip" if MUST_LOG else f"{tempfile.mkdtemp()}/reload_model.zip"
		model.save(model_path)
		cmodel = MODEL_CLS.load(model_path, env=cenv, seed=SEED, **HYPERPARAMS, verbose=1, tensorboard_log=LOG_DIR)
	else:
		cmodel = MODEL_CLS("MlpPolicy", cenv, seed=SEED, **HYPERPARAMS, verbose=1, tensorboard_log=LOG_DIR)

	# Learn
	cmodel.learn(total_timesteps=NSTEPS, progress_bar=True, callback=checkpoint_callback)

	# Save file
	model_path = f"{LOG_DIR}/model.zip" if MUST_LOG else f"{tempfile.mkdtemp()}/model.zip"
	cmodel.save(model_path)

	env = exp.make_env(delays_sim, DELAY_FN, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
	                   scheduling=SCHEDULING, jitter=JITTER,
	                   env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True, clock=CLOCK, real_time_factor=REAL_TIME,
	                   max_steps=MAX_STEPS, use_delays=USE_DELAYS)

	# Load model
	model_reloaded = exp.load_model(model_path, MODEL_CLS, env=GymWrapper(env))

	# Evaluate model
	policy_reloaded = exp.make_policy(model_reloaded)
	record_post = exp.eval_env(env, policy_reloaded, n_eval_episodes=NUM_EVAL_POST, verbose=True, record_settings=RECORD_SETTINGS)

	# Log
	if MUST_LOG:
		# Save post-train record to file
		with open(LOG_DIR + "/record_post.pb", "wb") as f:
			f.write(record_post.SerializeToString())
