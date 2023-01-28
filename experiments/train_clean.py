# Script to define environments from
# Automatically register all environments in the envs folder
# Script to fit delays --> Save as proto for either ode or real
# Script to evaluate performance
import gym
import os
import tempfile
import datetime
from typing import Dict, List, Tuple, Union, Callable, Any, Type
from types import ModuleType
import dill as pickle
import time
import jax
import numpy as onp
import jumpy
import jumpy.numpy as jp
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
from rex.distributions import Distribution, Gaussian
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, SEQUENTIAL, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, GRAPH_MODES
import rex.open_colors as oc
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import experiments as exp
import envs.pendulum as pend
import envs.double_pendulum as dpend
from jax.tree_util import tree_map


class SysIdPolicy:
	def __init__(self, duration: float = 5.0, min: float = -8, max: float = 8, seed: int = 0, model=None):

		self._model = model
		self._duration = duration
		self._last_time = 0
		self._max = max
		self._min = min
		self._action = jp.array([0], dtype=jp.float32)
		self._use_model = False
		self._rng = jumpy.random.PRNGKey(seed)

	def predict(self, obs, deterministic: bool = True):
		tstep = time.time()
		if tstep - self._last_time > self._duration:
			self._last_time = tstep
			if model is not None and not self._use_model:
				self._use_model = True
			else:
				self._use_model = False
				self._rng, rng_action = jumpy.random.split(self._rng)
				action = jumpy.random.uniform(rng_action, low=self._min, high=self._max)
				self._action = jp.array([action], dtype=jp.float32)
		if self._use_model:
			return self._model.predict(obs, deterministic=deterministic)
		else:
			return self._action, None


if __name__ == "__main__":
	# todo: place render.step in separate process?
	# todo: remove 1.01 from phase. It's a hack to make sure that the phase is always slightly greater than the BUFFER.
	# Logging
	NAME = "double_pendulum_sim"
	LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
	MUST_LOG = False
	MUST_PLOT = True

	# Environment
	ENV = "double_pendulum"  # "disc_pendulum"
	DIST_FILE = f"21eps_pretrained_sbx_sac_gmms_2comps.pkl"
	JITTER = LATEST
	SCHEDULING = PHASE
	MAX_STEPS = 5*80
	WIN_ACTION = 0
	WIN_OBS = 1
	BLOCKING = True
	ADVANCE = False
	ENV_FN = dpend.ode.build_double_pendulum  #  dpend.ode.build_double_pendulum  # pend.ode.build_pendulum
	ENV_CLS = dpend.env.DoublePendulumEnv  # dpend.env.DoublePendulumEnv  # pend.env.PendulumEnv
	CLOCK = SIMULATED
	RTF = FAST_AS_POSSIBLE
	RATES = dict(world=150, agent=80, actuator=80, sensor=80, render=20)
	DELAY_FN = lambda d: d.high*1.0

	# Load models
	MODEL_CLS = sbx.SAC
	MODEL_MODULE = dpend.models
	MODEL_PRELOAD = "sb_sac_double_pendulum_runyu"  # sb_sac_pendulum
	# MODEL_PRELOAD = "sb_sac_pendulum"  # sb_sac_pendulum

	# Training
	SEED = 0
	LEARNING_RATE = 1e-2
	NUM_ENVS = 10
	NSTEPS = 100_000
	NUM_EVAL_PRE = 20
	NUM_EVAL_POST = 20

	# Load distributions & set rates
	delays_sim = exp.load_distributions(DIST_FILE)
	delays_sim["step"]["world"] = Gaussian(0.)
	delays_sim["inputs"]["world"]["actuator"] = Gaussian(0.)
	delays_sim["inputs"]["sensor"]["world"] = Gaussian(0.)
	delays_sim = tree_map(lambda d: Gaussian(0), delays_sim) # todo: REMOVE
	delays = jax.tree_map(DELAY_FN, delays_sim)

	# Prepare environment
	env = exp.make_env(delays_sim, delays, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
	                   scheduling=SCHEDULING, jitter=JITTER,
	                   env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True, clock=CLOCK, real_time_factor=RTF,
	                   max_steps=MAX_STEPS)
	gym_env = GymWrapper(env)

	# Load model
	try:
		model = exp.load_model(MODEL_PRELOAD, MODEL_CLS, env=gym_env, seed=SEED, module=MODEL_MODULE)
		policy = exp.make_policy(model, constant_action=0.)  # todo: REMOVE
	except ValueError as e:
		if "Observation spaces" in e.__str__():
			print("Observation spaces don't match. Loading a random model instead.")
			model = MODEL_CLS("MlpPolicy", env=gym_env, seed=0, verbose=1)
			policy = exp.make_policy(model, constant_action=0.)
		else:
			raise e

	# TODO: REMOVE SYS ID POLICY
	# sys_model = SysIdPolicy(duration=3.0, min=-1.5, max=1.5, seed=0, model=model)
	# policy = exp.make_policy(sys_model)

	# Evaluate model
	record_pre = exp.eval_env(gym_env, policy, n_eval_episodes=NUM_EVAL_PRE, verbose=True, seed=SEED)

	# Get data
	data = exp.RecordHelper(record_pre)

	# if MUST_LOG:
	# 	# Save pre-train record to file
	# 	os.mkdir(LOG_DIR)
	# 	with open(LOG_DIR + "/record_sysid.pb", "wb") as f:
	# 		f.write(record_pre.SerializeToString())
	# 	print("SAVED!")
	# exit()

	# Compile env
	cenv = exp.make_compiled_env(env, record_pre.episode[-1], max_steps=MAX_STEPS, eval_env=False, graph_type=SEQUENTIAL, plot=MUST_PLOT)

	# Plot
	if MUST_PLOT:
		exp.show_communication(record_pre.episode[-1])
		exp.show_grouped(record_pre.episode[-1].node[-1], "state")
		plt.show()

	if MUST_LOG:
		os.mkdir(LOG_DIR)
		env.unwrapped.save(f"{LOG_DIR}/{env.unwrapped.name}.pkl")
		cenv.unwrapped.save(f"{LOG_DIR}/{cenv.unwrapped.name}.pkl")
		# Save pre-train record to file
		with open(LOG_DIR + "/record_pre.pb", "wb") as f:
			f.write(record_pre.SerializeToString())

		from stable_baselines3.common.callbacks import CheckpointCallback

		checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=LOG_DIR, name_prefix="checkpoint")
	else:
		LOG_DIR = None
		checkpoint_callback = None

	# Wrap model
	cenv = AutoResetWrapper(cenv)  # Wrap into auto reset wrapper
	cenv = VecGymWrapper(cenv, num_envs=NUM_ENVS)  # Wrap into vectorized environment
	cenv = VecMonitor(cenv)  # Wrap into vectorized monitor
	cenv.jit()  # Jit

	# Learn
	cmodel = MODEL_CLS("MlpPolicy", cenv, learning_rate=LEARNING_RATE, verbose=1, tensorboard_log=LOG_DIR)
	cmodel.learn(total_timesteps=NSTEPS, progress_bar=True, callback=checkpoint_callback)

	# Save file
	model_path = f"{LOG_DIR}/model.zip" if MUST_LOG else f"{tempfile.mkdtemp()}/model.zip"
	cmodel.save(model_path)

	# Load model
	model_reloaded = exp.load_model(model_path, MODEL_CLS, env=GymWrapper(env))

	# Evaluate model
	policy_reloaded = exp.make_policy(model_reloaded)
	record_post = exp.eval_env(env, policy_reloaded, n_eval_episodes=NUM_EVAL_POST, verbose=True)

	# Log
	if MUST_LOG:
		# Save post-train record to file
		with open(LOG_DIR + "/record_post.pb", "wb") as f:
			f.write(record_post.SerializeToString())
