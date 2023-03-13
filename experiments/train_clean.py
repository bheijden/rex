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
import rex.tracer as tracer
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import experiments as exp
import envs.pendulum as pend
import envs.double_pendulum as dpend


HYPERPARAMS = {
	# "gamma": 0.95,
	"learning_rate": 0.01,
	"gradient_steps": 10,  # Antonin
	"train_freq": 10,  # Antonin
	"qf_learning_rate": 1e-3,  # Antonin
	# "batch_size": 1024, # todo: maybe this was turned on?
	# "buffer_size": 10000,
	# "learning_starts": 0,
	# "train_freq": 4,
	# "gradient_steps": 4,  # same as train_freq
	# "ent_coef": "auto",
	# "tau": 0.08,
	# "target_entropy": "auto",
	# "policy_kwargs": dict(log_std_init=-0.07520582048294414, net_arch=[256, 256], use_sde=False),  # todo: ADD?
}


RECORD_SETTINGS = {"agent": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "world": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "actuator": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "sensor": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "render": dict(node=True, outputs=False, rngs=True, states=True, params=True, step_states=True)}

if __name__ == "__main__":
	# todo: DoublePendulum env notes:
	#     - Thdot2 sometimes saturates above 50 rad/s. This may be a problem for the root.
	#     - Increase max angular velocity (and take change of observation space into account).
	#     - Change initial state distribution (ie thdot, thdot2 = 0., 0.).
	#     - High sensor rate (100 Hz), low actuator rate (30 Hz). Increase sensor window.
	#     - Increase penalty on action changes
	#     - Increase noise (th, th2, thdot, thdot2)
	#     - Consistently learn
	#     - Add process noise to actuator scaled by delta action.
	#     - Add coulomb friction to make the learning easier in the upright position.
	#     - Use a stabilizing controller for upright position, and augment actions with a switching parameter.
	# todo: Verify that measured delays are working correctly
	# todo: Make new process wrapper that unpickle nodes of inputs as BaseNodes (enables access to e.g. phase, msg structures).
	# todo: Modify graph sorting that traces multiple outputs (i.e. add rendering to compiled graph).

	# Environment
	ENV = "double_pendulum"  # "disc_pendulum"
	# DIST_FILE = f"21eps_pretrained_sbx_sac_gmms_2comps.pkl"
	DIST_FILE = f"record_sysid-gmms-2comps.pkl"
	# DIST_FILE = f"real_pendulum_2023-01-27-1806_phase_blocking_120s_record_sysid-gmms-2comps.pkl"
	JITTER = BUFFER
	SCHEDULING = PHASE
	MAX_STEPS = 5*80
	START_STEPS = 0*MAX_STEPS
	WIN_ACTION = 2
	WIN_OBS = 3
	BLOCKING = True
	ADVANCE = False
	ENV_FN = dpend.ode.build_double_pendulum  #  dpend.ode.build_double_pendulum  # pend.ode.build_pendulum
	ENV_CLS = dpend.env.DoublePendulumEnv  # dpend.env.DoublePendulumEnv  # pend.env.PendulumEnv
	CLOCK = SIMULATED
	RTF = FAST_AS_POSSIBLE
	RATES = dict(world=150, agent=80, actuator=80, sensor=80, render=20)
	USE_DELAYS = True
	DELAY_FN = lambda d: d.quantile(0.99)*int(USE_DELAYS)  # todo: this is slow (takes 3 seconds).

	# Logging
	NAME = f"sysid_{ENV}"
	LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
	MUST_LOG = False
	SHOW_PLOTS = True

	# Load models
	MODEL_CLS = sbx.SAC  # sbx.SAC  sb3.SAC
	MODEL_MODULE = dpend.models
	MODEL_PRELOAD = "/home/r2ci/rex/logs/sim_dp_sbx_2023-02-01-1510/model.zip"  # todo: very good sbx model.
	# MODEL_PRELOAD = "/home/r2ci/rex/logs/50maxvel_8torque_30hz_005deltaact_agentpolicy_double_pendulum_2023-02-03-1905/model.zip"
	# MODEL_PRELOAD = "/home/r2ci/rex/logs/lessnoise_50maxvel_8torque_30hz_005deltaact_agentpolicy_double_pendulum_2023-02-03-1942/model.zip"
	# MODEL_PRELOAD = "/home/r2ci/rex/logs/continued_8torque_80hz_02deltaact_agentpolicy_double_pendulum_2023-02-03-1809/model.zip"
	# MODEL_PRELOAD = "/home/r2ci/rex/logs/testready_continue_good_20nenvs_withnoise_double_pendulum_2023-02-03-1526/model.zip"
	# MODEL_PRELOAD = "sbx_sac_pendulum"  # sb_sac_pendulum

	# Training
	CONTINUE = True
	SEED = 0
	NUM_ENVS = 10
	SAVE_FREQ = 40_000
	NSTEPS = 200_000
	NUM_EVAL_PRE = 1
	NUM_EVAL_POST = 20

	# Load distributions
	delays_sim = exp.load_distributions(DIST_FILE, module=dpend.dists)

	# Prepare environment
	env = exp.make_env(delays_sim, DELAY_FN, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
	                   scheduling=SCHEDULING, jitter=JITTER,
	                   env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True, clock=CLOCK, real_time_factor=RTF,
	                   max_steps=MAX_STEPS + START_STEPS, use_delays=USE_DELAYS)
	gym_env = GymWrapper(env)

	# Reload protobuf record
	# from rex.proto import log_pb2
	# NAME = f"sysid_double_pendulum_2023-02-06-0914"
	# LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}"
	# ENV = "double_pendulum_ode_train_buffer_phase_awin2_owin3_blocking_noadvance_compiled_vectorized.pkl"
	# RECORD = "record_pre.pb"
	# record_pre = log_pb2.ExperimentRecord()
	# with open(f"{LOG_DIR}/{RECORD}", "rb") as f:
	# 	record_pre.ParseFromString(f.read())
	# data = exp.RecordHelper(record_pre)
	#
	# from rex.base import StepState
	# import rex.jumpy as rjp
	#
	# def make_replay_step(node, outputs, step_states: StepState = None):
	# 	def _replay_step(step_state: StepState):
	# 		seq = step_state.seq
	# 		output = rjp.tree_take(outputs, seq, axis=0)
	# 		if step_states is not None:
	# 			new_step_state = rjp.tree_take(step_states, seq+1, axis=0)
	# 		else:
	# 			new_step_state = step_state
	# 		return new_step_state, output
	# 	return _replay_step
	#
	#
	# outputs = data._data[0]["actuator"]["outputs"].tree
	# actuator = env.unwrapped.graph.nodes["actuator"]
	# actuator.step = make_replay_step(actuator, outputs)

	# Load model
	model: sbx.SAC = exp.load_model(MODEL_PRELOAD, MODEL_CLS, env=gym_env, seed=SEED, module=MODEL_MODULE)

	sys_model = exp.SysIdPolicy(rate=RATES["agent"], duration=5.0, min=-1.0, max=1.0, seed=0, model=model, use_ros=False)
	policy = exp.make_policy(sys_model)
	# policy = exp.make_policy(model)

	# Evaluate model
	record_pre = exp.eval_env(gym_env, policy, n_eval_episodes=NUM_EVAL_PRE, verbose=True, seed=SEED, record_settings=RECORD_SETTINGS)

	# Get data
	# data = exp.RecordHelper(record_pre)

	# if MUST_LOG:
	# 	env.unwrapped.save(f"{LOG_DIR}/{env.unwrapped.name}.pkl")
	# 	# Save pre-train record to file
	# 	os.mkdir(LOG_DIR)
	# 	with open(LOG_DIR + "/record_sysid.pb", "wb") as f:
	# 		f.write(record_pre.SerializeToString())
	# 	print("SAVED!")
	# exit()

	# Compile env
	cenv = exp.make_compiled_env(env, record_pre.episode[-1], max_steps=MAX_STEPS, eval_env=False)

	# Plot
	G = tracer.create_graph(record_pre.episode[-1])
	fig_cg, _ = exp.show_computation_graph(G, cenv.graph.MCS, root="agent", plot_type="computation")
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
		# cmodel = MODEL_CLS.load(model_path, env=cenv, verbose=1, tensorboard_log=LOG_DIR)
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
