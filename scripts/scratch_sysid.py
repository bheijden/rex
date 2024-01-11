import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tempfile
import datetime
from typing import Dict, List, Tuple, Union, Callable, Any, Type
from types import ModuleType
import dill as pickle
import time
import jax
import jumpy
import jax.numpy as jnp
import numpy as onp
import rex.jax_utils as rjax
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
from rex.proto import log_pb2
from rex.env import BaseEnv
from rex.supergraph import get_network_record
from rex.compiled import CompiledGraph
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import experiments as exp
import envs.double_pendulum as dpend


import matplotlib.pyplot as plt



RECORD_SETTINGS = {"agent": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "world": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "actuator": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "sensor": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "render": dict(node=True, outputs=False, rngs=True, states=True, params=True, step_states=True)}


if __name__ == "__main__":
	# todo: Trace new episode based on pre_record, where:
	# 1. Recorded delays are used (or modify trace record?)
	# 4. Upsample world node to 150 Hz.
	# 6. Set default outputs in compiled graph.
	# 8. How to reuse outputs of previous batches during training? --> reuse outputs in graph_state --> How to spread info across samples in a batches?
	# 10: Do not re-index topological sort using depths. This improves distribution of steps across isolated depths.
	# 12: Define loss mask for unwrapped (e.g. [0., inf] mask for loss scaling)
	# 13: Learn (measurement & process) noise levels for sensors & actuator?
	# 14: Write node wrapper that takes care of replacing references in Output and connected Inputs.
	# steps/sec scale linearly with rollout length.
	# steps/sec unaffected by num_batches.

	# Environment
	ENV = "double_pendulum"
	DIST_FILE = f"record_sysid-gmms-2comps.pkl"
	# DIST_FILE = f"real_pendulum_2023-01-27-1806_phase_blocking_120s_record_sysid-gmms-2comps.pkl"
	JITTER = BUFFER
	SCHEDULING = PHASE
	MAX_STEPS = int(5*80)
	START_STEPS = 0*MAX_STEPS
	WIN_ACTION = 2
	WIN_OBS = 3
	BLOCKING = True
	ADVANCE = False
	ENV_FN = dpend.ode.build_double_pendulum  # dpend.ode.build_double_pendulum  # pend.ode.build_pendulum
	ENV_CLS = dpend.env.DoublePendulumEnv  # dpend.env.DoublePendulumEnv  # pend.env.PendulumEnv
	CLOCK = SIMULATED
	RTF = FAST_AS_POSSIBLE
	RATE_ESTIMATOR = 40
	RATES = dict(world=150, agent=80, actuator=80, sensor=80, render=20)
	USE_DELAYS = True
	# DELAY_FN = lambda d: d.quantile(0.99)*int(USE_DELAYS)
	DELAY_FN = lambda d: d.high*int(USE_DELAYS)

	# Logging
	NAME = f"sysid_{ENV}"
	LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
	MUST_LOG = False
	SHOW_PLOTS = False

	# Load models
	# MODEL_CLS = sbx.SAC  # sbx.SAC  sb3.SAC
	# MODEL_MODULE = dpend.models
	# MODEL_PRELOAD = "/home/r2ci/rex/logs/sim_dp_sbx_2023-02-01-1510/model.zip"
	MODEL_CLS = sb3.SAC  # sbx.SAC
	MODEL_MODULE = dpend.models
	MODEL_PRELOAD = "sb_sac_model"

	# Training
	CONTINUE = True
	SEED = 0
	NUM_ENVS = 10
	SAVE_FREQ = 40_000
	NSTEPS = 200_000
	NUM_EVAL_PRE = 2
	NUM_EVAL_POST = 20

	# Load distributions
	delays_sim = exp.load_distributions(DIST_FILE, module=dpend.dists)

	# Prepare environment
	env = exp.make_env(delays_sim, DELAY_FN, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
	                   scheduling=SCHEDULING, jitter=JITTER,
	                   env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True, clock=CLOCK, real_time_factor=RTF,
	                   max_steps=MAX_STEPS + START_STEPS, use_delays=USE_DELAYS)
	gym_env = GymWrapper(env)

	# Load model
	# model: sbx.SAC = exp.load_model(MODEL_PRELOAD, MODEL_CLS, env=gym_env, seed=SEED, module=MODEL_MODULE)

	# Prepare policy
	sys_model = exp.SysIdPolicy(rate=RATES["agent"], duration=0.1, min=-8.0, max=8.0, seed=0, model=None, use_ros=False)
	policy = exp.make_policy(sys_model)

	# Evaluate model
	record_pre = exp.eval_env(gym_env, policy, n_eval_episodes=NUM_EVAL_PRE, verbose=True, seed=SEED,
	                          record_settings=RECORD_SETTINGS)

	# Prepare data
	data = exp.RecordHelper(record_pre)

	# Build estimator
	from estimator import build_estimator
	record, nodes, excludes_inputs = build_estimator(record_pre, rate=RATE_ESTIMATOR, data=data)
 
	# Trace
	record_network, S, S_init_to_S, Gs, Gs_monomorphism = get_network_record(record.episode, "estimator", excludes_inputs=excludes_inputs)

	# Only show
	if SHOW_PLOTS:
		# Plot
		fig_cg, _ = exp.show_computation_graph(Gs[0], S, "estimator", plot_type="computation", xmax=2.0)
		fig_dp, _ = exp.show_computation_graph(Gs[0], S, "estimator", plot_type="depth", xmax=2.0)
		fig_tp, _ = exp.show_computation_graph(Gs[0], S, "estimator", plot_type="topological", xmax=2.0)
		# fig_com, _ = exp.show_communication(record_eps)
		# fig_grp, _ = exp.show_grouped(record_eps.node[-1], "state")
		plt.show()

	# Add Estimator node to record
	import flax.struct as struct

	@struct.dataclass
	class Loss:
		loss: Union[float, jax.typing.ArrayLike]
		rloss: Union[float, jax.typing.ArrayLike]
		dloss: Union[float, jax.typing.ArrayLike]


	alpha_dloss = nodes["world"].default_state(jax.random.PRNGKey(0)).replace(th=jp.float32(0.e0), th2=jp.float32(0.e0), thdot=jp.float32(0.e-2), thdot2=jp.float32(0.e-3))

	def loss_fn(graph_state):
		"""Get loss."""
		# Calculate reconstruction loss
		rloss_sensor = graph_state.nodes["sensor"].state.cum_loss
		rloss = rloss_sensor.cos_th + rloss_sensor.sin_th + rloss_sensor.sin_th2 + rloss_sensor.cos_th2
		# rloss += 1e-1*rloss_sensor.thdot
		rloss += 1e-3*rloss_sensor.thdot + 1e-3*rloss_sensor.thdot2

		# Calculate transition loss
		fwd_state = graph_state.nodes["world"].state
		# NOTE: mode="clip" disables negative indexing.
		est_state = rjax.tree_take(graph_state.nodes["estimator"].params.world_states, graph_state.step + 1, mode="clip")
		dloss = jax.tree_util.tree_map(lambda x, y: (x - y) ** 2, est_state, fwd_state)
		dloss = jax.tree_util.tree_map(lambda e, a: a*e, dloss, alpha_dloss)
		dloss = jax.tree_util.tree_reduce(lambda acc, l: acc + jp.sum(l), dloss, 0.)
		loss = rloss + dloss
		output = Loss(loss=loss, rloss=rloss, dloss=dloss)
		return output

	# Compile env
	from estimator import EstimatorEnv
	graph = CompiledGraph(nodes, nodes["estimator"], S)
	cenv = EstimatorEnv(graph, loss_fn=loss_fn)

	# Prepare initial graph_state
	from estimator import init_graph_state
	plt.ion()
	init_gs = init_graph_state(cenv, nodes, record_network, S, Gs, Gs_monomorphism, data)

	# Define initial params
	p_tree = jax.tree_util.tree_map(lambda x: None, nodes["world"].default_params(jax.random.PRNGKey(0)))
	# p_world = p_tree.replace(mass=jp.float32(0.3), mass2=jp.float32(0.3), K=jp.float32(1.0), J=jp.float32(0.02))
	p_world = jax.tree_util.tree_map(lambda x: x * 1.2, p_tree.replace(# J=jp.float32(0.037),
	                                                                   # J2=jp.float32(0.000111608131930852),
	                                                                   mass=jp.float32(0.18),
	                                                                   mass2=jp.float32(0.0691843934004535),
	                                                                   # length=jp.float32(0.1),
	                                                                   # length2=jp.float32(0.1),
	                                                                   b=jp.float32(0.975872107940422),
	                                                                   # b2=jp.float32(1.07098956449896e-05),
	                                                                   # c=jp.float32(0.06),
	                                                                   # c2=jp.float32(0.0185223578523340),
	                                                                   K=jp.float32(1.09724557347983),
	                                                                   ))
	p_est = init_gs.nodes["estimator"].params
	initial_params = {"estimator": p_est, "world": p_world}

	# Define prior
	def make_prior_fn(guess, multiplier):
		def prior_fn(params):
			loss = jax.tree_util.tree_map(lambda x: None, params)
			if params.get("world", None) is not None:
				wloss = jax.tree_util.tree_map(lambda x, g: 1/(multiplier*(x/g)), params["world"], guess)
				loss["world"] = wloss
			return loss
		return prior_fn

	guess = nodes["world"].default_params(jax.random.PRNGKey(0))
	prior_fn = make_prior_fn(guess, multiplier=10000)

	import optax
	optimizer = optax.adam(learning_rate=5e-2)

	# Define callbacks
	from estimator.callback import LogCallback, StateFitCallback, ParamFitCallback
	targets = nodes["world"].default_params(jax.random.PRNGKey(0))
	callbacks = {"log": LogCallback(visualize=True),
	             "state_fit": StateFitCallback(visualize=True),
	             "param_fit": ParamFitCallback(targets=targets, visualize=True)}

	# Optimize
	from estimator import fit
	metrics, opt_params, opt_state, opt_gs = fit(cenv, initial_params, optimizer, init_gs,
	                                             # num_steps=10, num_batches=50, lr=1e-2 works, 1e-3 thdot.
	                                             prior_fn=prior_fn, num_batches=50, num_steps=5, num_training_steps=10_000,
	                                             num_training_steps_per_epoch=200, callbacks=callbacks)
	# Print results
	print(jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y], axis=0), opt_params["world"],
	                             nodes["world"].default_params(jax.random.PRNGKey(0))))

	pause = input("Press any key to create an animation...")
	print(pause)

	ani_duration = 15.0
	num_frames = len(callbacks["log"]._frames)
	interval = int(1e3*ani_duration / num_frames)

	animation = callbacks["log"].get_animation(interval=interval)
	animation.save("loss.mp4")
	animation = callbacks["state_fit"].get_animation(interval=interval)
	animation.save("state_fit.mp4")
	animation = callbacks["param_fit"].get_animation(interval=interval)
	animation.save("param_fit.mp4")

	pause = input("Press any key to continue...")
