import os
import time
import tempfile
import datetime
import tqdm
import yaml

import jumpy.numpy as jp
import jumpy.random
import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

import rex
import rex.utils as utils
from rex.supergraph import create_graph
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, INFO, DEBUG
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import experiments as exp
import envs.vx300s as vx300s

RECORD_SETTINGS = {"planner": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "world": dict(node=False, outputs=False, rngs=False, states=True, params=False, step_states=False),
                   "armactuator": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "controller": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "armsensor": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "boxsensor": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   # "render": dict(node=True, outputs=False, rngs=True, states=True, params=True, step_states=True)
                   }

if __name__ == "__main__":
	# todo: set advance=true && LATEST for armsensor and boxsensor --> why does planner have num_msgs=2 for t=0?

	# Environment
	ENV = "vx300s"  # "disc_pendulum"
	DIST_FILE = f"vx300s_vx300s_2023-10-19-1629.pkl"
	JITTER = BUFFER
	SCHEDULING = PHASE

	BLOCKING = True
	ADVANCE = False
	ENV_FN = vx300s.brax.build_vx300s  #  dpend.ode.build_double_pendulum  # pend.ode.build_pendulum
	ENV_CLS = vx300s.env.Vx300sEnv  # dpend.env.DoublePendulumEnv  # pend.env.PendulumEnv
	CLOCK = SIMULATED
	RTF = FAST_AS_POSSIBLE
	RATES = dict(world=20, planner=0.5, controller=5, armactuator=5, armsensor=5, boxsensor=5)
	# RATES = dict(world=25, planner=25, controller=25, armactuator=25, armsensor=25, boxsensor=25)
	MAX_STEPS = int(10 * RATES["planner"])
	USE_DELAYS = True
	DELAY_FN = lambda d: d.quantile(0.99)*int(USE_DELAYS)  # todo: this is slow (takes 3 seconds).

	# Logging
	NAME = f"vx300s_{ENV}"
	LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
	MUST_LOG = False
	MUST_DIST = True
	MUST_PLOT = True
	SHOW_PLOTS = True

	# Training
	CONTINUE = True
	SEED = 0
	NUM_ENVS = 10
	SAVE_FREQ = 40_000
	NSTEPS = 200_000
	NUM_EVAL_PRE = 1
	NUM_EVAL_POST = 20

	# Load distributions
	delays_sim = exp.load_distributions(DIST_FILE, module=vx300s.dists)
	fig_sim_step, fig_sim_inputs = vx300s.plot_dists(delays_sim) if MUST_DIST and MUST_PLOT else (None, None)
	# delays_sim = {}

	# Prepare environment
	with jax.default_device(cpu_device):
		env = vx300s.make_env(delays_sim, DELAY_FN, RATES, win_planner=3, scheduling=SCHEDULING, jitter=JITTER,
		                      env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, clock=CLOCK, real_time_factor=RTF,
		                      max_steps=MAX_STEPS, use_delays=USE_DELAYS)

		# Set up logging
		# utils.set_log_level(INFO)
		# utils.set_log_level(INFO, env.world, "red")
		# utils.set_log_level(INFO, env.planner, "blue")
		# utils.set_log_level(INFO, env.nodes["armsensor"], "green")
		# utils.set_log_level(INFO, env.nodes["boxsensor"], "green")
		# utils.set_log_level(DEBUG, env.nodes["controller"], "cyan")
		# utils.set_log_level(DEBUG, env.nodes["armactuator"], "cyan")

		# Jit functions
		# todo: MPC loop: jit, warmup & specify gpu.
		env._get_graph_state = jax.jit(env._get_graph_state) # todo: warmup & specify cpu
		env.world.step = jax.jit(env.world.step)  # todo: warmup & specify cpu
		# env.nodes["controller"].step = jax.jit(env.nodes["controller"].step)  # todo: warmup & specify cpu

		# Warmup reset
		graph_state, ss_planner, info = env.reset(jumpy.random.PRNGKey(0))

		# Step
		default_plan = env.planner.default_output(jumpy.random.PRNGKey(0), graph_state)


		def update_plan(last_plan, next_ts, i):
			jpos = vx300s.env.get_next_jpos(last_plan, next_ts)
			timestamps = default_plan.timestamps + next_ts
			jvel = 0.1*jp.ones_like(default_plan.jvel)
			return default_plan.replace(jpos=jpos, timestamps=timestamps, jvel=jvel)


		# Jit and warmup
		jit_update_plan = jax.jit(update_plan)
		_ = jit_update_plan(ss_planner.state.last_plan, 0., 0)

		def policy(step_state):
			next_ts = ss_planner.ts + (1 / env.planner.rate)
			next_plan = jit_update_plan(ss_planner.state.last_plan, next_ts, 0.1)
			# time.sleep(0.95*1/env.planner.rate)
			return next_plan

		_, _ = vx300s.eval_env(env, policy, 1, progress_bar=True, record_settings=RECORD_SETTINGS, seed=0)
		record_pre, rwds = vx300s.eval_env(env, policy, 4, progress_bar=True, record_settings=RECORD_SETTINGS, seed=0)

		# Save html
		# todo: brax timestamps are not correct.
		from brax.io import html
		html.save("./rex_render.html", env.world.sys, [ss.state.pipeline_state for ss in env.world._record_step_states])

		# Plot computation graph
		G = create_graph(record_pre.episode[-1])
		fig_gr, _ = vx300s.show_graph(record_pre.episode[-1]) if MUST_PLOT else (None, None)
		fig_cg, _ = vx300s.show_computation_graph(G, root="planner", xmax=6.5)

		# Fit distributions
		data, info, est, dist = vx300s.make_delay_distributions(record_pre, num_steps=500, num_components=4, step_size=0.05,
		                                                        seed=SEED) if MUST_DIST else (None, None, None, None)
		fig_step, fig_inputs = vx300s.plot_dists(dist, data, info, est) if MUST_DIST and MUST_PLOT else (None, None)

		# Only show
		plt.show() if SHOW_PLOTS else None

	# Log
	if MUST_LOG:
		os.mkdir(LOG_DIR)
		# Identify all capitalized variables & save them to file
		capitalized_vars = {k: v for k, v in globals().items() if k.isupper()}
		with open(f"{LOG_DIR}/params.yaml", 'w') as file:
			yaml.dump(capitalized_vars, file)
		# Save envs
		# env.unwrapped.save(f"{LOG_DIR}/{env.unwrapped.name}.pkl")
		# cenv.unwrapped.save(f"{LOG_DIR}/{cenv.unwrapped.name}.pkl")
		# Save pre-train record to file
		with open(LOG_DIR + "/record_pre.pb", "wb") as f:
			f.write(record_pre.SerializeToString())
		# Save plots
		# fig_gr.savefig(LOG_DIR + "/robotic_system.png") if fig_gr is not None else None
		fig_cg.savefig(LOG_DIR + "/computation_graph.png") if fig_gr is not None else None
		# fig_com.savefig(LOG_DIR + "/communication.png") if fig_gr is not None else None
		# fig_grp.savefig(LOG_DIR + "/grouped_agent_sensor.png") if fig_gr is not None else None
		fig_sim_step.savefig(LOG_DIR + "/delay_sim_step.png") if fig_sim_step is not None else None
		fig_sim_inputs.savefig(LOG_DIR + "/delay_sim_inputs.png") if fig_sim_inputs is not None else None
		fig_step.savefig(LOG_DIR + "/delay_step.png") if fig_step is not None else None
		fig_inputs.savefig(LOG_DIR + "/delay_inputs.png") if fig_inputs is not None else None
		# Save to file
		import dill as pickle

		if MUST_DIST:
			with open(LOG_DIR + "/distributions.pkl", "wb") as f:
				pickle.dump(dist, f)
