import os
import time
import tempfile
import datetime
import tqdm
import yaml

import jumpy.numpy as jp
import jumpy.random
import numpy as onp
import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]

# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("./cache")
# import logging
# logging.getLogger("jax").setLevel(logging.INFO)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

import rex
import rex.jumpy as rjp
from rex.utils import timer, make_put_output_on_device
import rex.utils as utils
from rex.base import StepState
from rex.supergraph import create_graph
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, INFO, DEBUG
import experiments as exp
import envs.vx300s as vx300s

RECORD_SETTINGS = {"planner": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "world": dict(node=False, outputs=False, rngs=True, states=False, params=False, step_states=False),
                   "armactuator": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "controller": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "armsensor": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "boxsensor": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "supervisor": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   "viewer": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False),
                   }
PATH_VX300S = os.path.dirname(vx300s.__file__)
CONFIG = {"mjx": {"xml_path": f"{PATH_VX300S}/assets/vx300s_mjx.xml"},
          "brax": {"xml_path": f"{PATH_VX300S}/assets/vx300s_brax.xml"},
          "real": {"cam_trans": [0.589, 0.598, 0.355], "cam_rot": [0.252,  0.855, -0.436, -0.125], "cam_idx": 2,
                   "z_fixed": 0.051,
                   "cam_intrinsics": f"{PATH_VX300S}/assets/logitech_c170.yaml"},
          "planner": {"type": "brax",
	                  "brax_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_brax.xml",
                      "rex_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_brax.xml",
                      "mjx_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_mjx.xml",
                      # "rex_graph_path": "/home/r2ci/rex/logs/real_0.35umax_vx300s_2023-11-20-1759/record_pre.pb",
                      "rex_graph_path": "/home/r2ci/rex/logs/real_3winplanner_2eps_vx300s_2023-11-27-1708/record_pre.pb",
                      # "rex_graph_path": "/home/r2ci/rex/logs/vx300s_3winplanner_vx300s_2023-11-23-1722/record_pre.pb",
                      "supergraph_mode": "MCS",
                      "horizon": 2, "u_max": 0.35*3.14, "dt": 0.15, "dt_substeps": 0.015, "num_samples": 75, "max_iter": 3},
          "viewer": {"xml_path": f"{PATH_VX300S}/assets/vx300s_cem_mjx.xml"}}

if __name__ == "__main__":
	# todo: set advance=true && LATEST for armsensor and boxsensor --> why does planner have num_msgs=2 for t=0?
	# jax.config.update("jax_debug_nans", True)
	# Environment
	ENV = "vx300s"  # "disc_pendulum"
	DIST_FILE = "real_3winplanner_vx300s_2023-11-27-1647.pkl"  # "real_faster_nomovement_vx300s_2023-11-20-1548.pkl"  # f"vx300s_vx300s_2023-10-19-1629.pkl"
	JITTER = BUFFER
	SCHEDULING = PHASE

	ENV_FN = vx300s.brax.build_vx300s  # vx300s.brax.build_vx300s
	CLOCK = SIMULATED
	RTF = FAST_AS_POSSIBLE
	RATES = dict(world=80, supervisor=8, planner=5.0, controller=20, armactuator=20, armsensor=80, boxsensor=10, viewer=20)
	# RATES = dict(world=25, planner=25, controller=25, armactuator=25, armsensor=25, boxsensor=25)
	MAX_STEPS = int(10 * RATES["supervisor"])
	WIN_PLANNER = 3
	USE_DELAYS = True
	DELAY_FN = lambda d: d.quantile(0.99)*int(USE_DELAYS)  # todo: this is slow (takes 3 seconds).

	# Logging
	NAME = f"real_3winplanner_2eps_{ENV}"
	LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{NAME}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
	MUST_LOG = False
	MUST_DIST = False
	MUST_PLOT = True
	SHOW_PLOTS = True

	# Training
	SEED = 0
	RNG = jumpy.random.PRNGKey(0)
	NUM_EVAL = 20

	# Load distributions
	delays_sim = exp.load_distributions(DIST_FILE, module=vx300s.dists) if DIST_FILE is not None else vx300s.get_default_distributions()
	# delays_sim = vx300s.get_nodelay_distributions()
	fig_sim_step, fig_sim_inputs = vx300s.plot_dists(delays_sim) if MUST_DIST and MUST_PLOT else (None, None)

	# Prepare environment
	env = vx300s.make_env(delays_sim, DELAY_FN, RATES, CONFIG, win_planner=WIN_PLANNER, scheduling=SCHEDULING, jitter=JITTER,
	                      env_fn=ENV_FN, name=ENV, clock=CLOCK, real_time_factor=RTF,
	                      max_steps=MAX_STEPS, use_delays=USE_DELAYS)

	# Jit functions
	env.graph.init = jax.jit(env.graph.init, static_argnames=["order"], device=cpu_device)
	env._get_cost = jax.jit(env._get_cost, device=cpu_device)
	env.graph.nodes_and_root["world"].step = jax.jit(env.graph.nodes_and_root["world"].step, device=cpu_device)
	env.graph.nodes_and_root["planner"].step = make_put_output_on_device(jax.jit(env.graph.nodes_and_root["planner"].step, device=gpu_device), cpu_device)
	# env.graph.nodes_and_root["planner"].step = jax.jit(env.graph.nodes_and_root["planner"].step, device=gpu_device)
	env.graph.nodes_and_root["controller"].step = jax.jit(env.graph.nodes_and_root["controller"].step, device=cpu_device)
	if "viewer" in env.graph.nodes:
		env.graph.nodes_and_root["viewer"].step = jax.jit(env.graph.nodes_and_root["viewer"].step, device=cpu_device)

	# Warmup
	with timer(f"warmup[graph_state]", log_level=100):
		graph_state = env.graph.init(RNG, order=("supervisor", "world"))
	with timer(f"eval[graph_state]", log_level=100):
		_ = env.graph.init(RNG, order=("supervisor", "world"))
	# with rjp.use("numpy"):
	# 	np_ss_planner = jax.tree_util.tree_map(lambda x: onp.array(x), graph_state.nodes["planner"])
	# 	with timer(f"warmup[planner]", log_level=100):
	# 		_, _ = env.graph.nodes["planner"].step(np_ss_planner)
	with timer(f"warmup[cost]", log_level=100):
		cost, info = env._get_cost(graph_state)
	with timer(f"eval[cost]", log_level=100):
		_, _ = env._get_cost(graph_state)
	for name, node in env.graph.nodes.items():
		if name not in ["world", "planner", "controller", "viewer"]:
			continue
		with timer(f"warmup[{name}]", log_level=100):
			ss, o = node.step(graph_state.nodes[name])
			if name == "planner":
				print(o.jpos.device())
		with timer(f"eval[{name}]", log_level=100):
			_, _ = node.step(graph_state.nodes[name])
			jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
	with timer(f"warmup[dist]", log_level=100):
		for name, node in env.graph.nodes_and_root.items():
			node.warmup()

	policy = lambda step_state: 1
	# _, _ = vx300s.eval_env(env, policy, 1, progress_bar=True, record_settings=RECORD_SETTINGS, seed=0)
	# with jax.log_compiles(True):
	record_pre, rwds = vx300s.eval_env(env, policy, NUM_EVAL, progress_bar=True, record_settings=RECORD_SETTINGS, seed=0)

	# Save html
	# todo: brax timestamps are not correct.
	# todo: modify timestep.
	if "viewer" in env.graph.nodes:
		env.graph.nodes["viewer"].close()
	# rollout = [ss.state.pipeline_state for ss in env.world._record_step_states]
	# env.world.view_rollout(rollout=rollout, path="./vx300s_render.html")

	# Plot computation graph
	G = create_graph(record_pre.episode[-1])
	fig_gr, _ = vx300s.show_graph(record_pre.episode[-1]) if MUST_PLOT else (None, None)
	fig_cg, _ = vx300s.show_computation_graph(G, root="supervisor", xmax=2.0) if MUST_PLOT else (None, None)
	fig_com, _ = vx300s.show_communication(record_pre.episode[-1]) if MUST_PLOT else (None, None)

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
		fig_gr.savefig(LOG_DIR + "/robotic_system.png") if fig_gr is not None else None
		fig_cg.savefig(LOG_DIR + "/computation_graph.png") if fig_gr is not None else None
		fig_com.savefig(LOG_DIR + "/communication.png") if fig_gr is not None else None
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
