from typing import Union, Dict
import dill as pickle
import os
import time
import jax
import numpy as onp
import jumpy.numpy as jp
from stable_baselines3.common.vec_env import VecMonitor

from sbx import SAC
# from stable_baselines3.sac import SAC

import rex.utils as utils
from rex.tracer import trace
from rex.proto import log_pb2
from rex.distributions import Gaussian, Distribution
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, SEQUENTIAL, WARN, REAL_TIME, ASYNC, WALL_CLOCK

import envs.pendulum.dists
from envs.pendulum.env import PendulumEnv, Agent
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper


def load_distributions(file: str, module=envs.pendulum.dists) -> Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]]:
	# Append pkl extension
	if not file.endswith(".pkl"):
		file = file + ".pkl"

	# If not an absolute path, load from within dists
	if not file.startswith("/"):
		module_path = os.path.dirname(os.path.abspath(module.__file__))
		with open(f"{module_path}/{file}", "rb") as f:
			dists = pickle.load(f)
	else:
		with open(f"{file}", "rb") as f:
			dists = pickle.load(f)
	return dists


if __name__ == "__main__":
	# todo: Log experiments (trace, hyper parameters, model, tensorboard, w & b, data?)
	# todo: Visualise performance on simulated pendulum
	# todo: Visualise communication
	# todo: load dists outside of build_pendulum.
	utils.set_log_level(WARN)

	# Parameters
	log_dir = "/home/r2ci/rex/logs"
	name_proto = "pretrained-sbx-sac"
	preload_model = "sbx_sac_pendulum"
	new_model_name = "sbx_sac_pendulum"
	name_env = "disc-pendulum-real"
	dist_file = f"21eps_pretrained_sbx_sac_gmms_2comps.pkl"
	num_init_eps = 21
	scheduling = PHASE
	jitter = BUFFER
	clock = WALL_CLOCK  # WALL_CLOCK, SIMULATED
	real_time_factor = REAL_TIME  # REAL_TIME, FAST_AS_POSSIBLE
	max_steps = 100
	win_action = 1
	win_state = 2

	# Load distributions
	delays_sim = load_distributions(dist_file)
	delays_sim["step"]["world"] = Gaussian(0.)
	delays_sim["inputs"]["world"]["actuator"] = Gaussian(0.)
	delays_sim["inputs"]["sensor"]["world"] = Gaussian(0.)
	delays = jax.tree_map(lambda d: d.high, delays_sim)
	rates = dict(world=20, agent=20, actuator=20, sensor=20, render=20)

	# Make real environment
	import envs.pendulum.real as real
	nodes_real = real.build_pendulum(rates, delays_sim, delays, scheduling=scheduling, advance=False)
	# import envs.pendulum.ode as ode
	# nodes_real = ode.build_pendulum(rates, delays_sim, delays, scheduling=scheduling, advance=False)

	# todo: add dist to agent, and connections.
	# todo: advance actuator, agent?
	world_real, actuator_real, sensor_real = nodes_real["world"], nodes_real["actuator"], nodes_real["sensor"]
	agent_real = Agent("agent", rate=rates["agent"], delay_sim=delays_sim["step"]["agent"], delay=delays["step"]["agent"])
	nodes_real["agent"] = agent_real

	# Turn on evaluation
	assert hasattr(world_real, "eval_env"), "World node must have an eval_env attribute"
	world_real.eval_env = True

	# Connect
	agent_real.connect(agent_real, name="last_action", window=win_action, blocking=True, jitter=LATEST, skip=True)
	agent_real.connect(sensor_real, name="state",  window=win_state, blocking=True, jitter=jitter, delay_sim=delays_sim["inputs"]["agent"]["state"], delay=delays["inputs"]["agent"]["state"])
	actuator_real.connect(agent_real, name="action", window=1, blocking=True, jitter=jitter, delay_sim=delays_sim["inputs"]["actuator"]["action"], delay=delays["inputs"]["actuator"]["action"])

	# Warmup nodes_real (pre-compile jitted functions)
	[n.warmup() for n in nodes_real.values()]

	# Create environment
	env_real = PendulumEnv(nodes_real, agent=agent_real, max_steps=max_steps, clock=clock, real_time_factor=real_time_factor)
	env_real = GymWrapper(env_real)  # Wrap into gym wrapper

	# Initialize model
	try:
		try:
			model_real = SAC.load(f"{log_dir}/{preload_model}", env=env_real)
		except AttributeError as e:
			print("Could not load model. Chech whether the model was trained with sb or sbx.")
			raise e
		new_model = False
	except FileNotFoundError as e:
		print(e)
		new_model = True
		model_real = SAC("MlpPolicy", env_real)
	action, _states = model_real.predict(env_real.observation_space.sample(), deterministic=True)

	# Run environment
	exp_record = log_pb2.ExperimentRecord()
	for eps in range(num_init_eps):
		cum_reward = 0.
		done, obs = False, env_real.reset()
		tstart = time.time()
		while not done:
			# Call model.predict to include transmission overhead.
			action, _states = model_real.predict(obs, deterministic=True)
			if new_model:  # Apply constant action if new (untrained) model to avoid wear-and-tear
				action[0] = 2.0
			obs, reward, done, info = env_real.step(action)
			cum_reward += reward
			# print(f"{action=} | {obs=}")
		tend = time.time()
		print(f"ASYNC | eps={eps} | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f} | cum_reward={onp.mean(cum_reward)}")
		env_real.stop()  # NOTE: This is required to make the number of ticks equal to max_steps for compiled graph

		# Save record
		_kwargs = dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True)
		# eps_record = log_pb2.EpisodeRecord()
		# [eps_record.node.append(node.record(**_kwargs)) if name != "render" else eps_record.node.append(node.record(node=True)) for name, node in nodes_real.items()]
		node_records = [node.record(**_kwargs) if name != "render" else node.record(node=True) for name, node in nodes_real.items()]
		exp_record.episode.append(log_pb2.EpisodeRecord(node=node_records))

	exit()

	# Save experiment record
	with open(f"/home/r2ci/rex/logs/{name_env}/{num_init_eps}eps-{name_proto}.pb", "wb") as f:
		f.write(exp_record.SerializeToString())
	exit()

	# Trace record
	trace_record = trace(exp_record.episode[0], "agent")

	# Visualize trace
	must_plot = True
	if must_plot:
		import matplotlib
		import matplotlib.pyplot as plt
		import seaborn as sns
		from rex.plot import plot_computation_graph
		import rex.open_colors as oc
		matplotlib.use("TkAgg")
		sns.set()

		# Create new plot
		fig, ax = plt.subplots()
		fig.set_size_inches(12, 5)
		ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
		order = ["world", "sensor", "agent", "actuator"]
		cscheme = {"world": "gray", "sensor": "grape",  "agent": "teal", "actuator": "indigo"}
		plot_computation_graph(ax, trace_record, order=order, cscheme=cscheme, xmax=1.0, node_size=200, draw_excluded=True,
		                       draw_stateless=False, draw_nodelabels=True, node_labeltype="tick", connectionstyle="arc3,rad=0.1")
		# Plot legend
		handles, labels = ax.get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		by_label = dict(sorted(by_label.items()))
		ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
		          bbox_to_anchor=(1.0, 0.50))
		plt.show()

	# Make ode environment
	import envs.pendulum.ode as ode
	nodes_ode = ode.build_pendulum(rate=dict(world=world_rate, actuator=rate, sensor=rate, render=rate))

	world_ode, actuator_ode, sensor_ode = nodes_ode["world"], nodes_ode["actuator"], nodes_ode["sensor"]
	agent_ode = Agent("agent", rate=rate, delay_sim=Gaussian(process["agent"]), delay=process["agent"])
	nodes_ode["agent"] = agent_ode

	# Connect
	agent_ode.connect(agent_ode, name="last_action", window=win_action, blocking=True, skip=True, delay_sim=Gaussian(0.), delay=0., jitter=LATEST)
	agent_ode.connect(sensor_ode, name="state",  window=win_state, blocking=True, delay_sim=Gaussian(trans["sensor"]), delay=trans["sensor"], jitter=LATEST)
	actuator_ode.connect(agent_ode, name="action", window=1, blocking=True, delay_sim=Gaussian(trans["actuator"]), delay=trans["actuator"], jitter=LATEST)

	# Create trace environment
	env_ode = PendulumEnv(nodes_ode, agent=agent_ode, max_steps=max_steps, trace=trace_record, graph=SEQUENTIAL)
	# env_ode = GymWrapper(env_ode)  # Wrap into gym wrapper
	env_ode = AutoResetWrapper(env_ode)  # Wrap into auto reset wrapper
	env_ode = VecGymWrapper(env_ode, num_envs=10)  # Wrap into vectorized environment
	env_ode = VecMonitor(env_ode)  # Wrap into vectorized monitor

	# Jit
	env_ode.jit()

	# Initialize model
	model_ode = SAC("MlpPolicy", env_ode, verbose=1)
	model_ode.learn(total_timesteps=80_000, progress_bar=True)
	model_ode.save(f"{log_dir}/{new_model_name}")

	# Reload real model
	model_real = SAC.load(f"{log_dir}/{new_model_name}", env=env_real)
	model_real.predict(env_real.observation_space.sample(), deterministic=True)  # warm up

	# Run environment (simulation environment)
	ode_obs_eps = []
	for _ in range(10):
		tstart = time.time()
		done, obs = False, env_ode.reset()
		cum_reward = 0.
		obs_lst = [obs]
		ode_obs_eps.append(obs_lst)
		while not done:
			action, _states = model_ode.predict(obs, deterministic=True)
			obs, reward, dones, info = env_ode.step(action)
			obs_lst.append(obs)
			cum_reward += reward
			done = dones.any() if hasattr(dones, "__len__") else dones
		tend = time.time()
		print(f"COMPILED | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f} | cum_reward={onp.mean(cum_reward)}")
	env_ode.stop()

	# Check differences
	obs_eps = onp.concatenate([onp.array(e).reshape(-1, 7) for e in ode_obs_eps])
	ode_actions = model_ode.predict(obs_eps, deterministic=True)[0]
	real_actions = model_real.predict(obs_eps, deterministic=True)[0]
	env_ode_diff = real_actions - ode_actions

	print(f"env_ode | max diff: {onp.max(onp.abs(env_ode_diff))}")

	# Run environment
	real_obs_eps = []
	for _ in range(10):
		tstart = time.time()
		done, obs = False, env_real.reset()
		cum_reward = 0.
		obs_lst = [obs]
		real_obs_eps.append(obs_lst)
		while not done:
			action, _states = model_real.predict(obs, deterministic=True)
			obs, reward, done, info = env_real.step(action)
			obs_lst.append(obs)
			cum_reward += reward
		tend = time.time()
		print(f"ASYNC | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f} | cum_reward={onp.mean(cum_reward)}")
	env_real.stop()

	# Check differences
	obs_eps = onp.concatenate([onp.array(e).reshape(-1, 7) for e in real_obs_eps])
	ode_actions = model_ode.predict(obs_eps, deterministic=True)[0]
	real_actions = model_real.predict(obs_eps, deterministic=True)[0]
	env_real_diff = real_actions - ode_actions

	print(f"env_real | max diff: {onp.max(onp.abs(env_real_diff))}")
