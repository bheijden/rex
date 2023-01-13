import time
import numpy as onp
from stable_baselines3.common.vec_env import VecMonitor
from sbx import SAC

import rex.utils as utils
from rex.tracer import trace
from rex.proto import log_pb2
from rex.distributions import Gaussian
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, SEQUENTIAL, WARN, REAL_TIME, ASYNC, WALL_CLOCK

from envs.pendulum.env import PendulumEnv, Agent
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper


if __name__ == "__main__":
	# todo: Visualise performance on simulated pendulum
	# todo: Visualise communication
	# todo: Evaluate trained policy
	utils.set_log_level(WARN)

	# Parameters
	scheduling = FREQUENCY
	jitter = LATEST
	rate = 30.0
	max_steps = 100
	win_action = 3
	win_state = 4

	# Make real environment
	# import envs.pendulum.real as real
	# nodes_real = real.build_pendulum(rate=dict(world=100, actuator=rate, sensor=rate))
	import envs.pendulum.ode as ode
	nodes_real = ode.build_pendulum(rate=dict(world=100, actuator=rate, sensor=rate))

	world_real, actuator_real, sensor_real = nodes_real["world"], nodes_real["actuator"], nodes_real["sensor"]
	agent_real = Agent("agent", rate=rate, delay_sim=Gaussian(0.))
	nodes_real["agent"] = agent_real

	# Connect
	agent_real.connect(agent_real, name="last_action", window=win_action, blocking=True, skip=True, delay_sim=Gaussian(0.), delay=0., jitter=LATEST)
	agent_real.connect(sensor_real, name="state",  window=win_state, blocking=True, delay_sim=Gaussian(0.), jitter=jitter)
	actuator_real.connect(agent_real, name="action", window=1, blocking=True, delay_sim=Gaussian(0.), jitter=jitter)

	# Warmup nodes_real (pre-compile jitted functions)
	[n.warmup() for n in nodes_real.values()]

	# Create environment
	env_real = PendulumEnv(nodes_real, agent=agent_real, max_steps=max_steps, sync=ASYNC, clock=WALL_CLOCK, scheduling=scheduling, real_time_factor=REAL_TIME)
	env_real = GymWrapper(env_real)  # Wrap into gym wrapper

	# Initialize model
	model = SAC("MlpPolicy", env_real, verbose=1)
	action, _states = model.predict(env_real.observation_space.sample(), deterministic=True)
	# model.learn(total_timesteps=60_000, progress_bar=True)

	# Run environment
	for _ in range(1):
		tstart = time.time()
		done, obs = False, env_real.reset()
		while not done:
			# Call model.predict to include transmission overhead.
			action, _states = model.predict(obs, deterministic=True)
			# Apply constant action (to avoid wear-and-tear)
			action[0] = 2.0
			obs, reward, done, info = env_real.step(action)
			# print(f"{action=} | {obs=}")
		tend = time.time()
		print(f"ASYNC | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f}")
	env_real.stop()

	# Trace record
	record = log_pb2.EpisodeRecord()
	[record.node.append(node.record) for node in nodes_real.values()]
	trace_record = trace(record, "agent")

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
	nodes_ode = ode.build_pendulum(rate=dict(world=100, actuator=rate, sensor=rate))

	world_ode, actuator_ode, sensor_ode = nodes_ode["world"], nodes_ode["actuator"], nodes_ode["sensor"]
	agent_ode = Agent("agent", rate=rate, delay_sim=Gaussian(0.))
	nodes_ode["agent"] = agent_ode

	# Connect
	agent_ode.connect(agent_ode, name="last_action", window=win_action, blocking=True, skip=True, delay_sim=Gaussian(0.), delay=0., jitter=LATEST)
	agent_ode.connect(sensor_ode, name="state",  window=win_state, blocking=True, delay_sim=Gaussian(0.), jitter=LATEST)
	actuator_ode.connect(agent_ode, name="action", window=1, blocking=True, delay_sim=Gaussian(0.), jitter=LATEST)

	# Create trace environment
	env_ode = PendulumEnv(nodes_ode, agent=agent_ode, max_steps=max_steps, trace=trace_record, graph=SEQUENTIAL)
	env_ode = AutoResetWrapper(env_ode)  # Wrap into auto reset wrapper
	env_ode = VecGymWrapper(env_ode, num_envs=10)  # Wrap into vectorized environment
	env_ode = VecMonitor(env_ode)  # Wrap into vectorized monitor

	# Jit
	env_ode.jit()

	# Initialize model
	model = SAC("MlpPolicy", env_ode, verbose=1)
	model.learn(total_timesteps=200_000, progress_bar=True)

	model.save("pendulum")

	# Run environment
	for _ in range(10):
		tstart = time.time()
		done, obs = False, env_ode.reset()
		cum_reward = 0.
		while not done:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, dones, info = env_ode.step(action)
			cum_reward += reward
			done = dones.any()
		tend = time.time()
		print(f"COMPILED | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f} | cum_reward={onp.mean(cum_reward)}")
	env_ode.stop()

	# todo: evaluate model on real environment




