import time
import numpy as onp
from stable_baselines3.common.vec_env import VecMonitor
from sbx import SAC

import rex.utils as utils
from rex.tracer import trace
from rex.proto import log_pb2
from rex.distributions import Gaussian
from rex.constants import LATEST, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, SEQUENTIAL, WARN

from envs.pendulum.env import PendulumEnv, Agent
from envs.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper

if __name__ == "__main__":
	utils.set_log_level(WARN)

	# Define nodes
	import envs.pendulum.ode as ode
	nodes = ode.build_pendulum(rate=dict(world=20.0))
	# import envs.pendulum.real as real
	# nodes = real.build_pendulum(rate=dict(world=20.0))

	world, actuator, sensor = nodes["world"], nodes["actuator"], nodes["sensor"]
	agent = Agent("agent", rate=20., delay_sim=Gaussian(0.))
	nodes["agent"] = agent

	# Connect
	agent.connect(sensor, name="state",  window=1, blocking=True, delay_sim=Gaussian(0.), jitter=LATEST)
	actuator.connect(agent, name="action", window=1, blocking=True, delay_sim=Gaussian(0.), jitter=LATEST)

	# Warmup nodes (pre-compile jitted functions)
	[n.warmup() for n in nodes.values()]

	# Create environment
	max_steps = 100
	env = PendulumEnv(nodes, agent=agent, max_steps=max_steps, sync=SYNC, clock=SIMULATED, scheduling=PHASE, real_time_factor=FAST_AS_POSSIBLE)
	env = GymWrapper(env)  # Wrap into gym wrapper

	# Get spaces
	action_space = env.action_space
	observation_space = env.observation_space

	# Run environment
	for _ in range(1):
		tstart = time.time()
		done, obs = False, env.reset()
		while not done:
			action = action_space.sample()
			obs, reward, done, info = env.step(action)
		tend = time.time()
		print(f"ASYNC | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f}")
	env.close()

	# Trace record
	record = log_pb2.EpisodeRecord()
	[record.node.append(node.record) for node in nodes.values()]
	trace_record = trace(record, "agent")

	# Create trace environment
	env = PendulumEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_record, graph=SEQUENTIAL)
	env = AutoResetWrapper(env)  # Wrap into auto reset wrapper
	# env = GymWrapper(env)  # Wrap into gym wrapper
	env = VecGymWrapper(env, num_envs=10)  # Wrap into vectorized environment
	env = VecMonitor(env)  # Wrap into vectorized monitor

	# Jit
	env.jit()
	# env.reset()
	# action = jp.array([env.action_space.sample() for _ in range(env.num_envs)]) if isinstance(env, VecGymWrapper) else env.action_space.sample()
	# env.step(action)

	# Visualize trace
	must_plot = False
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
		plot_computation_graph(ax, env.unwrapped.graph.trace, order=order, cscheme=cscheme, xmax=0.6, node_size=200, draw_excluded=True,
		                       draw_stateless=False, draw_nodelabels=True, node_labeltype="tick", connectionstyle="arc3,rad=0.1")
		# Plot legend
		handles, labels = ax.get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		by_label = dict(sorted(by_label.items()))
		ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
		          bbox_to_anchor=(1.0, 0.50))
		plt.show()

	# Initialize model
	model = SAC("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=60_000, progress_bar=True)

	# Run environment
	for _ in range(10):
		tstart = time.time()
		done, obs = False, env.reset()
		cum_reward = 0.
		while not done:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, dones, info = env.step(action)
			cum_reward += reward
			done = dones.any()
		tend = time.time()
		print(f"COMPILED | steps={max_steps} | t_s={(tstart - tend): 2.4f} sec | fps={max_steps / (tend - tstart): 2.4f} | cum_reward={onp.mean(cum_reward)}")
	env.close()

	exit()

