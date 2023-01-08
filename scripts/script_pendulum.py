import time
import jax
import jumpy as jp
import jax.numpy as jnp
import numpy as onp

import rex.jumpy as rjp
from rex.proto import log_pb2
from rex.jumpy import use
from rex.tracer import trace
from rex.utils import timer
from rex.distributions import Gaussian
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, REAL_TIME, FAST_AS_POSSIBLE, SIMULATED, \
	WALL_CLOCK, SYNC, ASYNC, FREQUENCY, PHASE, SEQUENTIAL

from envs.pendulum.world_ode import World
from envs.pendulum.env import PendulumEnv, Agent


def evaluate(env, name: str = "env", backend: str = "numpy", use_jit: bool = False, seed: int = 0, vmap: int = 1):
	# Record
	gs_lst = []
	obs_lst = []
	ss_lst = []

	# Get action space
	action_space = env.action_space()

	use_jit = use_jit and backend == "jax"
	with use(backend=backend):
		rng = rjp.random_prngkey(jp.int32(seed))

		# vmap env
		# if vmap > 1:
		env_reset = rjp.vmap(env.reset)
		env_step = rjp.vmap(env.step)
		rng = jp.random_split(rng, num=vmap)
		action = rjp.vmap(action_space.sample)(rng)
		# else:
		# 	env_reset = env.reset
		# 	env_step = env.step
		# 	action = action_space.sample(rng)

		# Get reset and step function
		env_reset = jax.jit(env_reset) if use_jit else env_reset
		env_step = jax.jit(env_step) if use_jit else env_step

		# Reset environment
		with timer(f"{name} | jit reset", log_level=WARN):
			graph_state, obs = env_reset(rng)
			gs_lst.append(graph_state)
			obs_lst.append(obs)
			ss_lst.append(graph_state.nodes["agent"])

		# Initial step (warmup)
		with timer(f"{name} | jit step", log_level=WARN):
			graph_state, obs, reward, done, info = env_step(graph_state, action)
			obs_lst.append(obs)
			gs_lst.append(graph_state)
			ss_lst.append(graph_state.nodes["agent"])

		# Run environment
		tstart = time.time()
		eps_steps = 1
		while True:
			graph_state, obs, reward, done, info = env_step(graph_state, action)
			obs_lst.append(obs)
			gs_lst.append(graph_state)
			ss_lst.append(graph_state.nodes["agent"])
			eps_steps += 1
			# done = done[0] if vmap > 1 else done
			if done[0]:
				# Time env stopping
				tend = time.time()
				env.stop()
				tstop = time.time()

				# Print timings
				print(
					f"{name=} | agent_steps={eps_steps} | chunk_index={graph_state.step} | t={(tstop - tstart): 2.4f} sec | t_s={(tstop - tend): 2.4f} sec | fps={eps_steps / (tend - tstart): 2.4f} | fps={eps_steps / (tstop - tstart): 2.4f} (incl. stop)")
				break
	return gs_lst, obs_lst, ss_lst


if __name__ == "__main__":
	# Define nodes
	world = World("world", rate=20, delay_sim=Gaussian(0.))
	agent = Agent("agent", rate=20, delay_sim=Gaussian(0.))
	nodes = {n.name: n for n in [world, agent]}

	# Connect
	agent.connect(world, name="state",  window=1, blocking=True, delay_sim=Gaussian(0.), jitter=LATEST, skip=True)
	world.connect(agent, name="action", window=1, blocking=True, delay_sim=Gaussian(0.), jitter=LATEST)

	# Warmup nodes (pre-compile jitted functions)
	[n.warmup() for n in nodes.values()]

	# Create trace environment
	trace_steps = 100
	trace_env = PendulumEnv(nodes, agent=agent, max_steps=trace_steps, sync=SYNC, clock=SIMULATED, scheduling=PHASE, real_time_factor=FAST_AS_POSSIBLE)

	# Evaluate async env
	gs_trace, obs_trace, ss_trace = evaluate(trace_env, name="trace_env", backend="numpy", use_jit=False, seed=0)

	# Gather record
	record = log_pb2.EpisodeRecord()
	[record.node.append(node.record) for node in nodes.values()]
	r = {n.info.name: n for n in record.node}

	# Trace
	trace_record = trace(record, "agent")

	# Create compiled environment
	max_steps = trace_steps
	env = PendulumEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_record, graph=SEQUENTIAL)

	# Evaluate async env
	gs, obs, ss = evaluate(env, name="env", backend="numpy", use_jit=False, seed=0, vmap=20)
	# gs, obs, ss = evaluate(env, name="env", backend="jax", use_jit=True, seed=0, vmap=20)

	# Compare
	def compare(_trace, _opt):
		if not isinstance(_trace, (onp.ndarray, jnp.ndarray)):
			_equal_all = onp.allclose(_trace, _opt)
			_op_all = "==" if _equal_all else "!="
			msg = f"{_trace} {_op_all} {_opt}"
			assert _equal_all, msg
		else:
			for i in range(len(_trace)):
				_equal_all = onp.allclose(_trace[i], _opt[i])
				_op_all = "==" if _equal_all else "!="
				msg = f"{_trace} {_op_all} {_opt}"
				assert _equal_all, msg

	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use("TkAgg")
	arr_obs_trace = jp.array(obs_trace)
	arr_obs = jp.array(obs)
	fig, axes = plt.subplots(3, 1, sharex=True)
	axes = axes.flatten()
	for i, ax in enumerate(axes):
		ax.plot(arr_obs_trace[:, :, i], color="blue", label="async")
		ax.plot(arr_obs[:, :, i], label="jit")
		# if arr_obs.ndim < 3:
		# 	ax.plot(arr_obs[:, i], color="red", label="jit")
		# else:
	plt.show()

	# Compare observations and agent step states
	# jp.tree_map(compare, obs_trace, obs)
	# jp.tree_map(compare, ss_trace, ss)

	# Compare
	compare_obs = jp.tree_map(lambda *args: args, obs_trace, obs)
	compare_ss = jp.tree_map(lambda *args: args, ss_trace, ss)

	print("finished")