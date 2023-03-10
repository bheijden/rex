import time
import jax
import jumpy
import jumpy.numpy as jp
import jax.numpy as jnp
import numpy as onp

import rex.jumpy as rjp
from rex.proto import log_pb2
from rex.jumpy import use
from rex.tracer import trace
from rex.utils import timer
from rex.distributions import Gaussian
from rex.constants import LATEST, WARN, FAST_AS_POSSIBLE, SIMULATED, \
	SYNC, PHASE, SEQUENTIAL, VECTORIZED, GRAPH_MODES

from envs.pendulum.ode.world import World
from envs.pendulum.env import PendulumEnv, Agent
from dummy import build_dummy_env, DummyEnv


def evaluate(env, name: str = "env", backend: str = "numpy", use_jit: bool = False, seed: int = 0, vmap: int = 1, device: str = "cpu", num_eps: int = 1):

	# Get action space
	action_space = env.action_space()

	use_jit = use_jit and backend == "jax"
	with use(backend=backend):
		rng = jumpy.random.PRNGKey(jp.int32(seed))

		# vmap env
		env_reset = rjp.vmap(env.reset)
		env_step = rjp.vmap(env.step)
		rng = jumpy.random.split(rng, num=vmap)
		action = rjp.vmap(action_space.sample)(rng)

		# Get reset and step function
		env_reset = jax.jit(env_reset, backend=device) if use_jit else env_reset
		env_step = jax.jit(env_step, backend=device) if use_jit else env_step

		# Reset environment
		with timer(f"{name} | jit reset", log_level=WARN):
			graph_state, obs = env_reset(rng)

		# Initial step (warmup)
		with timer(f"{name} | jit step", log_level=WARN):
			graph_state, obs, reward, done, info = env_step(graph_state, action)

		# Run environment
		fps = []
		for _ in range(num_eps):
			# New record
			gs_lst = []
			obs_lst = []
			ss_lst = []

			# Reset
			graph_state, obs = env_reset(rng)
			gs_lst.append(graph_state)
			obs_lst.append(obs)
			ss_lst.append(graph_state.nodes["root"])

			tstart = time.time()
			eps_steps = 0
			while True:
				graph_state, obs, reward, done, info = env_step(graph_state, action)
				obs_lst.append(obs)
				gs_lst.append(graph_state)
				ss_lst.append(graph_state.nodes["root"])
				eps_steps += 1
				# done = done[0] if vmap > 1 else done
				if done[0]:
					# Time env stopping
					tend = time.time()
					env.stop()
					tstop = time.time()

					# Print timings
					fps.append(vmap*eps_steps / (tstop - tstart))
					print(
						f"{name=} | agent_steps={eps_steps} | chunk_index={graph_state.step} | t={(tstop - tstart): 2.4f} sec | t_s={(tstop - tend): 2.4f} sec | fps={vmap*eps_steps / (tend - tstart): 2.4f} | fps={vmap*eps_steps / (tstop - tstart): 2.4f} (incl. stop)")
					break
		print(f"{name=} | fps={onp.mean(fps): 2.4f} +/- {onp.std(fps): 2.4f}")
	return gs_lst, obs_lst, ss_lst


if __name__ == "__main__":

	# Define nodes
	world = World("world", rate=20)
	agent = Agent("root", rate=20)
	nodes = {n.name: n for n in [world, agent]}

	# Connect
	agent.connect(world, name="state",  window=1, blocking=True, jitter=LATEST, skip=True)
	world.connect(agent, name="action", window=1, blocking=True, jitter=LATEST)

	# Warmup nodes (pre-compile jitted functions)
	[n.warmup() for n in nodes.values()]

	# Create trace environment
	trace_steps = 100
	trace_env = PendulumEnv(nodes, root=agent, max_steps=trace_steps, clock=SIMULATED, real_time_factor=FAST_AS_POSSIBLE)
	# trace_env, nodes = build_dummy_env()
	# root = nodes["root"]

	# Evaluate async env
	gs_trace, obs_trace, ss_trace = evaluate(trace_env, name="trace_env", backend="numpy", use_jit=False, seed=0)

	# Gather record
	record = log_pb2.EpisodeRecord()
	[record.node.append(node.record()) for node in nodes.values()]
	r = {n.info.name: n for n in record.node}

	# Trace
	trace_record = trace(record, "root")

	# Settings
	use_jit = True
	vmap = 2000
	backend = "jax"
	jit = "jit" if (use_jit and backend == "jax") else "nojit"
	graph_type = VECTORIZED
	device = "gpu"

	# Define name
	name = []
	name.append(f"{vmap}")
	name.append(f"{backend}")
	name.append("jit" if (use_jit and backend == "jax") else "nojit")
	name.append(GRAPH_MODES[graph_type])
	name.append(device)
	name = "-".join(name)

	# Create compiled environment
	max_steps = trace_steps
	env = PendulumEnv(nodes, root=agent, max_steps=max_steps, trace=trace_record, graph=graph_type)
	# env = DummyEnv(nodes, root=root, max_steps=max_steps, trace=trace_record, graph=graph_type)

	# Evaluate async env
	gs, obs, ss = evaluate(env, name=name, backend=backend, use_jit=use_jit, seed=0, vmap=vmap, device=device, num_eps=30)

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

	# Compare observations and root step states
	# jax.tree_map(compare, obs_trace, obs)
	# jax.tree_map(compare, ss_trace, ss)

	# Compare
	compare_obs = jax.tree_map(lambda *args: args, obs_trace, obs)
	compare_ss = jax.tree_map(lambda *args: args, ss_trace, ss)

	print("finished")