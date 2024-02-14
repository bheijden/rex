import pytest
import dill as pickle
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as onp
import tempfile

from rex.utils import set_log_level
from rex.wrappers import GymWrapper, VecGymWrapper
from rex.constants import WARN, ERROR, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, DEBUG
from rex.proto import log_pb2
from rex.asynchronous import AsyncGraph
from tests.dummy import build_dummy_env, DummyEnv


def process_record(record: log_pb2.ExperimentRecord):
	data_copy = []
	for e in record.episode:
		node = {n.info.name: dict(obj=None, outputs=None, rngs=None, states=None, params=None, step_states=None) for n in
		        e.node}
		data_copy.append(node)

		# Reinitialize nodes
		for n in e.node:
			obj = pickle.loads(n.info.state)
			node[obj.name]["obj"] = obj

		# Finalize unpickling
		objs = {name: n["obj"] for name, n in node.items()}
		for n in node.values():
			n["obj"].unpickle(objs)

		for n in e.node:
			# Reinitialize outputs
			target = pickle.loads(n.outputs.target)
			encoded_bytes = n.outputs.encoded_bytes
			outputs = [serialization.from_bytes(target, b) for b in encoded_bytes]
			node[n.info.name]["outputs"] = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *outputs)

			# Reinitialize rngs
			target = pickle.loads(n.rngs.target)
			encoded_bytes = n.rngs.encoded_bytes
			rngs = [serialization.from_bytes(target, b) for b in encoded_bytes]
			node[n.info.name]["rngs"] = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *rngs)

			# Reinitialize states
			target = pickle.loads(n.states.target)
			encoded_bytes = n.states.encoded_bytes
			states = [serialization.from_bytes(target, b) for b in encoded_bytes]
			node[n.info.name]["states"] = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *states)

			# Reinitialize params
			target = pickle.loads(n.params.target)
			encoded_bytes = n.params.encoded_bytes
			params = [serialization.from_bytes(target, b) for b in encoded_bytes]
			node[n.info.name]["params"] = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *params)

			# Reinitialize step states
			target = pickle.loads(n.step_states.target)
			encoded_bytes = n.step_states.encoded_bytes
			step_states = [serialization.from_bytes(target, b) for b in encoded_bytes]
			node[n.info.name]["step_states"] = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *step_states)
	return data_copy


# @pytest.mark.parametrize("backend, jit", [("numpy", False), ("jax", False), ("jax", True)])
def test_reinitialize_nodes_from_recording():
	# Grab the dummy environment
	env, nodes = build_dummy_env()

	# Apply wrapper
	env = GymWrapper(env)  # Wrap into gym wrapper

	# Test wrapper api
	try:
		env.save("")
	except AttributeError:
		pass

	# Pickle & reload the environment
	tmp = tempfile.NamedTemporaryFile()
	env.unwrapped.save(tmp.name)
	env = env.unwrapped.load(tmp.name)
	env = GymWrapper(env)
	nodes = env.graph.nodes_and_root

	# Seed the environment
	env.seed(0)
	action_space = env.action_space

	# Run environment
	exp_record = log_pb2.ExperimentRecord()
	for _ in range(2):
		done, obs = False, env.reset()
		while not done:
			action = action_space.sample()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated | truncated
		env.stop()

		# Save record
		eps_record = log_pb2.EpisodeRecord()
		[eps_record.node.append(node.record(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True))
		 for node in nodes.values()]
		exp_record.episode.append(eps_record)

	# Re-initialize  nodes, serialized data (outputs, rngs, states, params, step_states)
	new_record = log_pb2.ExperimentRecord()
	new_record.ParseFromString(exp_record.SerializeToString())
	exp_record = new_record
	data = process_record(exp_record)
	nodes_copy = {name: n["obj"] for name, n in data[0].items()}
	agent_copy = nodes_copy["agent"]  # type: ignore
	graph = AsyncGraph(nodes_copy, agent_copy, clock=SIMULATED, real_time_factor=FAST_AS_POSSIBLE)
	env_copy = DummyEnv(graph=graph, max_steps=100, name="env_copy")

	# Apply wrapper
	env_copy = GymWrapper(env_copy)  # Wrap into gym wrapper
	env_copy.seed(0)
	action_space = env_copy.action_space

	# Run environment
	exp_record_copy = log_pb2.ExperimentRecord()
	for _ in range(2):
		done, obs = False, env_copy.reset()
		while not done:
			action = action_space.sample()
			obs, reward, terminated, truncated, info = env_copy.step(action)
			done = terminated | truncated
		env_copy.stop()

		# Save record
		eps_record = log_pb2.EpisodeRecord()
		[eps_record.node.append(node.record(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True))
		 for node in nodes_copy.values()]
		exp_record_copy.episode.append(eps_record)

	# Re-initialize  nodes, serialized data (outputs, rngs, states, params, step_states)
	data_copy = process_record(exp_record_copy)

	def compare_array(x, y):
		# only compare equal sized array
		num_steps = min(x.shape[0], y.shape[0])
		# NOTE: -1 to avoid comparing the last step, because they are unequal when lengths are different
		num_steps -= 1
		try:
			onp.testing.assert_array_equal(x[:num_steps], y[:num_steps])
		except AssertionError as e:
			print(f"FAILED | name={name} | eps={idx} | `{k}`")
			raise e

	# Compare
	has_failed = False
	for idx, (d, dc) in enumerate(zip(data, data_copy)):
		for name, n in d.items():
			# Remove object
			d[name].pop("obj"), dc[name].pop("obj")
			for k, v in n.items():
				try:
					jax.tree_util.tree_map(compare_array, d[name][k], dc[name][k])
				except AssertionError as e:
					print(f"FAILED | name={name} | eps={idx} | `{k}`")
					has_failed = True
					# raise e
	assert not has_failed, "Failed to reinitialize nodes from recording"


def test_record_overflow():
	# Grab the dummy environment
	env, nodes = build_dummy_env()

	# Grab record before any simulation
	_ = nodes["agent"].record()

	# Apply wrapper
	env = GymWrapper(env)  # Wrap into gym wrapper

	# Test env api
	env.log("test_api", "test_api", log_level=WARN)

	# Set max record size
	nodes["agent"]._max_records = 10

	# Seed the environment
	env.seed(0)
	action_space = env.action_space

	# Run environment
	exp_record = log_pb2.ExperimentRecord()
	for _ in range(2):
		done, obs = False, env.reset()
		while not done:
			action = action_space.sample()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated | truncated
		env.stop()

		# Save record
		eps_record = log_pb2.EpisodeRecord()
		[eps_record.node.append(node.record(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True))
		 for node in nodes.values()]
		exp_record.episode.append(eps_record)
