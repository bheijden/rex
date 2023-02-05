from functools import partial
from typing import Any, Dict, List, Tuple, Union, Optional
from pickle import UnpicklingError
import dill as pickle
import jax
import jumpy.numpy as jp
import numpy as onp
from jax.tree_util import tree_map
from flax import serialization
from scripts.dummy import DummyNode, DummyAgent
from rex.distributions import Distribution, Gaussian, GMM
from rex.constants import LATEST, BUFFER, WARN, DEBUG, ERROR
import rex.utils as utils
from rex.tracer import trace
from rex.proto import log_pb2
from rex.node import Node
from rex.agent import Agent
from rex.gmm_estimator import GMMEstimator
from rex.plot import get_subplots
from rex.open_colors import ecolor, fcolor

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


class _NoValue: pass


class _HasValue:
	def __init__(self, tree):
		self.tree = tree


def _truncated_stack(*data: jp.ndarray) -> Optional[jp.ndarray]:
	has_empty = any([isinstance(d, _NoValue) for d in data])
	if has_empty:
		return None
	elif all([isinstance(d, _HasValue) for d in data]):
		return tree_map(_truncated_stack, *[d.tree for d in data])
	assert all([not isinstance(d, _HasValue) for d in data])

	# Determine min_length
	min_length = min([x.shape[0] for x in data if x.ndim > 0])

	# Truncate data to min_length
	data = tree_map(lambda d: d[:min_length], data)

	# Stack data
	data_stacked = tree_map(lambda *d: jp.stack(d), *data)
	return data_stacked


class RecordHelper:
	def __init__(self, record: Union[log_pb2.ExperimentRecord, log_pb2.EpisodeRecord], trace: log_pb2.TraceRecord = None,
	             validate: bool = True, stack: bool = True):
		# todo: graph_states can only be constructed from a trace
		# todo: When rebuilding graph_states from trace: depth vs order?
		self.trace = trace
		self.record = record
		self.validate = validate
		self.stack = stack

		# Convert to experiment record
		if isinstance(record, log_pb2.EpisodeRecord):
			self._record = log_pb2.ExperimentRecord(episode=[record])
		else:
			self._record = record
		assert isinstance(self._record, log_pb2.ExperimentRecord), "Record must be an ExperimentRecord or EpisodeRecord"

		# Store preprocessed data in convenient format
		self._delays: List[Dict[str, Dict[str, Union[jp.ndarray, Dict[str, jp.ndarray]]]]]
		self._delays_stacked: Dict[str, Dict[str, Union[jp.ndarray, Dict[str, jp.ndarray]]]]
		self._data: List[Dict[str, Dict[str, Any]]]
		self._data_stacked: Dict[str, Dict[str, Any]]
		self._nodes: List[Dict[str, Union[str, Node, Agent]]] = []

		# Pre-process record data
		self._preprocess_data()

		# Validate record
		if self.validate:
			self._validate_data()

		# Stack data
		if self.stack:
			assert self.validate, "Stacking requires validation. Set validate=True."
			self._stack_data()

	def get_nodes(self, episode: int = -1) -> Dict[str, Union[Node, Agent]]:
		nodes = {}
		for name, node_bytes in self._nodes[episode].items():
			assert len(node_bytes) > 0, "Node state must be non-empty."
			nodes[name] = pickle.loads(node_bytes)

		# Fully restore node by unpickling (re-connects to other nodes, execute custom unpickling routines if any)
		for n in nodes.values():
			n.unpickle(nodes)
		return nodes

	def _unpickle_data(self, record: log_pb2.Serialization):
		if len(record.encoded_bytes) == 0:
			return _NoValue()
		encoded_bytes = record.encoded_bytes
		try:
			target = pickle.loads(record.target)
			data = [serialization.from_bytes(target, b) for b in encoded_bytes]
		except (UnpicklingError, ValueError) as e:
			print(f"Failed to load target. Unpickling to state_dict instead: {e}")
			data = [serialization.msgpack_restore(b) for b in encoded_bytes]
		return _HasValue(tree_map(lambda *x: jp.stack(x), *data))

	def _preprocess_data(self):
		# Get delays
		self._delays, _ = utils.get_delay_data(self._record, concatenate=False)

		# Get data
		self._data = []
		self._nodes = []
		for i, e in enumerate(self._record.episode):
			# Store nodes
			eps_nodes = {}
			self._nodes.append(eps_nodes)
			for n in e.node:
				eps_nodes[n.info.name] = n.info.state

			# Store data
			eps_data = {n.info.name: dict(outputs=None, rngs=None, states=None, params=None, step_states=None) for n in e.node}
			self._data.append(eps_data)
			for n in e.node:
				# Store outputs
				eps_data[n.info.name]["outputs"] = self._unpickle_data(n.outputs)
				eps_data[n.info.name]["rngs"] = self._unpickle_data(n.rngs)
				eps_data[n.info.name]["states"] = self._unpickle_data(n.states)
				eps_data[n.info.name]["params"] = self._unpickle_data(n.params)
				eps_data[n.info.name]["step_states"] = self._unpickle_data(n.step_states)

	def _validate_data(self):
		# todo: Do not raise errors, but rather set flags. Check flags in stack and truncate.
		# todo: check if all data is present
		# todo: Check if data is of same length?
		# todo: Check if computation graph is the same
		# todo: Check if all nodes are present
		# todo: Check if step_states correspond to logged inputs
		# Stack
		# self._lengths = tree_map(lambda *l: list(l), *self._lengths)
		# self._max_lengths = tree_map(lambda *l: max(l), *self._lengths
		pass

	def _stack_data(self):
		# Stack data
		self._data_stacked = tree_map(_truncated_stack, *self._data)
		self._delays_stacked = tree_map(_truncated_stack, *self._delays)


def make_delay_distributions(record: Union[RecordHelper, log_pb2.ExperimentRecord],
                             num_steps: int = 100,
                             num_components: int = 2,
                             step_size: float = 0.05,
                             seed: int = 0):
	# Prepare data
	if isinstance(record, log_pb2.ExperimentRecord):
		data, info = utils.get_delay_data(record, concatenate=True)
	elif isinstance(record, RecordHelper):
		data, info = utils.get_delay_data(record.trace, concatenate=True)
		# data = tree_map(lambda *x: jp.concatenate(x, axis=0), *record._delays)
	else:
		raise NotImplementedError

	def init_estimator(x, i):
		name = i.name if not isinstance(i, tuple) else f"{i[0].name}.input({i[1].name})"
		est = GMMEstimator(x, name)
		return est

	# Initialize estimators
	est = jax.tree_map(lambda x, i: init_estimator(x, i), data, info)

	# Fit estimators
	jax.tree_map(lambda e: e.fit(num_steps=num_steps, num_components=num_components, step_size=step_size, seed=seed), est)

	# Get distributions
	dist = jax.tree_map(lambda e: e.get_dist(include_data=True), est)
	return data, info, est, dist


if __name__ == "__main__":

	SAVE_DELAY_DISTRIBUTIONS = True
	FIT_DELAY_DISTRIBUTIONS = True
	SHOW_PLOTS = False

	# Load episode record
	# name = "21eps-pretrained-sbx-sac"
	# log_dir = "/home/r2ci/rex/logs/disc-pendulum-real"
	name = "record_sysid"
	# log_dir = "/home/r2ci/rex/logs/real_norl_double_pendulum_2023-02-03-1635"
	# log_dir = "/home/r2ci/rex/logs/real_more_double_pendulum_2023-01-27-1638_freq_nonblocking_120s"
	# log_dir = "/home/r2ci/rex/logs/real_pendulum_2023-01-27-1806_phase_blocking_120s"
	# log_dir = "/home/r2ci/rex/logs/real_pendulum_2023-01-27-1806"
	# log_dir = "/home/r2ci/rex/logs/real_withrl_double_pendulum_2023-02-03-1656"
	log_dir = "/home/r2ci/rex/logs/45sec_real_8torque_80hz_02deltaact_agentpolicy_double_pendulum_2023-02-03-1856"
	# log_dir = "/home/r2ci/rex/logs/45sec_real_50maxvel_8torque_80hz_01deltaact_agentpolicy_double_pendulum_2023-02-03-1854"
	# log_dir = "/home/r2ci/rex/logs/45sec_real_50maxvel_8torque_30hz_005deltaact_agentpolicy_double_pendulum_2023-02-03-1914"
	num_components = 2
	num_steps = 500
	exp_record = log_pb2.ExperimentRecord()
	with open(f"{log_dir}/{name}.pb", "rb") as f:
		exp_record.ParseFromString(f.read())

	if FIT_DELAY_DISTRIBUTIONS:
		# Get delay distributions
		data, info, est, dist = make_delay_distributions(exp_record, num_steps=num_steps, num_components=num_components, step_size=0.05, seed=0)

		# Split
		est_inputs, est_step = est["inputs"], est["step"]
		data_inputs, data_step = data["inputs"], data["step"]
		info_inputs, info_step = info["inputs"], info["step"]
		dist_inputs, dist_step = dist["inputs"], dist["step"]

		if SHOW_PLOTS:
			# Plot gmm
			def plot_gmm(ax, dist, delays, i, edgecolor):
				x = onp.linspace(0, onp.max(delays), 1000)
				y = dist.pdf(x)
				if hasattr(i, "cls"):
					ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
					ax.set_title(f"{i.name}")
				else:
					node_info, input_info = i
					ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
					ax.set_title(f"{input_info.output} -> {node_info.name} ({input_info.name})")
				ax.legend()

			# Plot distributions
			fig_step, axes_step = get_subplots(est_step, figsize=(10, 10), sharex=True, sharey=False, major="row")
			fig_inputs, axes_inputs = get_subplots(est_inputs, figsize=(10, 10), sharex=True, sharey=False, major="row")

			# Plot measured delays
			jax.tree_map(lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.computation, facecolor=fcolor.computation), axes_step, est_step)
			jax.tree_map(lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.communication, facecolor=fcolor.communication), axes_inputs, est_inputs)

			# Plot gmm
			jax.tree_map(partial(plot_gmm, edgecolor=ecolor.computation), axes_step, dist_step, data_step, info_step)
			jax.tree_map(partial(plot_gmm, edgecolor=ecolor.communication), axes_inputs, dist_inputs, data_inputs, info_inputs)

			plt.show()

		if SAVE_DELAY_DISTRIBUTIONS:
			# Save distributions
			with open(f"{log_dir}/{name}-gmms-{num_components}comps.pkl", "wb") as f:
				pickle.dump(dist, f)
			print(f"Saved distributions to {log_dir}/{name}-gmms-{num_components}comps.pkl")

	if SHOW_PLOTS:
		t = trace(exp_record.episode[0], "agent")

		data = RecordHelper(exp_record, t)

		# Plot delays
		from rex.open_colors import cscheme_fn

		cscheme = {"agent": "blue", "sensor": "grape", "world": "green", "render": "orange", "actuator": "yellow"}
		ecolor, fcolor = cscheme_fn(cscheme)

		fig, axes = plt.subplots(nrows=3, ncols=2)
		axes = axes.flatten()
		axes[-1].clear()
		for idx, (name, delay) in enumerate(data._delays_stacked["step"].items()):
			x = list(range(delay.shape[-1]))
			mean = delay.mean(axis=0)
			std = delay.std(axis=0)
			axes[idx].plot(x, mean, color=ecolor[name], label=name)
			# axes[idx].plot(x, delay[0], color="black", label="first")
			# axes[idx].plot(x, delay[-1], color="red", label="last")
			axes[idx].fill_between(x, mean-std, mean+std, color=fcolor[name], alpha=0.5)
			axes[idx].set_title(name)

		is_state_dict = isinstance(data._data_stacked["sensor"]["outputs"], (dict, tuple, list))

		# Plot states
		fig, axes = plt.subplots(6)
		if is_state_dict:
			for e in range(data._data_stacked["sensor"]["outputs"]["cos_th"].shape[0]):
				axes[0].plot(data._data_stacked["agent"]["outputs"].action[e])
				# axes[0].plot(data._data_stacked["actuator"]["step_states"].inputs["action"].data.action[e, :, 0, 0], color="red")
				axes[1].plot(data._data_stacked["sensor"]["outputs"]["cos_th"][e])
				# axes[1].plot(data._data_stacked["sensor"]["outputs"]["th_enc"][e])
				# axes[2].plot(data._data_stacked["sensor"]["outputs"]["th2"][e])
				# axes[3].plot(data._data_stacked["sensor"]["outputs"]["volt"][e])
				# axes[4].plot(data._data_stacked["sensor"]["outputs"]["volt2"][e])
				axes[2].plot(data._data_stacked["sensor"]["outputs"]["thdot"][e])
		else:
			for e in range(data._data_stacked["sensor"]["outputs"].cos_th.shape[0]):
				action = data._data_stacked["agent"]["outputs"].action[e]
				cos_th = data._data_stacked["sensor"]["outputs"].cos_th[e]
				sin_th = data._data_stacked["sensor"]["outputs"].sin_th[e]
				th = onp.arctan2(sin_th, cos_th)
				cos_th2 = data._data_stacked["sensor"]["outputs"].cos_th2[e]
				sin_th2 = data._data_stacked["sensor"]["outputs"].sin_th2[e]
				th2 = onp.arctan2(sin_th2, cos_th2)
				thdot = data._data_stacked["sensor"]["outputs"].thdot[e]
				thdot2 = data._data_stacked["sensor"]["outputs"].thdot2[e]
				delta_goal = (jp.pi - jp.abs(th + th2))


				axes[0].plot(delta_goal)
				axes[1].plot(action)
				axes[2].plot(cos_th)
				axes[2].plot(sin_th)
				axes[3].plot(cos_th2)
				axes[3].plot(sin_th2)
				axes[4].plot(thdot)
				axes[5].plot(thdot2)

				# axes[1].plot(data._data_stacked["sensor"]["outputs"].th_enc[e])
				# axes[2].plot(data._data_stacked["sensor"]["outputs"].th2[e])
				# axes[3].plot(data._data_stacked["sensor"]["outputs"].volt[e])
				# axes[4].plot(data._data_stacked["sensor"]["outputs"].volt2[e])

		plt.show()


	# # reload nodes
	# eps = []
	# for e in exp_record.episode:
	# 	node = {n.info.name: dict(obj=None, outputs=None, rngs=None, states=None, params=None, step_states=None) for n in
	# 	        e.node}
	# 	eps.append(node)
	#
	# 	# Reinitialize nodes
	# 	for n in e.node:
	# 		obj = pickle.loads(n.info.state)
	# 		node[obj.name]["obj"] = obj
	#
	# 	# Finalize unpickling
	# 	objs = {name: n["obj"] for name, n in node.items()}
	# 	for n in node.values():
	# 		n["obj"].unpickle(objs)
	#
	# 	for n in e.node:
	# 		# Reinitialize outputs
	# 		target = pickle.loads(n.outputs.target)
	# 		encoded_bytes = n.outputs.encoded_bytes
	# 		outputs = [serialization.from_bytes(target, b) for b in encoded_bytes]
	# 		node[n.info.name]["outputs"] = outputs
	#
	# 		# Reinitialize rngs
	# 		target = pickle.loads(n.rngs.target)
	# 		encoded_bytes = n.rngs.encoded_bytes
	# 		rngs = [serialization.from_bytes(target, b) for b in encoded_bytes]
	# 		node[n.info.name]["rngs"] = rngs
	#
	# 		# Reinitialize states
	# 		target = pickle.loads(n.states.target)
	# 		encoded_bytes = n.states.encoded_bytes
	# 		states = [serialization.from_bytes(target, b) for b in encoded_bytes]
	# 		node[n.info.name]["states"] = states
	#
	# 		# Reinitialize params
	# 		target = pickle.loads(n.params.target)
	# 		encoded_bytes = n.params.encoded_bytes
	# 		params = [serialization.from_bytes(target, b) for b in encoded_bytes]
	# 		node[n.info.name]["params"] = params
	#
	# 		# Reinitialize step states
	# 		target = pickle.loads(n.step_states.target)
	# 		encoded_bytes = n.step_states.encoded_bytes
	# 		step_states = [serialization.from_bytes(target, b) for b in encoded_bytes]
	# 		node[n.info.name]["step_states"] = step_states
