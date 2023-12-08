from estimator.env import Estimator, Replay, ReconstructionLoss, EstimatorParams, EstimatorEnv

from functools import partial
from math import ceil
from typing import Tuple, Dict, Callable, Union, Any, Sequence, List, TypeVar, TYPE_CHECKING
import flax.struct as struct
from flax.core import FrozenDict
import optax
import jax
import jumpy
import jax.numpy as jnp
import jumpy.numpy as jp

import networkx as nx

from rex.supergraph import get_timings_from_network_record, get_outputs_from_timings, get_graph_buffer, get_step_seqs_mapping, get_seqs_mapping
from rex.proto import log_pb2
from rex.base import GraphState, Empty, StepState, Output, GraphBuffer, SeqsMapping
from rex.env import BaseEnv
import rex.utils as utils
import rex.jumpy as rjp

if TYPE_CHECKING:
	from estimator.callback import BaseCallback


def single_loss(env: BaseEnv, graph_state: GraphState, seqs_step: SeqsMapping, params: optax.Params, outputs: Dict[str, Output],
                rng: jp.array, starting_eps: jp.int32, starting_step: jp.int32, num_steps: jp.int32) -> Tuple[jp.float32, GraphBuffer]:
	# Overwrite params in graph state with optimizer params
	new_nodes = {}
	for node_name, params in params.items():
		if params is None:
			continue
		new_params = jax.tree_util.tree_map(lambda x, y: x if y is None else y, graph_state.nodes[node_name].params, params)
		new_nodes[node_name] = graph_state.nodes[node_name].replace(params=new_params)

	# Determine episode timings
	max_eps, max_step = next(iter(graph_state.timings[-1].values()))["run"].shape
	eps = jp.clip(starting_eps, jp.int32(0), max_eps - 1)
	step = jp.clip(starting_step, jp.int32(0), max_step - 1)
	eps_timings = rjp.tree_take(graph_state.timings, eps)

	def _fill_buffer(seqs, arr):
		exp_dims = [1] * (arr.ndim - 1)
		exp_seqs = jp.expand_dims(seqs, exp_dims)
		return rjp.take_along_axis(arr, exp_seqs, axis=0)

	# Fill buffer with outputs
	new_outputs = {}
	eps_outputs = rjp.tree_take(outputs, eps, axis=0)
	for output_name, b in graph_state.buffer.outputs.items():
		bsize = seqs_step[output_name].shape[-1]
		seqs = rjp.dynamic_slice(seqs_step[output_name], [eps, step, 0], [1, 1, bsize])[0, 0]
		o = jax.tree_util.tree_map(partial(_fill_buffer, seqs), eps_outputs[output_name])
		new_outputs[output_name] = o

	# Update graph state
	buffer = graph_state.buffer.replace(outputs=FrozenDict(new_outputs), timings=eps_timings)
	gs = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes), eps=eps, step=step, buffer=buffer)

	# Reset env
	gs, loss_init, _info = env.reset(rng, gs)

	# Return buffer after first step (likely has most accurate output estimates).
	new_buffer = gs.buffer

	# Perform steps
	def _step(carry, x):
		gs, loss_step, loss = carry
		new_gs, new_loss_step, _, terminated, truncated, info = env.step(gs, loss_step)
		done = jp.logical_or(terminated, truncated)  # todo: correct?
		# Only add dynamic loss to total loss
		new_loss = (1-done)*(new_loss_step.loss) + loss
		new_carry = (new_gs, new_loss_step, new_loss)
		return new_carry, new_loss

	carry_out, scan_out = jumpy.lax.scan(_step, (gs, loss_init, loss_init.loss), None, length=num_steps)  # todo: unroll?
	loss = scan_out[-1] if len(scan_out) > 0 else loss_init.loss

	return loss, new_buffer


Metric = TypeVar('Metric')
Metrics = Dict[str, Metric]
Carry = Tuple[Metrics, Dict, Dict, Dict[str, Output]]
ScanOut = Any


@struct.dataclass
class Config:
	epoch: int
	env: BaseEnv
	optimizer: optax.GradientTransformation
	opt_state: optax.OptState
	params: optax.Params
	graph_state: GraphState
	num_epochs: int
	num_steps: int
	num_training_steps: int
	num_training_steps_per_epoch: int
	num_batches: int
	prior_fn: Callable[[optax.Params], Any]


def fit(env: BaseEnv, params: optax.Params, optimizer: optax.GradientTransformation, graph_state: GraphState, outputs: Dict[str, Output], num_batches: int = 20, num_steps: int = 3, num_training_steps: int = 1000,
        num_training_steps_per_epoch: int = 100, prior_fn: Callable[[optax.Params], Any] = lambda x: jp.float32(0),
        callbacks: Dict[str, "BaseCallback"] = None) -> optax.Params:
	max_eps, max_steps = next(iter(graph_state.timings[-1].values()))["run"].shape
	start_eps, start_steps = jp.meshgrid(jp.arange(0, max_eps), jp.arange(0, max_steps))
	start_eps, start_steps = start_eps.flatten(), start_steps.flatten()

	# Determine step_update_mask
	seqs_step, updated_step = get_step_seqs_mapping(env.graph.S, graph_state.timings, graph_state.buffer)

	# todo: test out single_loss without vmap.
	# loss, new_buffer = single_loss(env, graph_state, seqs_step, params, outputs, jumpy.random.PRNGKey(0), jp.int32(0),
	#                                jp.int32(12), num_steps=jp.int32(2))

	# Determine batch_loss
	# todo: avoid having to switch
	batch_loss = jax.vmap(partial(single_loss, env, graph_state, seqs_step), in_axes=(None, None, 0, 0, 0, None), out_axes=0)
	# batch_loss = jumpy.vmap(partial(single_loss, env, graph_state, seqs_step), include=(False, False, True, True, True, False))

	def loss(params: optax.Params, _outputs: Dict[str, Output], _rng: jp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Output]]:
		# Determine batch eps and steps
		_rngs = jumpy.random.split(_rng, num_batches+1)
		_rng, _rngs = _rngs[0], _rngs[1:]
		choice = jumpy.random.choice(_rng, start_steps.shape[0], shape=(num_batches,), replace=False)
		steps = jp.take(start_steps, choice)
		eps = jp.take(start_eps, choice)

		# Perform batch rollout
		loss_batch, buffer_batch = batch_loss(params, _outputs, _rngs, eps, steps, jp.int32(num_steps))

		# Add prior loss on params
		ploss = prior_fn(params)
		ploss = jax.tree_util.tree_reduce(lambda acc, l: acc + jp.sum(l), ploss, jp.float32(0.))

		# Update outputs with buffer_batch
		# todo: Determine mask that identifies which outputs have been updated after reset.
		# todo: Use updated_step for this.
		_new_outputs = _outputs
		return loss_batch.mean() + ploss, _new_outputs

	# todo: test out loss with vmap.
	# loss_batch, new_outputs = loss(params, outputs, jumpy.random.PRNGKey(0))

	@jax.jit
	def train_step(carry: Carry, _rng):
		metrics, params, opt_state, _outputs = carry
		(loss_value, _outputs), grads = jax.value_and_grad(loss, argnums=0, has_aux=True)(params, _outputs, _rng)
		updates, opt_state = optimizer.update(grads, opt_state, params)
		params = optax.apply_updates(params, updates)
		# Make all params positive todo: (hack)
		params["world"] = jax.tree_util.tree_map(lambda x: jp.abs(x), params["world"])
		carry = (metrics, params, opt_state, _outputs)
		return carry, (loss_value, params)

	# Initialize optimizer
	opt_state = optimizer.init(params)

	# Determine number of epochs
	num_epochs = ceil(num_training_steps // num_training_steps_per_epoch)

	# Prepare config
	config = Config(epoch=jp.int32(0), env=env, optimizer=optimizer, opt_state=opt_state, params=params, graph_state=graph_state,
	                num_epochs=num_epochs, num_steps=num_steps, num_training_steps=num_training_steps,
	                num_training_steps_per_epoch=num_training_steps_per_epoch, num_batches=num_batches, prior_fn=prior_fn)

	# Prepare callbacks
	callbacks = callbacks or {}
	metrics: Metrics = {cb_name: cb.on_training_start(config) for cb_name, cb in callbacks.items()}

	# Run training epochs
	rngs_epoch = jumpy.random.split(jax.random.PRNGKey(0), num=num_epochs)
	for i, rng in enumerate(rngs_epoch):
		# Update metrics
		config = config.replace(epoch=i)

		# Update metrics
		carry = (metrics, params, opt_state, outputs)
		metrics = {cb_name: cb.on_epoch_start(config, metrics[cb_name], carry) for cb_name, cb in callbacks.items()}
		carry = (metrics, params, opt_state, outputs)

		# Run epoch
		rngs_step = jumpy.random.split(rng, num=num_training_steps_per_epoch)
		carry, scan_out = jax.lax.scan(train_step, carry, rngs_step, length=num_training_steps_per_epoch, unroll=1)

		# Update metrics
		metrics, params, opt_state, outputs = carry
		metrics = {cb_name: cb.on_epoch_end(config, metrics[cb_name], carry, scan_out) for cb_name, cb in callbacks.items()}

		# timer = utils.timer(f"epoch {i}")
		# with timer:
		# todo: why jumpy.lax.scan only works with jit?
		# carry, scan_out = jumpy.lax.scan(train_step, carry, rngs_step, length=num_training_steps_per_epoch, unroll=1)
		# fps = num_training_steps_per_epoch / timer.duration
		# metrics = Metrics(epoch=i+1, step=(i+1)*num_training_steps_per_epoch, duration=timer.duration, fps=fps)
		# progress_fn(metrics, carry, scan_out)

	# Update metrics
	carry = (metrics, params, opt_state, graph_state)
	metrics = {cb_name: cb.on_training_end(config, metrics[cb_name], carry) for cb_name, cb in callbacks.items()}
	return metrics, params, opt_state, graph_state


from rex.open_colors import ewheel, fwheel
from rex.node import Node
from rex.asynchronous import Agent
import dill as pickle
import experiments as exp
import matplotlib.pyplot as plt


def build_estimator(record: log_pb2.ExperimentRecord, rate: float, data: exp.RecordHelper) -> Tuple[log_pb2.EpisodeRecord, Dict[str, Node], List[str]]:
	# Copy episode record
	# todo: does this truly deepcopy?
	record.CopyFrom(record)

	# Re-initialize nodes
	nodes: Dict[str, Union[Node, Agent]] = {}
	for n in record.episode[-1].node:
		nodes[n.info.name] = pickle.loads(n.info.state)
	[n.unpickle(nodes) for n in nodes.values()]

	# Initialize estimator
	estimator = Estimator("estimator", rate=rate, delay=0.)
	estimator.connect(nodes["sensor"], window=1, blocking=False, delay=0., skip=True)  # todo: skip to avoid phase shift.
	nodes["estimator"] = estimator

	# Define nodes for which we have recorded outputs
	excludes_inputs = ["actuator", "agent"]
	for name in excludes_inputs:
		outputs = data._data_stacked[name]["outputs"]
		states = data._data_stacked[name]["states"]
		params = data._data_stacked[name]["params"]
		nodes[name] = Replay(nodes[name], outputs, states, params)

	# Wrap sensors to calculate reconstruction loss
	outputs = data._data_stacked["sensor"]["outputs"]
	nodes["sensor"] = ReconstructionLoss(nodes["sensor"], outputs)

	# Recreate estimator record
	for record_eps in record.episode:
		record_sensor = [n for n in record_eps.node if n.info.name == "sensor"][0]

		# Add estimator record to episode record
		record_est = get_estimator_record(estimator, [record_sensor])
		record_eps.node.extend([record_est])
	return record, nodes, excludes_inputs


def get_estimator_record(estimator: Node, records: List[log_pb2.NodeRecord]):
	# Get record
	record_est = estimator.record()
	rate = estimator.rate

	# Check that number of input records matches number of inputs
	assert len(records) == len(estimator.inputs), "Number of input records does not match number of inputs."

	# Create step records
	max_time = max([r.steps[-1].ts_output for r in records])
	# last_ts_sensor = record_sensor.steps[-1].ts_output
	# todo: Verify that env._cgraph.max_steps is equal to max_num_steps (perhaps +1 needed?).
	max_num_steps = ceil(rate * max_time)+1
	ts_prev = 0.
	for i in range(max_num_steps):
		ts = i / rate

		# Create step record
		sent = log_pb2.Header(eps=0, seq=i, ts=log_pb2.Time(sc=ts, wc=ts))
		comp_delay = log_pb2.Time(sc=0., wc=0.)
		step = log_pb2.StepRecord(tick=i, ts_scheduled=ts, ts_max=0., ts_output_prev=ts_prev, ts_step=ts, ts_output=ts,
		                          phase=0., phase_scheduled=0., phase_inputs=0., phase_last=0., sent=sent,
		                          delay=0., comp_delay=comp_delay)
		record_est.steps.append(step)

		# Update ts_prev
		ts_prev = ts

	# Add sensor record to estimator record
	for r in records:
		record_in = [n for n in record_est.inputs if n.info.output == r.info.name][0]
		record_in.grouped.extend([log_pb2.GroupedRecord(num_msgs=0) for _ in record_est.steps])
		for s in r.steps:
			comm_delay = log_pb2.Time(sc=0., wc=0.)
			sent = log_pb2.Header()
			received = log_pb2.Header()
			sent.CopyFrom(s.sent)
			received.CopyFrom(s.sent)
			msg = log_pb2.MessageRecord(sent=sent, received=received, delay=0., comm_delay=comm_delay)

			# Add message to grouped record
			ts = s.sent.ts.sc
			tick = ceil(ts * rate)

			record_in.grouped[tick].num_msgs += 1
			record_in.grouped[tick].messages.append(msg)
	return record_est


def plot_trajectory(targets: jp.ndarray = None, preds_hidden: jp.ndarray = None, label: str = None, ax: plt.Axes = None, size: float = 2):
	label = label or ""
	if ax is None:
		fig, ax = plt.subplots()
	if targets is not None:
		art_target = ax.scatter(*targets, color=ewheel["blue"], label=f"targets", s=size)
	else:
		art_target = None
	if preds_hidden is not None:
		art_hidden = ax.scatter(*preds_hidden, color=ewheel["orange"], label=f"hidden", s=size)
	else:
		art_hidden = None
	ax.set_title(f"Trajectory {label}")
	return ax, (art_target, art_hidden)


def visualize(_world_states, nodes, ts_actuator, ts_sensor, ts_world):
	fig, axes = plt.subplots(nrows=5, ncols=1)
	axes = axes.flatten()

	targets = jp.stack((ts_sensor, nodes["sensor"]._outputs.cos_th))
	preds_hidden = jp.stack((ts_world, jp.cos(_world_states.th)))
	_, art_cos_th = plot_trajectory(targets, preds_hidden, label="cos(th)", ax=axes[0])

	targets = jp.stack((ts_sensor, nodes["sensor"]._outputs.cos_th2))
	preds_hidden = jp.stack((ts_world, jp.cos(_world_states.th2)))
	_, art_cos_th2 = plot_trajectory(targets, preds_hidden, label="cos(th2)", ax=axes[1])

	targets = jp.stack((ts_sensor, nodes["sensor"]._outputs.thdot))
	preds_hidden = jp.stack((ts_world, _world_states.thdot))
	_, art_thdot = plot_trajectory(targets, preds_hidden, label="thdot", ax=axes[2])

	targets = jp.stack((ts_sensor, nodes["sensor"]._outputs.thdot2))
	preds_hidden = jp.stack((ts_world, _world_states.thdot2))
	_, art_thdot2 = plot_trajectory(targets, preds_hidden, label="thdot2", ax=axes[3])

	# targets = jp.stack((ts_actuator[:gs.outputs["actuator"].action.shape[0]], gs.outputs["actuator"].action[:, 0]))
	targets = jp.stack((ts_actuator, nodes["actuator"]._outputs.action[:, 0]))
	_, art_actions = plot_trajectory(targets, None, label="actions", ax=axes[4])
	return fig, axes, dict(cos_th=art_cos_th, cos_th2=art_cos_th2, thdot=art_thdot, thdot2=art_thdot2, actions=art_actions)


def init_graph_state(env: BaseEnv, nodes: Dict[str, Node], record: log_pb2.NetworkRecord, MCS: nx.DiGraph, Gs: List[nx.DiGraph], Gs_monomorphism: List[Dict[str, Tuple[int, str]]], data: exp.RecordHelper):
	# Get timings & buffers
	timings = get_timings_from_network_record(record, Gs, Gs_monomorphism)
	buffer = get_graph_buffer(MCS, timings, nodes, extra_padding=0)
	seqs_step, updated_step = get_step_seqs_mapping(MCS, timings, buffer)
	num_eps, num_steps = next(iter(timings[0].values()))["run"].shape

	def _update_buffer(_data: jp.ndarray, _buffer: jp.ndarray) -> jp.ndarray:
		# todo: do not overwrite last entry
		if len(_buffer.shape) != len(_data.shape):
			raise ValueError(f"Warning: buffer shape {tuple(_buffer.shape)} does not match data shape {tuple(_data.shape)}")
		if _buffer.shape[1]-1 > _data.shape[1]:
			_buffer[:, :_data.shape[1]] = _data
		else:
			_buffer[:, :-1] = _data[:, :(_buffer.shape[1]-1)]
		return _buffer

	# Fill output buffer with data
	outputs = get_outputs_from_timings(MCS, timings, nodes, extra_padding=0)
	output_data = {k: d["outputs"] for k, d in data._data_stacked.items() if k in outputs}
	jax.tree_util.tree_map(_update_buffer, output_data, {k: v for k, v in outputs.items() if k in output_data})

	# Define initial sensor params (no noise)
	params = nodes["sensor"].default_params(jumpy.random.PRNGKey(0))
	params = params.replace(th_std=jp.float32(0.), th2_std=jp.float32(0.), thdot_std=jp.float32(0.), thdot2_std=jp.float32(0.))
	ss_sensor = StepState(rng=None, state=None, inputs=None, params=params)

	# Define initial estimator state
	def _fill_buffer(seqs, arr):
		exp_dims = [1] * (arr.ndim - 1)
		exp_seqs = jp.expand_dims(seqs, exp_dims)
		return rjp.take_along_axis(arr, exp_seqs, axis=0)

	world_states = []
	seqs_world = seqs_step["world"].max(axis=-1)
	for eps in range(num_eps):
		world_outputs = rjp.tree_take(outputs["world"], eps, axis=0)
		o = jax.tree_util.tree_map(partial(_fill_buffer, seqs_world[eps]), world_outputs)
		world_states.append(o)
	world_states = jax.tree_util.tree_map(lambda *x: jp.stack(x, axis=0), *world_states)

	# Define initial estimator params
	# num_steps = env._cgraph.max_steps + 1
	# world_state = nodes["world"].default_state(jumpy.random.PRNGKey(0))
	# world_states = jax.tree_map(lambda *x: jp.repeat(jp.stack(x)[None], num_eps, axis=0), *([world_state] * (num_steps+1)))
	p_est = EstimatorParams(world_states=world_states)
	ss_est = StepState(rng=None, state=None, inputs=None, params=p_est)

	# Define initial graph state
	ndict = dict(sensor=ss_sensor, estimator=ss_est)
	init_gs = GraphState(nodes=FrozenDict(ndict), timings=timings, buffer=buffer)
	init_gs, _, _ = env.reset(jumpy.random.PRNGKey(0), init_gs)

	return init_gs, outputs


def _init_progress(record: log_pb2.EpisodeRecord, nodes: Dict[str, Node], graph_state: GraphState, wseqs):
	record_est = [n for n in record.node if n.info.name == "estimator"][0]
	record_sensor = [n for n in record.node if n.info.name == "sensor"][0]
	record_actuator = [n for n in record.node if n.info.name == "actuator"][0]
	record_world = [n for n in record.node if n.info.name == "world"][0]

	# Get ts
	ts_actuator = jp.array([s.ts_output for s in record_actuator.steps])
	ts_sensor = jp.array([s.ts_step for s in record_sensor.steps])
	ts_estimator = jp.array([s.ts_step for s in record_est.steps])
	ts_world = jp.array(
		[record_world.steps[s].ts_step if s < len(record_world.steps) else record_world.steps[-1].ts_step for s in wseqs])

	# Plot initial params
	fig, ax, artists = visualize(graph_state.nodes["estimator"].params.world_states, nodes, ts_actuator, ts_sensor, ts_world)
	fig: plt.Figure

	def progress_fn(metrics, carry, scan_out):
		params, opt_state, _gs = carry

		# Plot
		_world_states = _gs.nodes["estimator"].params.world_states
		_, art_hidden = artists["cos_th"]
		art_hidden.set_offsets(jp.stack((ts_world, jp.cos(_world_states.th)), axis=1))
		_, art_hidden = artists["cos_th2"]
		art_hidden.set_offsets(jp.stack((ts_world, jp.cos(_world_states.th2)), axis=1))
		_, art_hidden = artists["thdot"]
		art_hidden.set_offsets(jp.stack((ts_world, _world_states.thdot), axis=1))
		_, art_hidden = artists["thdot2"]
		art_hidden.set_offsets(jp.stack((ts_world, _world_states.thdot2), axis=1))

		fig.canvas.draw()
		fig.canvas.flush_events()
		print(
			f"epoch {metrics.epoch} | step {metrics.step} | {metrics.fps:.2f} steps/sec | min(loss)= {scan_out.min():.3f} | loss: {scan_out.mean():.3f} +/- {scan_out.std():.3f} | max(loss)= {scan_out.max():.3f}")

	return progress_fn
