from estimator.env import Estimator, Replay, ReconstructionLoss, EstimatorParams, EstimatorEnv

from functools import partial
from math import ceil
from typing import Tuple, Dict, Callable, Union, Any, Sequence, List, TypeVar, TYPE_CHECKING
import flax.struct as struct
import optax
import jax
import jumpy
import jax.numpy as jnp
import jumpy.numpy as jp

from rex.proto import log_pb2
from rex.base import GraphState, Empty
from rex.env import BaseEnv
import rex.utils as utils
import rex.jumpy as rjp

if TYPE_CHECKING:
	from estimator.callback import BaseCallback


def single_loss(env: BaseEnv, params: optax.Params, graph_state: GraphState, rng: jp.array, starting_step: jp.int32, num_steps: jp.int32) -> Tuple[jp.float32, GraphState]:
	# Overwrite params in graph state with optimizer params
	new_nodes = {}
	for node_name, params in params.items():
		if params is None:
			continue
		new_params = jax.tree_util.tree_map(lambda x, y: x if y is None else y, graph_state.nodes[node_name].params, params)
		new_nodes[node_name] = graph_state.nodes[node_name].replace(params=new_params)

	# Update graph state
	gs = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes), step=starting_step)

	# Reset env
	gs, loss_init = env.reset(rng, gs)

	# Save initial graph state for preparing outputs
	# todo: determine mask that identifies which outputs have been updated after reset.

	# Perform steps
	def _step(carry, x):
		gs, loss_step, loss = carry
		new_gs, new_loss_step, _, done, info = env.step(gs, loss_step)
		# Only add dynamic loss to total loss
		new_loss = (1-done)*(new_loss_step.loss) + loss
		new_carry = (new_gs, new_loss_step, new_loss)
		return new_carry, loss

	carry_out, scan_out = jumpy.lax.scan(_step, (gs, loss_init, loss_init.loss), None, length=num_steps)  # todo: unroll?
	loss = scan_out[-1]

	return loss, gs


Metric = TypeVar('Metric')
Metrics = Dict[str, Metric]
Carry = Tuple[Metrics, Dict, Dict, GraphState]
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


def fit(env: BaseEnv, params: optax.Params, optimizer: optax.GradientTransformation, graph_state: GraphState, num_batches: int = 20, num_steps: int = 3, num_training_steps: int = 1000,
        num_training_steps_per_epoch: int = 100, prior_fn: Callable[[optax.Params], Any] = lambda x: jp.float32(0),
        callbacks: Dict[str, "BaseCallback"] = None) -> optax.Params:
	max_num_steps = env._cgraph.max_steps + 1  # todo: why + 1?
	batch_loss = jax.vmap(partial(single_loss, env), in_axes=(None, None, 0, 0, None), out_axes=0)

	def loss(params: optax.Params, _gs, _rng: jp.ndarray) -> Tuple[jnp.ndarray, GraphState]:
		_rngs = jax.random.split(_rng, num_batches+1)
		_rng, _rngs = _rngs[0], _rngs[1:]
		starting_steps = jax.random.choice(_rng, max_num_steps, shape=(num_batches,), replace=False)

		# Perform batch rollout
		loss_batch, gs_batch = batch_loss(params, _gs, _rngs, starting_steps, jp.int32(num_steps))

		# Add prior loss
		ploss = prior_fn(params)
		ploss = jax.tree_util.tree_reduce(lambda acc, l: acc + jp.sum(l), ploss, jp.float32(0.))
		# todo: determine mask that identifies which outputs have been updated
		return loss_batch.mean() + ploss, jax.tree_util.tree_map(lambda x: x[0], gs_batch)

	@jax.jit
	def train_step(carry: Carry, _rng):
		metrics, params, opt_state, _gs = carry
		(loss_value, _gs), grads = jax.value_and_grad(loss, has_aux=True)(params, _gs, _rng)
		updates, opt_state = optimizer.update(grads, opt_state, params)
		params = optax.apply_updates(params, updates)
		# Make all params positive
		params["world"] = jax.tree_util.tree_map(lambda x: jp.abs(x), params["world"])
		carry = (metrics, params, opt_state, _gs)
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
		carry = (metrics, params, opt_state, graph_state)
		metrics = {cb_name: cb.on_epoch_start(config, metrics[cb_name], carry) for cb_name, cb in callbacks.items()}
		carry = (metrics, params, opt_state, graph_state)

		# Run epoch
		rngs_step = jumpy.random.split(rng, num=num_training_steps_per_epoch)
		carry, scan_out = jax.lax.scan(train_step, carry, rngs_step, length=num_training_steps_per_epoch, unroll=1)

		# Update metrics
		metrics, params, opt_state, graph_state = carry
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
from rex.agent import Agent
import dill as pickle
import experiments as exp
import matplotlib.pyplot as plt


def build_estimator(record: log_pb2.EpisodeRecord, rate: float) -> Tuple[log_pb2.EpisodeRecord, Dict[str, Node], List[str]]:
	# Copy episode record
	# todo: does this truly deepcopy?
	record.CopyFrom(record)

	# Re-initialize nodes
	nodes: Dict[str, Union[Node, Agent]] = {}
	for n in record.node:
		nodes[n.info.name] = pickle.loads(n.info.state)
	[n.unpickle(nodes) for n in nodes.values()]

	# Initialize estimator
	estimator = Estimator("estimator", rate=rate, delay=0.)
	estimator.connect(nodes["sensor"], window=1, blocking=False, delay=0., skip=True)  # todo: skip to avoid phase shift.
	nodes["estimator"] = estimator

	# # Get data
	record_exp = log_pb2.ExperimentRecord(episode=[record])
	data = exp.RecordHelper(record_exp)

	# Define nodes for which we have recorded outputs
	excludes_inputs = ["actuator", "agent"]
	for name in excludes_inputs:
		outputs = data._data[0][name]["outputs"].tree
		states = data._data[0][name]["states"].tree
		params = data._data[0][name]["params"].tree
		nodes[name] = Replay(nodes[name], outputs, states, params)

	# Wrap sensors to calculate reconstruction loss
	outputs = data._data[0]["sensor"]["outputs"].tree
	nodes["sensor"] = ReconstructionLoss(nodes["sensor"], outputs)

	# Recreate estimator record
	record_sensor = [n for n in record.node if n.info.name == "sensor"][0]

	# Add estimator record to episode record
	record_est = get_estimator_record(estimator, [record_sensor])
	record.node.extend([record_est])
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


def _init(env: BaseEnv, nodes: Dict[str, Node]):
	# Get jit functions
	jit_reset = env.reset
	jit_step = env.step

	# Define initial params
	from rex.base import StepState

	# Define initial sensor params
	params = nodes["sensor"].default_params(jumpy.random.PRNGKey(0))
	params = params.replace(th_std=jp.float32(0.), th2_std=jp.float32(0.), thdot_std=jp.float32(0.), thdot2_std=jp.float32(0.))
	ss_sensor = StepState(rng=None, state=None, inputs=None, params=params)

	# Define initial estimator params
	# record_est = [n for n in env._cgraph.trace.episode.node if n.info.name == "estimator"][0]
	max_num_steps = env._cgraph.max_steps + 1
	world_state = nodes["world"].default_state(jumpy.random.PRNGKey(0))
	world_states = jax.tree_map(lambda *x: jp.stack(x), *([world_state] * (max_num_steps+1)))
	p_est = EstimatorParams(world_states=world_states)
	ss_est = StepState(rng=None, state=None, inputs=None, params=p_est)

	# Define initial graph state
	from rex.base import GraphState
	from flax.core import FrozenDict
	ndict = dict(sensor=ss_sensor, estimator=ss_est)
	default_outputs = env._cgraph._default_outputs
	init_gs = GraphState(nodes=FrozenDict(ndict), step=jp.int32(0), outputs=FrozenDict(default_outputs))
	init_gs, _ = jit_reset(jumpy.random.PRNGKey(0), init_gs)

	return init_gs


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
