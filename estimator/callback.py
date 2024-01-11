import flax.struct as struct
from typing import TypeVar, Any, Tuple, List, Dict, Optional, Union
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax.numpy as jnp
import numpy as onp
import time

from rex.supergraph import get_timings_after_root_split, get_chronological_timings
from rex.plot import get_subplots
from rex.open_colors import ewheel, fwheel
from rex.base import Empty
from estimator import Config, Carry, ScanOut, Metric


class BaseCallback:
	def on_training_start(self, config: Config) -> Metric:
		"""Called at the start of training. Is not jit-compiled."""
		return Empty()

	def on_epoch_start(self, config: Config, metric: Metric, carry: Carry) -> Metric:
		"""Called at the start of each epoch. Is not jit-compiled."""
		return metric

	# def on_rollout_start(self):
	# 	pass

	# def on_step(self):
	# 	pass

	# def on_event(self):
	# 	pass

	# def on_rollout_end(self):
	# 	pass

	def on_epoch_end(self, config: Config, metric: Metric, carry: Carry, scan_out: ScanOut) -> Metric:
		"""Called at the end of each epoch. Is not jit-compiled."""
		return metric

	def on_training_end(self, config: Config, metric: Metric, carry: Carry) -> Metric:
		"""Called at the end of training. Is not jit-compiled."""
		return metric


@struct.dataclass
class LogMetric:
	mask: jax.typing.ArrayLike
	ts_epoch_sec: jax.typing.ArrayLike
	ts_epoch_nsec: jax.typing.ArrayLike
	loss: jax.typing.ArrayLike
	fps: jax.typing.ArrayLike


class LogCallback(BaseCallback):
	def __init__(self, visualize: bool = True, fig_size: Tuple[float, float] = (8, 6)):
		self._visualize = visualize
		self._frames = []
		self._artists: Tuple[plt.Artist] = None
		self._fig = None
		self._ax = None
		self._fig_size = fig_size

	def _init_plot(self):
		import matplotlib.pyplot as plt
		last_frame = self._frames[-1]
		self._fig, self._ax = plt.subplots(1, 1, figsize=self._fig_size)
		self._ax.set_xlabel("Steps")
		self._ax.set_ylabel("Loss")
		self._ax.set_title("Training Loss")
		self._ax.set_xlim(0, last_frame.loss.shape[0])
		self._ax.set_ylim(max(1e-5, last_frame.loss.min()), max(1.1e-5, last_frame.loss.max()))
		self._ax.set_yscale("log")

		# Plot loss
		num_epochs = last_frame.mask.shape[0]
		num_steps = last_frame.loss.shape[0]
		num_steps_per_epoch = num_steps // num_epochs
		steps = onp.sum(last_frame.mask) * num_steps_per_epoch
		loss = last_frame.loss[:steps]

		art_loss_line, = self._ax.plot(loss, color=ewheel["blue"], label="loss")
		self._ax.legend()
		self._artists = (art_loss_line,)

	def _update_plot(self, frame: LogMetric):
		num_epochs = frame.mask.shape[0]
		num_steps = frame.loss.shape[0]
		num_steps_per_epoch = num_steps // num_epochs
		steps = onp.sum(frame.mask) * num_steps_per_epoch
		loss = frame.loss[:steps]

		art_loss_line = self._artists[0]
		art_loss_line.set_data(jnp.arange(0, loss.shape[0]), loss)

		return self._artists

	def on_training_start(self, config: Config) -> LogMetric:
		num_epochs = config.num_epochs
		num_steps = config.num_training_steps_per_epoch

		mask = onp.zeros((num_epochs,), dtype=bool)
		ts_epoch_sec = onp.zeros((num_epochs+1,), dtype=onp.int32)
		ts_epoch_nsec = onp.zeros((num_epochs+1,), dtype=onp.int32)
		fps = onp.zeros((num_epochs,), dtype=onp.float32)
		loss = onp.zeros((num_epochs*num_steps,), dtype=onp.float32)

		# Get time
		ts = time.time()
		ts_sec = int(ts)
		ts_nsec = int((ts - ts_sec) * 1e9)
		ts_epoch_sec[0] = ts_sec
		ts_epoch_nsec[0] = ts_nsec

		metric = LogMetric(mask, ts_epoch_sec, ts_epoch_nsec, loss, fps)
		self._frames.append(metric)

		if self._visualize and self._fig is None:
			self._init_plot()

		return metric

	def on_epoch_end(self, config: Config, metric: LogMetric, carry: Carry, scan_out: ScanOut) -> LogMetric:
		epoch = config.epoch
		num_steps = config.num_training_steps_per_epoch
		steps = (epoch+1) * num_steps

		# Determine epoch time
		ts_epoch = time.time()
		ts_prev = int(metric.ts_epoch_sec[epoch]) + int(metric.ts_epoch_nsec[epoch]) * 1e-9
		ts_epoch_sec = int(ts_epoch)
		ts_epoch_nsec = int((ts_epoch - ts_epoch_sec) * 1e9)

		# Determine steps per second
		_fps = num_steps / (ts_epoch - ts_prev)
		_loss, _params = scan_out

		# Update metric
		mask = metric.mask.at[epoch].set(True)
		ts_epoch_sec = metric.ts_epoch_sec.at[epoch+1].set(ts_epoch_sec)
		ts_epoch_nsec = metric.ts_epoch_nsec.at[epoch+1].set(ts_epoch_nsec)
		fps = metric.fps.at[epoch].set(_fps)
		loss = metric.loss.at[slice(epoch*num_steps, steps)].set(_loss)
		print(f"epoch {epoch} | step {steps} | {_fps:.2f} steps/sec | min(loss)= {_loss.min():.3f} | loss: {_loss.mean():.3f} +/- {_loss.std():.3f} | max(loss)= {_loss.max():.3f}")

		# Update plot
		metric = LogMetric(mask, ts_epoch_sec, ts_epoch_nsec, loss, fps)
		self._frames.append(metric)
		if self._visualize:
			self._update_plot(metric)
			self._ax.set_ylim(max(metric.loss.min(), 1e-9), max(metric.loss.max(), 1.1e-5))
			self._fig.canvas.draw()
			self._fig.canvas.flush_events()
		return metric

	def on_training_end(self, config: Config, metric: LogMetric, carry: Carry) -> LogMetric:
		ts_start = int(metric.ts_epoch_sec[0]) + int(metric.ts_epoch_nsec[0]) * 1e-9
		total_time = time.time() - ts_start
		train_fps = metric.fps.mean()
		total_steps = config.num_epochs * config.num_training_steps_per_epoch
		print(f"total_steps: {total_steps} | train_fps: {train_fps:.2f} steps/sec | total_time: {total_time:.2f} sec | ")
		return metric

	def get_animation(self, interval: int = 750):
		self._init_plot()
		init_func = lambda: self._artists
		ani = animation.FuncAnimation(self._fig, self._update_plot, init_func=init_func, frames=self._frames, interval=interval, blit=True)
		return ani


class StateFitCallback(BaseCallback):
	def __init__(self, visualize: bool = True, size: float = 3, fig_size: Tuple[float, float] = (12, 6), eps: int = 0):
		self._eps = eps
		self._visualize = visualize
		self._frames = []
		self._artists: Tuple[plt.Artist] = None
		self._art_dict: Dict[str, plt.Artist] = None
		self._fig = None
		self._fig_size = fig_size
		self._axes = None
		self._size = size
		self._record = None
		self._nodes = None
		self._ts_world = None

	def _init_plot(self):
		# Get ts
		num_sensor = self._ts_sensor.shape[0]
		num_actuator = self._ts_actuator.shape[0]

		# Create figure
		self._fig, self._axes = plt.subplots(nrows=5, ncols=1, figsize=self._fig_size)
		self._axes = self._axes.flatten()

		# Plot targets
		targets = jnp.stack((self._ts_sensor, self._nodes["sensor"]._outputs.cos_th[self._eps, :num_sensor]))
		_ = self._axes[0].scatter(*targets, color=ewheel["blue"], label=f"target", s=self._size)
		art_cos_th = self._axes[0].scatter(*([], []), color=ewheel["orange"], label=f"cos(th)", s=self._size)
		targets = jnp.stack((self._ts_sensor, self._nodes["sensor"]._outputs.cos_th2[self._eps, :num_sensor]))
		_ = self._axes[1].scatter(*targets, color=ewheel["blue"], label=f"target", s=self._size)
		art_cos_th2 = self._axes[1].scatter(*([], []), color=ewheel["orange"], label=f"cos(th2)", s=self._size)
		targets = jnp.stack((self._ts_sensor, self._nodes["sensor"]._outputs.thdot[self._eps, :num_sensor]))
		_ = self._axes[2].scatter(*targets, color=ewheel["blue"], label=f"target", s=self._size)
		art_thdot = self._axes[2].scatter(*([], []), color=ewheel["orange"], label=f"thdot", s=self._size)
		targets = jnp.stack((self._ts_sensor, self._nodes["sensor"]._outputs.thdot2[self._eps, :num_sensor]))
		_ = self._axes[3].scatter(*targets, color=ewheel["blue"], label=f"target", s=self._size)
		art_thdot2 = self._axes[3].scatter(*([], []), color=ewheel["orange"], label=f"thdot2", s=self._size)
		targets = jnp.stack((self._ts_actuator, self._nodes["actuator"]._outputs.action[self._eps, :num_actuator, 0]))

		art_action = self._axes[4].scatter(*targets, color=ewheel["blue"], label=f"action", s=self._size)

		# Add legend
		for i, ax in enumerate(self._axes):
			if i < (len(self._axes)-1):
				ax.xaxis.set_ticklabels([])
			ax.legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))

		# Prepare artists
		self._art_dict = dict(cos_th=art_cos_th, cos_th2=art_cos_th2, thdot=art_thdot, thdot2=art_thdot2, action=art_action)

	def _update_plot(self, frame: jax.typing.ArrayLike):
		artists = self._art_dict
		ws = frame
		art_hidden = artists["cos_th"]
		art_hidden.set_offsets(jnp.stack((self._ts_world, jnp.cos(ws.th[self._eps])), axis=1))
		art_hidden = artists["cos_th2"]
		art_hidden.set_offsets(jnp.stack((self._ts_world, jnp.cos(ws.th2[self._eps])), axis=1))
		art_hidden = artists["thdot"]
		art_hidden.set_offsets(jnp.stack((self._ts_world, ws.thdot[self._eps]), axis=1))
		art_hidden = artists["thdot2"]
		art_hidden.set_offsets(jnp.stack((self._ts_world, ws.thdot2[self._eps]), axis=1))
		return (art for art in self._art_dict.values())

	def on_training_start(self, config: Config) -> Metric:
		self._nodes = config.env.unwrapped.graph.nodes_and_root
		# NOTE: we add [0] to indicate the initial state (before step [0]).
		timings = config.graph_state.timings
		timings_root = get_timings_after_root_split(config.env.graph.S, timings)
		self._ts_world = onp.concatenate(([0], timings_root["world"]["ts_step"][self._eps]))
		timings_chron = get_chronological_timings(config.env.graph.S, timings, self._eps)
		self._ts_actuator = timings_chron["actuator"]["ts_step"]
		self._ts_sensor = timings_chron["sensor"]["ts_step"]

		# prepare initial frame
		params = config.params
		graph_state = config.graph_state
		if params.get("estimator") is not None:
			world_states = jax.tree_util.tree_map(lambda x, y: x if y is None else y, graph_state.nodes["estimator"].params,
			                                    params.get("estimator")).world_states
		else:
			world_states = graph_state.nodes["estimator"].params.world_states
		self._frames.append(world_states)

		# Visualize
		if self._visualize and self._fig is None:
			self._init_plot()
			self._update_plot(world_states)
			self._fig.canvas.draw()
			self._fig.canvas.flush_events()

		return Empty()

	def on_epoch_end(self, config: Config, metric: Metric, carry: Carry, scan_out: ScanOut) -> Metric:
		metrics, params, opt_state, outputs = carry

		# Update plot
		if "estimator" in params:
			world_states = params["estimator"].world_states
		else:
			world_states = config.graph_state.nodes["estimator"].params.world_states
		self._frames.append(world_states)

		# Visualize
		if self._visualize:
			self._update_plot(world_states)
			self._fig.canvas.draw()
			self._fig.canvas.flush_events()
		return metric

	def get_animation(self, interval: int = 750):
		self._init_plot()
		init_func = lambda: (art for art in self._art_dict.values())
		ani = animation.FuncAnimation(self._fig, self._update_plot, init_func=init_func, frames=self._frames, interval=interval, blit=True)
		return ani


@struct.dataclass
class ParamFitMetric:
	mask: jax.typing.ArrayLike
	params: Any


class _NoValue: pass


class ParamFitCallback(BaseCallback):
	def __init__(self, targets: Any = None, visualize: bool = True, fig_size: Tuple[float, float] = (8, 6)):
		self._visualize = visualize
		self._frames = []
		self._artists: Tuple[plt.Artist] = None
		self._art_tree: Dict[str, plt.Artist] = None
		self._fig = None
		self._axes_tree = None
		self._fig_size = fig_size
		self._targets = targets

	def _init_plot(self):
		last_frame = self._frames[-1]
		self._fig, self._axes_tree = get_subplots(last_frame.params, self._fig_size)

		# Plot targets
		num_epochs = last_frame.mask.shape[0]
		num_steps = jax.tree_util.tree_leaves(last_frame.params)[0].shape[0]
		num_steps_per_epoch = num_steps // num_epochs
		steps = onp.sum(last_frame.mask) * num_steps_per_epoch + 1  # NOTE: +1 to include initial params
		params = jax.tree_util.tree_map(lambda p: p[:steps], last_frame.params)

		# Prepare labels
		_, pytree_def = jax.tree_util.tree_flatten(params)
		labels_flat = [key for key, val in params.__dict__.items() if val is not None]
		labels = jax.tree_util.tree_unflatten(pytree_def, labels_flat)

		# Plot
		def plot_targets(ax, t):
			ax.set_xlim(0, num_steps+1)
			if not isinstance(t, _NoValue):
				x = jnp.arange(0, num_steps)
				art_line, = ax.plot(x, onp.ones_like(x)*t, color=ewheel["grape"], label="hidden")
				return art_line

		def plot_preds(ax, p, label):
			x = jnp.arange(0, steps)
			art_line, = ax.plot(x, onp.ones_like(x)*p, color=ewheel["orange"], label=label)
			return art_line

		self._targets = jax.tree_util.tree_map(lambda p, t: t if t is not None else _NoValue(), params, self._targets)
		_art_targets = jax.tree_util.tree_map(lambda ax, t: plot_targets(ax, t), self._axes_tree, self._targets)
		art_preds = jax.tree_util.tree_map(lambda ax, p, l: plot_preds(ax, p, l), self._axes_tree, params, labels)

		# Turn on legends
		jax.tree_util.tree_map(lambda ax: ax.legend(ncol=1, fancybox=True, shadow=False), self._axes_tree)

		# Prepare artists
		self._art_tree = art_preds

	def _update_plot(self, frame: ParamFitMetric):
		num_epochs = frame.mask.shape[0]
		num_steps = jax.tree_util.tree_leaves(frame.params)[0].shape[0]
		num_steps_per_epoch = num_steps // num_epochs
		steps = onp.sum(frame.mask) * num_steps_per_epoch + 1  # NOTE: +1 to include initial params
		params = jax.tree_util.tree_map(lambda p: p[:steps], frame.params)

		def update_preds(art, p, ):
			x = jnp.arange(0, steps)
			art.set_data(x, p)
			return art

		jax.tree_util.tree_map(lambda art, p: update_preds(art, p), self._art_tree, params)
		return tuple(jax.tree_util.tree_leaves(self._art_tree))

	def on_training_start(self, config: Config) -> ParamFitMetric:
		# Prepare initial frame
		num_epochs = config.num_epochs
		num_steps = config.num_training_steps_per_epoch
		mask = onp.zeros((num_epochs,), dtype=bool)
		params = jax.tree_util.tree_map(lambda x: onp.zeros((num_epochs*num_steps+1,), dtype=onp.float32), config.params.get("world", None))
		params = jax.tree_util.tree_map(lambda x, y: x.at[0].set(y), params, config.params.get("world", None))
		metric = ParamFitMetric(mask=mask, params=params)

		self._frames.append(metric)

		# Visualize
		if self._visualize and self._fig is None:
			self._init_plot()
			self._update_plot(metric)
			self._fig.canvas.draw()
			self._fig.canvas.flush_events()

		return metric

	def on_epoch_end(self, config: Config, metric: ParamFitMetric, carry: Carry, scan_out: ScanOut) -> ParamFitMetric:
		metrics, params, opt_state, outputs = carry
		epoch = config.epoch
		num_steps = config.num_training_steps_per_epoch
		steps = (epoch+1) * num_steps + 1

		_loss, _params = scan_out
		params_epoch = _params.get("world", None)
		params = jax.tree_util.tree_map(lambda x, y: x.at[epoch*num_steps+1:steps].set(y), metric.params, params_epoch)
		mask = metric.mask.at[epoch].set(True)

		# Update Metric
		metric = ParamFitMetric(mask, params)

		# Update plot
		self._frames.append(metric)

		# Visualize
		if self._visualize:
			self._update_plot(metric)

			def update_ylim(ax, p, t):
				if not isinstance(t, _NoValue):
					ax.set_ylim(min(t, p.min()), max(t, p.max()))
				else:
					ax.set_ylim(p.min(), p.max())

			jax.tree_map(update_ylim, self._axes_tree, metric.params, self._targets)
			self._fig.canvas.draw()
			self._fig.canvas.flush_events()
		return metric

	def get_animation(self, interval: int = 750):
		self._init_plot()
		init_func = lambda: tuple(jax.tree_util.tree_leaves(self._art_tree))
		ani = animation.FuncAnimation(self._fig, self._update_plot, init_func=init_func, frames=self._frames, interval=interval, blit=True)
		return ani