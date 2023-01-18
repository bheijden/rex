from typing import Dict, List, Tuple, Union
from functools import partial
import matplotlib.pyplot as plt
from typing import List
import numpy as onp
import jax
import jumpy.numpy as jp

from rex.proto import log_pb2
from rex.open_colors import ecolor, fcolor
from rex.distributions import GMM
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, SEQUENTIAL, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK

from rex.gmm_estimator import GMMEstimator


def get_subplots(tree, figsize=(10, 10), sharex=False, sharey=False, major="row"):
	_, treedef = jax.tree_util.tree_flatten(tree)
	num = treedef.num_leaves
	nrows, ncols = onp.ceil(onp.sqrt(num)).astype(int), onp.ceil(onp.sqrt(num)).astype(int)
	if nrows * (ncols - 1) >= num:
		if major == "row":
			ncols -= 1
		else:
			nrows -= 1
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
	tree_axes = jax.tree_util.tree_unflatten(treedef, axes.flatten()[0:num].tolist())
	if len(axes.flatten()) > num:
		for ax in axes.flatten()[num:]:
			ax.remove()
	return fig, tree_axes


def get_delay_data(record: log_pb2.ExperimentRecord):
	get_step_delay = lambda s: s.delay  # todo: use comp_delay?
	get_input_delay = lambda m: m.delay  # todo: use comm_delay?

	exp_data, exp_info = [], []
	for e in record.episode:
		data, info = dict(inputs=dict(), step=dict()), dict(inputs=dict(), step=dict())
		exp_data.append(data), exp_info.append(info)
		for n in e.node:
			node_name = n.info.name
			# Fill info tree
			info["inputs"][node_name] = dict()
			info["step"][node_name] = n.info
			for i in n.inputs:
				input_name = i.info.name
				info["inputs"][node_name][input_name] = (n.info, i.info)

			# Fill data tree
			delays = [get_step_delay(s) for s in n.steps]
			data["step"][node_name] = onp.array(delays)
			data["inputs"][node_name] = dict()
			for i in n.inputs:
				input_name = i.info.name
				delays = [get_input_delay(m) for g in i.grouped for m in g.messages]
				data["inputs"][node_name][input_name] = onp.array(delays)
	data = jax.tree_map(lambda *x: onp.concatenate(x, axis=0), *exp_data)
	info = jax.tree_map(lambda *x: x[0], *exp_info)
	return data, info


if __name__ == "__main__":
	import seaborn as sns

	sns.set()

	log_dir = "/home/r2ci/rex/logs"

	# Load episode record
	exp_record = log_pb2.ExperimentRecord()
	with open(f"{log_dir}/sac_pendulum.pb", "rb") as f:
		exp_record.ParseFromString(f.read())

	# Prepare data
	data, info = get_delay_data(exp_record)

	# Initialize estimators
	est = jax.tree_map(lambda x: GMMEstimator(x), data)

	# Split
	est_inputs, est_step = est["inputs"], est["step"]
	data_inputs, data_step = data["inputs"], data["step"]
	info_inputs, info_step = info["inputs"], info["step"]

	# Prepare figures for plotting
	fig_step, axes_step = get_subplots(est_step, figsize=(10, 10), sharex=True, sharey=False, major="row")
	fig_inputs, axes_inputs = get_subplots(est_inputs, figsize=(10, 10), sharex=True, sharey=False, major="row")

	# Plot step delays
	jax.tree_map(lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.computation, facecolor=fcolor.computation), axes_step,
	             est_step)
	jax.tree_map(lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.communication, facecolor=fcolor.communication), axes_inputs,
	             est_inputs)

	# Fit estimators
	jax.tree_map(lambda e: e.fit(num_steps=100, num_components=2, step_size=0.05, seed=1), est_step)
	jax.tree_map(lambda e: e.fit(num_steps=100, num_components=2, step_size=0.05, seed=1), est_inputs)

	# Get distributions
	dist_step = jax.tree_map(lambda e: e.get_dist(), est_step)
	dist_inputs = jax.tree_map(lambda e: e.get_dist(), est_inputs)

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

	jax.tree_map(partial(plot_gmm, edgecolor=ecolor.computation), axes_step, dist_step, data_step, info_step)
	jax.tree_map(partial(plot_gmm, edgecolor=ecolor.communication), axes_inputs, dist_inputs, data_inputs, info_inputs)

	# Animate agent training
	est_step: Dict[str, GMMEstimator]
	anim_step = est_step["agent"].animate_training(fig=fig_step, ax=axes_step["agent"], edgecolor=ecolor.computation, facecolor=fcolor.computation)
	anim_step.save("gmm_step.mp4")

	est_inputs: Dict[str, Dict[str, GMMEstimator]]
	anim_inputs = est_inputs["world"]["actuator"].animate_training(fig=fig_step, ax=axes_inputs["world"]["actuator"], edgecolor=ecolor.communication, facecolor=fcolor.communication)
	anim_inputs.save("gmm_inputs.mp4")

	plt.show()
