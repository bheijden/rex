import time
from typing import Dict
from functools import partial
import numpy as onp
import jax
import matplotlib.pyplot as plt

from rex.gmm_estimator import GMMEstimator
from rex.proto import log_pb2
from rex.open_colors import ecolor, fcolor
import rex.utils as utils
from rex.plot import get_subplots
from rex.wrappers import GymWrapper
from scripts.dummy import build_dummy_env


def test_gmm_estimator():
	# Create dummy graph
	env, nodes = build_dummy_env()

	# Apply wrapper
	env = GymWrapper(env)  # Wrap into gym wrapper
	env.seed(1)  # Set seed

	# Get spaces
	action_space = env.action_space

	# Run environment
	exp_record = log_pb2.ExperimentRecord()
	for _ in range(5):
		done, obs = False, env.reset()
		while not done:
			action = action_space.sample()
			obs, reward, truncated, done, info = env.step(action)
		env.stop()

		# Save record
		eps_record = log_pb2.EpisodeRecord()
		[eps_record.node.append(node.record()) for node in nodes.values()]
		exp_record.episode.append(eps_record)

	# Prepare data
	data, info = utils.get_delay_data(exp_record, concatenate=True)

	def init_estimator(x, i):
		name = i.name if not isinstance(i, tuple) else f"{i[0].name}.input({i[1].name})"
		est = GMMEstimator(x, name)
		return est

	# Initialize estimators
	est = jax.tree_map(lambda x, i: init_estimator(x, i), data, info)

	# Split
	est_inputs, est_step = est["inputs"], est["step"]
	data_inputs, data_step = data["inputs"], data["step"]
	info_inputs, info_step = info["inputs"], info["step"]

	# Prepare figures for plotting
	fig_step, axes_step = get_subplots(est_step, figsize=(10, 10), sharex=True, sharey=False, major="row")
	# fig_inputs, axes_inputs = get_subplots(est_inputs, figsize=(10, 10), sharex=True, sharey=False, major="row")

	# Plot step delays
	jax.tree_map(lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.computation, facecolor=fcolor.computation), axes_step,
	             est_step)
	# jax.tree_map(lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.communication, facecolor=fcolor.communication), axes_inputs,
	#              est_inputs)

	# Fit estimators
	jax.tree_map(lambda e: e.fit(num_steps=100, num_components=2, step_size=0.05, seed=1), est_step)
	# jax.tree_map(lambda e: e.fit(num_steps=100, num_components=2, step_size=0.05, seed=1), est_inputs)

	# Plot fitting results
	_, axes_loss_step = get_subplots(est_step, figsize=(10, 10), sharex=True, sharey=False, major="row")
	jax.tree_map(lambda ax, e: e.plot_loss(ax=ax), axes_loss_step, est_step)
	_, axes_weights_step = get_subplots(est_step, figsize=(10, 10), sharex=True, sharey=False, major="row")
	jax.tree_map(lambda ax, e: e.plot_normalized_weights(ax=ax), axes_weights_step, est_step)

	# Get distributions
	dist_step = jax.tree_map(lambda e: e.get_dist(), est_step)
	# dist_inputs = jax.tree_map(lambda e: e.get_dist(), est_inputs)

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
	# jax.tree_map(partial(plot_gmm, edgecolor=ecolor.communication), axes_inputs, dist_inputs, data_inputs, info_inputs)

	# Animate root training
	est_step: Dict[str, GMMEstimator]
	anim_step = est_step["agent"].animate_training(fig=fig_step, ax=axes_step["agent"], edgecolor=ecolor.computation, facecolor=fcolor.computation)
	anim_step.save("gmm_step.mp4")

	# est_inputs: Dict[str, Dict[str, GMMEstimator]]
	# anim_inputs = est_inputs["world"]["actuator"].animate_training(fig=fig_step, ax=axes_inputs["world"]["actuator"], edgecolor=ecolor.communication, facecolor=fcolor.communication)
	# anim_inputs.save("gmm_inputs.mp4")

	est_step["agent"].plot_hist()
	est_step["agent"].plot_loss()
	est_step["agent"].plot_normalized_weights()
	anim_agent = est_step["agent"].animate_training()

	# Wait a few seconds
	# plt.show(block=False)
	# time.sleep(3.0)
