import pytest
from concurrent.futures.process import BrokenProcessPool
import os
import jumpy
import jax
import rex.jumpy as rjp
import jumpy.numpy as jp
import numpy as onp

from rex.base import StepState
from rex.multiprocessing import new_process
from rex.constants import WARN, ERROR
from rex.utils import log
from rex.base import StepState, Output
from scripts.dummy import build_dummy_env


# Create initialize function that updates the pid (as an example)
def initializer(fn, raise_error_in_initializer: bool = False, raise_error_in_step: bool = False):
	fn.__self__.pid = os.getpid()  # Sets the PID of the worker process
	fn.__self__.raise_error_in_step = raise_error_in_step

	# Raise error if desired
	if raise_error_in_initializer:
		raise ValueError("Trigger error in initializer for testing.")


# Define dummy class
class Node:
	def __init__(self):
		self.pid = os.getpid()  # Sets the PID of the main process
		self.raise_error_in_step = False

	def step(self, pytree_in):
		assert self.pid == os.getpid(), "Process ID changed"
		log("worker", "blue", WARN, "worker", f"pid={self.pid}")
		pytree_out = jax.tree_map(lambda x: x + 1, pytree_in)
		if self.raise_error_in_step:
			raise ValueError("Trigger error in step for testing.")
		return pytree_out


@pytest.mark.parametrize("backend, jit", [("numpy", False), ("jax", False), ("jax", True)])
def test_mp(backend, jit):
	# Log PID of mainprocess
	log(name="mainprocess", color="red", log_level=WARN, id="main_thread", msg=f"START")

	# Create dummy class instance
	n = Node()

	# Wrap the step function into a new process
	n.step = new_process(n.step, max_workers=2, initializer=initializer, initargs=(False,))

	# Optionally jit the step function
	n.step = jax.jit(n.step) if jit else n.step

	# Steps
	with rjp.use(backend=backend):
		for i in range(5):
			pytree_in = [jp.array([1, 2, 3]) + i, jp.array([4, 5, 6]) + i]
			pytree_out = n.step(pytree_in)
			pytree_equal = jax.tree_map(lambda x, y: onp.isclose(x + 1, y).all(), pytree_in, pytree_out)
			assert all(pytree_equal), "Pytree not equal"


@pytest.mark.parametrize(
    "raise_error_in_initializer, raise_error_in_step, exception_type",
    [(True, False, BrokenProcessPool), (False, True, ValueError)],
)
def test_mp_error(raise_error_in_initializer, raise_error_in_step, exception_type):
	# Log PID of mainprocess
	log(name="mainprocess", color="red", log_level=WARN, id="main_thread", msg=f"START")

	# Create dummy class instance
	n = Node()

	# Wrap the step function into a new process
	n.step = new_process(n.step, max_workers=2, initializer=initializer, initargs=(raise_error_in_initializer, raise_error_in_step))

	# Try to step
	pytree_in = [jp.array([1, 2, 3]), jp.array([4, 5, 6])]
	with pytest.raises(exception_type):
		pytree_out = n.step(pytree_in)


def test_mp_with_rex_nodes():
	# Log PID of mainprocess
	log(name="mainprocess", color="red", log_level=WARN, id="main_thread", msg=f"START")

	# Grab the dummy environment
	env, nodes = build_dummy_env()
	sensor = nodes["sensor"]

	# Grab initial step state
	graph_state, obs, info = env.reset(jumpy.random.PRNGKey(0))
	env.stop()
	step_state = graph_state.nodes[sensor.name]

	# Wrap the step function into a new process
	sensor.step = new_process(sensor.step, max_workers=2, initializer=initializer, initargs=(False,))

	# Step
	step_state = sensor.step(step_state)
