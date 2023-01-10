from typing import Any, Dict, Tuple, Union
import jumpy as jp
import numpy as onp
from math import ceil
from flax import struct
from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST
from rex.base import StepState, GraphState
from rex.node import Node

from envs.pendulum.env import Output as ActuatorOutput


def build_pendulum(rate: Dict[str, float] = None,
                   process: Dict[str, float] = None,
                   process_sim: Dict[str, Distribution] = None,
                   trans: Dict[str, float] = None,
                   trans_sim: Dict[str, Distribution] = None,
                   log_level: Dict[str, int] = None
                   ) -> Dict[str, Node]:
	rate = rate or {}
	process = process or {}
	process_sim = process_sim or {}
	trans = trans or {}
	trans_sim = trans_sim or {}
	log_level = log_level or {}

	# Fill in default values
	names = ["world", "actuator", "sensor"]
	for name in names:
		rate[name] = rate.get(name, rate.get("world", 30.0))
		process[name] = process.get(name, None)
		process_sim[name] = process_sim.get(name, Gaussian(mean=0., var=0.))
		log_level[name] = log_level.get(name, WARN)
		if name in ["actuator", "sensor"]:
			trans[name] = trans.get(name, None)
			trans_sim[name] = trans_sim.get(name, Gaussian(mean=0., var=0.))

	# Create nodes
	world = World(name="world", rate=rate["world"], delay=process["world"], delay_sim=process_sim["world"], log_level=log_level["world"], color="blue")
	actuator = Actuator(name="actuator", rate=rate["actuator"], delay=process["actuator"], delay_sim=process_sim["actuator"], log_level=log_level["actuator"], color="green")
	sensor = Sensor(name="sensor", rate=rate["sensor"], delay=process["sensor"], delay_sim=process_sim["sensor"], log_level=log_level["sensor"], color="yellow")

	# Connect nodes
	world.connect(actuator, window=1, blocking=False, skip=False, delay_sim=trans_sim["actuator"], delay=trans["actuator"], jitter=LATEST)
	sensor.connect(world, window=1, blocking=False, skip=True, delay_sim=trans_sim["sensor"], delay=trans["sensor"], jitter=LATEST)
	return dict(world=world, actuator=actuator, sensor=sensor)


@struct.dataclass
class State:
	"""Pendulum ode state definition"""

	th: jp.float32
	thdot: jp.float32


@struct.dataclass
class Output:
	"""Pendulum ode output definition"""

	th: jp.float32
	thdot: jp.float32


@struct.dataclass
class Empty: pass


class World(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default params of the node."""
		# Try to grab params from graph_state
		return Empty()

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default state of the node."""
		return Empty()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default output of the node."""
		# Grab output from state
		return Empty()

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jp.random_split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

	def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Empty]:
		"""Step the node."""
		# Prepare output
		new_step_state = step_state
		return new_step_state, Empty()


class Sensor(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default params of the node."""
		return Empty()

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default state of the node."""
		return Empty()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the node."""
		# Grab output from state
		return Output(th=jp.float32(0.), thdot=jp.float32(0.))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jp.random_split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

	def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""
		# Update state
		new_step_state = step_state

		# Prepare output
		# todo: Call ROS service to receive th, thdot from mops
		th, thdot = 0., 0.
		output = Output(th=jp.float32(th), thdot=jp.float32(thdot))
		return new_step_state, output


class Actuator(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default params of the node."""
		return Empty()

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		"""Default state of the node."""
		return Empty()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the node."""
		return ActuatorOutput(action=jp.array([0.0], dtype=jp.float32))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jp.random_split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

	def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Update state
		new_step_state = step_state

		# Prepare output
		output: ActuatorOutput = list(inputs.values())[0][-1].data

		# todo: Call ROS service to send action to mops
		action = output.action[0]
		return new_step_state, output