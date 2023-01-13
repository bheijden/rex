from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
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
	actuator = Actuator(name="actuator", rate=rate["actuator"], delay=process["actuator"], delay_sim=process_sim["actuator"], log_level=log_level["actuator"], color="green", advance=False)
	sensor = Sensor(name="sensor", rate=rate["sensor"], delay=process["sensor"], delay_sim=process_sim["sensor"], log_level=log_level["sensor"], color="yellow")

	# Connect nodes
	world.connect(actuator, window=1, blocking=False, skip=False, delay_sim=trans_sim["actuator"], delay=trans["actuator"], jitter=LATEST)
	sensor.connect(world, window=1, blocking=False, skip=True, delay_sim=trans_sim["sensor"], delay=trans["sensor"], jitter=LATEST)
	return dict(world=world, actuator=actuator, sensor=sensor)


@struct.dataclass
class Params:
	"""Pendulum ode param definition"""
	max_speed: jp.float32
	J: jp.float32
	mass: jp.float32
	length: jp.float32
	b: jp.float32
	K: jp.float32
	R: jp.float32
	c: jp.float32
	d: jp.float32


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


def runge_kutta4(ode, dt, params, x, u):
	k1 = ode(params, x, u)
	k2 = ode(params, x + 0.5 * dt * k1, u)
	k3 = ode(params, x + 0.5 * dt * k2, u)
	k4 = ode(params, x + dt * k3, u)
	return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def ode_pendulum(params: Params, x, u):
	g, J, m, l, b, K, R, c, d = 9.81, params.J, params.mass, params.length, params.b, params.K, params.R, params.c, params.d

	ddx = (u * K / R + m * g * l * jp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * (2 * sigmoid(d * x[1]) - 1)) / J

	return jp.array([x[1], ddx])


def sigmoid(x):
	pos = 1.0 / (1.0 + jp.exp(-x))
	neg = jp.exp(x) / (1.0 + jp.exp(x))
	return jp.where(x >= 0, pos, neg)


class World(Node):
	def __init__(self, *args, dt_ode: float = 1/100, **kwargs):
		super().__init__(*args, **kwargs)
		dt = 1/self.rate
		self.substeps = ceil(dt / dt_ode)
		self.dt_ode = dt / self.substeps

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Params:
		"""Default params of the node."""
		# Try to grab params from graph_state
		try:
			params = graph_state.nodes[self.name].params
			if params is not None:
				return params
		except (AttributeError, KeyError):
			pass
		return Params(max_speed=jp.float32(22.0),
		              J=jp.float32(0.000159931461600856),
		              mass=jp.float32(0.0508581731919534),
		              length=jp.float32(0.0415233722862552),
		              b=jp.float32(1.43298488358436e-05),
		              K=jp.float32(0.0333391179016334),
		              R=jp.float32(7.73125142447252),
		              c=jp.float32(0.000975041213361349),
		              d=jp.float32(165.417960777425))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
		"""Default state of the node."""
		# Try to grab state from graph_state
		try:
			state = graph_state.nodes[self.name].state
			if state is not None:
				return state
		except (AttributeError, KeyError):
			pass
		# Else, return default state
		rng_th, rng_thdot = jumpy.random.split(rng, num=2)
		th = jumpy.random.uniform(rng_th, shape=(), low=-3.14, high=3.14)
		thdot = 0.  # jumpy.random.uniform(rng_thdot, shape=(), low=-9., high=9.)
		return State(th=th, thdot=thdot)

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the node."""
		# Grab output from state
		try:
			th = graph_state.nodes["world"].state.th
			thdot = graph_state.nodes["world"].state.thdot
		except (AttributeError):
			th = jp.float32(0.)
			thdot = jp.float32(0.)
		return Output(th=th, thdot=thdot)

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

	def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Get action
		u = list(inputs.values())[0].data.action[-1][0]
		x = jp.array([state.th, state.thdot])
		next_x = x

		# Calculate next state
		for _ in range(self.substeps):
			next_x = runge_kutta4(ode_pendulum, self.dt_ode, params, next_x, u)

		# Update state
		next_th, next_thdot = next_x
		new_state = state.replace(th=next_th, thdot=jp.clip(next_thdot, -params.max_speed, params.max_speed))
		new_step_state = step_state.replace(state=new_state)

		# Prepare output
		output = Output(th=next_th, thdot=next_thdot)
		# print(f"{self.name.ljust(14)} | x: {x} | u: {u} -> next_x: {next_x}")
		return new_step_state, output


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
		try:
			th = graph_state.nodes["world"].state.th
			thdot = graph_state.nodes["world"].state.thdot
		except (AttributeError):
			th = jp.float32(0.)
			thdot = jp.float32(0.)
		return Output(th=th, thdot=thdot)

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
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
		output = inputs["world"][-1].data
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
		rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
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
		output = list(inputs.values())[0][-1].data
		return new_step_state, output