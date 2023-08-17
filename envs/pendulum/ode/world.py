from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
from math import ceil
from flax import struct
from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.multiprocessing import new_process

from envs.pendulum.env import Output as ActuatorOutput
from envs.pendulum.render import Render


def build_pendulum(rates: Dict[str, float],
                   delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
                   delays: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                   scheduling: int = PHASE,
                   advance: bool = False,
                   ) -> Dict[str, Node]:

	# Prepare delays
	process_sim = delays_sim["step"]
	process = delays["step"]
	trans_sim = delays_sim["inputs"]
	trans = delays["inputs"]

	# Create nodes
	world = World(name="world", rate=rates["world"], scheduling=scheduling,
	              delay=process["world"], delay_sim=process_sim["world"])
	actuator = Actuator(name="actuator", rate=rates["actuator"], scheduling=scheduling, advance=advance,
	                    delay=process["actuator"], delay_sim=process_sim["actuator"])
	sensor = Sensor(name="sensor", rate=rates["sensor"], scheduling=scheduling,
	                delay=process["sensor"], delay_sim=process_sim["sensor"])
	render = Render(name="render", rate=rates["render"], scheduling=scheduling,
	                delay=process["render"], delay_sim=process_sim["render"])

	# Connect nodes
	world.connect(actuator, window=1, blocking=False, skip=True, jitter=LATEST,
	              delay_sim=trans_sim["world"]["actuator"], delay=trans["world"]["actuator"])
	sensor.connect(world, window=1, blocking=False, skip=False, jitter=LATEST,  # todo: CHANGED skip=False
	               delay_sim=trans_sim["sensor"]["world"], delay=trans["sensor"]["world"])
	render.connect(sensor, window=1, blocking=False, skip=False, jitter=LATEST,
	               delay_sim=trans_sim["render"]["sensor"], delay=trans["render"]["sensor"])

	# render.connect(actuator, window=1, blocking=False, skip=False, delay_sim=Gaussian(mean=0., std=0.), delay=0.0, jitter=LATEST)

	render.step = new_process(render.step)  # todo: same process ?

	return dict(world=world, actuator=actuator, sensor=sensor, render=render)


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
	# For very large negative values of x, the sigmoid is very close to 0.
	# For very large positive values of x, the sigmoid is very close to 1.
	# We can use these properties to avoid computing the exponential in these cases.
	large_positive = x > 35
	large_negative = x < -35
	safe_x = jp.where(large_positive, 0, jp.where(large_negative, 0, x))

	pos = 1.0 / (1.0 + jp.exp(-safe_x))
	neg = 1.0 - pos
	return jp.where(x >= 0, pos, neg)


class World(Node):
	def __init__(self, *args, dt_ode: float = 1 / 100, eval_env: bool = False, **kwargs):
		super().__init__(*args, **kwargs)
		dt = 1 / self.rate
		self.substeps = ceil(dt / dt_ode)
		self.dt_ode = dt / self.substeps
		self.eval_env = eval_env

	def __getstate__(self):
		args, kwargs, inputs = super().__getstate__()
		kwargs.update(dict(dt_ode=self.dt_ode, eval_env=self.eval_env))
		return args, kwargs, inputs

	def __setstate__(self, state):
		args, kwargs, inputs = state
		self.__init__(*args, **kwargs)
		# At this point, the inputs are not yet fully unpickled.
		self.inputs = inputs

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
		if not self.eval_env:
			th = jumpy.random.uniform(rng_th, shape=(), low=-3.14, high=3.14)
			thdot = jumpy.random.uniform(rng_thdot, shape=(), low=-9., high=9.)
		else:
			th = jumpy.random.uniform(rng_th, shape=(), low=-0.2, high=0.2) + 3.14
			thdot = jumpy.random.uniform(rng_thdot, shape=(), low=0., high=0.)
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

	def step(self, step_state: StepState) -> Tuple[StepState, Output]:
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
		next_thdot = jp.clip(next_thdot, -params.max_speed, params.max_speed)
		new_state = state.replace(th=next_th, thdot=next_thdot)
		new_step_state = step_state.replace(state=new_state)

		# Prepare output
		output = Output(th=next_th, thdot=next_thdot)
		# print(f"{self.name.ljust(14)} | x: {x} | u: {u} -> next_x: {next_x}")
		return new_step_state, output


class Sensor(Node):

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

	def step(self, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Update state
		new_step_state = step_state

		# Prepare output
		output = inputs["world"][-1].data
		return new_step_state, output


class Actuator(Node):

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ActuatorOutput:
		"""Default output of the node."""
		return ActuatorOutput(action=jp.array([0.0], dtype=jp.float32))

	def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Update state
		new_step_state = step_state

		# Prepare output
		output = list(inputs.values())[0][-1].data
		return new_step_state, output
