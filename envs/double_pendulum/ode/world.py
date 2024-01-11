from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from math import ceil
from flax import struct
from flax.core import FrozenDict

from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE, FAST_AS_POSSIBLE, SIMULATED
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.multiprocessing import new_process

from envs.double_pendulum.env import Output as ActuatorOutput
from envs.double_pendulum.render import Render


def build_double_pendulum(rates: Dict[str, float],
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
	sensor.connect(world, window=1, blocking=False, skip=True, jitter=LATEST,
	               delay_sim=trans_sim["sensor"]["world"], delay=trans["sensor"]["world"])
	render.connect(sensor, window=1, blocking=False, skip=False, jitter=LATEST,
	               delay_sim=trans_sim["render"]["sensor"], delay=trans["render"]["sensor"])

	# render.connect(actuator, window=1, blocking=False, skip=False, delay_sim=Gaussian(mean=0., std=0.), delay=0.0, jitter=LATEST)

	# render.step = new_process(render.step)

	return dict(world=world, actuator=actuator, sensor=sensor, render=render)


@struct.dataclass
class Params:
	max_torque: Union[float, jax.typing.ArrayLike]
	max_speed: Union[float, jax.typing.ArrayLike]
	max_speed2: Union[float, jax.typing.ArrayLike]
	J: Union[float, jax.typing.ArrayLike]
	J2: Union[float, jax.typing.ArrayLike]
	mass: Union[float, jax.typing.ArrayLike]
	mass2: Union[float, jax.typing.ArrayLike]
	length: Union[float, jax.typing.ArrayLike]
	length2: Union[float, jax.typing.ArrayLike]
	b: Union[float, jax.typing.ArrayLike]
	b2: Union[float, jax.typing.ArrayLike]
	c: Union[float, jax.typing.ArrayLike]
	c2: Union[float, jax.typing.ArrayLike]
	K: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class State:
	th: Union[float, jax.typing.ArrayLike]
	th2: Union[float, jax.typing.ArrayLike]
	thdot: Union[float, jax.typing.ArrayLike]
	thdot2: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorParams:
	th_std: Union[float, jax.typing.ArrayLike]
	th2_std: Union[float, jax.typing.ArrayLike]
	thdot_std: Union[float, jax.typing.ArrayLike]
	thdot2_std: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class Output:
	cos_th: Union[float, jax.typing.ArrayLike]
	sin_th: Union[float, jax.typing.ArrayLike]
	cos_th2: Union[float, jax.typing.ArrayLike]
	sin_th2: Union[float, jax.typing.ArrayLike]
	thdot: Union[float, jax.typing.ArrayLike]
	thdot2: Union[float, jax.typing.ArrayLike]


def runge_kutta4(ode, dt, params, x, u):
	k1 = ode(params, x, u)
	k2 = ode(params, x + 0.5 * dt * k1, u)
	k3 = ode(params, x + 0.5 * dt * k2, u)
	k4 = ode(params, x + dt * k3, u)
	return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def ode_double_pendulum(params: Params, x, u):
	"""source: http://underactuated.mit.edu/acrobot.html"""
	g, K = 9.81, params.K
	J1, m1, l1, b1, c1, = params.J, params.mass, params.length, params.b, params.c
	J2, m2, l2, b2, c2 = params.J2, params.mass2, params.length2, params.b2, params.c2

	g = 9.81
	P3 = m2 * l1 * c2
	F1 = (m1 * c1 + m2 * l1) * g
	F2 = m2 * c2 * g
	alpha1 = x[0]
	alpha2 = x[1]
	alpha_dot1 = x[2]
	alpha_dot2 = x[3]
	M = jnp.array([[J1 + J2 + m2 * l1 * l1 + 2 * P3 * jnp.cos(alpha2), J2 + P3 * jnp.cos(alpha2)],
	              [J2 + P3 * jnp.cos(alpha2), J2]])
	C = jnp.array([[b1 - 2 * P3 * alpha_dot2 * jnp.sin(alpha2), -P3 * (alpha_dot2) * jnp.sin(alpha2)],
	              [P3 * alpha_dot1 * jnp.sin(alpha2), b2]])
	G = jnp.array([[-F1 * jnp.sin(alpha1) - F2 * jnp.sin(alpha1 + alpha2)],
	              [-F2 * jnp.sin(alpha1 + alpha2)]])
	U = jnp.array([[K * u],
	              [0]])
	alpha_dot = jnp.array([[alpha_dot1],
	                      [alpha_dot2]])
	# Minv = 1/(jnp.linalg.det(M)+0.000001)*jnp.array([[J2, -(J2 + P3 * jnp.cos(alpha2))],
	#                                             [-(J2 + P3 * cos(alpha2)), J1 + J2 + 2 * P3 * cos(alpha2)]])
	Minv = jnp.linalg.inv(M)
	totoal_torque = U + G
	coli = C @ alpha_dot
	ddx = Minv @ (totoal_torque - coli)
	# print(x,[x[2], x[3], ddx1, ddx2],u,np.linalg.det(M))
	# print("force",totoal_torque,coli)
	return jnp.array([x[2], x[3], ddx[0][0], ddx[1][0]])


def _angle_normalize(th: jax.typing.ArrayLike):
	th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
	return th_norm


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

	def default_params(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> Params:
		"""Default params of the node."""
		if graph_state is None or graph_state.nodes.get("root", None) is None:
			max_torque, max_speed, max_speed2 = 8.0, 50.0, 50.0
			length, length2 = 0.1, 0.1
		else:
			agent_params = graph_state.nodes["root"].params
			max_torque = agent_params.max_torque
			max_speed, max_speed2 = agent_params.max_speed, agent_params.max_speed2
			length, length2 = agent_params.length, agent_params.length2
		return Params(max_torque=max_torque,
		              max_speed=max_speed,
		              max_speed2=max_speed2,
		              J=0.037,
		              J2=0.000111608131930852,
		              mass=0.18,
		              mass2=0.0691843934004535,
		              length=length,
		              length2=length2,
		              b=0.975872107940422,
		              b2=1.07098956449896e-05,
		              c=0.06,
		              c2=0.0185223578523340,
		              K=1.09724557347983,
		              )

	def default_state(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> State:
		"""Default state of the node."""
		rng_th, rng_thdot = jax.random.split(rng, num=2)
		if not self.eval_env:
			th = jax.random.uniform(rng_th, shape=(), minval=-3.14, maxval=3.14)
			th2 = jax.random.uniform(rng_th, shape=(), minval=-3.14, maxval=3.14)
			thdot = jax.random.uniform(rng_th, shape=(), minval=-3.14, maxval=3.14)
			thdot2 = jax.random.uniform(rng_th, shape=(), minval=-3.14, maxval=3.14)
		else:
			# goal: th2 = jnp.pi - th
			# [th, th2] = [0, 0] = th=down, th2=down,
			# [th, th2] = [pi, 0] = th=up, th2=up
			# [th, th2] = [0, pi] = th=down , th2=up
			# [th, th2] = [pi, pi] = th=up, th2=down
			# alpha = jax.random.uniform(rng_th, shape=(), minval=-jnp.pi, maxval=jnp.pi)
			# th = alpha
			# th2 = jnp.pi - alpha
			th = jax.random.uniform(rng_th, shape=(), minval=-0.3, maxval=0.3)*0
			th2 = jax.random.uniform(rng_th, shape=(), minval=-0.3, maxval=0.3)*0
			thdot = 0.  # jax.random.uniform(rng_thdot, shape=(), low=-0.05, high=0.05)
			thdot2 = 0.  # jax.random.uniform(rng_thdot, shape=(), low=-0.1, high=0.1)
		return State(th=th, th2=th2, thdot=thdot, thdot2=thdot2)

	def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> State:
		"""Default output of the node."""
		# Grab output from state
		try:
			th = graph_state.nodes["world"].state.th
			th2 = graph_state.nodes["world"].state.th2
			thdot = graph_state.nodes["world"].state.thdot
			thdot2 = graph_state.nodes["world"].state.thdot2
		except (AttributeError):
			th = 0.
			th2 = 0.
			thdot = 0.
			thdot2 = 0.
		return State(th=th, th2=th2, thdot=thdot, thdot2=thdot2)

	def step(self, step_state: StepState) -> Tuple[StepState, State]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Get action
		u = list(inputs.values())[0].data.action[-1][0]
		x = jnp.array([state.th, state.th2, state.thdot, state.thdot2])
		# print(step_state.eps, step_state.seq, list(inputs.values())[0].seq[0], f"u={u} || x={x}")
		next_x = x

		# Calculate next state
		for _ in range(self.substeps):
			next_x = runge_kutta4(ode_double_pendulum, self.dt_ode, params, next_x, u)

		# Update state
		next_th, next_th2, next_thdot, next_thdot2 = next_x

		# Clip speed
		next_thdot = jnp.clip(next_thdot, -params.max_speed, params.max_speed)
		next_thdot2 = jnp.clip(next_thdot2, -params.max_speed, params.max_speed)

		# Update state
		new_state = state.replace(th=next_th, th2=next_th2, thdot=next_thdot, thdot2=next_thdot2)
		new_step_state = step_state.replace(state=new_state)

		# Prepare output
		output = State(th=next_th, th2=next_th2, thdot=next_thdot, thdot2=next_thdot2)
		return new_step_state, output


class Sensor(Node):

	def default_params(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> SensorParams:
		return SensorParams(th_std=0.0, th2_std=0.0, thdot_std=0.0, thdot2_std=0.0)

	def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> Output:
		"""Default output of the node."""
		# Grab output from state
		try:
			th = graph_state.nodes["world"].state.th
			th2 = graph_state.nodes["world"].state.th2
			thdot = graph_state.nodes["world"].state.thdot
			thdot2 = graph_state.nodes["world"].state.thdot2
		except (AttributeError):
			th = 0.
			th2 = 0.
			thdot = 0.
			thdot2 = 0.
		return Output(cos_th=jnp.cos(th), sin_th=jnp.sin(th), cos_th2=jnp.cos(th2), sin_th2=jnp.sin(th2), thdot=thdot, thdot2=thdot2)

	def step(self, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""

		# Unpack StepState
		rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# World state
		dp_state = inputs["world"][-1].data
		th, th2 = dp_state.th, dp_state.th2
		thdot, thdot2 = dp_state.thdot, dp_state.thdot2

		# Add noise
		th_std, th2_std, thdot_std, thdot2_std = params.th_std, params.th2_std, params.thdot_std, params.thdot2_std
		new_rng, rng_th, rng_th2, rng_thdot, rng_thdot2 = jax.random.split(rng, num=5)
		th = th + th_std * jax.random.normal(rng_th, shape=th.shape, dtype=jnp.float32)
		th2 = th2 + th2_std * jax.random.normal(rng_th2, shape=th2.shape, dtype=jnp.float32)
		thdot = thdot + thdot_std * jax.random.normal(rng_thdot, shape=thdot.shape, dtype=jnp.float32)
		thdot2 = thdot2 + thdot2_std * jax.random.normal(rng_thdot2, shape=thdot2.shape, dtype=jnp.float32)

		# Update state
		new_step_state = step_state.replace(rng=new_rng)

		# Prepare output
		output = Output(cos_th=jnp.cos(th), sin_th=jnp.sin(th), cos_th2=jnp.cos(th2), sin_th2=jnp.sin(th2), thdot=thdot,
		                thdot2=thdot2)
		return new_step_state, output


class Actuator(Node):

	def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> ActuatorOutput:
		"""Default output of the node."""
		return ActuatorOutput(action=onp.array([0.0]))

	def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Update state
		new_step_state = step_state

		# Prepare output
		output = list(inputs.values())[0][-1].data
		return new_step_state, output
