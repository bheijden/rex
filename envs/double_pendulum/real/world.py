from typing import Any, Dict, Tuple, Union, List
import dill as pickle
import numpy as onp
import jax
import jax.numpy as jnp
from flax import struct

from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.multiprocessing import new_process

from envs.double_pendulum.env import Output as ActuatorOutput
from envs.pendulum.real.pid import PID
from envs.double_pendulum.render import Render


# Import ROS specific functions
try:
	import rospy
	from std_msgs.msg import Float32
	error_rospy = None
except ImportError as e:
	error_rospy = e
	rospy, Float32 = None, None

try:
	import dcsc_setups.srv as dcsc_setups_srv
	import dcsc_setups.msg as dcsc_setups_msg
	error_dcsc_setups = None
except ImportError as e:
	error_dcsc_setups = e
	dcsc_setups_srv = None
	dcsc_setups_msg = None


def _convert_to_radians(volt, max_volt, offset_volt):
	# todo: take shifts into account
	return 2 * jnp.pi * (volt - offset_volt) / max_volt


def build_double_pendulum(rates: Dict[str, float],
                          delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
                          delays: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                          scheduling: int = PHASE,
                          advance: bool = False,
                          ) -> Dict[str, Node]:
	# Initialize main process as Node
	rospy.init_node("pendulum_client", anonymous=True)

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
	# render.connect(actuator, window=1, blocking=False, skip=False, jitter=LATEST,
	#                delay_sim=Gaussian(0.), delay=0.0)

	render.step = new_process(render.step)

	return dict(world=world, actuator=actuator, sensor=sensor, render=render)


@struct.dataclass
class State:
	"""Pendulum ode state definition"""

	th: Union[float, jax.typing.ArrayLike]
	th2: Union[float, jax.typing.ArrayLike]
	thdot: Union[float, jax.typing.ArrayLike]
	thdot2: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorOutput:
	"""Pendulum ode state definition"""

	cos_th: Union[float, jax.typing.ArrayLike]
	sin_th: Union[float, jax.typing.ArrayLike]
	cos_th2: Union[float, jax.typing.ArrayLike]
	sin_th2: Union[float, jax.typing.ArrayLike]
	thdot: Union[float, jax.typing.ArrayLike]
	thdot2: Union[float, jax.typing.ArrayLike]
	volt: Union[float, jax.typing.ArrayLike]
	volt2: Union[float, jax.typing.ArrayLike]
	th_enc: Union[float, jax.typing.ArrayLike]


class World(Node):
	def __init__(self, *args, eval_env: bool = False, gains: List[float] = None, downward_reset: bool = False, **kwargs):
		super().__init__(*args, **kwargs)
		assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
		assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
		self.srv_write = rospy.ServiceProxy("/pendulum/write", dcsc_setups_srv.PendulumWrite)
		self.srv_write.wait_for_service()

		# Prepare
		self.sub_read = rospy.Subscriber("/pendulum/obs", dcsc_setups_msg.PendulumSensors, self._update_reading)
		self._last_msg = None

		# Prepare for downward reset
		self._rospy_rate = rospy.Rate(50)
		gains = gains or [2.0, 0.2, 0.5]
		self._downward_reset = downward_reset
		self.eval_env = eval_env
		self._controller = PID(u0=0.0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

	def reset(self, rng: jax.random.KeyArray, graph_state: GraphState = None):
		"""Reset the node."""
		if self._downward_reset:
			self._to_downward()

	def step(self, step_state: StepState) -> Tuple[StepState, Empty]:
		"""Step the node."""
		# Prepare output
		new_step_state = step_state
		return new_step_state, Empty()

	def _wrap_angle(self, angle):
		return angle - 2 * jnp.pi * jnp.floor((angle + jnp.pi) / (2 * jnp.pi))

	def _apply_action(self, action):
		"""Call ROS service and send action to mops."""
		req = dcsc_setups_srv.PendulumWriteRequest()
		req.actuators.digital_outputs = 1
		req.actuators.voltage0 = action
		req.actuators.voltage1 = 0.0
		req.actuators.timeout = 0.02
		self.srv_write.call(req)

	def _to_downward(self, goal_th: float = 0.):
		"""Set the pendulum to downward position."""
		while self._last_msg is None:
			self.log("", "Waiting for sensor reading...", log_level=WARN)
			rospy.sleep(0.01)

		self._controller.reset()
		goal = jnp.array([goal_th, 0.0])
		done = False
		tstart = rospy.get_time()
		while not done:
			# Grab last sensor measurement
			obs: dcsc_setups_msg.PendulumSensors = self._last_msg
			th = obs.angle_beam
			th = self._wrap_angle(th)
			thdot, thdot2 = obs.velocity_beam, obs.velocity_pendulum

			# Get action
			action = self._controller.next_action(th, ref=goal[0])
			action = jnp.clip(action, -2.0, 2.0)
			self._apply_action(action)

			# Sleep
			self._rospy_rate.sleep()

			# Determine if we have reached our goal state
			done = onp.isclose(jnp.array([th, thdot]), goal, atol=0.1).all()

			# Check timeout
			now = rospy.get_time()
			done = done or (now - tstart > 5.0)

		# Set initial action to 0
		self._apply_action(0.)

	def _update_reading(self, msg):
		self._last_msg = msg


class Sensor(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
		assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)

		# Prepare
		self.sub_read = rospy.Subscriber("/pendulum/obs", dcsc_setups_msg.PendulumSensors, self._update_reading)
		self._last_msg: dcsc_setups_msg.PendulumSensors = None

	def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> SensorOutput:
		"""Default output of the node."""
		return self._read_output()  # Read output from ROS service

	def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
		"""Step the node."""
		# Update state
		new_step_state = step_state

		# Read output
		output = self._read_output()
		return new_step_state, output

	def _update_reading(self, msg):
		self._last_msg = msg

	def _read_output(self):
		"""Read output from ROS."""
		while self._last_msg is None:
			self.log("", "Waiting for sensor reading...", log_level=WARN)
			rospy.sleep(0.01)

		# Grab last sensor measurement
		obs: dcsc_setups_msg.PendulumSensors = self._last_msg

		# Prepare output
		v, v2 = obs.voltage_beam, obs.voltage_pendulum
		th_enc = obs.position0
		th = obs.angle_beam
		th2 = obs.angle_pendulum
		thdot, thdot2 = obs.velocity_beam, obs.velocity_pendulum
		return SensorOutput(cos_th=jnp.cos(th),
		                    sin_th=jnp.sin(th),
		                    cos_th2=jnp.cos(th2),
		                    sin_th2=jnp.sin(th2),
		                    thdot=thdot,
		                    thdot2=thdot2,
		                    volt=v,
		                    volt2=v2,
		                    th_enc=th_enc)


class Actuator(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
		assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
		self.srv_write = rospy.ServiceProxy("/pendulum/write", dcsc_setups_srv.PendulumWrite)
		self.srv_write.wait_for_service()

	def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> ActuatorOutput:
		"""Default output of the node."""
		return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

	def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Update state
		new_step_state = step_state

		# Prepare output
		output: ActuatorOutput = list(inputs.values())[0][-1].data
		action = jnp.clip(output.action[0], -8, 8)

		# Call ROS service and send action to mops
		self._apply_action(action)

		return new_step_state, output

	def _apply_action(self, action):
		"""Call ROS service and send action to mops."""
		req = dcsc_setups_srv.PendulumWriteRequest()
		req.actuators.digital_outputs = 1
		req.actuators.voltage0 = action
		req.actuators.voltage1 = 0.0
		req.actuators.timeout = 0.5
		self.srv_write.call(req)
