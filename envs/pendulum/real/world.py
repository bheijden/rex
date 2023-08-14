from typing import Any, Dict, Tuple, Union, List
import dill as pickle
import jumpy
import numpy as onp
import jumpy.numpy as jp
from flax import struct

from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE
from rex.base import StepState, GraphState, Empty
from rex.node import Node

from envs.pendulum.env import Output as ActuatorOutput
from envs.pendulum.real.pid import PID
from envs.pendulum.render import Render


# Import ROS specific functions
try:
	import rospy
	from std_msgs.msg import Float32
	error_rospy = None
except ImportError as e:
	error_rospy = e
	rospy, Float32 = None, None

try:
	import dcsc_setups.srv as dcsc_setups
	error_dcsc_setups = None
except ImportError as e:
	error_dcsc_setups = e
	dcsc_setups = None


def build_pendulum(rates: Dict[str, float],
                   delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
                   delays: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
				   scheduling: int = PHASE,
                   advance: bool = False,
                   ) -> Dict[str, Node]:
	# Initialize main process as Node
	rospy.init_node("mops_client", anonymous=True)

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
	sensor.connect(world, window=1, blocking=False, skip=False, jitter=LATEST,  # TODO: skip=True?
	               delay_sim=trans_sim["sensor"]["world"], delay=trans["sensor"]["world"])
	render.connect(sensor, window=1, blocking=False, skip=False, jitter=LATEST,
	               delay_sim=trans_sim["render"]["sensor"], delay=trans["render"]["sensor"])
	# render.connect(actuator, window=1, blocking=False, skip=False, jitter=LATEST,
	#                delay_sim=Gaussian(0.), delay=0.0)

	# render.step = new_process(render.step)  # TODO: new process?

	return dict(world=world, actuator=actuator, sensor=sensor, render=render)


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


class World(Node):
	def __init__(self, *args, eval_env: bool = False, gains: List[float] = None, downward_reset: bool = True, **kwargs):
		super().__init__(*args, **kwargs)
		assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
		assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
		self.srv_read = rospy.ServiceProxy("/mops/read", dcsc_setups.MopsRead)
		self.srv_write = rospy.ServiceProxy("/mops/write", dcsc_setups.MopsWrite)
		self.srv_read.wait_for_service()
		self.srv_write.wait_for_service()
		self._rospy_rate = rospy.Rate(20)
		gains = gains or [2.0, 0.2, 1.0]
		self._downward_reset = downward_reset
		self.eval_env = eval_env
		self._controller = PID(u0=0.0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)

		if self._downward_reset:
			self._to_downward()
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

	def step(self, step_state: StepState) -> Tuple[StepState, Empty]:
		"""Step the node."""
		# Prepare output
		new_step_state = step_state
		return new_step_state, Empty()

	def _wrap_angle(self, angle):
		return angle - 2 * jp.pi * jp.floor((angle + jp.pi) / (2 * jp.pi))

	def _apply_action(self, action):
		"""Call ROS service and send action to mops."""
		req = dcsc_setups.MopsWriteRequest()
		req.actuators.digital_outputs = 1
		req.actuators.voltage0 = action
		req.actuators.voltage1 = 0.0
		req.actuators.timeout = 0.5
		self.srv_write.call(req)

	def _to_downward(self, goal_th: float = 0.):
		"""Set the pendulum to downward position."""
		self._controller.reset()
		goal = jp.array([goal_th, 0.0])
		done = False
		tstart = rospy.get_time()
		while not done:
			# Theta=0 here is downward position (different from the simulation)
			res = self.srv_read.call(dcsc_setups.MopsReadRequest())
			th, thdot = res.sensors.position0, res.sensors.speed
			th = self._wrap_angle(th)  # Wrap angle

			# Get action
			action = self._controller.next_action(th, ref=goal[0])
			action = jp.clip(action, -2.0, 2.0)
			self._apply_action(action)

			# Sleep
			self._rospy_rate.sleep()

			# Determine if we have reached our goal state
			done = onp.isclose(jp.array([th, thdot]), goal, atol=0.1).all()

			# Check timeout
			now = rospy.get_time()
			done = done or (now - tstart > 5.0)

		# Set initial action to 0
		self._apply_action(0.)


class Sensor(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
		assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
		self.srv_read = rospy.ServiceProxy("/mops/read", dcsc_setups.MopsRead)
		self.srv_read.wait_for_service()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the node."""
		return self._read_output()  # Read output from ROS service

	def step(self, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""
		# Update state
		new_step_state = step_state

		# Read output
		output = self._read_output()
		return new_step_state, output

	def _read_output(self):
		"""Read output from ROS."""
		res = self.srv_read.call(dcsc_setups.MopsReadRequest())
		th, thdot = res.sensors.position0 + jp.pi, res.sensors.speed
		return Output(th=jp.float32(th), thdot=jp.float32(thdot))


class Actuator(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
		assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
		self.srv_write = rospy.ServiceProxy("/mops/write", dcsc_setups.MopsWrite)
		self.srv_write.wait_for_service()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the node."""
		return ActuatorOutput(action=jp.array([0.0], dtype=jp.float32))

	def step(self, step_state: StepState) -> Tuple[StepState, Output]:
		"""Step the node."""

		# Unpack StepState
		_, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Update state
		new_step_state = step_state

		# Prepare output
		output: ActuatorOutput = list(inputs.values())[0][-1].data
		action = jp.clip(output.action[0], -2, 2)

		# Call ROS service and send action to mops
		self._apply_action(action)

		return new_step_state, output

	def _apply_action(self, action):
		"""Call ROS service and send action to mops."""
		req = dcsc_setups.MopsWriteRequest()
		req.actuators.digital_outputs = 1
		req.actuators.voltage0 = action
		req.actuators.voltage1 = 0.0
		req.actuators.timeout = 0.5
		self.srv_write.call(req)