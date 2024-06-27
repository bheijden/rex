from typing import Dict, Tuple, Union, List
import jax
import jax.experimental.host_callback as hcb
import numpy as onp
import jax.numpy as jnp
from flax import struct

from rexv2 import base, constants
from rexv2.node import BaseNode

from envs.pendulum.base import ActuatorOutput, SensorOutput, SensorParams
from envs.pendulum.pid import PID

# Import ROS specific functions
try:
    import rospy
    from std_msgs.msg import Float32

    error_rospy = None
    ROS_AVAILABLE = True
except ImportError as e:
    error_rospy = e
    rospy, Float32 = None, None
    ROS_AVAILABLE = False

try:
    import dcsc_setups.srv as dcsc_setups

    error_dcsc_setups = None
    ROS_AVAILABLE = True and ROS_AVAILABLE
except ImportError as e:
    error_dcsc_setups = e
    dcsc_setups = None
    ROS_AVAILABLE = False
# ROS_AVAILABLE = False  # TODO: REMOVE


class Sensor(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not ROS_AVAILABLE:
            self.log("ROS", "ROS not available, using dummy output.", log_level=constants.LogLevel.WARN)
        else:
            assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
            assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
            self.srv_read = rospy.ServiceProxy("/mops/read", dcsc_setups.MopsRead)
            self.srv_read.wait_for_service()
        self._dummy_output = self._read_output(None)

    def _read_output(self, dummy):
        """Read output from ROS."""
        if ROS_AVAILABLE:
            res = self.srv_read.call(dcsc_setups.MopsReadRequest())
            th, thdot = res.sensors.position0 + jnp.pi, res.sensors.speed
        else:
            th, thdot = 0., 0.
        ts = self.now()
        return SensorOutput(th=onp.array(th, dtype=onp.float32), thdot=onp.array(thdot, dtype=onp.float32), ts=onp.array(ts, dtype=onp.float32))

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> SensorParams:
        """Default params of the node."""
        sensor_delay = base.TrainableDist.create(alpha=0., min=0.0, max=1 / self.rate)
        return SensorParams(sensor_delay=sensor_delay)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> SensorOutput:
        """Default output of the node."""
        graph_state = graph_state or base.GraphState()
        # Account for sensor delay
        output = jax.experimental.io_callback(self._read_output, self._dummy_output, 0)  # Read output from ROS service
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        output = output.replace(ts=-1. / self.rate - sensor_delay)
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, SensorOutput]:
        """Step the node."""
        # Read output
        output = jax.experimental.io_callback(self._read_output, self._dummy_output, 0)

        # Update ts of step_state
        new_step_state = step_state.replace(ts=output.ts)

        # Correct for sensor delay
        delay = step_state.params.sensor_delay.mean()
        output = output.replace(ts=new_step_state.ts - delay)
        return new_step_state, output


class Actuator(BaseNode):
    def __init__(self, *args, gains: List[float] = None, downward_reset: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if not ROS_AVAILABLE:
            self.log("ROS", "ROS not available, using dummy output.", log_level=constants.LogLevel.WARN)
        else:
            assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
            assert error_dcsc_setups is None, "Failed to import dcsc_setups: {}".format(error_dcsc_setups)
            self.srv_write = rospy.ServiceProxy("/mops/write", dcsc_setups.MopsWrite)
            self.srv_write.wait_for_service()

            # For resetting:
            self.srv_read = rospy.ServiceProxy("/mops/read", dcsc_setups.MopsRead)
            self.srv_read.wait_for_service()
            self._rospy_rate = rospy.Rate(20)
        gains = gains or [2.0, 0.2, 1.0]
        self._downward_reset = downward_reset
        self._controller = PID(u0=0.0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=float))

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        controller_output = list(inputs.values())[0][-1].data
        output = ActuatorOutput(action=controller_output.action)
        action = jnp.clip(output.action[0], -3, 3)

        # Call ROS service and send action to mops
        _ = jax.experimental.io_callback(self._apply_action, jnp.array(1.0), action)

        return new_step_state, output

    def startup(self, graph_state: base.GraphState = None, timeout: float = None) -> Union[bool, jax.Array]:
        """Reset the node."""
        state = graph_state.state.get("supervisor", None)
        init_th = state.init_th if state is not None else onp.pi
        if self._downward_reset and ROS_AVAILABLE:
            self._to_downward(init_th)
        return True

    def _wrap_angle(self, angle):
        return angle - 2 * jnp.pi * jnp.floor((angle + jnp.pi) / (2 * jnp.pi))

    def _apply_action(self, action):
        """Call ROS service and send action to mops."""
        if not ROS_AVAILABLE:
            return jnp.array(1.0)
        req = dcsc_setups.MopsWriteRequest()
        req.actuators.digital_outputs = 1
        req.actuators.voltage0 = action
        req.actuators.voltage1 = 0.0
        req.actuators.timeout = 0.5
        self.srv_write.call(req)
        return jnp.array(1.0, dtype=onp.float32)

    def _to_downward(self, goal_th: float = 0.):
        """Set the pendulum to downward position."""
        self._controller.reset()
        goal_th = self._wrap_angle(goal_th - onp.pi)
        goal = jnp.array([goal_th, 0.0])
        done = False
        tstart = rospy.get_time()
        while not done:
            # Theta=0 here is downward position (different from the simulation)
            res = self.srv_read.call(dcsc_setups.MopsReadRequest())
            th, thdot = res.sensors.position0, res.sensors.speed
            th = self._wrap_angle(th)  # Wrap angle

            # Get action
            action = self._controller.next_action(th, ref=float(goal[0]))
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
