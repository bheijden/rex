from typing import Dict, Tuple, Union, List, Optional
import jax
import jax.experimental.host_callback as hcb
import numpy as onp
import jax.numpy as jnp
from flax import struct

from rexv2 import base, constants
from rexv2.node import BaseNode
from envs.crazyflie.ode import MoCapOutput, force_to_pwm
from envs.crazyflie.pid import PIDOutput

# Import ROS specific functions
try:
    import rospy

    error_rospy = None
    ROS_AVAILABLE = True
except ImportError as e:
    error_rospy = e
    rospy = None, None
    ROS_AVAILABLE = False

try:
    from crazyflie_ros.client import Client

    error_crazyflie_ros = None
    ROS_AVAILABLE = True and ROS_AVAILABLE
except ImportError as e:
    error_crazyflie_ros = e
    Client = None
    ROS_AVAILABLE = False
# ROS_AVAILABLE = False  # TODO: REMOVE


def rpy_to_R(rpy, convention="xyz"):
    phi, theta, psi = rpy
    Rz = onp.array([[onp.cos(psi), -onp.sin(psi), 0],
                   [onp.sin(psi), onp.cos(psi), 0],
                   [0, 0, 1]])
    Ry = onp.array([[onp.cos(theta), 0, onp.sin(theta)],
                   [0, 1, 0],
                   [-onp.sin(theta), 0, onp.cos(theta)]])
    Rx = onp.array([[1, 0, 0],
                   [0, onp.cos(phi), -onp.sin(phi)],
                   [0, onp.sin(phi), onp.cos(phi)]])
    # Define below which one is Tait-bryan and which one is Euler
    if convention == "xyz":
        R = Rx @ Ry @ Rz  # This uses Tait-Bryan angles (XYZ sequence)
    elif convention == "zyx":
        R = Rz @ Ry @ Rx  # This uses Tait-Bryan angles (ZYX sequence)
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return R


@struct.dataclass
class MoCapParams(base.Base):
    sensor_delay: base.TrainableDist


class MoCap(BaseNode):
    def __init__(self, *args, pos_offset: jax.typing.ArrayLike = None, copilot_name: str = "cf", mock: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if not ROS_AVAILABLE:
            self.log("ROS", "ROS not available, using dummy output.", log_level=constants.LogLevel.WARN)
        else:
            assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
            assert error_crazyflie_ros is None, "Failed to import crazyflie_ros: {}".format(error_crazyflie_ros)
        # From top mocap board to control board (corrects vicon position to cf center in xml)
        self._pos_offset = pos_offset if pos_offset is not None else onp.array([0.0, 0.0, -0.018], dtype=onp.float32)
        self._copilot_name = copilot_name
        self._mock = mock
        self._client = MockClient(copilot_name) if mock or Client is None else Client(copilot_name)
        self._dummy_output = self._read_output(None)

    def _read_output(self, dummy) -> MoCapOutput:
        """Read output from ROS."""
        # Get relevant data
        pos = self._client.position
        att = self._client.attitude
        vel = self._client.velocity
        ts = onp.array(self.now(), dtype=onp.float32)  # Get current time
        ang_vel = onp.zeros_like(att)  # Not available
        # Correct for the position offset
        cf2w_R = rpy_to_R(att)
        pos_corrected = pos + cf2w_R @ self._pos_offset
        return MoCapOutput(pos=pos_corrected.astype(onp.float32), vel=vel.astype(onp.float32),
                           att=att.astype(onp.float32), ang_vel=ang_vel.astype(onp.float32), ts=ts)

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> MoCapParams:
        """Default params of the node."""
        sensor_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05)
        return MoCapParams(sensor_delay=sensor_delay)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> MoCapOutput:
        """Default output of the node."""
        graph_state = graph_state or base.GraphState()
        # Account for sensor delay
        output = jax.experimental.io_callback(self._read_output, self._dummy_output, 0)  # Read output from ROS service
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        output = output.replace(ts=-1. / self.rate - sensor_delay)
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, MoCapOutput]:
        """Step the node."""
        # Read output
        output = jax.experimental.io_callback(self._read_output, self._dummy_output, 0)

        # Update ts of step_state
        new_step_state = step_state.replace(ts=output.ts)

        # Correct for sensor delay
        delay = step_state.params.sensor_delay.mean()
        output = output.replace(ts=new_step_state.ts - delay)
        return new_step_state, output


@struct.dataclass
class PIDParams(base.Base):
    actuator_delay: base.TrainableDist


class PID(BaseNode):
    def __init__(self, *args, copilot_name: str = "cf", mock: bool = False, feedthrough: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if not ROS_AVAILABLE:
            self.log("ROS", "ROS not available, using dummy output.", log_level=constants.LogLevel.WARN)
        else:
            assert error_rospy is None, "Failed to import rospy: {}".format(error_rospy)
            assert error_crazyflie_ros is None, "Failed to import crazyflie_ros: {}".format(error_crazyflie_ros)
        # From top mocap board to control board (corrects vicon position to cf center in xml)
        self._copilot_name = copilot_name
        self._mock = mock
        self._feedthrough = feedthrough
        self._client = MockClient(copilot_name) if mock or Client is None else Client(copilot_name)

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDParams:
        """Default params of the node."""
        actuator_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05)
        return PIDParams(actuator_delay=actuator_delay)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDOutput:
        assert "agent" in self.inputs, "No agent connected"
        graph_state = graph_state or base.GraphState()
        # Get base output
        output = self.inputs["agent"].output_node.init_output(rng, graph_state)
        # Fill pwm_ref with default hover_pwm
        params_sup = graph_state.params.get("supervisor")
        pwm_hover = force_to_pwm(params_sup.pwm_constants, params_sup.mass * 9.81)
        output = output.replace(pwm_ref=pwm_hover)
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, PIDOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Grab ctrl output
        assert len(self.inputs) > 0, "Nothing connected to pid"
        output: PIDOutput = list(inputs.values())[0][-1].data

        # Call ROS service and send action to mops
        _ = jax.experimental.io_callback(self._apply_action, jnp.array(1.0), output)

        return new_step_state, output

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        # Get initial position
        x, y, z = graph_state.state["supervisor"].init_pos
        _, _, yaw = graph_state.state["supervisor"].init_att
        # Go to position
        self.log("Starting", f"Going to position (in world frame): x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
        while True:
            self.log("Starting", "Waiting for feedthrough to be enabled...")
            self._client.wait_for_feedthrough()  # This blocks until feedthrough is enabled
            self.log("Starting", "Going to position...")
            success = self._client.go_to(x, y, z, yaw, timeout=10)
            if success:
                self.log("Starting", "Success!")
                break
            self.log("Starting", "Failed to reach the position. Retrying...")
        return True

    def stop(self, timeout: float = None) -> bool:
        self._client.disable_feedthrough()  # This should halt the quadcopter in place
        self.log("Stopping", "Feedthrough disabled.")
        return True

    def _apply_action(self, output: PIDOutput, ts_throttle=None):
        """Call ROS service and send action to mops."""
        # Forcefully fly a circle with x=cos(x) and y=sin(x) with z=1.5 and a period of 10 seconds and a radius of 1m
        # x = np.cos(2 * np.pi * t_n / 10)
        # y = np.sin(2 * np.pi * t_n / 10)
        # yaw = (360 * t_n / 10) % 360
        # self._client.send_position_setpoint(x, y, 1.75, yaw)

        # Apply command
        # if output.state_estimate.ts > 1.5: # Skip the first 1.5 seconds
        phi_ref, theta_ref, psi_ref, z_ref = output.phi_ref, output.theta_ref, output.psi_ref, output.z_ref
        if self._feedthrough:
            self._client.send_rpyz_setpoint(phi_ref, theta_ref, psi_ref, z_ref)

        return jnp.array(1.0, dtype=onp.float32)


class MockClient:
    def __init__(self, name: str="cf"):
        self._name = name
        self._pos = onp.array([0., 0., 0.], dtype="float32")
        self._vel = onp.array([0., 0., 0.], dtype="float32")
        self._att = onp.array([0., 0., 0.], dtype="float32")

    @property
    def position(self):
        return self._pos

    @property
    def attitude(self):
        return self._att

    @property
    def velocity(self):
        return self._vel

    def wait_for_feedthrough(self, timeout: Optional[float] = None) -> bool:
        return True

    def _cf_state_callback(self, pos: Tuple[float, float, float], vel: Tuple[float, float, float], att: Tuple[float, float, float]):
        self._pos = onp.array(pos, dtype="float32")
        self._vel = onp.array(vel, dtype="float32")
        self._att = onp.array(att, dtype="float32")

    def _toggle_feedthrough(self, feedthrough: bool):
        self._feedthrough = feedthrough

    def go_to(self, x: float, y: float, z: float, yaw: float, timeout: float, threshold: float = 0.05) -> bool:
        return True

    def disable_feedthrough(self):
        pass

    def send_position_setpoint(self, x: float, y: float, z: float, yaw: float):
        pass

    def send_setpoint(self, roll: float, pitch: float, yawrate: float, thrust: float, landed: bool = False):
        pass

    def send_rpyz_setpoint(self, roll: float, pitch: float, yawrate: float, z: float, landed: bool = False):
        pass

    def shutdown(self):
        pass