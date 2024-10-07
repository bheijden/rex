import time
from typing import Dict, Tuple, Union, List, Optional
import jax
import jax.experimental.host_callback as hcb
import numpy as onp
import jax.numpy as jnp
from flax import struct

from rex import base, constants
from rex.node import BaseNode
from envs.crazyflie.ode import PlatformOutput, MoCapOutput, rpy_to_spherical
from envs.crazyflie.pid import PIDOutput

try:
    from pyvicon_datastream import tools as pvtools

    error_vicon = None
    VICON_AVAILABLE = True
except ImportError as e:
    error_vicon = e
    pvtools = None
    VICON_AVAILABLE = False

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
        sensor_delay = base.TrainableDist.create(delay=0., min=0.0, max=0.05)
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


class Platform(BaseNode):
    def __init__(self, *args, vicon_ip: str = "192.168.0.232", vicon_platform: str = "inclined_surface1", mock: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if not VICON_AVAILABLE:
            self.log("VICON", "`pyvicon_datastream` not available, using dummy output.", log_level=constants.LogLevel.WARN)
        else:
            assert error_vicon is None, "Failed to import vicon: {}".format(error_vicon)
        self._mock = mock
        self._vicon_ip = vicon_ip
        self._vicon_name = vicon_platform
        self._vicon_tracker = MockObjectTracker(vicon_ip) if self._mock or pvtools is None else pvtools.ObjectTracker(vicon_ip)
        self._dummy_output = self._read_output(None)
        self._last_pose = onp.zeros(3), onp.zeros(3)

    def _get_platform_pose(self):
        res = self._vicon_tracker.get_position(self._vicon_name)
        data = res[2] if isinstance(res, tuple) else []
        if len(data) > 0 and len(data[0]) == 8:
            pose = [
                data[0][2] / 1000,  # x
                data[0][3] / 1000,  # y
                data[0][4] / 1000,  # z
                data[0][5],  # roll
                data[0][6],  # pitch
                data[0][7],  # yaw
            ]
            position = onp.array(pose[:3], dtype=onp.float32)
            orientation = onp.array(pose[3:], dtype=onp.float32)
            return position, orientation
        else:
            return False

    def _read_output(self, dummy) -> PlatformOutput:
        """Read output from ROS."""
        res = self._get_platform_pose()
        if not res:  # If no data, use the last known pose
            pos, att = self._last_pose
        else:
            pos, att = res
            self._last_pose = pos, att
        # Get relevant data
        # **IMPORTANT**: forcefully set vel[2] == 0.0 if policy is trained this way in simulation.
        vel = onp.zeros_like(pos)  # Not available for now
        ts = onp.array(self.now(), dtype=onp.float32)  # Get current time
        return PlatformOutput(pos=pos.astype(onp.float32), vel=vel.astype(onp.float32), att=att.astype(onp.float32), ts=ts)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PlatformOutput:
        """Default output of the node."""
        graph_state = graph_state or base.GraphState()
        # Account for sensor delay
        output = jax.experimental.io_callback(self._read_output, self._dummy_output, 0)  # Read output from ROS service
        # params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        # sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        # output = output.replace(ts=-1. / self.rate - sensor_delay)
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, MoCapOutput]:
        """Step the node."""
        # Read output
        output = jax.experimental.io_callback(self._read_output, self._dummy_output, 0)

        # Update ts of step_state
        new_step_state = step_state.replace(ts=output.ts)

        # Correct for sensor delay
        # delay = step_state.params.sensor_delay.mean()
        # output = output.replace(ts=new_step_state.ts - delay)
        return new_step_state, output

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        # Wait for the vicon tracker to find the object
        sleep_time = 0.1
        start = time.time()
        while True:
            now = time.time()
            res = self._get_platform_pose()
            if res:
                break
            if now - start > 5:
                start = now
                self.backend.logwarn(f"Vicon tracker {self._vicon_name} not found.")
            time.sleep(sleep_time)
        return True

    def stop(self, timeout: float = None) -> bool:
        # todo: Reset _last_pose?
        return True


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
        actuator_delay = base.TrainableDist.create(delay=0., min=0.0, max=0.05)
        return PIDParams(actuator_delay=actuator_delay)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDOutput:
        assert "agent" in self.inputs, "No agent connected"
        graph_state = graph_state or base.GraphState()
        # Get base output
        output = self.inputs["agent"].output_node.init_output(rng, graph_state)
        output = output.replace(pwm_ref=40_000)
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
        x, y, z = graph_state.state["agent"].init_pos
        _, _, yaw = graph_state.state["agent"].init_att
        # Go to position
        self.log("Starting", f"Going to position (in world frame): x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
        while True:
            self.log("Starting", "Waiting for feedthrough to be enabled...")
            self._client.wait_for_feedthrough()  # This blocks until feedthrough is enabled
            # self.log("Starting", "Going to position...")
            success = self._client.go_to(x, y, z, yaw, timeout=10)
            if success:
                # self.log("Starting", "Success!")
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
        if True:#output.state_estimate.ts > 1.5:  # Skip the first 1.5 seconds
            if self._feedthrough:
                if output.has_landed:
                    print(f"pid | has landed: {output.has_landed}")
                self._client.send_rpyz_setpoint(float(output.phi_ref),
                                                float(output.theta_ref),
                                                float(output.psi_ref),
                                                float(output.z_ref),
                                                bool(output.has_landed))

        return jnp.array(1.0, dtype=onp.float32)


class InclinedPID(PID):
    def __init__(self, *args, vicon_ip: str = "192.168.0.232", vicon_platform: str = "inclined_surface1",
                 reset_in_platform_frame: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if not VICON_AVAILABLE:
            self.log("VICON", "`pyvicon_datastream` not available, using dummy output.", log_level=constants.LogLevel.WARN)
        else:
            assert error_vicon is None, "Failed to import vicon: {}".format(error_vicon)
        self._reset_in_platform_frame = reset_in_platform_frame
        self._vicon_ip = vicon_ip
        self._vicon_name = vicon_platform
        self._vicon_tracker = MockObjectTracker(vicon_ip) if self._mock or pvtools is None else pvtools.ObjectTracker(vicon_ip)

    def get_platform_pose(self):
        res = self._vicon_tracker.get_position(self._vicon_name)
        data = res[2] if isinstance(res, tuple) else []
        if len(data) > 0 and len(data[0]) == 8:
            pose = [
                data[0][2] / 1000,  # x
                data[0][3] / 1000,  # y
                data[0][4] / 1000,  # z
                data[0][5],  # roll
                data[0][6],  # pitch
                data[0][7],  # yaw
            ]
            position = onp.array(pose[:3], dtype=onp.float32)
            orientation = onp.array(pose[3:], dtype=onp.float32)
            return position, orientation
        else:
            return False

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        # Get initial position
        x, y, z = graph_state.state["agent"].init_pos
        _, _, yaw = graph_state.state["agent"].init_att
        # Go to position
        while True:
            self.log("Startup", "Waiting for feedthrough to be enabled...")
            self._client.wait_for_feedthrough()  # This blocks until feedthrough is enabled
            if self._reset_in_platform_frame:
                self.log("Startup", f"Going to position (in PLATFORM frame): x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
                x, y, z = graph_state.state["agent"].init_pos  # Reset in case we've overridden already
                _, _, yaw = graph_state.state["agent"].init_att  # Reset in case we've overridden already
                # Wait for the vicon tracker to find the platform
                sleep_time = 1.0
                start = time.time()
                while True:
                    now = time.time()
                    if now - start > 5:
                        start = now
                        self.log("Startup", f"Vicon tracker {self._vicon_name} not found.", log_level=constants.LogLevel.WARN)
                    pose_is = self.get_platform_pose()
                    if pose_is:
                        break
                    time.sleep(sleep_time)
                is_pos, is_att = pose_is
                is_polar, is_azimuth = rpy_to_spherical(is_att)
                self.log("Startup", f"Found inclined surface at x={is_pos[0]:.2f}, y={is_pos[1]:.2f}, z={is_pos[2]:.2f} with azimuth {is_azimuth:.2f} and inclination {is_polar: .2f}")

                # Make is=inclined surface rotation matrix
                Rz = onp.array([[onp.cos(is_azimuth), -onp.sin(is_azimuth), 0],
                               [onp.sin(is_azimuth), onp.cos(is_azimuth), 0],
                               [0, 0, 1]])
                is2w_R = Rz

                # World to is=inclined surface
                is2w_H = onp.eye(4)
                is2w_H[:3, :3] = is2w_R
                is2w_H[:3, 3] = is_pos

                cf_pos = is2w_H @ onp.array([x, y, z, 1.0], dtype=onp.float32)
                cf_pos = cf_pos[:3]
                cf_yaw_w = yaw + is_azimuth

                x, y, z = cf_pos
                yaw = cf_yaw_w

            # self.log("Starting", "Going to position...")
            self.log("Startup", f"Going to position (in WORLD frame): x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
            success = self._client.go_to(x, y, z, yaw, timeout=10)
            if success:
                # self.log("Starting", "Success!")
                break
            self.log("Starting", "Failed to reach the position. Retrying...")
        return True

    def stop(self, timeout: float = None) -> bool:
        # todo: How to make sure to stay landed?
        self._client.disable_feedthrough()  # This should halt the quadcopter in place
        self.log("Stopping", "Feedthrough disabled.")
        return True


class MockObjectTracker:
    def __init__(self, ip: str):
        pass

    def get_position(self, name: str):
        latency = 0
        framenumber = 1
        positions = []
        position_entry = [
                    name,
                    name,
                    0.2 * 1000,  # position_x in mm
                    0.1 * 1000,  # position_y in mm
                    1.0 * 1000,  # position_z in mm
                    -onp.pi / 8,  # euler_x,
                    0.0,  # euler_y,
                    0.0,  # euler_z
                ]
        positions.append(position_entry)
        return latency, framenumber, positions


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


