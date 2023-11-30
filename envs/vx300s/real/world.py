import os
from typing import Any, Dict, Tuple, Union, List
import time
import cv2
import atexit
import yaml

import jumpy
import jumpy.numpy as jp
import numpy as np
from scipy.spatial.transform import Rotation as R
import jax
from math import ceil
from flax import struct

from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.multiprocessing import new_process

from envs.vx300s.env import ActuatorOutput, ArmOutput, BoxOutput
# from envs.vx300s.render import Render

try:
    from interbotix_copilot.client import Client
except ImportError as e:
    error_interbotix_copilot = e
    Client = None

try:
    from interbotix_xs_modules import core, gripper as interbotix_gripper
except ImportError as e:
    error_interbotix_xs_modules = e
    core, interbotix_gripper = None, None

GRIPPER_STATES = {"open": 1.0,
                  "closed": 0.0}


def build_vx300s(rates: Dict[str, float],
                 delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
                 delays: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                 config: Dict[str, Dict[str, Any]],
                 scheduling: int = PHASE,
                 ) -> Dict[str, Node]:
    robot_type = "vx300s"
    arm_name = robot_type
    if Client is None:
        print("make sure interbotix_copilot is installed & sourced!")
        raise error_interbotix_copilot
    if interbotix_gripper is None or core is None:
        print("make sure interbotix_xs_modules is installed & sourced!")
        raise error_interbotix_xs_modules

    # Create interbotix client
    client = Client(robot_type, arm_name, group_name="arm")
    client.set_joint_remapping(["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"])
    # Set operating mode
    client.set_operating_mode(
            mode=Client.POSITION,
            profile_type="velocity",
            profile_velocity=131,
            profile_acceleration=13,
        )
    # Set gains
    client.set_pid_gains(kp_pos=800, ki_pos=0, kd_pos=1000)

    # Create interbotix gripper client
    gripper = interbotix_gripper.InterbotixGripperXSInterface(
        core.InterbotixRobotXSCore(robot_type, arm_name, False),
        "gripper",
        gripper_pressure=0.5,
        gripper_pressure_lower_limit=150,
        gripper_pressure_upper_limit=350,
    )

    # Load camera
    with open(config["real"]["cam_intrinsics"], 'r') as file:
        ci = yaml.safe_load(file)
    # ce = yaml.load("/home/r2ci/rex/envs/vx300s/assets/eye_hand_calibration_2022-08-10-1757.yaml")

    # Prepare delays
    process_sim = delays_sim["step"]
    process = delays["step"]
    trans_sim = delays_sim["inputs"]
    trans = delays["inputs"]

    # Create nodes
    world = World(client, gripper, name="world", rate=rates["world"], scheduling=scheduling, advance=False,
                  delay=process["world"], delay_sim=process_sim["world"])
    armsensor = ArmSensor(client, name="armsensor", rate=rates["armsensor"], scheduling=scheduling, advance=False,
                          delay=process["armsensor"], delay_sim=process_sim["armsensor"])
    boxsensor = BoxSensor(aruco_id=25,
                          aruco_size=0.08,
                          aruco_type="DICT_ARUCO_ORIGINAL",
                          aruco_trans=[0, 0, -0.05],
                          cam_trans=config["real"]["cam_trans"],
                          cam_rot=config["real"]["cam_rot"],
                          cam_intrinsics=ci,
                          cam_idx=config["real"]["cam_idx"],
                          z_fixed=config["real"]["z_fixed"],
                          name="boxsensor", rate=rates["boxsensor"], scheduling=scheduling, advance=False,
                          delay=process["boxsensor"], delay_sim=process_sim["boxsensor"])
    armactuator = ArmActuator(client, name="armactuator", rate=rates["armactuator"], scheduling=scheduling, advance=False,
                              delay=process["armactuator"], delay_sim=process_sim["armactuator"])

    # Connect nodes
    world.connect(armactuator, window=1, blocking=False, skip=True, jitter=LATEST,
                  delay_sim=trans_sim["world"]["armactuator"], delay=trans["world"]["armactuator"])
    armsensor.connect(world, window=1, blocking=False, skip=False, jitter=LATEST,
                      delay_sim=trans_sim["armsensor"]["world"], delay=trans["armsensor"]["world"])
    boxsensor.connect(world, window=1, blocking=False, skip=False, jitter=LATEST,
                      delay_sim=trans_sim["boxsensor"]["world"], delay=trans["boxsensor"]["world"])

    return dict(world=world, armactuator=armactuator, armsensor=armsensor, boxsensor=boxsensor)


@struct.dataclass
class State:
    pipeline_state: Empty


class World(Node):
    def __init__(self, client: Client, gripper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = client
        self._gripper: interbotix_gripper.InterbotixRobotXSCore = gripper
        f = self._client.wait_for_feedthrough()
        f.result()  # Wait for feedthrough to be toggled on.
        print("Arm feedthrough enabled.")

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
        return State(pipeline_state=Empty())

    def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
        # rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
        params = self.default_params(rng, graph_state)
        state = self.default_state(rng, graph_state)
        inputs = self.default_inputs(rng, graph_state)

        # Reset arm
        # todo: wrap in jumpy.host_callback (with halting).
        home_jpos = graph_state.nodes["supervisor"].params.home_jpos
        f = self._client.wait_for_feedthrough()
        f.result()  # Wait for feedthrough to be toggled on.
        print("Arm feedthrough enabled.")
        f = self._client.go_to(points=home_jpos, timestamps=5.0, remap=True)  # Move to home
        # print(f"Going to home position: {home_jpos}")
        self._gripper.close(delay=0.)  # Close gripper
        f.result()  # Block until at home position
        print("Arm at home position.")
        return StepState(rng=rng, params=params, state=state, inputs=inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, Empty]:
        """Step the node."""
        return step_state, Empty()

    def view_rollout(self, rollout: List[Empty], m=None, verbose=False, **kwargs):
        pass


class ArmSensor(Node):
    def __init__(self, client: Client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = client

    def _get_output(self):
        jpos = self._client.get_joint_states().position
        jpos = np.array(jpos, dtype="float32")
        rot_matrix, eepos, eeorn = self._client.get_ee_pose()
        eepos = eepos.astype("float32")
        eeorn = R.from_matrix(rot_matrix).as_quat().astype("float32")  # quaternion: (x, y, z, w)
        output = ArmOutput(jpos=jpos, eepos=eepos, eeorn=eeorn)
        # output = ArmOutput(jpos=jp.zeros((6,), dtype=jp.float32),
        #                    eepos=jp.array([0.2, 0.2, 0.2], dtype=jp.float32),
        #                    eeorn=jp.array([0, 0, 0, 1], dtype=jp.float32))
        return output

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ArmOutput:
        """Default output of the node."""
        arm_output = self._get_output()
        return arm_output

    def step(self, step_state: StepState) -> Tuple[StepState, ArmOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        arm_output = self._get_output()
        return new_step_state, arm_output


@struct.dataclass
class BoxState:
    last_detection: BoxOutput


class BoxSensor(Node):
    def __init__(self, aruco_id: int, aruco_size: float, aruco_type: str,
                 aruco_trans: List[float], cam_trans: List[float], cam_rot: List[float],
                 cam_intrinsics: Dict[str, Any], cam_idx: int, z_fixed: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._z_fixed = z_fixed

        # Initialize detector
        from envs.vx300s.real.aruco_detector import ArucoPoseDetector

        self._aruco_id = aruco_id
        self._ci = cam_intrinsics
        self._height, self._width = self._ci["image_height"], self._ci["image_width"]
        cam_matrix = np.array(self._ci["camera_matrix"]["data"], dtype="float32").reshape(
            self._ci["camera_matrix"]["rows"], self._ci["camera_matrix"]["cols"]
        )
        dist_coeffs = np.array(self._ci["distortion_coefficients"]["data"], dtype="float32").reshape(
            self._ci["distortion_coefficients"]["rows"], self._ci["distortion_coefficients"]["cols"]
        )
        self._detector = ArucoPoseDetector(self._height, self._width, aruco_size, cam_matrix, dist_coeffs, aruco_type)

        # Calculate cam_to_base transformation matrix
        self._T_c2b = np.zeros((4, 4), dtype="float32")
        self._T_c2b[3, 3] = 1
        self._T_c2b[:3, :3] = R.from_quat(cam_rot).as_matrix()
        self._T_c2b[:3, 3] = cam_trans

        # Object translation offset
        self._aruco_trans = np.array([[aruco_trans[0]], [aruco_trans[1]], [aruco_trans[2]], [1.0]], dtype="float32")

        # Cam
        self._cam_idx = cam_idx
        self._cam = None

    def _init_cam(self):
        if self._cam is None:
            from envs.vx300s.real.aruco_detector import VideoCapture

            self._cam = VideoCapture(self._cam_idx)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            # Register shutdown
            atexit.register(self._shutdown)

            # Wait until
            iter = 0
            while True:
                time.sleep(0.05)
                ret, image_raw = self._cam.read()
                iter += 1
                if ret:
                    break
                elif iter > 100:
                    raise ValueError(f"Failed to initialize camera with index {self._cam_idx}.")

    def _shutdown(self):
        # Release camera resources
        if self._cam is not None:
            self._cam.release()
            cv2.destroyAllWindows()
            self._cam = None

    def _get_output(self):
        self._init_cam()  # Make sure camera is initialized

        # Get image
        i = 0
        while True:
            ret, image_raw = self._cam.read()  # cv_img is in BGR format
            i += 1
            if i % 100 == 0:
                print("Waiting for image...")
            if ret:
                break
            time.sleep(0.01)

        # Undistort image
        image = self._detector.undistort(image_raw)

        # Get pose
        image, corners, ids, rvec, tvec = self._detector.estimate_pose(image, draw=True)

        # boxpos = np.array([0.35, 0.0, 0.051], dtype="float32")
        if rvec is not None and (ids == self._aruco_id)[:, 0].any():
            mask = (ids == self._aruco_id)[:, 0]
            rvec = rvec[mask]
            tvec = tvec[mask]
            # Position aruco (with offset) in camera frame (aic).
            pos_aic = self._apply_aruco_translation(rvec, tvec)
            # Position aruco (with offset) in base frame (aib).
            pos_aib = self._T_c2b[:3, :3] @ pos_aic + self._T_c2b[:3, 3, None]
            # Rotation matrix from aruco to base frame
            rmat_a2c = R.from_rotvec(rvec[:, 0, :]).as_matrix()
            rmat_a2b = self._T_c2b[:3, :3] @ rmat_a2c
            quat_a2b = R.from_matrix(rmat_a2b).as_quat().astype("float32")
            # Get wrapped orientation
            wrapped_yaws = []
            for q in quat_a2b:
                wrapped_yaws.append(BoxOutput(boxpos=None, boxorn=q).wrapped_yaw)
            yaw = mean_angle(np.array(wrapped_yaws, dtype="float32"))
            cos_half_yaw = jp.cos(yaw / 2)
            sin_half_yaw = jp.sin(yaw / 2)
            boxorn = jp.array([0, 0, sin_half_yaw, cos_half_yaw], dtype=jp.float32)
            # Store last position
            boxpos = np.mean(pos_aib, axis=0, dtype="float32")[:, 0]
            boxpos[2] = self._z_fixed

            # Plot position
            cv2.putText(image, f"pos: {[round(pos, 2) for pos in boxpos]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"orn: {[round(orn, 2) for orn in boxorn]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            boxorn = None
            boxpos = None
        # boxorn = np.array([0.0, 0.0, 0.0, 1.0], dtype="float32")   # quaternion: (x, y, z, w)
        return BoxOutput(boxpos=boxpos, boxorn=boxorn), image

    def _apply_aruco_translation(self, rvec, tvec):
        T_a2c = np.zeros((rvec.shape[0], 4, 4), dtype="float32")
        T_a2c[:, 3, 3] = 1
        # Set all rotation matrices
        rmat = R.from_rotvec(rvec[:, 0, :])
        T_a2c[:, :3, :3] = rmat.as_matrix()
        # Set all translations
        T_a2c[:, :3, 3] = tvec[:, 0, :]
        # Offset measurements
        position = (T_a2c @ self._aruco_trans[:, :])[:, :3, :]
        return position

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> BoxOutput:
        """Default output of the node."""
        box_output, image = self._get_output()
        if box_output.boxpos is None or box_output.boxorn is None:
            box_output = graph_state.nodes[self.name].state.last_detection
        return box_output

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> BoxState:
        """Default state of the node."""
        box_output, image = self._get_output()
        while box_output.boxpos is None or box_output.boxorn is None:
            # Wait until box is detected
            input("Please place the box in front of the camera and press enter.")
            box_output, image = self._get_output()

        return BoxState(last_detection=box_output)

    def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
        self._init_cam()

        return graph_state.nodes[self.name]

    def step(self, step_state: StepState) -> Tuple[StepState, BoxOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        box_output, image = self._get_output()
        if box_output.boxpos is None or box_output.boxorn is None:
            box_output = state.last_detection
            new_step_state = step_state
        else:
            # Update state
            new_state = state.replace(last_detection=box_output)
            new_step_state = step_state.replace(state=new_state)

        # Display result frame
        cv2.imshow("image", image)
        key = cv2.waitKey(1)

        # print(f"Box position: {box_output.boxpos} | Box orientation: {box_output.boxorn}")
        return new_step_state, box_output


class ArmActuator(Node):
    def __init__(self, client: Client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = client

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        jpos = self._client.get_joint_states().position
        jpos = np.array(jpos, dtype="float32")
        # jpos = np.zeros((6,), dtype="float32")
        return ActuatorOutput(jpos=jpos)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        actuator_output = inputs["controller"][-1].data
        # self._client.write_commands(actuator_output.jpos.tolist())
        return new_step_state, actuator_output


def unwrap_angles(angles, wrap_point = np.pi/4):   # --> wrap_point=half the domain
    """
    Unwrap angles by adjusting those that exceed the wrap point.
    """
    unwrapped_angles = np.copy(angles)
    for i in range(len(angles)):
        if angles[i] > wrap_point:
            unwrapped_angles[i] -= wrap_point * 2
    return unwrapped_angles


def mean_angle(angles):
    """
    Calculate the mean of a list of angles after unwrapping.
    """
    unwrapped = unwrap_angles(angles)
    x = np.cos(unwrapped)
    y = np.sin(unwrapped)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.arctan2(mean_y, mean_x)
