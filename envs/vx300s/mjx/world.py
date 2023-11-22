import os
import time
from typing import Any, Dict, Tuple, Union, List
import jumpy
import numpy as np
import jumpy.numpy as jp
import jax
import jax.experimental.host_callback as host_callback
import jax.numpy as jnp
from math import ceil
from flax import struct

import mujoco
import mujoco.viewer
from mujoco import mjx
from mujoco.mjx._src import types, math
from jax.debug import print as jax_print

from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.utils import timer
from rex.multiprocessing import new_process

from envs.vx300s.env import ActuatorOutput, ArmOutput, BoxOutput, SupervisorOutput


def build_vx300s(rates: Dict[str, float],
                 delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
                 delays: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                 config: Dict[str, Dict[str, Any]],
                 scheduling: int = PHASE,
                 ) -> Dict[str, Node]:
    # Prepare delays
    process_sim = delays_sim["step"]
    process = delays["step"]
    trans_sim = delays_sim["inputs"]
    trans = delays["inputs"]

    # Create nodes
    world = World(xml_path=config["mjx"]["xml_path"], name="world", rate=rates["world"], scheduling=scheduling, advance=False,
                  delay=process["world"], delay_sim=process_sim["world"])
    armsensor = ArmSensor(world, name="armsensor", rate=rates["armsensor"], scheduling=scheduling, advance=False,
                          delay=process["armsensor"], delay_sim=process_sim["armsensor"])
    boxsensor = BoxSensor(world, name="boxsensor", rate=rates["boxsensor"], scheduling=scheduling, advance=False,
                          delay=process["boxsensor"], delay_sim=process_sim["boxsensor"])
    armactuator = ArmActuator(world, name="armactuator", rate=rates["armactuator"], scheduling=scheduling, advance=False,
                              delay=process["armactuator"], delay_sim=process_sim["armactuator"])
    # Connect nodes
    world.connect(armactuator, window=1, blocking=False, skip=False, jitter=LATEST,
                  delay_sim=trans_sim["world"]["armactuator"], delay=trans["world"]["armactuator"])
    armsensor.connect(world, window=1, blocking=False, skip=True, jitter=LATEST,
                      delay_sim=trans_sim["armsensor"]["world"], delay=trans["armsensor"]["world"])
    boxsensor.connect(world, window=1, blocking=False, skip=True, jitter=LATEST,
                      delay_sim=trans_sim["boxsensor"]["world"], delay=trans["boxsensor"]["world"])
    return dict(world=world, armactuator=armactuator, armsensor=armsensor, boxsensor=boxsensor)


@struct.dataclass
class Params:
    # sys: BraxSystem
    pass


@struct.dataclass
class State:
    pipeline_state: types.Data


@struct.dataclass
class MjxOutput:
    jpos: jp.ndarray
    eepos: jp.ndarray
    eeorn: jp.ndarray
    boxpos: jp.ndarray
    boxorn: jp.ndarray


class World(Node):
    def __init__(self, xml_path: str, *args, dt_mjx: float = 0.04, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_mjx)
        self.dt_mjx = dt / self.substeps

        # Load system
        self._xml_path = xml_path

        self._mj_m = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mjx_m: types.Model = mjx.device_put(self._mj_m)
        self._mjx_m = self._mjx_m.replace(opt=self._mjx_m.opt.replace(timestep=self.dt_mjx))

        # Get indices
        self._joint_idx = self._mjx_m.actuator_trnid.tolist()
        self._joint_slice = slice(self._joint_idx[0], self._joint_idx[-1] + 1)
        self._ee_arm_idx = self._mj_m.body("ee_link").geomadr[0]
        self._box_idx = self._mj_m.body("box").geomadr[0]
        self._goal_idx = self._mj_m.body("goal").geomadr[0]

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Params:
        """Default params of the node."""
        return Params()

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
        """Default state of the node."""
        # Try to grab state from graph_state
        goalpos = graph_state.nodes["supervisor"].state.goalpos
        goalyaw = graph_state.nodes["supervisor"].state.goalyaw
        home_boxpos = graph_state.nodes["supervisor"].params.home_boxpos
        home_boxyaw = graph_state.nodes["supervisor"].params.home_boxyaw
        home_jpos = graph_state.nodes["supervisor"].params.home_jpos

        mj_d = mujoco.MjData(self._mj_m)
        d = mjx.device_put(mj_d)

        qpos = jnp.concatenate([home_boxpos, home_boxyaw, goalpos, home_jpos, jnp.array([0])])
        d = d.replace(qpos=qpos, ctrl=jnp.zeros(self._mjx_m.nu))
        d = mjx.forward(self._mjx_m, d)
        return State(pipeline_state=d)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> MjxOutput:
        """Default output of the node."""
        # Grab output from state
        pipeline_state = graph_state.nodes["world"].state.pipeline_state
        mjx_output = self._get_output(pipeline_state)
        return mjx_output

    def step(self, step_state: StepState) -> Tuple[StepState, MjxOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        action = inputs["armactuator"].data.jpos[-1]
        pipeline_state = state.pipeline_state.replace(ctrl=action)

        def f(state, _):
            return mjx.step(self._mjx_m, state), None

        new_pipeline_state = jax.lax.scan(f, pipeline_state, (), self.substeps)[0]

        # Update state
        new_state = state.replace(pipeline_state=new_pipeline_state)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        mjx_output = self._get_output(new_pipeline_state)
        return new_step_state, mjx_output

    def _get_output(self, pipeline_state: types.Data) -> MjxOutput:
        x_i = pipeline_state.xipos
        x_quat = pipeline_state.xquat
        jpos = pipeline_state.qpos[self._joint_slice]
        eepos = x_i[self._ee_arm_idx]
        eeorn = self._convert_wxyz_to_xyzw(x_quat[self._ee_arm_idx])  # quaternion (w,x,y,z) -> (x,y,z,w)
        boxpos = x_i[self._box_idx]
        boxorn = self._convert_wxyz_to_xyzw(x_quat[self._box_idx])  # quaternion (w,x,y,z) -> (x,y,z,w)
        return MjxOutput(jpos=jpos, eepos=eepos, eeorn=eeorn, boxpos=boxpos, boxorn=boxorn)

    def _convert_wxyz_to_xyzw(self, quat: jp.ndarray):
        """Convert quaternion (w,x,y,z) -> (x,y,z,w)"""
        return jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype="float32")

    def view_state(self, mjx_d: types.Data, m=None):
        from mujoco import Renderer
        m = self._mj_m if m is None else m
        d = mujoco.MjData(m)
        viewer = mujoco.viewer.launch_passive(m, d)
        mjx.device_get_into(d, mjx_d)
        viewer.sync()
        return viewer

    def view_rollout(self, rollout: List[types.Data], m=None, verbose=False, **kwargs):
        m = self._mj_m if m is None else m

        d = mujoco.MjData(m)

        _paused = False
        _must_quit = False
        _must_reset = False
        _real_time_factor = 1.0

        def _key_callback(keycode):
            nonlocal _paused, _must_quit, _must_reset, _real_time_factor
            if chr(keycode) == ' ':
                _paused = not _paused
                print(f'{"paused" if _paused else "unpaused"}') if verbose else None
            elif chr(keycode) == 'Q':
                print("quitting") if verbose else None
                _must_quit = True
            elif chr(keycode) == 'R':
                print("resetting") if verbose else None
                _must_reset = True
            # Speed up or slow down simulation
            elif keycode == 265:  # up arrow
                _real_time_factor *= 2
                print(f"real_time_factor: {_real_time_factor}") if verbose else None
            elif keycode == 264:  # down arrow
                _real_time_factor /= 2
                print(f"real_time_factor: {_real_time_factor}") if verbose else None
            else:
                print(f"keycode: {keycode} | chr: {chr(keycode)}")  if verbose else None

        with mujoco.viewer.launch_passive(m, d, key_callback=_key_callback) as viewer:
            nsamples = len(rollout)
            dt = m.opt.timestep
            i = 0
            while viewer.is_running():
                start = time.time()
                if _must_quit:
                    break
                if _must_reset:
                    i = 0
                    pipeline_state = rollout[i]
                    mjx.device_get_into(d, pipeline_state)
                    viewer.sync()
                    _must_reset = False
                if not _paused:
                    pipeline_state = rollout[i]
                    # jax_print(f"max(xfrc_applied)={pipeline_state.xfrc_applied.max()} |  min(xfrc_applied)={pipeline_state.xfrc_applied.min()} | qfrc_applied={pipeline_state.qfrc_applied}")
                    mjx.device_get_into(d, pipeline_state)
                    viewer.sync()
                    end = time.time()
                    elapsed = end - start
                    i = (i + 1) % nsamples
                time.sleep(max(0, dt / _real_time_factor - elapsed))


class ArmSensor(Node):
    def __init__(self, world, *args, **kwargs):
        self._world = world
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ArmOutput:
        """Default output of the node."""
        mjx_output = self._world.default_output(rng, graph_state)
        return ArmOutput(jpos=mjx_output.jpos, eepos=mjx_output.eepos, eeorn=mjx_output.eeorn)

    def step(self, step_state: StepState) -> Tuple[StepState, ArmOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        mjx_output = inputs["world"][-1].data
        return new_step_state, ArmOutput(jpos=mjx_output.jpos, eepos=mjx_output.eepos, eeorn=mjx_output.eeorn)


class BoxSensor(Node):
    def __init__(self, world, *args, **kwargs):
        self._world = world
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> BoxOutput:
        """Default output of the node."""
        mjx_output = self._world.default_output(rng, graph_state)
        return BoxOutput(boxpos=mjx_output.boxpos, boxorn=mjx_output.boxorn)

    def step(self, step_state: StepState) -> Tuple[StepState, BoxOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        mjx_output = inputs["world"][-1].data
        return new_step_state, BoxOutput(boxpos=mjx_output.boxpos, boxorn=mjx_output.boxorn)


class ArmActuator(Node):
    def __init__(self, world, *args, **kwargs):
        self._world = world
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        mjx_output = self._world.default_output(rng, graph_state)
        return ActuatorOutput(jpos=mjx_output.jpos)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        actuator_output = inputs["controller"][-1].data
        return new_step_state, actuator_output


@struct.dataclass
class ViewerOutput:
    boxsensor: BoxOutput
    armsensor: ArmOutput
    supervisor: SupervisorOutput
    qpos: jp.ndarray


class Viewer(Node):
    def __init__(self, xml_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load system
        self._xml_path = xml_path
        self._mj_m = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_d = mujoco.MjData(self._mj_m)
        self._mjx_m = mjx.device_put(self._mj_m)
        self._mjx_d = mjx.device_put(self._mj_d)
        self._viewer = None
        self.open()

        # boxpos = np.array([0.1, 0.0, 0.00])
        # boxyaw = np.array([3.14/4])
        # goalpos = np.array([0.0, 0.0])
        # jpos = np.array([0, 0, 0, 0, 3.14/2, 0])
        # qpos = np.concatenate([boxpos, boxyaw, goalpos, jpos, [0]])
        #
        # self._mj_d.qpos[:] = qpos
        # mujoco.mj_forward(self._mj_m, self._mj_d)
        # self._viewer.sync()
        #
        #
        # mjx_d = self._mjx_d.replace(qpos=qpos)
        # mjx_d = mjx.forward(self._mjx_m, mjx_d)
        # mjx.device_get_into(self._mj_d, mjx_d)
        # self._viewer.sync()

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
        """Default params of the node."""
        return Empty()

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
        return Empty()

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ViewerOutput:
        """Default output of the node."""
        # Grab output from state
        default_outputs = {}
        for i in self.inputs:
            default_outputs[i.input_name] = i.output.node.default_output(rng, graph_state)
        viewer_output = ViewerOutput(**default_outputs, qpos=None)
        qpos = self._get_qpos(viewer_output)
        viewer_output = ViewerOutput(**default_outputs, qpos=qpos)
        return viewer_output

    def step(self, step_state: StepState) -> Tuple[StepState, ViewerOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get output
        default_outputs = {}
        for i in self.inputs:
            default_outputs[i.input_name] = inputs[i.input_name][-1].data
        viewer_output = ViewerOutput(**default_outputs, qpos=None)
        qpos = self._get_qpos(viewer_output)
        viewer_output = ViewerOutput(**default_outputs, qpos=qpos)

        # Update viewer
        mjx_d = self._mjx_d.replace(qpos=qpos)
        mjx_d = mjx.forward(self._mjx_m, mjx_d)
        _ = host_callback.call(self._sync_viewer, mjx_d, result_shape=jax.ShapeDtypeStruct((), np.float32))

        # Update state
        new_step_state = step_state

        return new_step_state, viewer_output

    def close(self):
        self._viewer.close()
        self._viewer = None

    def open(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._mj_m, self._mj_d, key_callback=self._key_callback)
            self._paused = False

    def _sync_viewer(self, mjx_d):
        self.open()
        mjx.device_get_into(self._mj_d, mjx_d)
        if not self._paused:
            self._viewer.sync()
        return 1.

    def _get_qpos(self, viewer_output: ViewerOutput) -> jp.ndarray:
        boxyaw = jp.array([viewer_output.boxsensor.wrapped_yaw])
        boxpos = viewer_output.boxsensor.boxpos
        goalpos = viewer_output.supervisor.goalpos
        goalyaw = viewer_output.supervisor.goalyaw
        jpos = viewer_output.armsensor.jpos
        qpos = jnp.concatenate([boxpos, boxyaw, goalpos, jpos, jp.array([0])])
        # jax_print("qpos={qpos}", qpos=qpos)
        return qpos

    def _key_callback(self, keycode):
        if chr(keycode) == ' ':
            self._paused = not self._paused
