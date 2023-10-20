import os
from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
import jax
from math import ceil
from flax import struct

from brax.base import State as BraxState, System as BraxSystem, Transform
from brax.math import quat_to_euler
from brax.io import mjcf
from brax.generalized import pipeline as g_pipeline
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline

from rex.distributions import Distribution, Gaussian
from rex.constants import WARN, LATEST, PHASE
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.multiprocessing import new_process

from envs.vx300s.env import ActuatorOutput, ArmOutput, BoxOutput
# from envs.vx300s.render import Render


def build_vx300s(rates: Dict[str, float],
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
    world = World(name="world", rate=rates["world"], scheduling=scheduling, advance=False,
                  delay=process["world"], delay_sim=process_sim["world"])
    armsensor = ArmSensor(world, name="armsensor", rate=rates["armsensor"], scheduling=scheduling, advance=False,
                          delay=process["armsensor"], delay_sim=process_sim["armsensor"])
    boxsensor = BoxSensor(world, name="boxsensor", rate=rates["boxsensor"], scheduling=scheduling, advance=False,
                          delay=process["boxsensor"], delay_sim=process_sim["boxsensor"])
    armactuator = ArmActuator(world, name="armactuator", rate=rates["armactuator"], scheduling=scheduling, advance=False,
                              delay=process["armactuator"], delay_sim=process_sim["armactuator"])

    # Connect nodes
    world.connect(armactuator, window=1, blocking=True, skip=False, jitter=LATEST,
                  delay_sim=trans_sim["world"]["armactuator"], delay=trans["world"]["armactuator"])
    armsensor.connect(world, window=1, blocking=True, skip=False, jitter=LATEST,
                      delay_sim=trans_sim["armsensor"]["world"], delay=trans["armsensor"]["world"])
    boxsensor.connect(world, window=1, blocking=True, skip=False, jitter=LATEST,
                      delay_sim=trans_sim["boxsensor"]["world"], delay=trans["boxsensor"]["world"])

    return dict(world=world, armactuator=armactuator, armsensor=armsensor, boxsensor=boxsensor)


@struct.dataclass
class Params:
    # sys: BraxSystem
    pass


@struct.dataclass
class State:
    pipeline_state: BraxState


@struct.dataclass
class BraxOutput:
    jpos: jp.ndarray
    eepos: jp.ndarray
    eeorn: jp.ndarray
    boxpos: jp.ndarray
    boxorn: jp.ndarray


PIPELINE = {
    'generalized': g_pipeline,
    'spring': s_pipeline,
    'positional': p_pipeline,
}


class World(Node):
    def __init__(self, *args, dt_brax: float = 0.04, backend: str = "generalized", debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_brax)
        self.dt_brax = dt / self.substeps
        self.backend = backend
        self._pipeline = PIPELINE[backend]

        # Load system
        path = os.path.dirname(__file__) + "/../assets/vx300s.xml"
        self.sys: BraxSystem = mjcf.load(path)
        self.sys = self.sys.replace(dt=self.dt_brax)

        # Get indices
        self._joint_idx = self.sys.actuator.q_id.tolist()
        self._joint_slice = slice(self._joint_idx[0], self._joint_idx[-1] + 1)
        self._ee_arm_idx = self.sys.link_names.index("ee_link")
        self._box_idx = self.sys.link_names.index("box")
        self._goal_idx = self.sys.link_names.index("goal")

    def __getstate__(self):
        args, kwargs, inputs = super().__getstate__()
        kwargs.update(dict(dt_brax=self.dt_brax, backend=self.backend, debug=self.debug))
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
        return Params()

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
        """Default state of the node."""
        # Try to grab state from graph_state
        goalpos = graph_state.nodes["planner"].state.goalpos
        boxpos_home = graph_state.nodes["planner"].params.boxpos_home

        # Set joint positions
        qpos = self.sys.init_q
        qpos = qpos.at[1:5].set(jp.concatenate([boxpos_home, goalpos]))
        pipeline_state = self._pipeline.init(self.sys, qpos, jp.zeros(self.sys.qd_size()))
        return State(pipeline_state=pipeline_state)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> BraxOutput:
        """Default output of the node."""
        # Grab output from state
        try:
            pipeline_state = graph_state.nodes["world"].state.pipeline_state
            brax_output = self._get_output(pipeline_state)
        except (AttributeError):
            jpos = jp.zeros(self.sys.actuator.q_id.shape, dtype=jp.float32)
            eepos = jp.zeros((3,), dtype=jp.float32)
            eeorn = jp.zeros((3,), dtype=jp.float32)
            boxpos = jp.zeros((3,), dtype=jp.float32)
            boxorn = jp.zeros((3,), dtype=jp.float32)
            brax_output = BraxOutput(jpos=jpos, eepos=eepos, eeorn=eeorn, boxpos=boxpos, boxorn=boxorn)
        return brax_output

    def _get_output(self, pipeline_state: BraxState) -> BraxOutput:
        x_i = pipeline_state.x.vmap().do(
            Transform.create(pos=self.sys.link.inertia.transform.pos)
        )
        jpos = pipeline_state.q[self._joint_slice]
        eepos = x_i.pos[self._ee_arm_idx]
        eeorn = quat_to_euler(x_i.rot[self._ee_arm_idx])
        boxpos = x_i.pos[self._box_idx]
        boxorn = quat_to_euler(x_i.rot[self._box_idx])
        return BraxOutput(jpos=jpos, eepos=eepos, eeorn=eeorn, boxpos=boxpos, boxorn=boxorn)

    def step(self, step_state: StepState) -> Tuple[StepState, BraxOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        action = inputs["armactuator"].data.jpos[-1]

        def f(state, _):
            return (
                self._pipeline.step(self.sys, state, action, self.debug),
                None,
            )

        new_pipeline_state = jax.lax.scan(f, state.pipeline_state, (), self.substeps)[0]

        # Update state
        new_state = state.replace(pipeline_state=new_pipeline_state)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        brax_output = self._get_output(new_pipeline_state)
        return new_step_state, brax_output


class ArmSensor(Node):
    def __init__(self, world, *args, **kwargs):
        self._world = world
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ArmOutput:
        """Default output of the node."""
        brax_output = self._world.default_output(rng, graph_state)
        return ArmOutput(jpos=brax_output.jpos, eepos=brax_output.eepos, eeorn=brax_output.eeorn)

    def step(self, step_state: StepState) -> Tuple[StepState, ArmOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        brax_output = inputs["world"][-1].data
        return new_step_state, ArmOutput(jpos=brax_output.jpos, eepos=brax_output.eepos, eeorn=brax_output.eeorn)


class BoxSensor(Node):
    def __init__(self, world, *args, **kwargs):
        self._world = world
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> BoxOutput:
        """Default output of the node."""
        brax_output = self._world.default_output(rng, graph_state)
        return BoxOutput(boxpos=brax_output.boxpos, boxorn=brax_output.boxorn)

    def step(self, step_state: StepState) -> Tuple[StepState, BoxOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        brax_output = inputs["world"][-1].data
        return new_step_state, BoxOutput(boxpos=brax_output.boxpos, boxorn=brax_output.boxorn)


class ArmActuator(Node):
    def __init__(self, world, *args, **kwargs):
        self._world = world
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        brax_output = self._world.default_output(rng, graph_state)
        return ActuatorOutput(jpos=brax_output.jpos)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        actuator_output = inputs["controller"][-1].data
        return new_step_state, actuator_output