import os
from typing import Any, Dict, Tuple, Union, List
import numpy as np
import brax
from brax.io import html
import jax
import jax.experimental.host_callback as hcb
from jax.debug import print as jax_print
import jax.numpy as jnp
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

from envs.vx300s.env import ActuatorOutput, ArmOutput, BoxOutput
# from envs.vx300s.render import Render


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
    world = World(xml_path=config["brax"]["xml_path"], name="world", rate=rates["world"], scheduling=scheduling, advance=False,
                  delay=process["world"], delay_sim=process_sim["world"])
    armsensor = ArmSensor(name="armsensor", rate=rates["armsensor"], scheduling=scheduling, advance=False,
                          delay=process["armsensor"], delay_sim=process_sim["armsensor"] )
    boxsensor = BoxSensor(name="boxsensor", rate=rates["boxsensor"], scheduling=scheduling, advance=False,
                          delay=process["boxsensor"], delay_sim=process_sim["boxsensor"])
    armactuator = ArmActuator(name="armactuator", rate=rates["armactuator"], scheduling=scheduling, advance=False,
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
    pipeline_state: BraxState


@struct.dataclass
class BraxOutput:
    jpos: jax.typing.ArrayLike
    eepos: jax.typing.ArrayLike
    eeorn: jax.typing.ArrayLike
    boxpos: jax.typing.ArrayLike
    boxorn: jax.typing.ArrayLike


PIPELINE = {
    'generalized': g_pipeline,
    'spring': s_pipeline,
    'positional': p_pipeline,
}


class World(Node):
    def __init__(self, xml_path: str, *args, dt_brax: float = 0.015, backend: str = "generalized", debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_brax)
        self.dt_brax = dt / self.substeps
        self.backend = backend
        self._pipeline = PIPELINE[backend]

        # Load system
        self._xml_path = xml_path
        self.sys: BraxSystem = mjcf.load(xml_path)
        self.sys = self.sys.replace(dt=self.dt_brax)

        # Get indices
        self._joint_idx = self.sys.actuator.q_id[:6].tolist()
        self._joint_slice = slice(self._joint_idx[0], self._joint_idx[-1] + 1)
        self._ee_arm_idx = self.sys.link_names.index("ee_link")
        self._box_idx = self.sys.link_names.index("box")
        self._goal_idx = self.sys.link_names.index("goal")

    def __getstate__(self):
        args, kwargs, inputs = super().__getstate__()
        kwargs.update(dict(xml_path=self._xml_path, dt_brax=self.dt_brax, backend=self.backend, debug=self.debug))
        return args, kwargs, inputs

    def default_params(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> Params:
        """Default params of the node."""
        # Try to grab params from graph_state
        try:
            params = graph_state.nodes[self.name].params
            if params is not None:
                return params
        except (AttributeError, KeyError):
            pass
        return Params()

    def default_state(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> State:
        """Default state of the node."""
        # if graph_state is not None and "supervisor" in graph_state.nodes:
        # Try to grab state from graph_state
        goalpos = graph_state.nodes["supervisor"].state.goalpos
        home_boxpos = graph_state.nodes["supervisor"].params.home_boxpos
        home_boxyaw = graph_state.nodes["supervisor"].params.home_boxyaw
        home_jpos = graph_state.nodes["supervisor"].params.home_jpos
        # Set joint positions
        qpos = jnp.concatenate([home_boxpos, home_boxyaw, goalpos, home_jpos, np.array([0.])], dtype=np.float32)
        # else:
        #     qpos = self.sys.init_q
        qd = jnp.zeros(self.sys.qd_size(), dtype=jnp.float32)
        pipeline_state = self._pipeline.init(self.sys, qpos, qd)
        return State(pipeline_state=pipeline_state)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> BraxOutput:
        """Default output of the node."""
        # Grab output from state
        # if graph_state is not None:
        pipeline_state = graph_state.nodes["world"].state.pipeline_state
        brax_output = self._get_output(pipeline_state)
        # else:
        #     brax_output = self._get_output()
        return brax_output

    def _get_output(self, pipeline_state: BraxState = None) -> BraxOutput:
        if pipeline_state is None:
            jpos = jnp.zeros((self.sys.act_size(),), dtype=jnp.float32)
            eepos = jnp.zeros((3,), dtype=jnp.float32)
            eeorn = jnp.array([0, 0, 0, 1], dtype=jnp.float32)
            boxpos = jnp.zeros((3,), dtype=jnp.float32)
            boxorn = jnp.array([0, 0, 0, 1], dtype=jnp.float32)
        else:
            x_i = pipeline_state.x.vmap().do(
                Transform.create(pos=self.sys.link.inertia.transform.pos)
            )
            jpos = pipeline_state.q[self._joint_slice]
            eepos = x_i.pos[self._ee_arm_idx]
            eeorn = self._convert_wxyz_to_xyzw(x_i.rot[self._ee_arm_idx])  # quaternion (w,x,y,z) -> (x,y,z,w)
            boxpos = x_i.pos[self._box_idx]
            boxorn = self._convert_wxyz_to_xyzw(x_i.rot[self._box_idx])  # quaternion (w,x,y,z) -> (x,y,z,w)
        return BraxOutput(jpos=jpos, eepos=eepos, eeorn=eeorn, boxpos=boxpos, boxorn=boxorn)

    def _convert_wxyz_to_xyzw(self, quat: jax.typing.ArrayLike):
        """Convert quaternion (w,x,y,z) -> (x,y,z,w)"""
        return jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype="float32")

    def _get_PD(self, sys: BraxSystem, q_des):
        if sys.act_size() == 12:
            action = jnp.concatenate([q_des, 0 * jnp.ones(q_des.shape)])
        else:
            action = q_des
        return action

    def step(self, step_state: StepState) -> Tuple[StepState, BraxOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        action = inputs["armactuator"].data.jpos[-1]

        # Convert to PD action (if needed)
        action_pd = self._get_PD(self.sys, action)

        def f(state, _):
            return (
                self._pipeline.step(self.sys, state, action_pd, self.debug),
                None,
            )

        new_pipeline_state = jax.lax.scan(f, state.pipeline_state, (), self.substeps)[0]

        # jax_print("qf_constraint: {qf_constraint}", qf_constraint=new_pipeline_state.qf_constraint[:3])
        # hcb.call(lambda qf_constraint: print(f"qf_constraint: {qf_constraint}"), new_pipeline_state.qf_constraint)

        # Update state
        new_state = state.replace(pipeline_state=new_pipeline_state)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        brax_output = self._get_output(new_pipeline_state)
        return new_step_state, brax_output

    def view_rollout(self, rollout: List[BraxState], sys=None, verbose=False, path: str = None, dt: float = None, **kwargs):
        """Render a rollout."""
        sys = self.sys if sys is None else sys
        path = "./brax_render.html" if path is None else path
        dt = sys.dt if dt is None else dt
        sys = sys.replace(dt=dt)

        # Check if directory to path exists
        assert os.path.exists(os.path.dirname(path)), f"Directory {os.path.dirname(path)} does not exist"

        # save rollout
        html.save(path, sys, rollout)
        if verbose:
            print(f"Saved rollout to {path}")


class ArmSensor(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> ArmOutput:
        """Default output of the node."""
        world = self.get_connected_output("world").node
        brax_output = world.default_output(rng, graph_state)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> BoxOutput:
        """Default output of the node."""
        world = self.get_connected_output("world").node
        brax_output = world.default_output(rng, graph_state)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        world = self.get_connected_input("world").node
        brax_output = world.default_output(rng, graph_state)
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