from typing import Any, Dict, Tuple, Union
import jax
import jax.numpy as jnp
from math import ceil
from flax import struct
from flax.core import FrozenDict
from rexv2.base import StepState, GraphState, Empty, TrainableDist, Base
from rexv2.node import BaseNode

from envs.pendulum.base import ActuatorOutput, WorldState, SensorOutput, WorldParams, SensorParams, ActuatorParams

try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.spring import pipeline as spring_pipeline
    from brax.positional import pipeline as pos_pipeline
    from brax.base import System, State
    from brax.io import mjcf

    Systems = Union[gen_pipeline.System, spring_pipeline.System, pos_pipeline.System]
    Pipelines = Union[gen_pipeline.State, spring_pipeline.State, pos_pipeline.State]
    BRAX_INSTALLED = True
except ModuleNotFoundError as e:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False
    raise e


@struct.dataclass
class BraxParams(WorldParams):
    max_speed: Union[float, jax.typing.ArrayLike]
    damping: Union[float, jax.typing.ArrayLike]
    armature: Union[float, jax.typing.ArrayLike]
    gear: Union[float, jax.typing.ArrayLike]
    mass_weight: Union[float, jax.typing.ArrayLike]
    radius_weight: Union[float, jax.typing.ArrayLike]
    offset: Union[float, jax.typing.ArrayLike]
    friction_loss: Union[float, jax.typing.ArrayLike]
    backend: str = struct.field(pytree_node=False)
    base_sys: Systems = struct.field(pytree_node=False)
    dt: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False)

    @property
    def substeps(self) -> int:
        dt_substeps_per_backend = {
            "generalized": 1 / 100,
            "spring": 1 / 100,
            "positional": 1 / 100
        }[self.backend]
        substeps = ceil(self.dt / dt_substeps_per_backend)
        return int(substeps)

    @property
    def dt_substeps(self) -> float:
        substeps = self.substeps
        dt_substeps = self.dt / substeps
        return dt_substeps

    @property
    def pipeline(self) -> Pipelines:
        return {
            "generalized": gen_pipeline,
            "spring": spring_pipeline,
            "positional": pos_pipeline
        }[self.backend]

    @property
    def sys(self) -> Systems:
        # Appropriately replace parameters for the disk pendulum
        itransform = self.base_sys.link.inertia.transform.replace(pos=jnp.array([[0.0, self.offset, 0.0]]))
        i = self.base_sys.link.inertia.i.at[0, 0, 0].set(
            0.5 * self.mass_weight * self.radius_weight**2
        )  # inertia of cylinder in local frame.
        inertia = self.base_sys.link.inertia.replace(transform=itransform, mass=jnp.array([self.mass_weight]), i=i)
        link = self.base_sys.link.replace(inertia=inertia)
        actuator = self.base_sys.actuator.replace(gear=jnp.array([self.gear]))
        dof = self.base_sys.dof.replace(armature=jnp.array([self.armature]), damping=jnp.array([self.damping]))
        opt = self.base_sys.opt.replace(timestep=self.dt_substeps)
        new_sys = self.base_sys.replace(link=link, actuator=actuator, dof=dof, opt=opt)
        return new_sys

    def step(self, substeps: int, dt_substeps: Union[float, jax.typing.ArrayLike], x: Pipelines, us: jax.typing.ArrayLike) -> Tuple[Pipelines, Pipelines]:
        """Step the pendulum ode."""
        # Appropriately replace timestep for the disk pendulum
        sys = self.sys.replace(opt=self.sys.opt.replace(timestep=self.dt_substeps))

        def _scan_fn(_x, _u):
            # Add friction loss
            thdot = x.qd[0]
            activation = jnp.sign(thdot)
            friction = self.friction_loss * activation / sys.actuator.gear[0]
            _u_friction = _u - friction
            # Step
            next_x = gen_pipeline.step(sys, _x, jnp.array(_u_friction)[None])
            # Clip velocity
            next_x = next_x.replace(qd=jnp.clip(next_x.qd, -self.max_speed, self.max_speed))
            return next_x, next_x

        x_final, x_substeps = jax.lax.scan(_scan_fn, x, us, length=substeps)
        return x_final, x_substeps


@struct.dataclass
class BraxState:
    """Pendulum state definition"""
    last_ts: Union[int, jax.typing.ArrayLike]
    loss_th: Union[float, jax.typing.ArrayLike]
    loss_thdot: Union[float, jax.typing.ArrayLike]
    loss_ts: Union[float, jax.typing.ArrayLike]
    loss_task: Union[float, jax.typing.ArrayLike]
    pipeline_state: Pipelines

    @property
    def th(self):
        return self.pipeline_state.q[..., 0]

    @property
    def thdot(self):
        return self.pipeline_state.qd[..., 0]


class World(BaseNode):
    def __init__(self, *args, backend: str = "generalized", **kwargs):
        assert BRAX_INSTALLED, "Brax not installed. Install it with `pip install brax`"
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.sys = mjcf.loads(DISK_PENDULUM_XML)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxParams:
        """Default params of the node."""
        # Try to grab params from graph_state
        graph_state = graph_state or GraphState()
        actuator = self.inputs["actuator"].output_node
        actuator_delay = graph_state.params.get("actuator", actuator.init_params(rng, graph_state)).actuator_delay
        return BraxParams(
            actuator_delay=actuator_delay,
            # Realistic parameters for the disk pendulum
            max_speed=40.0,
            damping=0.00015877,
            armature=6.4940527e-06,
            gear=0.00428677,
            mass_weight=0.05076142,
            radius_weight=0.05121992,
            offset=0.04161447,
            friction_loss=0.00097525,
            # Backend parameters
            dt=1/self.rate,
            base_sys=self.sys,
            backend=self.backend,
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxState:
        """Default state of the node."""
        graph_state = graph_state or GraphState()

        # Try to grab state from graph_state
        state = graph_state.state.get("supervisor", None)
        th = state.init_th if state is not None else jnp.pi
        thdot = state.init_thdot if state is not None else 0.

        # Get params
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sys = params.sys
        q = sys.init_q.at[0].set(th)
        qd = jnp.array([thdot])
        pipeline_state = params.pipeline.init(sys, q, qd)
        return BraxState(pipeline_state=pipeline_state,
                         last_ts=0., loss_th=0.0, loss_thdot=0.0, loss_ts=0.0, loss_task=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldState:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        # Grab output from state
        state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return WorldState(th=state.th, thdot=state.thdot)

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        if isinstance(inputs["actuator"].delay_dist, TrainableDist):
            inputs["actuator"] = inputs["actuator"].replace(delay_dist=params.actuator_delay)
        return FrozenDict(inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, WorldState]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        params: BraxParams
        state: BraxState

        # Get state estimate
        data: ActuatorOutput = step_state.inputs["actuator"][-1].data
        ts = step_state.ts if data.state_estimate is None else data.state_estimate.ts
        mean = state if data.state_estimate is None else data.state_estimate.mean

        # Only update loss if actuator.seq is new (i.e. only penalize state mismatch at the start of a new action)
        alpha = state.last_ts < ts
        loss_th = state.loss_th + alpha * (jnp.sin(mean.th) - jnp.sin(state.th)) ** 2 + alpha * (jnp.cos(mean.th) - jnp.cos(state.th)) ** 2
        loss_thdot = state.loss_thdot + alpha * (mean.thdot - state.thdot) ** 2
        loss_ts = state.loss_ts + alpha * (ts - step_state.ts) ** 2
        new_last_ts = ts

        # Apply dynamics
        u = inputs["actuator"].data.action[-1][0]
        us = jnp.array([u] * params.substeps)
        x = state.pipeline_state
        next_x = params.step(params.substeps, params.dt_substeps, x, us)[0]
        new_state = state.replace(pipeline_state=next_x)
        next_th, next_thdot = new_state.th, new_state.thdot
        output = WorldState(th=next_th, thdot=next_thdot)  # Prepare output

        # Calculate cost (penalize angle error, angular velocity and input voltage)
        norm_next_th = self._angle_normalize(next_th)
        loss_task = state.loss_task + norm_next_th ** 2 + 0.1 * (next_thdot / (1 + 10 * abs(norm_next_th))) ** 2 + 0.01 * u ** 2
        # loss_task = state.loss_task +  norm_next_th ** 2 + 0.01 * next_thdot ** 2 + 0.001 * (u ** 2)

        # Update state
        new_state = new_state.replace(last_ts=new_last_ts, loss_th=loss_th, loss_thdot=loss_thdot, loss_ts=loss_ts, loss_task=loss_task)
        new_step_state = step_state.replace(state=new_state)

        # print(f"{self.name.ljust(14)} | x: {x} | u: {u} -> next_x: {next_x}")
        return new_step_state, output

    @staticmethod
    def _angle_normalize(th: jax.typing.ArrayLike):
        th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        return th_norm


DISK_PENDULUM_XML = """
<mujoco model="disk_pendulum">
    <compiler inertiafromgeom="auto" angle="radian" coordinate="local" eulerseq="xyz" autolimits="true"/>
    <option gravity="0 0 -9.81" timestep="0.01" iterations="10"/>
    <custom>
        <numeric data="10" name="constraint_ang_damping"/> <!-- positional & spring -->
        <numeric data="1" name="spring_inertia_scale"/>  <!-- positional & spring -->
        <numeric data="0" name="ang_damping"/>  <!-- positional & spring -->
        <numeric data="0" name="spring_mass_scale"/>  <!-- positional & spring -->
        <numeric data="0.5" name="joint_scale_pos"/> <!-- positional -->
        <numeric data="0.1" name="joint_scale_ang"/> <!-- positional -->
        <numeric data="3000" name="constraint_stiffness"/>  <!-- spring -->
        <numeric data="10000" name="constraint_limit_stiffness"/>  <!-- spring -->
        <numeric data="50" name="constraint_vel_damping"/>  <!-- spring -->
        <numeric data="10" name="solver_maxls"/>  <!-- generalized -->
    </custom>

    <asset>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>

    <default>
        <geom contype="0" friction="1 0.1 0.1" material="geom"/>
    </default>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom name="table" type="plane" pos="0 0.0 -0.1" size="1 1 0.1" contype="8" conaffinity="11" condim="3"/>
        <body name="disk" pos="0.0 0.0 0.0" euler="1.5708 0.0 0.0">
            <joint name="hinge_joint" type="hinge" axis="0 0 1" range="-180 180" armature="0.00022993" damping="0.0001" limited="false"/>
            <geom name="disk_geom" type="cylinder" size="0.06 0.001" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.2 1" mass="0.0"/>
            <geom name="mass_geom" type="cylinder" size="0.02 0.005" contype="0" conaffinity="0"  condim="3" rgba="0.3 0.08 0.08 1" pos="0.0 0.04 0." mass="0.05085817"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="hinge_joint" ctrllimited="false" ctrlrange="-3.0 3.0"  gear="0.01"/>
    </actuator>
</mujoco>
"""

CAM_DISK_PENDULUM_XML = """
<mujoco model="disk_pendulum">
    <compiler inertiafromgeom="auto" angle="radian" coordinate="local" eulerseq="xyz" autolimits="true" meshdir="/home/r2ci/rex/envs/pendulum/assets"/>
    <option gravity="0 0 -9.81" timestep="0.01" iterations="10"/>
    <custom>
        <numeric data="10" name="constraint_ang_damping"/> <!-- positional & spring -->
        <numeric data="1" name="spring_inertia_scale"/>  <!-- positional & spring -->
        <numeric data="0" name="ang_damping"/>  <!-- positional & spring -->
        <numeric data="0" name="spring_mass_scale"/>  <!-- positional & spring -->
        <numeric data="0.5" name="joint_scale_pos"/> <!-- positional -->
        <numeric data="0.1" name="joint_scale_ang"/> <!-- positional -->
        <numeric data="3000" name="constraint_stiffness"/>  <!-- spring -->
        <numeric data="10000" name="constraint_limit_stiffness"/>  <!-- spring -->
        <numeric data="50" name="constraint_vel_damping"/>  <!-- spring -->
        <numeric data="10" name="solver_maxls"/>  <!-- generalized -->
    </custom>

    <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true" specular="1" shininess="1.0"/>
        <material name="disk" reflectance="1.0" specular="1" shininess="1"/>
        
        <mesh file="d435i_0.obj"/>
        <mesh file="d435i_1.obj"/>
        <mesh file="d435i_2.obj"/>
        <mesh file="d435i_3.obj"/>
        <mesh file="d435i_4.obj"/>
        <mesh file="d435i_5.obj"/>
        <mesh file="d435i_6.obj"/>
        <mesh file="d435i_7.obj"/>
        <mesh file="d435i_8.obj"/>
    </asset>

    <default>
        <geom contype="0" friction="1 0.1 0.1" material="disk"/>
        
        <default class="d435i">
            <material specular="0" shininess="0.25"/>
            <default class="visual">
                <geom group="2" type="mesh" contype="0" conaffinity="0" mass="0"/>
            </default>
            <default class="collision">
                <geom group="3" type="mesh" mass="0"/>
            </default>
        </default>
    </default>

    <worldbody>
        <light cutoff="45" diffuse="1 1 1" dir="0 -1 -1" directional="true" exponent="1" pos="0. 1.0 1.0" specular="1 1 1"/>
        <geom name="floor" type="plane" conaffinity="0" condim="3" pos="0 0 -0.095" rgba="0.79 0.79 0.79 1.0" size="1 1 1" />
        <geom name="table" type="box" size="0.15 0.25 0.01" contype="0" conaffinity="0" condim="3" rgba="0.65 0.41 0.199 1.0" mass="0.0" pos="0. 0.0 -0.1"/>
        <geom name="light_geom" type="cylinder" size="0.003 0.001" contype="0" conaffinity="0" condim="3" rgba="0.2 0.2 1.0 1" mass="0.0" pos="-0.06 0. 0.06" euler="1.5708 0.0 0.0"/>
        <geom name="box_geom" type="box" size="0.08 0.12 0.08" contype="0" conaffinity="0" condim="3" rgba="0.8 0.8 0.8 1" mass="0.0" pos="0. -0.122 0."/>
        <geom name="corner1_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="-0.075 -0.12 -0.075" euler="1.5708 0.0 0.0"/>
        <geom name="corner2_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="0.075 -0.12 0.075" euler="1.5708 0.0 0.0"/>
        <geom name="corner3_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="-0.075 -0.12 0.075" euler="1.5708 0.0 0.0"/>
        <geom name="corner4_geom" type="cylinder" size="0.015 0.117" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0" pos="0.075 -0.12 -0.075" euler="1.5708 0.0 0.0"/>
        <body name="disk" pos="0.0 0.0 0.0" euler="1.5708 0.0 0.0">
            <joint name="hinge_joint" type="hinge" axis="0 0 1" range="-180 180" armature="0.00022993" damping="0.0001" limited="false"/>
            <geom name="hinge_geom" type="cylinder" size="0.014 0.007" contype="0" conaffinity="0" condim="3" rgba="0.6 0.6 0.6 1" mass="0.0"/>
            <geom name="screw_top_geom" type="cylinder" size="0.003 0.002" contype="0" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" mass="0.0" pos="0.0 -0.005 -0.007"/>
            <geom name="screw_right_geom" type="cylinder" size="0.003 0.002" contype="0" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" mass="0.0" pos="0.005 0.004 -0.007"/>
            <geom name="screw_left_geom" type="cylinder" size="0.003 0.002" contype="0" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" mass="0.0" pos="-0.005 0.004 -0.007"/>
            <geom name="disk_geom" type="cylinder" size="0.06 0.001" contype="0" conaffinity="0" condim="3" rgba="0.08 0.08 0.3 1" mass="0.0"/>
            <geom name="mass_geom" type="cylinder" size="0.02 0.005" contype="0" conaffinity="0"  condim="3" rgba="0.5 0.08 0.08 1" pos="0.0 0.04 0." mass="0.05085817"/>
            <geom name="hole_geom" type="cylinder" size="0.002 0.002" contype="0" conaffinity="0"  condim="3" rgba="0.8 0.8 0.8 1" pos="0.0 -0.04 0." mass="0.0"/>      
        </body>
        
        <!--
        <body name="world" pos="0.0 0.0 0.0" euler="0.0 0.0 0.0">
            <geom name="x_axis_w" type="cylinder" size="0.001 0.01" pos="0.01 0.03 0" euler="0 1.5707963 0" rgba="1 0 0 1" contype="0" conaffinity="0" condim="3"/>
            <geom name="y_axis_w" type="cylinder" size="0.001 0.01" pos="0 0.04 0" euler="1.5707963 0 0" rgba="0 1 0 1" contype="0" conaffinity="0" condim="3"/>
            <geom name="z_axis_w" type="cylinder" size="0.001 0.01" pos="0 0.03 0.01" euler="0 0 0" rgba="0 0 1 1" contype="0" conaffinity="0" condim="3"/>
        </body>
        -->
        
        <body name="camera" pos="0.0 0.0 0.0" euler="0.0 0.0 0.0">
            <camera name="d435i_render" mode="fixed" focalpixel="308.773 308.724" principalpixel="214.688 118.102" resolution="424 240" pos="0 0 0" sensorsize="0 0" euler="3.1413 0 0"/>
            <geom name="x_axis" type="cylinder" size="0.001 0.01" pos="0.01 0 0" euler="0 1.5707963 0" rgba="1 0 0 1" contype="0" conaffinity="0" condim="3"/>
            <geom name="y_axis" type="cylinder" size="0.001 0.01" pos="0 0.01 0" euler="1.5707963 0 0" rgba="0 1 0 1" contype="0" conaffinity="0" condim="3"/>
            <geom name="z_axis" type="cylinder" size="0.001 0.01" pos="0 0 0.01" euler="0 0 0" rgba="0 0 1 1" contype="0" conaffinity="0" condim="3"/>
            <joint name="camera_joint" type="free" damping="0.1" limited="false"/>
            <inertial pos="0.0 0.0 0.0" mass="0.0001" diaginertia="0.0001 0.0001 0.0001"/>
            <body name="d435i_" childclass="d435i">
                <geom name="d435i_case" type="box" size="0.007 0.005 0.039" class="visual" rgba="0.05 0.05 0.05 1" pos="0.033 0.0 -0.009" euler="1.5708 1.5708 0.0"/>
                <geom name="d435i_ball" type="sphere" size="0.005" class="visual" rgba="0.8 0.8 0.8 1" pos="0.033 0.005 -0.009" euler="0.0 0.0 0.0"/>
                <geom name="d435i_leg1" type="cylinder" size="0.003 0.075" class="visual" rgba="0.8 0.8 0.8 1" pos="0.033 0.07 -0.009" euler="1.5708 0.0 0.0"/>
            </body>
            <body name="d435i" childclass="d435i" pos="0.0 0.0 0.0">
                <geom mesh="d435i_0" rgba="0.035601 0.035601 0.035601 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
                <geom mesh="d435i_1" rgba="0.287440 0.665387 0.327778 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
                <geom mesh="d435i_2" rgba="0.799102 0.806952 0.799103 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
                <geom mesh="d435i_3" rgba="0.035601 0.035601 0.035601 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
                <geom mesh="d435i_5" rgba="0.070360 0.070360 0.070360 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
                <geom mesh="d435i_6" rgba="0.070360 0.070360 0.070360 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
                <geom mesh="d435i_7" rgba="0.087140 0.002866 0.009346 1" class="visual" pos="0.033 0.0 0.0" euler="0.0 0.0 0.0"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="hinge_joint" ctrllimited="false" ctrlrange="-3.0 3.0"  gear="0.01"/>
    </actuator>
</mujoco>
"""


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath

    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


def get_pipeline_state(th: jax.typing.ArrayLike, dt: float, thdot: jax.typing.ArrayLike = None, cam_pos: jax.typing.ArrayLike = None, cam_orn: jax.typing.ArrayLike = None, sys=None):
    thdot = jnp.zeros_like(th) if thdot is None else thdot
    assert (cam_pos is None) == (cam_orn is None), "cam_pos and cam_orn must be both None or both not None"
    if not (cam_pos is None or cam_orn is None):
        XML = CAM_DISK_PENDULUM_XML
    else:
        XML = DISK_PENDULUM_XML
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    assert th.shape == thdot.shape, "th and thdot must have the same shape"
    assert th.ndim == 1, "th and thdot must be 1D arrays"

    # Initialize system
    sys = mjcf.loads(XML) if sys is None else sys
    sys = sys.replace(opt=sys.opt.replace(timestep=dt))

    def _set_pipeline_state(_th, _thdot):
        qpos = sys.init_q.at[0].set(_th)
        if cam_pos is not None:
            qpos = qpos.at[1:4].set(cam_pos)
            qpos = qpos.at[4:9].set(cam_orn)
        qvel = jnp.zeros((sys.qd_size()))
        qvel = qvel.at[0].set(_thdot)
        pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    jit_set_pipeline_state = jax.jit(_set_pipeline_state).lower(th[0], thdot[0]).compile()
    pipeline_state_lst = []
    for _th, _thdot in zip(th, thdot):
        pipeline_state_i = jit_set_pipeline_state(_th, _thdot)
        pipeline_state_lst.append(pipeline_state_i)
    return sys, pipeline_state_lst


def render(th: jax.typing.ArrayLike, dt: float, thdot: jax.typing.ArrayLike = None, cam_pos: jax.typing.ArrayLike = None, cam_orn: jax.typing.ArrayLike = None, sys=None):
    """Render the rollout as an HTML file."""
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    from brax.io import html
    sys, pipeline_state_lst = get_pipeline_state(th, dt, thdot, cam_pos, cam_orn, sys)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json
