from typing import Union
import jax
from jax import numpy as jnp

try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.io import mjcf
    from brax.io import html
except ModuleNotFoundError as e:
    print("Brax not installed. Install it with `pip install brax`")
    raise e

from rex.pendulum.brax import BraxState
from rex.pendulum.ode import OdeState


DISK_PENDULUM_VISUAL_XML = """
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
    </asset>

    <default>
        <geom contype="0" friction="1 0.1 0.1" material="disk"/>
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


def render(rollout: Union[BraxState, OdeState], dt: Union[float, jax.typing.ArrayLike] = 0.02, xml_string: str = DISK_PENDULUM_VISUAL_XML):
    """Render the rollout as an HTML file.

    :param rollout: Rollout to render
    :param dt: Time step
    :param xml_string: XML string to render
    """

    # Initialize system
    sys = mjcf.loads(xml_string)
    sys = sys.replace(opt=sys.opt.replace(timestep=dt))

    def _set_pipeline_state(th, thdot):
        qpos = sys.init_q.at[0].set(th)
        qvel = jnp.array([thdot])
        pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    pipeline_state_rollout = jax.vmap(_set_pipeline_state)(rollout.th, rollout.thdot)
    pipeline_state_lst = []
    for i in range(rollout.th.shape[0]):
        pipeline_state_i = jax.tree_util.tree_map(lambda x: x[i], pipeline_state_rollout)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json
