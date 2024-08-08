from typing import Any, Dict, Tuple, Union, TYPE_CHECKING, List
import jax
import jax.numpy as jnp
import numpy as onp
from math import ceil
from flax import struct
from flax.core import FrozenDict
from rexv2.base import StepState, GraphState, Empty, TrainableDist, Base
from rexv2.node import BaseNode
import rexv2.rl as rl
from rexv2.jax_utils import tree_dynamic_slice
if TYPE_CHECKING:
    from envs.crazyflie.pid import PIDOutput


try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.io import mjcf

    BRAX_INSTALLED = True
except ModuleNotFoundError:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False


def pwm_to_force(pwm_constants: jax.typing.ArrayLike, pwm: Union[float, jax.typing.ArrayLike]) -> Union[float, jax.Array]:
    # Modified formula from Julian Forster's Crazyflie identification
    a, b, c = pwm_constants
    force = 4 * (a * (pwm ** 2) + b * pwm + c)
    return force


def force_to_pwm(pwm_constants: jax.typing.ArrayLike, force: Union[float, jax.typing.ArrayLike]) -> Union[float, jax.Array]:
    # Just the inversion of pwm_to_force
    a, b, c = pwm_constants
    a = 4 * a
    b = 4 * b
    c = 4 * c - force
    d = b ** 2 - 4 * a * c
    pwm = (-b + jnp.sqrt(d)) / (2 * a)
    return pwm


def rpy_to_R(rpy, convention="xyz"):
    phi, theta, psi = rpy
    Rz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                    [jnp.sin(psi), jnp.cos(psi), 0],
                    [0, 0, 1]])
    Ry = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                    [0, 1, 0],
                    [-jnp.sin(theta), 0, jnp.cos(theta)]])
    Rx = jnp.array([[1, 0, 0],
                    [0, jnp.cos(phi), -jnp.sin(phi)],
                    [0, jnp.sin(phi), jnp.cos(phi)]])
    if convention == "xyz":
        R = jnp.dot(Rx, jnp.dot(Ry, Rz))  # Tait-Bryan angles (XYZ sequence)
    elif convention == "zyx":
        R = jnp.dot(Rz, jnp.dot(Ry, Rx))  # Tait-Bryan angles (ZYX sequence)
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return R


def R_to_rpy(R: jax.typing.ArrayLike, convention="xyz") -> jax.typing.ArrayLike:
    p = jnp.arcsin(R[0, 2])
    def no_gimbal_lock(*_):
        r = jnp.arctan2(-R[1, 2] / jnp.cos(p), R[2, 2] / jnp.cos(p))
        y = jnp.arctan2(-R[0, 1] / jnp.cos(p), R[0, 0] / jnp.cos(p))
        return jnp.array([r, p, y])

    def gimbal_lock(*_):
        # When cos(p) is close to zero, gimbal lock occurs, and many solutions exist.
        # Here, we arbitrarily set roll to zero in this case.
        r = 0
        y = jnp.arctan2(R[1, 0], R[1, 1])
        return jnp.array([r, p, y])
    rpy = jax.lax.cond(jnp.abs(jnp.cos(p)) > 1e-6, no_gimbal_lock, gimbal_lock, None)
    return rpy


def rpy_to_wxyz(v: jax.typing.ArrayLike) -> jax.Array:
    """
    Converts euler rotations in degrees to quaternion.
    this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
    """
    c1, c2, c3 = jnp.cos(v / 2)
    s1, s2, s3 = jnp.sin(v / 2)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    return jnp.array([w, x, y, z])


def in_agent_frame(pos: jax.typing.ArrayLike, att: jax.typing.ArrayLike, vel: jax.typing.ArrayLike, center: jax.typing.ArrayLike):
    # return self
    pos_ciw = center  # Center of the platform in the world frame
    pos_qiw = pos  # Position of the quadrotor in the world frame
    vel_qiw = vel  # Velocity of the quadrotor in the world frame
    att_q2w = att  # Attitude of the quadrotor in the world frame

    # Get agent frame by rotating world frame with yaw of quadrotor
    Rz = jnp.array([[jnp.cos(att_q2w[2]), -jnp.sin(att_q2w[2]), 0],
                    [jnp.sin(att_q2w[2]), jnp.cos(att_q2w[2]), 0],
                    [0, 0, 1]])
    R_a2w = Rz

    # Agent frame transformation
    H_w2a = jnp.eye(4)
    H_w2a = H_w2a.at[:3, :3].set(R_a2w.T)  # TODO: COMMENT to circumvent XLA bug -> calculation will be wrong.
    H_w2a = H_w2a.at[:3, 3].set(-R_a2w.T @ pos_qiw)  # TODO: COMMENT to circumvent XLA bug -> calculation will be wrong.

    # Transform quad to agent frame
    R_q2w = rpy_to_R(att_q2w)
    att_q2a = R_to_rpy(R_a2w.T @ R_q2w)
    # att_q2a = att_q2w
    roll_q2a, pitch_q2a, yaw_q2a = att_q2a  # yaw2a should be zero...
    rp_q2a = jnp.array([roll_q2a, pitch_q2a])  # Roll and pitch of the quadrotor in the agent frame

    # Position of center w.r.t. quadrotor's local frame
    pos_ciq = (H_w2a @ jnp.concatenate([pos_ciw, jnp.array([1.0])]))[:3]
    pos_cia = -pos_ciq  # Position of the quadrotor from the center of the circle

    # Velocity of the quadrotor in the circle frame
    vel_qiq = H_w2a @ jnp.concatenate([vel_qiw, jnp.array([0.0])])
    # vel_qiq = vel_qiw

    # Tangent
    radial = jnp.linalg.norm(pos_cia[:2])
    unit_radial = jnp.where(radial > 1e-5, pos_cia[:2] / radial, onp.array([1.0, 0.0]))
    theta = jnp.arctan2(pos_cia[1], pos_cia[0])
    unit_tangent = jnp.array([-jnp.sin(theta), jnp.cos(theta)])  # xy-plane
    binormal = pos_cia[2]
    new_pos = jnp.array([radial, theta, binormal])

    v_tangent = jnp.dot(unit_tangent, vel_qiq[:2])
    v_radial = jnp.dot(unit_radial, vel_qiq[:2])
    v_binormal = vel_qiq[2]
    new_vel = jnp.array([v_radial, v_tangent, v_binormal])
    return new_pos, new_vel, rp_q2a


@struct.dataclass
class WorldState(Base):
    """Pendulum state definition"""
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state
    radius: jax.typing.ArrayLike
    center: jax.typing.ArrayLike

    def in_agent_frame(self):
        new_pos, new_vel, new_att = in_agent_frame(self.pos, self.att, self.vel, self.center)
        return self.replace(pos=new_pos, att=new_att, vel=new_vel)


@struct.dataclass
class WorldParams(Base):
    actuator_delay: TrainableDist
    # Domain randomization
    use_dr: Union[bool, jax.typing.ArrayLike]  # Domain randomization
    mass_var: jax.typing.ArrayLike  # [%]
    # Parameters
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    gain_constant: Union[float, jax.typing.ArrayLike]  # 1.1094
    time_constant: Union[float, jax.typing.ArrayLike]  # 0.183806
    state_space: jax.typing.ArrayLike  # [-15.4666, 1, 3.5616e-5, 7.2345e-8]  # [A,B,C,D]
    pwm_constants: jax.typing.ArrayLike  # [2.130295e-11, 1.032633e-6, 5.485e-4] # [a,b,c]
    dragxy_constants: jax.typing.ArrayLike  # [9.1785e-7, 0.04076521, 380.8359] # Fa,x
    dragz_constants: jax.typing.ArrayLike  # [10.311e-7, 0.04076521, 380.8359] # Fa,z
    clip_rad: jax.typing.ArrayLike  # 2.0
    clip_vel: jax.typing.ArrayLike  # [-10.0, 10.0]

    @property
    def pwm_hover(self) -> Union[float, jax.Array]:
        return force_to_pwm(self.pwm_constants, self.mass * 9.81)

    def step(self, substeps: int, dt_substeps: Union[float, jax.typing.ArrayLike], x: WorldState, action: "PIDOutput" = None, action_substeps: "PIDOutput" = None) -> Tuple[WorldState, WorldState]:
        """Step the pendulum ode."""
        assert (action is None) is not (action_substeps is None), "Either action or action_substeps should be provided."
        if action is not None:  # Fixed action across substeps
            def _scan_fn(_x, _):
                next_x = self._runge_kutta4(dt_substeps, _x, action)
                return next_x, next_x

            x_final, x_substeps = jax.lax.scan(_scan_fn, x, onp.arange(substeps), length=substeps)
        else:  # Different action for each substep
            def _scan_fn(_x, _u):
                next_x = self._runge_kutta4(dt_substeps, _x, _u)
                return next_x, next_x

            x_final, x_substeps = jax.lax.scan(_scan_fn, x, action_substeps, length=substeps)
        return x_final, x_substeps

    def _runge_kutta4(self, dt, state: WorldState, action: "PIDOutput"):
        k1 = self.ode(state, action)
        k2 = self.ode(state + k1 * dt * 0.5, action)
        k3 = self.ode(state + k2 * dt * 0.5, action)
        k4 = self.ode(state + k3 * dt, action)
        return state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    def ode(self, state: WorldState, u: "PIDOutput") -> WorldState:
        # Unpack params
        mass = self.mass
        gain_c = self.gain_constant
        time_c = self.time_constant
        A, B, C, D = self.state_space
        pwm_constants = self.pwm_constants
        dragxy_c = self.dragxy_constants  # [9.1785e-7, 0.04076521, 380.8359] # Fa,x
        dragz_c = self.dragz_constants  # [10.311e-7, 0.04076521, 380.8359] # Fa,z
        # Unpack state
        x, y, z = state.pos
        xdot, ydot, zdot = state.vel
        phi, theta, psi = state.att
        p, q, r = state.ang_vel
        thrust_state = state.thrust_state
        # Unpack action
        pwm = u.pwm_ref
        phi_ref = u.phi_ref
        theta_ref = u.theta_ref
        psi_ref = u.psi_ref

        # Calculate static thrust offset (from hover)
        # Difference between the steady state value of eq (3.16) and eq (3.3) at the hover point (mass*g)
        # System Identification of the Crazyflie 2.0 Nano Quadrocopter. Julian Forster, 2016.
        #  -https://www.research-collection.ethz.ch/handle/20.500.11850/214143
        hover_force = 9.81 * mass
        hover_pwm = force_to_pwm(pwm_constants, hover_force)  # eq (3.3)

        # Steady state thrust_state for the given pwm
        ss_thrust_state = B / (-A) * hover_pwm  # steady-state with eq (3.16)
        ss_force = 4 * (C * ss_thrust_state + D * hover_pwm)  # Thrust force at steady state
        force_offset = ss_force - hover_force  # Offset from hover

        # Calculate forces
        force_thrust = 4 * (C * thrust_state + D * pwm)  # Thrust force
        force_thrust = jnp.clip(force_thrust - force_offset, 0, None)  # Correct for offset

        # Calculate rotation matrix
        R = rpy_to_R(jnp.array([phi, theta, psi]))

        # Calculate drag matrix
        # pwm_drag = force_to_pwm(pwm_constants, force_thrust)  # Symbolic PWM to approximate rotor drag
        # dragxy = dragxy_c[0] * 4 * (dragxy_c[1] * pwm_drag + dragxy_c[2])  # Fa,x
        # dragz = dragz_c[0] * 4 * (dragz_c[1] * pwm_drag + dragz_c[2])  # Fa,z
        # drag_matrix = jnp.array([
        #     [dragxy, 0, 0],
        #     [0, dragxy, 0],
        #     [0, 0, dragz]
        # ])
        # force_drag = drag_matrix @ jnp.array([xdot, ydot, zdot]) # todo: should be vel in body frame...

        # Calculate dstate
        dpos = jnp.array([xdot, ydot, zdot])
        dvel = R @ jnp.array([0, 0, force_thrust / mass]) - jnp.array([0, 0, 9.81])
        datt = jnp.array([
            (gain_c * phi_ref - phi) / time_c,  # phi_dot
            (gain_c * theta_ref - theta) / time_c,  # theta_dot
            0.  # (gain_c * psi_ref - psi) / time_c  # psi_dot
        ])
        dang_vel = jnp.array([0.0, 0.0, 0.0])  # No angular velocity
        dthrust_state = A * thrust_state + B * pwm  # Thrust_state dot
        dmass = 0.0  # No mass change
        dradius = 0.0  # No radius change
        dcenter = jnp.array([0.0, 0.0, 0.0])  # No center change
        dstate = WorldState(mass=dmass, pos=dpos, vel=dvel, att=datt, ang_vel=dang_vel, thrust_state=dthrust_state, radius=dradius, center=dcenter)
        return dstate


@struct.dataclass
class WorldOutput(Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state

    def in_agent_frame(self, center: jax.typing.ArrayLike):
        new_pos, new_vel, new_att = in_agent_frame(self.pos, self.att, self.vel, center)
        return self.replace(pos=new_pos, att=new_att, vel=new_vel)


class World(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 50, **kwargs):
        super().__init__(*args, **kwargs)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldParams:
        graph_state = graph_state or GraphState()
        params = graph_state.params.get("supervisor")
        return WorldParams(
            actuator_delay=graph_state.params.get("pid").actuator_delay,
            use_dr=params.use_dr,  # Whether to perform domain randomization
            mass_var=0.02,
            mass=params.mass,
            gain_constant=1.1094,
            time_constant=0.183806,
            state_space=onp.array([-15.4666, 1, 3.5616e-5, 7.2345e-8]),  # [A,B,C,D]
            pwm_constants=params.pwm_constants,
            dragxy_constants=onp.array([9.1785e-7, 0.04076521, 380.8359]),
            dragz_constants=onp.array([10.311e-7, 0.04076521, 380.8359]),
            clip_rad=2.0,
            clip_vel=jnp.array([20., 20., 20.]),
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldState:
        """Default state of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        # Determine mass
        rng, rng_mass = jax.random.split(rng)
        use_dr = params.use_dr
        dmass = use_dr*params.mass*params.mass_var*jax.random.uniform(rng_mass, shape=(), minval=-1, maxval=1)
        mass = params.mass + dmass
        # Determine initial state
        A, B, C, D = params.state_space
        init_thrust_state = B * params.pwm_hover / (-A)  # Assumes dthrust = 0.
        # Get radius & radius of path from supervisor
        state_sup = graph_state.state.get("supervisor")
        params_sup = graph_state.params.get("supervisor")
        state = WorldState(
            mass=mass,
            pos=state_sup.init_pos,
            vel=state_sup.init_vel,
            att=state_sup.init_att,
            ang_vel=state_sup.init_ang_vel,
            thrust_state=init_thrust_state,
            radius=state_sup.radius,
            center=params_sup.center,
        )
        return state

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        state = graph_state.state.get(self.name, self.init_state(rng, graph_state))
        return WorldOutput(
                    pos=state.pos,
                    vel=state.vel,
                    att=state.att,
                    ang_vel=state.ang_vel,
                    thrust_state=state.thrust_state,
                )

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        if isinstance(inputs["pid"].delay_dist, TrainableDist):
            inputs["pid"] = inputs["pid"].replace(delay_dist=params.actuator_delay)
        return FrozenDict(inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        params: WorldParams

        # Get action
        action = inputs["pid"][-1].data

        # Step the ode
        next_state, _ = params.step(self.substeps, self.dt_substeps, state, action=action)

        # Clip position
        center = next_state.center
        high = center + params.clip_rad + next_state.radius
        low = center - params.clip_rad - next_state.radius
        pos_clip = jnp.clip(next_state.pos, low, high)
        # next_state_ia = next_state.in_agent_frame()
        # radial, theta, _ = next_state_ia.pos
        # radial_clip = jnp.clip(radial, 0.0, params.clip_rad + next_state.radius)
        # x_clip = radial_clip * jnp.cos(theta)
        # y_clip = radial_clip * jnp.sin(theta)
        # pos_clip = next_state.pos#.at[:2].set(jnp.array([x_clip, y_clip]))

        # Clip velocity
        vel_clip = jnp.clip(next_state.vel, -params.clip_vel, params.clip_vel)
        next_state = next_state.replace(pos=pos_clip, vel=vel_clip)

        # Update state
        new_step_state = step_state.replace(state=next_state)

        # Prepare output
        output = WorldOutput(
            pos=next_state.pos,
            vel=next_state.vel,
            att=next_state.att,
            ang_vel=next_state.ang_vel,
            thrust_state=next_state.thrust_state,
        )
        return new_step_state, output


@struct.dataclass
class MoCapParams(Base):
    sensor_delay: TrainableDist
    use_noise: Union[bool, jax.typing.ArrayLike]
    pos_std: jax.typing.ArrayLike  # [x, y, z]
    vel_std: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att_std: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel_std: jax.typing.ArrayLike  # [p, q, r]


@struct.dataclass
class MoCapState:
    loss_pos: Union[float, jax.typing.ArrayLike]
    loss_vel: Union[float, jax.typing.ArrayLike]
    loss_att: Union[float, jax.typing.ArrayLike]
    loss_ang_vel: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class MoCapOutput(Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    ts: Union[float, jax.typing.ArrayLike]

    def in_agent_frame(self, center: jax.typing.ArrayLike):
        new_pos, new_vel, new_att = in_agent_frame(self.pos, self.att, self.vel, center)
        return self.replace(pos=new_pos, att=new_att, vel=new_vel)


class MoCap(BaseNode):
    def __init__(self, *args, outputs: MoCapOutput = None, **kwargs):
        """Initialize

        Args:
        images: Recorded outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapParams:
        graph_state = graph_state or GraphState
        params_sup = graph_state.params.get("supervisor")
        sensor_delay = TrainableDist.create(alpha=0., min=0.0, max=0.05)
        return MoCapParams(
            sensor_delay=sensor_delay,
            use_noise=params_sup.use_noise,
            pos_std=onp.array([0.01, 0.01, 0.01], dtype=float),     # [x, y, z]
            vel_std=onp.array([0.02, 0.02, 0.02], dtype=float),        # [xdot, ydot, zdot]
            att_std=onp.array([0.02, 0.02, 0.02], dtype=float),     # [phi, theta, psi]
            ang_vel_std=onp.array([0.1, 0.1, 0.1], dtype=float),        # [p, q, r]
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapState:
        """Default state of the node."""
        return MoCapState(loss_pos=0.0, loss_vel=0.0, loss_att=0.0, loss_ang_vel=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        graph_state = graph_state or GraphState
        state_sup = graph_state.state.get("supervisor")
        # Account for sensor delay
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        ts = -1. / self.rate - sensor_delay
        output = MoCapOutput(
            pos=state_sup.init_pos,
            vel=state_sup.init_vel,
            att=state_sup.init_att,
            ang_vel=state_sup.init_ang_vel,
            ts=ts,
        )
        return output

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        if isinstance(inputs["world"].delay_dist, TrainableDist):
            inputs["world"] = inputs["world"].replace(delay_dist=params.sensor_delay)
        return FrozenDict(inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, MoCapOutput]:
        """Step the node."""
        world = step_state.inputs["world"][-1].data
        use_noise = step_state.params.use_noise
        params: MoCapParams = step_state.params

        # Adjust ts_start (i.e. step_state.ts) to reflect the timestamp of the world state that generated the sensor output
        ts = step_state.ts - step_state.params.sensor_delay.mean()  # Should be equal to world_interp.ts_sent[-1]?

        # Sample small amount of noise to pos, vel (std=0.05), pos(std=0.005)
        rngs = jax.random.split(step_state.rng, 8)
        new_rng = rngs[0]
        pos_noise = params.pos_std*jax.random.normal(rngs[1], world.pos.shape)
        vel_noise = params.vel_std*jax.random.normal(rngs[2], world.vel.shape)
        att_noise = params.att_std*jax.random.normal(rngs[3], world.att.shape)
        ang_vel_noise = params.ang_vel_std*jax.random.normal(rngs[4], world.ang_vel.shape)

        # Prepare output
        output = MoCapOutput(
            pos=world.pos + use_noise*pos_noise,
            vel=world.vel + use_noise*vel_noise,
            att=world.att + use_noise*att_noise,
            ang_vel=world.ang_vel + use_noise*ang_vel_noise,
            ts=ts,
        )

        # Update state
        new_step_state = step_state.replace(rng=new_rng)

        return new_step_state, output


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath

    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


def render(ts: jax.typing.ArrayLike, pos: jax.typing.ArrayLike, att: jax.typing.ArrayLike, radius: jax.typing.ArrayLike = None, center: jax.typing.ArrayLike = None, done: jax.Array = None):
    """Render the rollout as an HTML file."""
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    from brax.io import html

    # Determine fps
    max_ts = jnp.max(ts)
    dt = max_ts / ts.shape[-1]

    # CRAZYFLIE_BRAX_XML is defind relative to this __file__ as path_to_file/cf2_brax.xml
    import os
    CRAZYFLIE_BRAX_XML = os.path.join(os.path.dirname(__file__), "cf2_brax.xml")
    ASSET_PATH = os.path.join(os.path.dirname(__file__))

    # Read as string
    with open(CRAZYFLIE_BRAX_XML, "r") as f:
        CRAZYFLIE_BRAX_STRING = f.read()

    # Replace radius & center
    center = center if center is not None else jnp.array([0.0, 0.0, 0.0])
    if radius is None:
        radius = 1.0
    elif radius.ndim > 0:  # If radius is a list of radii
        radius = radius.reshape(-1)[-1]

    # Find and replace size="1.00 0.02" with size=
    CRAZYFLIE_BRAX_STRING = CRAZYFLIE_BRAX_STRING.replace('size="1.00', f'size="{radius}')

    # Initialize system
    sys = mjcf.loads(CRAZYFLIE_BRAX_STRING, asset_path=ASSET_PATH)
    sys = sys.replace(opt=sys.opt.replace(timestep=dt))

    def _set_pipeline_state(i):
        quat = rpy_to_wxyz(att[i])
        c = center[i] if center.ndim > 1 else center
        qpos = jnp.concatenate([pos[i], quat, jnp.array([0.0]), c])

        # Set initial state
        qvel = jnp.zeros_like(qpos)
        x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
        pipeline_state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
        # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    jit_set_pipeline_state = jax.jit(_set_pipeline_state).lower(0).compile()
    pipeline_state_lst = []
    for i in range(0, pos.shape[0]):
        if done is not None and done[i]:
            break
        pipeline_state_i = jit_set_pipeline_state(i)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json


@struct.dataclass
class Rollout:
    next_gs: GraphState
    next_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    info: Any


def env_rollout(env: Union[rl.Environment, rl.BaseWrapper], rng: jax.Array):
    init_gs, init_obs, info = env.reset(rng)

    def _scan(_carry, _):
        _gs, _obs = _carry
        _params = _gs.params["agent"]
        _action = _params.get_action(_obs)
        next_gs, next_obs, reward, terminated, truncated, info = env.step(_gs, _action)
        done = jnp.logical_or(terminated, truncated)
        r = Rollout(next_gs, next_obs, _action, reward, terminated, truncated, done, info)
        return (next_gs, next_obs), r

    carry = (init_gs, init_obs)
    _, r = jax.lax.scan(_scan, carry, jnp.arange(env.max_steps))
    return r


def plot_data(output, ts, ts_max=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    # Ensure output and ts have the same keys
    assert set(output.keys()) == set(ts.keys()), "Output and ts dictionaries must have the same keys"

    # Filter data based on ts_max if provided
    if ts_max is not None:
        for key in ts:
            mask = ts[key] <= ts_max
            ts[key] = ts[key][mask]
            if isinstance(output[key], onp.ndarray):
                output[key] = output[key][mask]
            else:
                output[key] = [output[key][i] for i in range(len(output[key])) if mask[i]]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # First row: PWM and x-y plot
    if "pwm_ref" in output and output["pwm_ref"] is not None:
        axes[0, 0].plot(ts["pwm_ref"], output["pwm_ref"], label="pwm_ref", color="green", linestyle="-")
        axes[0, 0].legend()
        axes[0, 0].set_title("PWM")
    else:
        axes[0, 0].axis("off")  # Empty plot

    # Replace the scatter plot with a line plot
    points = onp.array([output["pos"][:, 0], output["pos"][:, 1]]).T.reshape(-1, 1, 2)
    segments = onp.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(ts["pos"].min(), ts["pos"].max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(ts["pos"])
    lc.set_linewidth(2)
    line = axes[0, 1].add_collection(lc)
    xlim = [output["pos"][:, 0].min(), output["pos"][:, 0].max()]
    xlim = [xlim[0]-1, xlim[1]+1] if xlim[0] == xlim[1] else xlim
    ylim = [output["pos"][:, 1].min(), output["pos"][:, 1].max()]
    ylim = [ylim[0]-1, ylim[1]+1] if ylim[0] == ylim[1] else ylim
    axes[0, 1].set_xlim(xlim)
    axes[0, 1].set_ylim(ylim)
    fig.colorbar(line, ax=axes[0, 1])
    axes[0, 1].set_title("X-Y Position (color: time)")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    # scatter = axes[0, 1].scatter(output["pos"][:, 0], output["pos"][:, 1], c=ts["pos"], cmap="viridis")
    # plt.colorbar(scatter, ax=axes[0, 1])
    # axes[0, 1].set_title("X-Y Position (color: time)")
    # axes[0, 1].set_xlabel("X")
    # axes[0, 1].set_ylabel("Y")

    axes[0, 2].axis("off")  # Empty plot

    # Second row: Phi and Theta
    axes[1, 0].plot(ts["att"], output["att"][:, 0], label="phi", color="blue", linestyle="-")
    axes[1, 0].plot(ts["phi_ref"], output["phi_ref"], label="phi_ref", color="green", linestyle="-")
    axes[1, 0].legend()
    axes[1, 0].set_title("Phi")

    axes[1, 1].plot(ts["att"], output["att"][:, 1], label="theta", color="blue", linestyle="-")
    axes[1, 1].plot(ts["theta_ref"], output["theta_ref"], label="theta_ref", color="green", linestyle="-")
    axes[1, 1].legend()
    axes[1, 1].set_title("Theta")

    axes[1, 2].axis("off")  # Empty plot

    # Third row: X, Y, and Z
    axes[2, 0].plot(ts["pos"], output["pos"][:, 0], label="x", color="blue", linestyle="-")
    axes[2, 0].legend()
    axes[2, 0].set_title("X Position")

    axes[2, 1].plot(ts["pos"], output["pos"][:, 1], label="y", color="blue", linestyle="-")
    axes[2, 1].legend()
    axes[2, 1].set_title("Y Position")

    axes[2, 2].plot(ts["pos"], output["pos"][:, 2], label="z", color="blue", linestyle="-")
    axes[2, 2].plot(ts["z_ref"], output["z_ref"], label="z_ref", color="green", linestyle="-")
    axes[2, 2].legend()
    axes[2, 2].set_title("Z Position")
    fig.tight_layout()
    return fig, axes