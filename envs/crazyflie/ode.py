from typing import Any, Dict, Tuple, Union, TYPE_CHECKING, List
import jax
import jax.numpy as jnp
import numpy as onp
from math import ceil
from flax import struct
from flax.core import FrozenDict

from rex import base
from rex.base import StepState, GraphState, Empty, TrainableDist, Base
from rex.node import BaseNode
import rex.rl as rl
from rex.jax_utils import tree_dynamic_slice
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


def spherical_to_R(polar, azimuth):
    Rz = jnp.array([[jnp.cos(azimuth), -jnp.sin(azimuth), 0],
                    [jnp.sin(azimuth), jnp.cos(azimuth), 0],
                    [0, 0, 1]])
    Ry = jnp.array([[jnp.cos(polar), 0, jnp.sin(polar)],
                    [0, 1, 0],
                    [-jnp.sin(polar), 0, jnp.cos(polar)]])
    R = jnp.dot(Rz, Ry)
    return R


def R_to_spherical(R):
    polar = jnp.arccos(R[2, 2])
    azimuth = jnp.arctan2(R[1, 2], R[0, 2])
    return polar, azimuth


def spherical_to_rpy(polar, azimuth):
    R = spherical_to_R(polar, azimuth)
    rpy = R_to_rpy(R)
    return rpy


def rpy_to_spherical(rpy):
    R = rpy_to_R(rpy, convention="xyz")
    polar, azimuth = R_to_spherical(R)
    return polar, azimuth


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
    # pos_ciq = (H_w2a @ jnp.concatenate([pos_ciw, jnp.array([1.0])]))[:3]
    pos_ciq = jnp.dot(R_a2w.T, pos_ciw) - jnp.dot(R_a2w.T, pos_qiw)
    pos_cia = -pos_ciq  # Position of the quadrotor from the center of the circle

    # Velocity of the quadrotor in the circle frame
    # vel_qiq = H_w2a @ jnp.concatenate([vel_qiw, jnp.array([0.0])])
    # vel_qiq = H_w2a @ jnp.concatenate([vel_qiw, jnp.array([0.0])])
    vel_qiq = jnp.dot(R_a2w.T, vel_qiw)

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


def in_body_frame(att: jax.typing.ArrayLike, vel: jax.typing.ArrayLike):
    R_q2w = rpy_to_R(att)
    vel_qiw = vel
    vel_qib = jnp.dot(R_q2w.T, vel_qiw)
    return vel_qib


@struct.dataclass
class WorldState(Base):
    """Pendulum state definition"""
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state

    def in_agent_frame(self, center):
        new_pos, new_vel, new_att = in_agent_frame(self.pos, self.att, self.vel, center)
        return self.replace(pos=new_pos, att=new_att, vel=new_vel)


@struct.dataclass
class WorldParams(Base):
    actuator_delay: TrainableDist
    # Domain randomization
    mass_var: jax.typing.ArrayLike  # [%]
    # Parameters
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    gain_constant: Union[float, jax.typing.ArrayLike]  # 1.1094
    time_constant: Union[float, jax.typing.ArrayLike]  # 0.183806
    state_space: jax.typing.ArrayLike  # [-15.4666, 1, 3.5616e-5, 7.2345e-8]  # [A,B,C,D]
    init_thrust_scale: Union[float, jax.typing.ArrayLike]  # 1.0  # scales the initial thrust (for sysid)
    pwm_constants: jax.typing.ArrayLike  # [2.130295e-11, 1.032633e-6, 5.485e-4] # [a,b,c]
    rotor_constants: jax.typing.ArrayLike  # [0.04076521, 380.8359]
    dragxy: jax.typing.ArrayLike  # 9.1785e-7 # Fa,x
    dragz: jax.typing.ArrayLike  # 10.311e-7 # Fa,z

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
        A, B, C = self.state_space
        pwm_constants = self.pwm_constants
        rotor_constants = self.rotor_constants
        dragxy_c = self.dragxy  # [9.1785e-7, 0.04076521, 380.8359] # Fa,x
        dragz_c = self.dragz  # [10.311e-7, 0.04076521, 380.8359] # Fa,z
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
        ss_force = 4 * (C * ss_thrust_state)  # Thrust force at steady state
        force_offset = ss_force - hover_force  # Offset from hover

        # Calculate forces
        force_thrust = 4 * (C * thrust_state)  # Thrust force
        force_thrust = jnp.clip(force_thrust - force_offset, 0, None)  # Correct for offset

        # Calculate rotation matrix
        R = rpy_to_R(jnp.array([phi, theta, psi]))  # R_q2w

        # Calculate drag
        vel_qiw = state.vel
        vel_qib = jnp.dot(R.T, vel_qiw)
        pwm_drag = force_to_pwm(pwm_constants, force_thrust)  # Symbolic PWM to approximate rotor drag
        rotor_speed = 4 * (rotor_constants[0]*pwm_drag + rotor_constants[1]) # Rotor speed

        # dragxy_constants: [9.1785e-7, 0.04076521, 380.8359] # Fa,x
        # dragz_constants: [10.311e-7, 0.04076521, 380.8359] # Fa,z
        kappa = jnp.array([
            [dragxy_c, 0, 0],
            [0, dragxy_c, 0],
            [0, 0, dragz_c]
        ])
        force_drag = jnp.dot(kappa, -vel_qib) * rotor_speed

        # Calculate dstate
        dpos = jnp.array([xdot, ydot, zdot])

        dvel = R @ ((jnp.array([0, 0, force_thrust]) + force_drag) / mass) - jnp.array([0, 0, 9.81])
        datt = jnp.array([
            (gain_c * phi_ref - phi) / time_c,  # phi_dot
            (gain_c * theta_ref - theta) / time_c,  # theta_dot
            0.  # (gain_c * psi_ref - psi) / time_c  # psi_dot
        ])
        dang_vel = jnp.array([0.0, 0.0, 0.0])  # No angular velocity
        dthrust_state = A * thrust_state + B * pwm  # Thrust_state dot
        dmass = 0.0  # No mass change
        # dradius = 0.0  # No radius change
        # dcenter = jnp.array([0.0, 0.0, 0.0])  # No center change
        dstate = WorldState(mass=dmass, pos=dpos, vel=dvel, att=datt, ang_vel=dang_vel, thrust_state=dthrust_state)
        return dstate

    def sysid_range(self):
        actuator_delay = None #self.actuator_delay.replace(alpha=onp.array([0., 1.0]))
        # Domain randomization
        mass_var = None
        # Parameters
        mass = onp.array([0.02, 0.04])
        gain_constant = onp.array([0.2, 2])  # 1.1094
        time_constant = onp.array([0.02, 0.3]) # 0.183806
        state_space = None  # onp.array([[-100.4666, 0.5, 1e-6],
        #                          [-5.4666, 1.5, 1e-4]])
        init_thrust_scale = None  # onp.array([0.3, 3.0])
        pwm_constants = None
        rotor_constants = None  # [0.04076521, 380.8359]
        dragxy = onp.array([0.5*9.1785e-7, 3.*9.1785e-7])
        dragz = onp.array([0.5*10.311e-7, 1.*10.311e-7])
        return self.replace(
            actuator_delay=actuator_delay,
            mass_var=mass_var,
            mass=mass,
            gain_constant=gain_constant,
            time_constant=time_constant,
            state_space=state_space,
            init_thrust_scale=init_thrust_scale,
            pwm_constants=pwm_constants,
            rotor_constants=rotor_constants,
            dragxy=dragxy,
            dragz=dragz,
        )


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

    @staticmethod
    def static_in_agent_frame(self, center: jax.typing.ArrayLike):
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
        return WorldParams(
            actuator_delay=graph_state.params.get("pid").actuator_delay,
            mass_var=0.02,
            mass=0.033,
            gain_constant=1.1094,  # Attitude gain constant
            time_constant=0.183806,  # Attitude time constant
            state_space=onp.array([-15.4666, 1, 3.5616e-5]),  # [A,B,C,D]
            init_thrust_scale=1.0,
            pwm_constants=onp.array([2.130295e-11, 1.032633e-6, 5.485e-4]),
            rotor_constants=onp.array([0.04076521, 380.8359]),
            dragxy=9.1785e-7,
            dragz=10.311e-7,
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldState:
        """Default state of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        # Determine mass
        rng, rng_mass = jax.random.split(rng)
        params_sup = graph_state.params.get("agent")
        use_dr = params_sup.use_dr
        dmass = use_dr*params.mass*params.mass_var*jax.random.uniform(rng_mass, shape=(), minval=-1, maxval=1)
        mass = params.mass + dmass
        # Determine initial state
        A, B, C = params.state_space
        init_thrust_state = params.init_thrust_scale * B * params.pwm_hover / (-A)  # Assumes dthrust = 0.
        state_sup = graph_state.state.get("agent")
        state = WorldState(
            mass=mass,
            pos=state_sup.init_pos,
            vel=state_sup.init_vel,
            att=state_sup.init_att,
            ang_vel=state_sup.init_ang_vel,
            thrust_state=init_thrust_state,
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
    pos_std: jax.typing.ArrayLike  # [x, y, z]
    vel_std: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att_std: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel_std: jax.typing.ArrayLike  # [p, q, r]


@struct.dataclass
class MoCapState:
    use_noise: Union[bool, jax.typing.ArrayLike]
    loss_pos: Union[float, jax.typing.ArrayLike]
    loss_vel: Union[float, jax.typing.ArrayLike]
    loss_att: Union[float, jax.typing.ArrayLike]


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

    @staticmethod
    def static_in_agent_frame(self, center: jax.typing.ArrayLike):
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

        sensor_delay = TrainableDist.create(delay=0., min=0.0, max=0.05)
        return MoCapParams(
            sensor_delay=sensor_delay,

            pos_std=onp.array([0.01, 0.01, 0.01], dtype=float),     # [x, y, z]
            vel_std=onp.array([0.02, 0.02, 0.02], dtype=float),        # [xdot, ydot, zdot]
            att_std=onp.array([0.02, 0.02, 0.02], dtype=float),     # [phi, theta, psi]
            ang_vel_std=onp.array([0.1, 0.1, 0.1], dtype=float),        # [p, q, r]
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapState:
        """Default state of the node."""
        params_sup = graph_state.params.get("agent")
        return MoCapState(use_noise=params_sup.use_noise, loss_pos=0.0, loss_vel=0.0, loss_att=0.0)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        graph_state = graph_state or GraphState
        state_sup = graph_state.state.get("agent")
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
        rng = step_state.rng
        state: MoCapState = step_state.state
        params: MoCapParams = step_state.params
        world = step_state.inputs["world"][-1].data

        # Adjust ts_start (i.e. step_state.ts) to reflect the timestamp of the world state that generated the sensor output
        ts = step_state.ts - step_state.params.sensor_delay.mean()  # Should be equal to world_interp.ts_sent[-1]?

        # Determine output
        if self._outputs is not None:
            recorded_output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            output = recorded_output.replace(ts=ts)
        else:
            # Sample small amount of noise to pos, vel (std=0.05), pos(std=0.005)
            rngs = jax.random.split(rng, 8)
            rng = rngs[0]
            pos_noise = params.pos_std*jax.random.normal(rngs[1], world.pos.shape)
            vel_noise = params.vel_std*jax.random.normal(rngs[2], world.vel.shape)
            att_noise = params.att_std*jax.random.normal(rngs[3], world.att.shape)
            ang_vel_noise = params.ang_vel_std*jax.random.normal(rngs[4], world.ang_vel.shape)

            # Prepare output
            output = MoCapOutput(
                pos=world.pos + state.use_noise*pos_noise,
                vel=world.vel + state.use_noise*vel_noise,
                att=world.att + state.use_noise*att_noise,
                ang_vel=world.ang_vel + state.use_noise*ang_vel_noise,
                ts=ts,
            )

        vel_v = jnp.stack([world.vel, output.vel], axis=0)
        att_v = jnp.stack([world.att, output.att], axis=0)
        vel_qib = jax.vmap(in_body_frame)(att_v, vel_v)
        vel_qib_world = vel_qib[1]
        vel_qib_out = vel_qib[1]

        # Calculate velocity in-body frame
        # R_q2w = rpy_to_R(world.att)
        # vel_qiw = world.vel
        # vel_qib = jnp.dot(R_q2w.T, vel_qiw)
        # R_q2w_out = rpy_to_R(output.att)
        # vel_qiw_out = output.vel
        # vel_qib_out = jnp.dot(R_q2w_out.T, vel_qiw_out)

        # Calculate loss
        # until_ts = (ts < 3.0).astype(int)
        until_ts = True
        loss_pos = state.loss_pos + until_ts * jnp.linalg.norm(world.pos - output.pos)**2
        loss_att = state.loss_att + until_ts * jnp.linalg.norm(world.att - output.att)**2
        loss_vel = state.loss_vel + until_ts * jnp.linalg.norm(vel_qib_world - vel_qib_out)**2
        new_state = state.replace(loss_pos=loss_pos, loss_att=loss_att, loss_vel=loss_vel)

        # Update state
        new_step_state = step_state.replace(rng=rng, state=new_state)

        return new_step_state, output


@struct.dataclass
class PlatformParams(Base):
    pos_std: jax.typing.ArrayLike  # [x, y, z]
    vel_std: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att_std: jax.typing.ArrayLike  # [phi, theta, psi]

    def step(self, substeps: int, dt_substeps: Union[float, jax.typing.ArrayLike], x: "PlatformState") -> Tuple["PlatformState", "PlatformState"]:
        """Step the pendulum ode."""
        def _scan_fn(_x, _):
            next_x = self._runge_kutta4(dt_substeps, _x)
            return next_x, next_x

        x_final, x_substeps = jax.lax.scan(_scan_fn, x, onp.arange(substeps), length=substeps)
        return x_final, x_substeps

    def _runge_kutta4(self, dt, state: "PlatformState"):
        k1 = self.ode(state)
        k2 = self.ode(state + k1 * dt * 0.5)
        k3 = self.ode(state + k2 * dt * 0.5)
        k4 = self.ode(state + k3 * dt)
        return state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    def ode(self, state: "PlatformState") -> "PlatformState":
        # Calculate dstate
        dstate = state * 0  # Set all derivatives to zero
        dstate = dstate.replace(pos=state.vel)
        return dstate


@struct.dataclass
class PlatformState(Base):
    use_noise: Union[bool, jax.typing.ArrayLike]
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]


@struct.dataclass
class PlatformOutput(Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ts: Union[float, jax.typing.ArrayLike]


class Platform(BaseNode):

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> PlatformParams:
        return PlatformParams(
            pos_std=onp.array([0.01, 0.01, 0.01], dtype=float),     # [x, y, z]
            vel_std=onp.array([0.02, 0.02, 0.02], dtype=float),        # [xdot, ydot, zdot]
            att_std=onp.array([0.02, 0.02, 0.02], dtype=float),     # [phi, theta, psi]
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PlatformState:
        """Default state of the node."""
        graph_state = graph_state or GraphState
        params_sup = graph_state.params.get("agent")
        state_sup = graph_state.state.get("agent")
        return PlatformState(
            use_noise=jnp.float32(params_sup.use_noise),
            pos=state_sup.init_pos_plat,
            vel=state_sup.init_vel_plat,
            att=state_sup.init_att_plat,
        )

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> PlatformOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        graph_state = graph_state or GraphState
        state_sup = graph_state.state.get("agent")
        output = PlatformOutput(
            pos=state_sup.init_pos_plat,
            vel=state_sup.init_vel_plat,
            att=state_sup.init_att_plat,
            ts=0.0,
        )
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, PlatformOutput]:
        """Step the node."""
        rng = step_state.rng
        state: PlatformState = step_state.state
        params: PlatformParams = step_state.params
        ts = step_state.ts

        # Simulate platform dynamics
        substeps = 1
        dt_substeps = 1/self.rate
        next_state, _ = params.step(substeps, dt_substeps, state)

        # Sample small amount of noise to pos, vel
        rngs = jax.random.split(rng, 8)
        rng = rngs[0]
        pos_noise = params.pos_std*jax.random.normal(rngs[1], next_state.pos.shape)
        vel_noise = params.vel_std*jax.random.normal(rngs[2], next_state.vel.shape)
        att_noise = params.att_std*jax.random.normal(rngs[3], next_state.att.shape)

        # Prepare output
        output = PlatformOutput(
            pos=next_state.pos + state.use_noise*pos_noise,
            vel=next_state.vel + state.use_noise*vel_noise,
            att=next_state.att + state.use_noise*att_noise,
            ts=ts,
        )

        # Update state
        new_step_state = step_state.replace(rng=rng, state=next_state)

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

    # CRAZYFLIE_BRAX_XML is defind relative to this __file__ as path_to_file/cf2_pathfollowing.xml
    import os
    CRAZYFLIE_BRAX_XML = os.path.join(os.path.dirname(__file__), "cf2_pathfollowing.xml")
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

    # Plot Z
    axes[0, 2].plot(ts["pos"], output["pos"][:, 2], label="z", color="blue", linestyle="-")
    axes[0, 2].plot(ts["z_ref"], output["z_ref"], label="z_ref", color="green", linestyle="-")
    axes[0, 2].legend()
    axes[0, 2].set_title("Z Position")

    # Second row: Phi, Theta, Psi
    axes[1, 0].plot(ts["att"], output["att"][:, 0], label="phi", color="blue", linestyle="-")
    axes[1, 0].plot(ts["phi_ref"], output["phi_ref"], label="phi_ref", color="green", linestyle="-")
    axes[1, 0].set_ylim([-onp.pi / 5, onp.pi / 5])
    axes[1, 0].legend()
    axes[1, 0].set_title("Phi")

    axes[1, 1].plot(ts["att"], output["att"][:, 1], label="theta", color="blue", linestyle="-")
    axes[1, 1].plot(ts["theta_ref"], output["theta_ref"], label="theta_ref", color="green", linestyle="-")
    axes[1, 1].set_ylim([-onp.pi / 5, onp.pi / 5])
    axes[1, 1].legend()
    axes[1, 1].set_title("Theta")

    axes[1, 2].plot(ts["att"], output["att"][:, 2], label="psi", color="blue", linestyle="-")
    # axes[1, 2].plot(ts["psi_ref"], output["psi_ref"], label="psi_ref", color="green", linestyle="-")
    axes[1, 2].set_ylim([-onp.pi / 5, onp.pi / 5])
    axes[1, 2].legend()
    axes[1, 2].set_title("Psi")

    # Plot off-center position
    if "pos_ia" in output:
        pos_off = jnp.linalg.norm(jnp.concatenate([output["pos_ia"][:, [0]], output["pos_ia"][:, [2]]], axis=-1), axis=-1)
        axes[2, 0].plot(ts["pos_ia"], pos_off, label="pos_off", color="blue", linestyle="-")
        axes[2, 0].legend()
        axes[2, 0].set_title("Radial Position")
    else:
        axes[2, 0].axis("off")  # Empty plot

    # Plot velocities in agent frame
    if "vel_ia" in output:
        vel_on = output["vel_ia"][:, 1]
        vel_off = jnp.linalg.norm(jnp.concatenate([output["vel_ia"][:, [0]], output["vel_ia"][:, [2]]], axis=-1), axis=-1)
        axes[2, 1].plot(ts["vel_ia"], vel_off, label="vel_off", color="blue", linestyle="-")
        axes[2, 1].legend()
        axes[2, 1].set_title("Velocity (off-path)")
        axes[2, 2].plot(ts["vel_ia"], vel_on, label="vel_on", color="blue", linestyle="-")
        axes[2, 2].legend()
        axes[2, 2].set_title("Velocity (on-path)")
    else:
        axes[2, 1].axis("off")  # Empty plot
        axes[2, 2].axis("off")  # Empty plot

    fig.tight_layout()
    return fig, axes


def metrics(mocap: MoCapOutput, radius: jax.typing.ArrayLike, center: jax.typing.ArrayLike):
    mocap_ia = mocap.in_agent_frame(center)
    vel_on = mocap_ia.vel[1]
    pos_on = mocap_ia.pos[1]
    vel_off = jnp.linalg.norm(jnp.array([mocap_ia.vel[0], mocap_ia.vel[2]]))
    pos_off = jnp.linalg.norm(jnp.array([mocap_ia.pos[0] - radius, mocap_ia.pos[2]]))
    pos_on = (pos_on + onp.pi) / (2 * onp.pi) * radius  # Convert to meters
    return vel_on, vel_off, pos_on, pos_off