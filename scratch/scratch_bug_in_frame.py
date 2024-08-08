from typing import Union
import jax
import jax.numpy as jnp
import jax.numpy as onp
from rexv2.base import Base
from envs.crazyflie.ode import rpy_to_R, R_to_rpy
from flax import struct


def new_rpy_to_R(rpy, convention="xyz"):
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


def new_R_to_rpy(R: jax.typing.ArrayLike, convention="xyz") -> jax.typing.ArrayLike:
    p = jnp.arcsin(R[0, 2])

    def no_gimbal_lock(*_):
        r = jnp.arctan2(-R[1, 2] / jnp.cos(p), R[2, 2] / jnp.cos(p))
        y = jnp.arctan2(-R[0, 1] / jnp.cos(p), R[0, 0] / jnp.cos(p))
        return jnp.array([r, p, y])

    def gimbal_lock(*_):
        r = 0
        y = jnp.arctan2(R[1, 0], R[1, 1])
        return jnp.array([r, p, y])

    rpy = jax.lax.cond(jnp.abs(jnp.cos(p)) > 1e-6, no_gimbal_lock, gimbal_lock, None)
    return rpy


@struct.dataclass
class MoCapOutput(Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    ts: Union[float, jax.typing.ArrayLike]

    @classmethod
    def create(cls, N, M=None):
        if M is None:
            shape = (N,)
        else:
            shape = (M, N)
        pos = jnp.ones(shape + (3,))
        vel = jnp.ones(shape + (3,))
        att = jnp.ones(shape + (3,))
        ang_vel = jnp.zeros(shape + (3,))
        ts = jnp.zeros(shape)
        return cls(pos=pos, vel=vel, att=att, ang_vel=ang_vel, ts=ts)

    def in_agent_frame(self, center: jax.typing.ArrayLike):
        # return self
        pos_ciw = center  # Center of the platform in the world frame
        pos_qiw = self.pos  # Position of the quadrotor in the world frame
        vel_qiw = self.vel  # Velocity of the quadrotor in the world frame
        att_q2w = self.att  # Attitude of the quadrotor in the world frame

        # Get agent frame by rotating world frame with yaw of quadrotor
        Rz = jnp.array([[jnp.cos(att_q2w[2]), -jnp.sin(att_q2w[2]), 0],
                        [jnp.sin(att_q2w[2]), jnp.cos(att_q2w[2]), 0],
                        [0, 0, 1]])
        R_a2w = Rz

        # Agent frame transformation
        H_w2a = jnp.eye(4)
        H_w2a = H_w2a.at[:3, :3].set(R_a2w.T)
        H_w2a = H_w2a.at[:3, 3].set(-R_a2w.T @ pos_qiw)  # todo: UNCOMMENT

        # Transform quad to agent frame
        R_q2w = rpy_to_R(att_q2w)
        att_q2a = R_to_rpy(R_a2w.T @ R_q2w)  # todo: UNCOMMENT
        # att_q2a = att_q2w
        roll_q2a, pitch_q2a, yaw_q2a = att_q2a  # yaw2a should be zero...
        rp_q2a = jnp.array([roll_q2a, pitch_q2a])  # Roll and pitch of the quadrotor in the agent frame

        # Position of center w.r.t quadrotor's local frame
        pos_ciq = H_w2a @ jnp.concatenate([pos_ciw, jnp.array([1.0])]) # todo: UNCOMMENT
        # pos_ciq = H_w2a[:3, 3]
        pos_cia = -pos_ciq[:3]  # Position of the quadrotor from the center of the circle

        # Velocity of the quadrotor in the circle frame
        vel_qiq = H_w2a @ jnp.concatenate([vel_qiw, jnp.array([0.0])]) # todo: UNCOMMENT
        # vel_qiq = vel_qiw

        # Tangent
        radial = jnp.linalg.norm(pos_cia[:2])
        unit_radial = pos_cia[:2] / radial  # xy-plane
        theta = jnp.arctan2(pos_cia[1], pos_cia[0])
        unit_tangent = jnp.array([-jnp.sin(theta), jnp.cos(theta)])  # xy-plane
        binormal = pos_cia[2]
        new_pos = jnp.array([radial, theta, binormal])

        v_tangent = jnp.dot(unit_tangent, vel_qiq[:2])
        v_radial = jnp.dot(unit_radial, vel_qiq[:2])
        v_binormal = vel_qiq[2]
        new_vel = jnp.array([v_tangent, v_radial, v_binormal])
        return self.replace(pos=new_pos, vel=new_vel, att=rp_q2a)

    def _new_in_agent_frame(self, center: jax.typing.ArrayLike):
        # return self
        pos_ciw = center  # Center of the platform in the world frame
        pos_qiw = self.pos  # Position of the quadrotor in the world frame
        vel_qiw = self.vel  # Velocity of the quadrotor in the world frame
        att_q2w = self.att  # Attitude of the quadrotor in the world frame

        # Get agent frame by rotating world frame with yaw of quadrotor
        Rz = jnp.array([[jnp.cos(att_q2w[2]), -jnp.sin(att_q2w[2]), 0],
                        [jnp.sin(att_q2w[2]), jnp.cos(att_q2w[2]), 0],
                        [0, 0, 1]])
        R_a2w = Rz

        # Agent frame transformation
        H_w2a = jnp.eye(4)
        H_w2a = H_w2a.at[:3, :3].set(R_a2w.T)
        H_w2a = H_w2a.at[:3, 3].set(-R_a2w.T @ pos_qiw)  # todo: UNCOMMENT

        # Transform quad to agent frame
        R_q2w = rpy_to_R(att_q2w)
        att_q2a = R_to_rpy(R_a2w.T @ R_q2w)  # todo: UNCOMMENT
        # att_q2a = att_q2w
        roll_q2a, pitch_q2a, yaw_q2a = att_q2a  # yaw2a should be zero...
        rp_q2a = jnp.array([roll_q2a, pitch_q2a])  # Roll and pitch of the quadrotor in the agent frame

        # Position of center w.r.t quadrotor's local frame
        pos_ciq = H_w2a @ jnp.concatenate([pos_ciw, jnp.array([1.0])]) # todo: UNCOMMENT
        # pos_ciq = H_w2a[:3, 3]
        pos_cia = -pos_ciq[:3]  # Position of the quadrotor from the center of the circle

        # Velocity of the quadrotor in the circle frame
        vel_qiq = H_w2a @ jnp.concatenate([vel_qiw, jnp.array([0.0])]) # todo: UNCOMMENT
        # vel_qiq = vel_qiw

        # Tangent
        radial = jnp.linalg.norm(pos_cia[:2])
        unit_radial = pos_cia[:2] / radial  # xy-plane
        theta = jnp.arctan2(pos_cia[1], pos_cia[0])
        unit_tangent = jnp.array([-jnp.sin(theta), jnp.cos(theta)])  # xy-plane
        binormal = pos_cia[2]
        new_pos = jnp.array([radial, theta, binormal])

        v_tangent = jnp.dot(unit_tangent, vel_qiq[:2])
        v_radial = jnp.dot(unit_radial, vel_qiq[:2])
        v_binormal = vel_qiq[2]
        new_vel = jnp.array([v_tangent, v_radial, v_binormal])
        return self.replace(pos=new_pos, vel=new_vel, att=rp_q2a)

    def new_in_agent_frame(self, center: jax.typing.ArrayLike):
        pos_ciw = center
        pos_qiw = self.pos
        vel_qiw = self.vel
        att_q2w = self.att

        Rz = jnp.array([[jnp.cos(att_q2w[2]), -jnp.sin(att_q2w[2]), 0],
                        [jnp.sin(att_q2w[2]), jnp.cos(att_q2w[2]), 0],
                        [0, 0, 1]])
        R_a2w = Rz

        H_w2a = jnp.eye(4)
        H_w2a = H_w2a.at[:3, :3].set(R_a2w.T)
        H_w2a = H_w2a.at[:3, 3].set(-jnp.dot(R_a2w.T, pos_qiw))

        R_q2w = new_rpy_to_R(att_q2w)
        att_q2a = new_R_to_rpy(jnp.dot(R_a2w.T, R_q2w))
        roll_q2a, pitch_q2a, yaw_q2a = att_q2a
        rp_q2a = jnp.array([roll_q2a, pitch_q2a])

        pos_ciq = jnp.dot(H_w2a, jnp.concatenate([pos_ciw, jnp.array([1.0])]))
        pos_cia = -pos_ciq[:3]

        vel_qiq = jnp.dot(H_w2a, jnp.concatenate([vel_qiw, jnp.array([0.0])]))

        radial = jnp.linalg.norm(pos_cia[:2])  # todo: can be (close to) zero...
        unit_radial = pos_cia[:2] / radial
        theta = jnp.arctan2(pos_cia[1], pos_cia[0])
        unit_tangent = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
        binormal = pos_cia[2]
        new_pos = jnp.array([radial, theta, binormal])

        v_tangent = jnp.dot(unit_tangent, vel_qiq[:2])
        v_radial = jnp.dot(unit_radial, vel_qiq[:2])
        v_binormal = vel_qiq[2]
        new_vel = jnp.array([v_tangent, v_radial, v_binormal])

        return self.replace(pos=new_pos, vel=new_vel, att=rp_q2a)


# Define functions
def fn(x, center):
    return x.in_agent_frame(center)


def new_fn(x, center):
    return x.new_in_agent_frame(center)


def is_close(old, new):
    res = jax.tree_util.tree_map(lambda x, y: onp.allclose(jax.device_get(x), jax.device_get(y)), jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new))
    return all(res)



gpu = jax.devices("gpu")[0]
cpu = jax.devices("cpu")[0]
M, N = 4, 5

x_v = MoCapOutput.create(N)
center_v = -jnp.ones((N, 3))
x_vv = MoCapOutput.create(N, M=M)
center_vv = -jnp.ones((M, N, 3))

# Old
fn_v = jax.vmap(fn)
fn_jv_cpu = jax.jit(jax.vmap(fn), device=cpu)
fn_jv_gpu = jax.jit(jax.vmap(fn), device=gpu)
fn_jvv_gpu = jax.jit(jax.vmap(jax.vmap(fn)), device=gpu)

# New
new_fn_v = jax.vmap(new_fn)
new_fn_jv_cpu = jax.jit(jax.vmap(new_fn), device=cpu)
new_fn_jv_gpu = jax.jit(jax.vmap(new_fn), device=gpu)
new_fn_jvv_gpu = jax.jit(jax.vmap(jax.vmap(new_fn)), device=gpu)

new_res_vv_gpu = new_fn_jvv_gpu(x_vv, center_vv)
res_vv_gpu = fn_jvv_gpu(x_vv, center_vv)
assert is_close(res_vv_gpu, new_res_vv_gpu), "Double vmap + jit (GPU) failed"
print("Jit (GPU), double vmap: SUCCESS")
new_res_v = new_fn_v(x_v, center_v)
res_v = fn_v(x_v, center_v)
assert is_close(res_v, new_res_v), "Single vmap + jit (CPU) failed"
print("No jit, single vmap: SUCCESS")
new_res_v_cpu = new_fn_jv_cpu(x_v, center_v)
res_v_cpu = fn_jv_cpu(x_v, center_v)
assert is_close(res_v_cpu, new_res_v_cpu), "Single vmap + jit (CPU) failed"
print("Jit (CPU), single vmap: SUCCESS")
new_res_v_gpu = new_fn_jv_gpu(x_v, center_v)
assert is_close(res_v_cpu, new_res_v_gpu), "Single vmap + jit (GPU) failed"
print("[NEW] Jit (GPU), single vmap: SUCCESS")
res_v_gpu = fn_jv_gpu(x_v, center_v)  # Fails here...
assert is_close(res_v_gpu, new_res_v_gpu), "Single vmap + jit (GPU) failed"
print("Jit (GPU), single vmap: SUCCESS")
