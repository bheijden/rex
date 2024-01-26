from functools import partial
import hashlib
import os
import dill as pickle
from typing import Union, Tuple
from math import ceil
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import brax
from brax.io import mjcf
from brax.base import Transform, System
from brax.generalized import pipeline as gen_pipeline
from brax.positional import pipeline as pos_pipeline
from brax.spring import pipeline as spr_pipeline

import mujoco
from mujoco import mjx

from rex.jax_utils import tree_extend
import sysid.utils as sid


@struct.dataclass
class Params:
    damping: Union[float, jax.typing.ArrayLike]
    armature: Union[float, jax.typing.ArrayLike]
    gear: Union[float, jax.typing.ArrayLike]
    mass_weight: Union[float, jax.typing.ArrayLike]
    radius_weight: Union[float, jax.typing.ArrayLike]
    offset: Union[float, jax.typing.ArrayLike]
    friction_loss: Union[float, jax.typing.ArrayLike]
    activation: Union[float, jax.typing.ArrayLike]
    sys: Union[gen_pipeline.System]
    dt_sysid: Union[float, jax.typing.ArrayLike]
    substeps: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=None)

    @classmethod
    def default(cls, sys: gen_pipeline.System = None, dt_sysid: Union[float, jax.typing.ArrayLike] = None, substeps: int = None):
        # matlab nlgreyest params
        J = 0.000159931461600856
        M = 0.0508581731919534
        offset = 0.0415233722862552
        radius_weight = 0.0508581731919534
        b = 1.43298488e-05
        K = 0.03333912
        R = 7.73125142

        damping = (b + K**2 / R)
        armature = (J - 0.5*radius_weight**2*M - M*offset**2)
        gear = (K/R)
        mass_weight = M
        radius_weight = radius_weight
        offset = offset
        friction_loss = 0.000975041213361349
        activation = 165.417960777425
        return cls(damping=damping, armature=armature, gear=gear, mass_weight=mass_weight, radius_weight=radius_weight,
                   offset=offset, friction_loss=friction_loss, activation=activation,
                   sys=sys, dt_sysid=dt_sysid, substeps=substeps)


@struct.dataclass
class State:
    init_jpos: Union[float, jax.typing.ArrayLike]
    init_jvel: Union[float, jax.typing.ArrayLike]
    pipeline_state: Union[gen_pipeline.State, mjx.Data, jax.Array]


@struct.dataclass
class Action:
    voltage: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class Output:
    jpos: Union[float, jax.typing.ArrayLike]
    jvel: Union[float, jax.typing.ArrayLike]

@struct.dataclass
class ODESystem:
    """Pendulum ode param definition"""
    dt: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.01)
    J: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.000159931461600856)
    mass: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.0508581731919534)
    length: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.0415233722862552)
    b: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 1.43298488358436e-05)
    K: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.0333391179016334)
    R: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 7.73125142447252)
    c: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 0.000975041213361349)
    d: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: 165.417960777425)


def sigmoid(x):
    # For very large negative values of x, the sigmoid is very close to 0.
    # For very large positive values of x, the sigmoid is very close to 1.
    # We can use these properties to avoid computing the exponential in these cases.
    safe_x = jnp.clip(x, -35, 35)
    # large_positive = x > 35
    # large_negative = x < -35
    # safe_x = jnp.where(large_positive, 0, jnp.where(large_negative, 0, x))

    pos = 1.0 / (1.0 + jnp.exp(-safe_x))
    return pos


def runge_kutta4(ode, dt, params, x, u):
    k1 = ode(params, x, u)
    k2 = ode(params, x + 0.5 * dt * k1, u)
    k3 = ode(params, x + 0.5 * dt * k2, u)
    k4 = ode(params, x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class ODEBackend(sid.Backend):
    name = "ode"

    def init_backend(self, dt_sysid: float, dt: float = None, sys: ODESystem = None) -> Params:
        sys = ODESystem() if sys is None else sys
        sys = sys if dt is None else sys.replace(dt=dt)
        substeps = ceil(dt_sysid / sys.dt)
        if substeps > 1:
            raise NotImplementedError("substeps > 1 not implemented yet")
        dt_sysid = substeps * sys.dt
        print(f"\nTIME")
        print(f"dt: {sys.dt}, dt_sysid: {dt_sysid}, substeps: {substeps}")
        params = Params.default(sys=sys, dt_sysid=dt_sysid, substeps=substeps)
        return params

    def init_sys(self, pre_params: Params) -> Params:
        p = pre_params
        J = p.armature + 0.5 * p.radius_weight ** 2 * p.mass_weight + p.mass_weight * p.offset ** 2
        mass = p.mass_weight
        length = p.offset
        R = 7.73125142
        K = p.gear * R
        b = p.damping - K ** 2 / R
        c = p.friction_loss
        d = p.activation
        sys = ODESystem(dt=p.dt_sysid, J=J, mass=mass, length=length, b=b, K=K, R=R, c=c, d=d)
        post_params = pre_params.replace(sys=sys)
        return post_params

    def init_pipeline(self, params: Params, pre_state: State) -> State:
        init_jpos = pre_state.init_jpos
        init_jvel = pre_state.init_jvel
        pipeline_state = jnp.array([init_jpos, init_jvel])
        post_state = pre_state.replace(pipeline_state=pipeline_state)
        return post_state

    def get_output(self, params: Params, state: State) -> Output:
        y = Output(jpos=state.pipeline_state[0], jvel=state.pipeline_state[1])
        return y

    def step(self, params: Params, state: State, action: Action) -> Tuple[State, Output]:
        new_pipeline_state = runge_kutta4(self._ode_pendulum, params.sys.dt, params.sys, state.pipeline_state, action.voltage)
        new_state = state.replace(pipeline_state=new_pipeline_state)
        y = self.get_output(params, new_state)
        return new_state, y

    def _ode_pendulum(self, params: ODESystem, x: jax.Array, u: jax.typing.ArrayLike):
        g, J, m, l, b, K, R, c, d = 9.81, params.J, params.mass, params.length, params.b, params.K, params.R, params.c, params.d

        ddx = (u * K / R + m * g * l * jnp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * (2 * sigmoid(d * x[1]) - 1)) / J

        return jnp.array([x[1], ddx])


class BraxBackend(sid.Backend):
    name = "brax"
    xml_path = "/home/r2ci/rex/envs/pendulum/assets/disk_pendulum.xml"

    def init_backend(self, dt_sysid: float, dt: float = None, sys: gen_pipeline.System = None) -> Params:
        sys = mjcf.load(self.xml_path) if sys is None else sys
        sys = sys.replace(dt=dt) if dt is not None else sys

        print(f"degrees of freedom: {sys.qd_size()}")

        # Determine collision pairs
        print("\nCOLLISIONS")
        from brax.geometry.contact import _geom_pairs

        for (geom_i, geom_j) in _geom_pairs(sys):
            # print(geom_i.link_idx, geom_j.link_idx)
            name_i = sys.link_names[geom_i.link_idx[0]] if geom_i.link_idx is not None else "world"
            name_j = sys.link_names[geom_j.link_idx[0]] if geom_j.link_idx is not None else "world"
            print(f"collision pair: {name_i} --> {name_j}")

        # Actuators
        print("\nACTUATOR SIZE")
        print(f"actuator size: {sys.act_size()}")
        q_id = sys.actuator.q_id[:1]

        # DOFS
        print("\nDEGREES OF FREEDOM SIZE")
        print(f"degrees of freedom: {sys.qd_size()}")

        substeps = ceil(dt_sysid / sys.dt)
        dt_sysid = substeps * sys.dt
        print(f"\nTIME")
        print(f"dt: {sys.dt}, dt_sysid: {dt_sysid}, substeps: {substeps}")
        return Params.default(sys=sys, dt_sysid=dt_sysid, substeps=substeps)

    def init_sys(self, pre_params: Params) -> Params:
        _m = pre_params.sys
        itransform = _m.link.inertia.transform.replace(pos=jnp.array([[0., pre_params.offset, 0.]]))
        i = _m.link.inertia.i.at[0, 0, 0].set(0.5 * pre_params.mass_weight * pre_params.radius_weight ** 2)  # inertia of cylinder in local frame.
        inertia = _m.link.inertia.replace(transform=itransform, mass=jnp.array([pre_params.mass_weight]), i=i)
        link = _m.link.replace(inertia=inertia)
        actuator = _m.actuator.replace(gear=jnp.array([pre_params.gear]))
        dof = _m.dof.replace(armature=jnp.array([pre_params.armature]), damping=jnp.array([pre_params.damping]))
        new_m = _m.replace(link=link, actuator=actuator, dof=dof)
        post_params = pre_params.replace(sys=new_m)
        return post_params

    def init_pipeline(self, params: Params, pre_state: State) -> State:
        init_jpos = pre_state.init_jpos
        init_jvel = pre_state.init_jvel
        _m = params.sys
        # Set state.
        qpos = _m.init_q.at[0].set(init_jpos)
        qvel = jnp.array([init_jvel])
        pipeline_state = gen_pipeline.init(_m, qpos, qvel)
        post_state = pre_state.replace(pipeline_state=pipeline_state)
        return post_state

    def get_output(self, params: Params, state: State) -> Output:
        y = Output(jpos=state.pipeline_state.q[0], jvel=state.pipeline_state.qd[0])
        return y

    def step(self, params: Params, state: State, action: Action) -> Tuple[State, Output]:
        jvel = state.pipeline_state.qd[0]
        activation = 2 * sigmoid(params.activation * jvel) - 1
        friction = params.friction_loss * activation / params.sys.actuator.gear[0]
        a_friction = action.voltage - friction
        pipeline_state = gen_pipeline.step(params.sys, state.pipeline_state, jnp.array(a_friction)[None])
        new_state = state.replace(pipeline_state=pipeline_state)
        y = self.get_output(params, new_state)
        return new_state, y


def residual(backend: sid.Backend, opt_params: Params, args) -> Union[Output]:
    base_params, pre_s, actions, init_y_ys = args

    # Replace base_params with optimizable params and re-initialize system with new params
    # NOTE: ugly fix for substeps --> trees don't deal with static fields very well
    opt_params = opt_params.replace(substeps=base_params.substeps)
    # opt_params = jax.tree_util.tree_map(jnp.exp, opt_params)  # Transform to unconstrained space
    opt_params = tree_extend(base_params, opt_params)  # Extend opt_params to match base_params pytree structure
    pre_params = jax.tree_util.tree_map(lambda base_x, opt_x: base_x if opt_x is None else opt_x, base_params, opt_params)
    params = backend.init_sys(pre_params)  # Updates parameters in params.sys

    # Get initial state (incl. pipeline state)
    init_s = jax.vmap(backend.init_pipeline, in_axes=(None, 0))(params, pre_s)

    # Rollout with params
    pred_final_s, pred_init_y_ys = jax.vmap(backend.rollout, in_axes=(None, 0, 0))(params, init_s, actions)

    # Get residual (label - pred)
    pre_res = jax.tree_util.tree_map(lambda label, pred: label - pred, init_y_ys, pred_init_y_ys)

    # Replace with None if masking some residual components
    res = pre_res
    return res
