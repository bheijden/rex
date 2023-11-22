import jax
import numpy as np
import jax.numpy as jnp
from math import ceil
import os
import tqdm
import time

import mujoco
import mujoco.viewer
from mujoco import mjx
from mujoco.mjx._src import types, math
from jax.debug import print

# import brax
# from brax.io import html
# from brax.io import mjcf
# from brax.generalized import pipeline as gen_pipeline
# from brax.positional import pipeline as pos_pipeline
# from brax.spring import pipeline as spr_pipeline
# from brax import base, math
from flax import struct

from rex.utils import timer

jnp.set_printoptions(precision=3, suppress=True)

# todo: tune solver params to increase speed.

print("loading system")
mj_m = mujoco.MjModel.from_xml_path("/home/r2ci/rex/envs/vx300s/assets/vx300s_mjx.xml")
mjx_m: types.Model = mjx.device_put(mj_m)

# Get sampling time (0.8s horizon needed)
total_time = 5
cem_dt = 0.15
substeps = ceil(cem_dt / mjx_m.opt.timestep)
timesteps = ceil(total_time / cem_dt)
assert cem_dt > mjx_m.opt.timestep
dt = cem_dt / substeps
mjx_m = mjx_m.replace(opt=mjx_m.opt.replace(timestep=dt))
horizon = 4
print(f"\nTIME")
print(f"cem_dt: {cem_dt}, brax_dt: {dt}, substeps: {substeps}, horizon_steps: {horizon}, horizon_t: {horizon * cem_dt}, t_final: {timesteps*cem_dt}")

control_high = 0.2 * 3.1416 * jnp.ones(mjx_m.nu)
control_low = -control_high

cem_hyperparams = {
    'sampling_smoothing': 0.,
    'evolution_smoothing': 0.1,
    'elite_portion': 0.1,
    'max_iter': 3,
    'num_samples': 100
}


def safe_norm(x: jax.Array, axis = None) -> jax.Array:
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for jnp.linal.norm(0) see
    https://github.com/google/jax/issues/3058 for details
    Args:
      x: A jnp.array
      axis: The axis along which to compute the norm

    Returns:
      Norm of the array x.
    """

    is_zero = jnp.allclose(x, 0.0)
    # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis)
    n = jnp.where(is_zero, 0.0, n)
    return n


def view_state(m, d, mjx_d):
    viewer = mujoco.viewer.launch_passive(m, d)
    mjx.device_get_into(d, mjx_d)
    viewer.sync()


def view_rollout(m, rollout, verbose=False):
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
            print(f"keycode: {keycode} | chr: {chr(keycode)}")


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
                # print(f"max(xfrc_applied)={pipeline_state.xfrc_applied.max()} |  min(xfrc_applied)={pipeline_state.xfrc_applied.min()} | qfrc_applied={pipeline_state.qfrc_applied}")
                print(
                    f"max={pipeline_state.qfrc_constraint.max():.3f} | min={pipeline_state.qfrc_constraint.min():.3f} | sum={jnp.abs(pipeline_state.qfrc_constraint).sum():.3f} |  qfrc_constraint={pipeline_state.qfrc_constraint}"
                    # f"max(qfrc_constraint)={pipeline_state.qfrc_constraint.max()} |  min(qfrc_constraint)={pipeline_state.qfrc_constraint.min()} \n"
                    # f"min(dist)={min(0, pipeline_state.contact.dist.min()): .4f} | "
                )
                # print(f"max(xfrc_applied)={pipeline_state.xfrc_applied.max()} |  min(xfrc_applied)={pipeline_state.xfrc_applied.min()} | qfrc_applied={pipeline_state.qfrc_applied}")

                mjx.device_get_into(d, pipeline_state)
                viewer.sync()
                end = time.time()
                elapsed = end - start
                i = (i + 1) % nsamples
            time.sleep(max(0, dt / _real_time_factor - elapsed))


def format_info(info):
    formatted_items = []
    for key, value in info.items():
        try:
            value_list = value.tolist()
            if isinstance(value_list, list):
                value_list = [round(v, 2) for v in value_list]
            else:
                value_list = round(value_list, 2)
            formatted_items.append(f"{key}: {value_list}")
        except AttributeError:
            formatted_items.append(f"{key}: {round(value, 2)}")

    formatted_string = ' | '.join(formatted_items)
    return formatted_string


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath
    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


@struct.dataclass
class State:
    pipeline_state: types.Data
    q_des: jnp.ndarray


@struct.dataclass
class Params:
    substeps: jnp.ndarray
    sys: types.Model


def _cost(params: Params, state: State, action):
    pipeline_state = state.pipeline_state
    sys = params.sys

    # Get indices
    ee_arm_idx = mj_m.body("ee_link").geomadr[0]  # sys.link_names.index("ee_link")
    box_idx = mj_m.body("box").geomadr[0]
    goal_idx = mj_m.body("goal").geomadr[0]

    x_i = pipeline_state.xipos
    boxpos = x_i[box_idx]
    eepos = x_i[ee_arm_idx]
    goalpos = x_i[goal_idx][:2]

    rot_mat = pipeline_state.ximat[ee_arm_idx]
    ee_to_goal = goalpos - eepos[:2]
    box_to_goal = goalpos - boxpos[:2]

    # dot product of ee_yaxis with ee_to_goal (ee parallel to box?)
    norm_ee_to_goal = ee_to_goal / safe_norm(ee_to_goal)
    cost_orn = jnp.abs(jnp.dot(rot_mat[:2, 1], norm_ee_to_goal))

    norm_box_to_goal = box_to_goal / safe_norm(box_to_goal)
    target_ee_xaxis = jnp.concatenate([norm_box_to_goal, jnp.array([-5.0])])
    norm_target_ee_xaxis = target_ee_xaxis / safe_norm(target_ee_xaxis)
    cost_down = (1 - jnp.dot(rot_mat[:3, 0], norm_target_ee_xaxis))
    # cost_down = 0.5 * jnp.abs(rot_mat[2, 0]+1)

    # Distances in xy-plane
    box_to_ee = (eepos - boxpos)[:2]
    box_to_goal = (goalpos - boxpos[:2])
    dist_box_to_ee = safe_norm(box_to_ee)
    dist_box_to_goal = safe_norm(box_to_goal)
    cost_align = (jnp.sum(box_to_ee * box_to_goal) / (dist_box_to_ee * dist_box_to_goal) + 1)

    # Radius cost
    box_dist = safe_norm(box_to_goal)
    ee_dist = safe_norm(ee_to_goal)
    cost_radius = jnp.where(ee_dist <= (box_dist + 0.06), 15.0, 0.0)

    # Force cost
    # cost_force = (pipeline_state.qfrc_constraint[2] - 0.46) ** 2
    # cost_force = jnp.abs(pipeline_state.qfrc_constraint).sum()
    cost_force = (jnp.abs(pipeline_state.qfrc_constraint).max() - 0.46) ** 2

    # cost_z = 1.0 * jnp.abs(eepos[2] - 0.09)
    cost_z = jnp.abs(eepos[2] - 0.075)
    cost_near = safe_norm((boxpos - eepos)[:2])
    cost_dist = safe_norm(boxpos[:2] - goalpos)
    cost_ctrl = safe_norm(action)

    cm = cost_dist

    # Weight all costs
    alpha = 1 / (1 + 2.0 * jnp.abs(cost_down + cost_orn))
    cost_orn = 3.0 * cost_orn
    cost_down = 3.0 * cost_down
    cost_radius = 0.0 * cost_radius
    cost_force = 1.0 * cost_force
    cost_align = 2.0 * cost_align
    cost_z = 1.0 * cost_z * alpha
    cost_near = 2.0 * cost_near * alpha
    cost_dist = 20.0 * cost_dist * alpha
    cost_ctrl = 0.1 * cost_ctrl

    total_cost = cost_ctrl + cost_z + cost_near + cost_dist + cost_radius + cost_orn + cost_down + cost_align + cost_force
    info = {"cm": cm * 100, "cost": total_cost, "cost_orn": cost_orn, "cost_force": cost_force, "cost_down": cost_down,
            "cost_radius": cost_radius, "cost_align": cost_align, "cost_z": cost_z, "cost_near": cost_near,
            "cost_dist": cost_dist, "cost_ctrl": cost_ctrl, "alpha": alpha}
    # return cost_z + cost_dist + cost_near + cost_ctrl + cost_radius + cost_orn + cost_down
    return total_cost, info


def cost(params, state, action, time_step: int):
    total_cost, info = _cost(params, state, action)
    return total_cost


def dynamics(params: Params, state: State, action, time_step):
    def loop_cond(args):
        i, _ = args
        return i < params.substeps

    def loop_body(args):
        i, state = args
        q_des = state.q_des + action * params.sys.opt.timestep  # todo: clip to max angles?
        pipeline_state = state.pipeline_state.replace(ctrl=q_des)
        pipeline_state = mjx.step(params.sys, pipeline_state)
        # pipeline_state = pipeline.step(params.sys, state.pipeline_state, q_des)
        return i + 1, State(pipeline_state=pipeline_state, q_des=q_des)

    i, state = jax.lax.while_loop(loop_cond, loop_body, (0, state))

    return state


from trajax.optimizers import cem, random_shooting
from functools import partial

jit_cem = jax.jit(partial(cem, cost, dynamics, hyperparams=cem_hyperparams))
jit_cost = jax.jit(_cost)


mj_d = mujoco.MjData(mj_m)
mjx_d = mjx.device_put(mj_d)


def env_reset(m: types.Model, d: types.Data, boxpos_home: jax.Array, goalpos: jax.Array, joints_home: jax.Array):
    qpos = jnp.concatenate([boxpos_home, goalpos, joints_home, jnp.array([0])])
    d = d.replace(qpos=qpos, ctrl=jnp.zeros(m.nu))
    d = mjx.forward(m, d)
    return d


def env_step(m: types.Model, d: types.Data, ctrl: jax.Array) -> types.Data:
    d = d.replace(ctrl=ctrl)
    d = mjx.step(m, d)
    return d


# Initialize (NO CONTROL)
jit_env_reset = jax.jit(env_reset)
jit_env_step = jax.jit(env_step)
rng = jax.random.PRNGKey(seed=1)
with timer("jit[reset]", log_level=100):
    boxpos_home = jnp.array([0., 0.35, 0.0, 0.051])
    goalpos = boxpos_home[1:3] + jnp.array([-0.1, 0.45])
    joints_home = jnp.array([0., 0., 0., 0., jnp.pi/2, 0.])
    pipeline_state = jit_env_reset(mjx_m, mjx_d, boxpos_home, goalpos, joints_home)
with timer("eval[reset]", log_level=100):
    _ = jit_env_reset(mjx_m, mjx_d, boxpos_home, goalpos, joints_home)
if False:
    with timer("jit[step]", log_level=100):
        _ = jit_env_step(mjx_m, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(mjx_m.nu))
    with timer("eval[step]", log_level=100):
        _ = jit_env_step(mjx_m, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(mjx_m.nu))

# Rollouts (NO CONTROL)
if False:
    rollout = []
    pipeline_state = jit_env_reset(mjx_m, mjx_d, boxpos_home, goalpos, joints_home)
    params = Params(sys=mjx_m, substeps=substeps)
    next_q_des = pipeline_state.qpos[mjx_m.actuator_trnid]
    pbar = tqdm.tqdm(range(timesteps), desc=f"Episode")
    for i in pbar:
        next_state = State(pipeline_state=pipeline_state, q_des=next_q_des)
        total_cost, info = _cost(params, next_state, next_q_des)
        total_cost, info = jit_cost(params, next_state, next_q_des)
        pbar.set_postfix_str(format_info(info))
        # print("")  # Uncomment to print history.
        for j in range(substeps):
            mean = jnp.zeros((1, mjx_m.nu))
            mean = mean.at[0, 0].set(0.0)
            # next_q_des = next_q_des + mean[0] * m.dt  # todo: clip to max angles?
            pipeline_state = jit_env_step(mjx_m, pipeline_state, next_q_des)
            rollout.append(pipeline_state)

    # Visualize
    view_rollout(mj_m, rollout)

# Initialize (CEM)
init_state = State(pipeline_state=pipeline_state, q_des=pipeline_state.qpos[mjx_m.actuator_trnid])
params = Params(sys=mjx_m, substeps=substeps)
init_controls = jnp.zeros((horizon, mjx_m.nu))
with timer("jit[cem]", log_level=100):
    X, mean, obj = jit_cem(params, init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))
with timer("eval[cem]", log_level=100):
    _ = jit_cem(params, init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))

# Rollouts (CEM)
rollout = []
pipeline_state = jit_env_reset(mjx_m, mjx_d, boxpos_home, goalpos, joints_home)
params = Params(sys=mjx_m, substeps=substeps)
next_q_des = pipeline_state.qpos[mjx_m.actuator_trnid]
init_controls = jnp.zeros((horizon, mjx_m.nu))
_, next_controls, _ = jit_cem(params, init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))
pbar = tqdm.tqdm(range(timesteps), desc=f"Episode")
for i in pbar:
    next_state = State(pipeline_state=pipeline_state, q_des=next_q_des)
    X, mean, obj = jit_cem(params, next_state, next_controls, control_low, control_high, jax.random.PRNGKey(0))
    total_cost, info = jit_cost(params, next_state, mean[0])
    pbar.set_postfix_str(format_info(info))
    # print("")  # Uncomment to print history.
    for j in range(substeps):
        next_q_des = next_q_des + mean[0] * mjx_m.opt.timestep  # todo: clip to max angles?
        pipeline_state = jit_env_step(mjx_m, pipeline_state, next_q_des)
        rollout.append(pipeline_state)
    next_controls = jnp.vstack((mean[1:], jnp.zeros((1, mjx_m.nu))))

# Visualize
view_rollout(mj_m, rollout)
