import jax
import jax.numpy as jnp
import os
import tqdm

import brax
from brax.io import html
from brax.io import mjcf
from brax.generalized import pipeline
# from brax.positional import pipeline
# from brax.spring import pipeline
from brax import base, math
from flax import struct

from rex.utils import timer

print("loading system")
m = mjcf.load('/home/r2ci/rex/envs/vx300s/assets/vx300s.xml')

# Determine collision pairs
from brax.geometry.contact import _geom_pairs
for (geom_i, geom_j) in _geom_pairs(m):
    # print(geom_i.link_idx, geom_j.link_idx)
    name_i = m.link_names[geom_i.link_idx[0]] if geom_i.link_idx is not None else "world"
    name_j = m.link_names[geom_j.link_idx[0]] if geom_j.link_idx is not None else "world"
    print(f"collision pair: {name_i} --> {name_j}")

# Actuators
print(f"actuator size: {m.act_size()}")

# Get indices
ee_arm_idx = m.link_names.index("ee_link")
box_idx = m.link_names.index("box")
goal_idx = m.link_names.index("goal")

# Jit
jit_env_reset = jax.jit(pipeline.init)
jit_env_step = jax.jit(pipeline.step)
rng = jax.random.PRNGKey(seed=1)
with timer("jit[reset]", log_level=100):
    boxpos_home = jnp.array([0.35, 0.0, 0.051])
    goalpos = boxpos_home[:2] + jnp.array([-0.1, 0.45])
    qpos = m.init_q.at[9].set(jnp.pi/2)
    qpos = qpos.at[0:5].set(jnp.concatenate([boxpos_home, goalpos]))
    pipeline_state = jit_env_reset(m, qpos, jnp.zeros(m.qd_size()))
# with timer("jit[step]", log_level=100):
#     _ = jit_env_step(m, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(m.act_size()))

# Get sampling time (0.8s horizon needed)
dt = m.dt
substeps = 20
cem_dt = substeps * dt
horizon = 4
control_high = 0.2 * 3.1416 * jnp.ones(m.act_size())
control_low = -control_high
cem_hyperparams = {
      'sampling_smoothing': 0.,
      'evolution_smoothing': 0.1,
      'elite_portion': 0.1,
      'max_iter': 3,
      'num_samples': 100
}

@struct.dataclass
class State:
  pipeline_state: base.State
  q_des: jnp.ndarray


def _cost(state, action):
    pipeline_state = state.pipeline_state
    x_i = pipeline_state.x.vmap().do(
        base.Transform.create(pos=m.link.inertia.transform.pos)
    )
    boxpos = x_i.pos[box_idx]
    eepos = x_i.pos[ee_arm_idx]
    goalpos = x_i.pos[goal_idx][:2]

    rot_mat = math.quat_to_3x3(x_i.rot[ee_arm_idx])
    ee_to_goal = goalpos - eepos[:2]
    box_to_goal = goalpos - boxpos[:2]

    # dot product of ee_yaxis with ee_to_goal
    norm_ee_to_goal = ee_to_goal / math.safe_norm(ee_to_goal)
    cost_orn = jnp.abs(jnp.dot(rot_mat[:2, 1], norm_ee_to_goal))

    norm_box_to_goal = box_to_goal / math.safe_norm(box_to_goal)
    target_ee_xaxis = jnp.concatenate([norm_box_to_goal, jnp.array([-5.0])])
    norm_target_ee_xaxis = target_ee_xaxis / math.safe_norm(target_ee_xaxis)
    cost_down = (1-jnp.dot(rot_mat[:3, 0], norm_target_ee_xaxis))
    # cost_down = 0.5 * jnp.abs(rot_mat[2, 0]+1)

    # Radius cost
    box_dist = math.safe_norm(box_to_goal)
    ee_dist = math.safe_norm(ee_to_goal)
    cost_radius = jnp.where(ee_dist <= (box_dist+0.06), 15.0, 0.0)

    # cost_z = 1.0 * jnp.abs(eepos[2] - 0.09)
    cost_z = jnp.abs(eepos[2] - 0.075)
    cost_near = math.safe_norm((boxpos - eepos)[:2])
    cost_dist = math.safe_norm(boxpos[:2] - goalpos)
    cost_ctrl = math.safe_norm(action)

    cm = cost_dist

    # Weight all costs
    alpha = 1 / (1 + 2.0 * jnp.abs(cost_down + cost_orn))
    cost_orn = 3.0 * cost_orn
    cost_down = 3.0 * cost_down
    cost_radius = 1.0 * cost_radius
    cost_z = 1.0 * cost_z * alpha
    cost_near = 2.0 * cost_near * alpha
    cost_dist = 20.0 * cost_dist * alpha
    cost_ctrl = 0.1 * cost_ctrl

    total_cost = cost_ctrl + cost_z +  cost_near + cost_dist + cost_radius + cost_orn + cost_down
    info = {"cm": cm, "cost": total_cost, "cost_orn": cost_orn, "cost_down": cost_down, "cost_radius": cost_radius, "cost_z": cost_z, "cost_near": cost_near, "cost_dist": cost_dist, "cost_ctrl": cost_ctrl, "alpha": alpha}
    # return cost_z + cost_dist + cost_near + cost_ctrl + cost_radius + cost_orn + cost_down
    return total_cost, info


def cost(state, action, time_step: int):
    total_cost, info = _cost(state, action)
    return total_cost


def dynamics(state: State, action, time_step):

    def loop_body(_, args):
        state, = args
        q_des = state.q_des + action * dt  # todo: clip to max angles?
        pipeline_state = pipeline.step(m, state.pipeline_state, q_des)
        return State(pipeline_state=pipeline_state, q_des=q_des),

    state, = jax.lax.fori_loop(0, substeps, loop_body, (state,))

    return state


from trajax.optimizers import cem, random_shooting
from functools import partial
jit_cem = jax.jit(partial(cem, cost, dynamics, hyperparams=cem_hyperparams))
jit_cost = jax.jit(_cost)
# jit_cem = jax.jit(partial(random_shooting, cost, dynamics, hyperparams=cem_hyperparams))

init_state = State(pipeline_state=pipeline_state, q_des=pipeline_state.q[m.actuator.q_id])
init_controls = jnp.zeros((horizon, m.act_size()))
with timer("jit[cem]", log_level=100):
    X, mean, obj = jit_cem(init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))
with timer("eval[cem]", log_level=100):
    _ = jit_cem(init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))

def format_info(info):
    formatted_items = []
    for key, value in info.items():
        try:
            value_list = value.tolist()
            if isinstance(value_list, list):
                value_list= [round(v, 2) for v in value_list]
            else:
                value_list = round(value_list, 2)
            formatted_items.append(f"{key}: {value_list}")
        except AttributeError:
            formatted_items.append(f"{key}: {round(value, 2)}")

    formatted_string = ' | '.join(formatted_items)
    return formatted_string

# Rollouts
rollout = []
next_q_des = pipeline_state.q[m.actuator.q_id]
next_controls = mean
pbar = tqdm.tqdm(range(30), desc=f"Episode")
for i in pbar:
    next_state = State(pipeline_state=pipeline_state, q_des=next_q_des)
    X, mean, obj = jit_cem(next_state, next_controls, control_low, control_high, jax.random.PRNGKey(0))
    total_cost, info = jit_cost(next_state, mean[0])
    pbar.set_postfix_str(format_info(info))
    print("")
    for j in range(substeps):
        next_q_des = next_q_des + mean[0] * dt  # todo: clip to max angles?
        pipeline_state = jit_env_step(m, pipeline_state, next_q_des)
        rollout.append(pipeline_state)
    next_controls = jnp.vstack((mean[1:], jnp.zeros((1, m.act_size()))))
print("rendering")

html.save("./brax_render.html", m, rollout)
print("done")