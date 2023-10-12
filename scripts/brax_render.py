import jax
import jax.numpy as jnp
import os
import tqdm

import brax
from brax.io import html
from brax.io import mjcf
from brax.generalized import pipeline
# from brax.positional import pipeline
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
    pipeline_state = jit_env_reset(m, m.init_q, jnp.zeros(m.qd_size()))
# with timer("jit[step]", log_level=100):
#     _ = jit_env_step(m, pipeline_state, 10 * jnp.sin(1 / 100) * jnp.ones(m.act_size()))

# todo: define cost fn: cost(state: F[n], action: F[d], time_step: int) : float
#   - penalize pose from straight down position
#   - penalize distance ee_link to box (take offset object into account) --> only in xy?
#   - penalize distance box to goal --> only in xy
#   - penalize orientation box to orientation goal --> only yaw.
#   - penalize approaching box with ee_link from above --> penalize z-direction of ee_link

# Get sampling time
dt = m.dt
substeps = 10
cem_dt = substeps * dt
horizon = 3
control_high = 0.2 * 3.1416 * jnp.ones(m.act_size())
control_low = -control_high
cem_hyperparams = {
      'sampling_smoothing': 0.,
      'evolution_smoothing': 0.1,
      'elite_portion': 0.1,
      'max_iter': 5,
      'num_samples': 100
  }

@struct.dataclass
class State:
  pipeline_state: base.State
  q_des: jnp.ndarray


def cost(state, action, time_step: int):
    pipeline_state = state.pipeline_state
    q_des = state.q_des
    x_i = pipeline_state.x.vmap().do(
        base.Transform.create(pos=m.link.inertia.transform.pos)
    )
    cost_z = 1.0 * math.safe_norm(x_i.pos[ee_arm_idx][2]-0.075)
    cost_near = 0.4 * math.safe_norm((x_i.pos[box_idx] - x_i.pos[ee_arm_idx])[:2])
    cost_dist = 4.0 * math.safe_norm((x_i.pos[box_idx] - x_i.pos[goal_idx])[:2])
    cost_ctrl = 0.1 * math.safe_norm(action)
    return cost_z + cost_near + cost_dist + cost_ctrl


def dynamics(state: State, action, time_step):

    def loop_body(_, args):
        state, = args
        q_des = state.q_des + action * dt  # todo: clip to max angles?
        pipeline_state = pipeline.step(m, state.pipeline_state, q_des)
        return State(pipeline_state=pipeline_state, q_des=q_des),

    state, = jax.lax.fori_loop(0, substeps, loop_body, (state,))

    return state


from trajax.optimizers import cem
from functools import partial
jit_cem = jax.jit(partial(cem, cost, dynamics, hyperparams=cem_hyperparams))

init_state = State(pipeline_state=pipeline_state, q_des=pipeline_state.q[m.actuator.q_id])
init_controls = jnp.zeros((horizon, m.act_size()))
with timer("jit[cem]", log_level=100):
    X, mean, obj = jit_cem(init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))
with timer("eval[cem]", log_level=100):
    _ = jit_cem(init_state, init_controls, control_low, control_high, jax.random.PRNGKey(0))

# Rollouts
rollout = []
next_q_des = pipeline_state.q[m.actuator.q_id]
next_controls = mean
for i in tqdm.tqdm(range(30)):
    next_state = State(pipeline_state=pipeline_state, q_des=next_q_des)
    X, mean, obj = jit_cem(next_state, next_controls, control_low, control_high, jax.random.PRNGKey(0))
    for j in range(substeps):
        next_q_des = next_q_des + mean[0] * dt  # todo: clip to max angles?
        pipeline_state = jit_env_step(m, pipeline_state, next_q_des)
        rollout.append(pipeline_state)
    next_controls = jnp.vstack((mean[1:], jnp.zeros((1, m.act_size()))))
print("rendering")

html.save("./brax_render.html", m, rollout)
