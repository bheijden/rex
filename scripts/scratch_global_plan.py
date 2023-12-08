import matplotlib.pyplot as plt

import jumpy
import jax
import jumpy.numpy as jp
import rex.jumpy as rjp
import jax.numpy as jnp
import numpy as onp

from rex.base import InputState
from envs.vx300s.env import PlannerOutput


def get_next_jpos(plan, ts):
    """Interpolate next joint position"""
    jpos = plan.jpos
    jvel = plan.jvel
    timestamps = plan.timestamps
    next_jpos = jpos

    # Loop body function
    def loop_body(idx, _next_jpos):
        last_timestamp = timestamps[idx - 1]
        current_timestamp = timestamps[idx]
        current_delta = jvel[idx - 1]

        # Calculate the interpolation factor alpha
        alpha = (ts - last_timestamp) / (current_timestamp - last_timestamp + jp.float32(1e-6))
        alpha = jp.clip(alpha, jp.float32(0), jp.float32(1))

        # Determine interpolated delta
        interpolated_delta = alpha * current_delta * (current_timestamp - last_timestamp)

        # Update _next_jpos
        _next_jpos = _next_jpos + interpolated_delta
        return _next_jpos

    # Perform loop
    next_jpos = jumpy.lax.fori_loop(1, len(timestamps), loop_body, next_jpos)
    return next_jpos


def update_global_plan(timestamps_global, jvel_global, timestamps, jvel):
    idx = jp.argmax(timestamps_global > timestamps[0])
    timestamps_global = rjp.dynamic_update_slice(timestamps_global, timestamps, (idx,))
    jvel_global = rjp.dynamic_update_slice(jvel_global, jvel, (idx, 0))
    return timestamps_global, jvel_global


# @checkify.checkify
def get_global_plan(plan_history: PlannerOutput, debug: bool = False):
    num_plans = plan_history.jvel.shape[0]
    horizon = plan_history.jvel.shape[1]
    other_dims = plan_history.jvel.shape[2:]

    # Initialize global plan
    jpos_global = plan_history.jpos[0]
    jvel_global = jp.zeros((num_plans * horizon + num_plans - 1,) + other_dims, dtype=jp.float32)
    timestamps_global = jp.amax(plan_history.timestamps[-1]) + 1e-6*jp.arange(1, num_plans*horizon+num_plans+1, dtype=jp.float32)

    # Update global plan
    for i in range(num_plans):
        timestamps_global, jvel_global = update_global_plan(timestamps_global, jvel_global, plan_history.timestamps[i], plan_history.jvel[i])

    # Return global plan
    plan_global = PlannerOutput(jpos_global, jvel_global, timestamps_global)

    if debug:
        for (check_jpos, check_ts) in zip(plan_history.jpos[1:], plan_history.timestamps[1:, 0]):
            check_jpos_global = get_next_jpos(plan_global, check_ts)
            equal = jp.all(jax.numpy.isclose(check_jpos_global, check_jpos))
            jax.debug.print("EQUAL?={equal} {check_jpos_global} vs {check_jpos}", equal=equal, check_jpos_global=check_jpos_global, check_jpos=check_jpos)
            # checkify.check(equal, "NOT EQUAL! {check_jpos_global} vs {check_jpos}", check_jpos_global=check_jpos_global, check_jpos=check_jpos)  # convenient but effectful API
    return plan_global


def get_init_plan(last_plan: PlannerOutput, timestamps: jp.ndarray) -> PlannerOutput:
    get_next_jpos_vmap = jumpy.vmap(get_next_jpos, include=(False, True))
    jpos_timestamps = get_next_jpos_vmap(last_plan, timestamps)
    dt = timestamps[1:] - timestamps[:-1]
    jvel_timestamps = (jpos_timestamps[1:] - jpos_timestamps[:-1]) / dt[:, None]
    return PlannerOutput(jpos=jpos_timestamps[0], jvel=jvel_timestamps, timestamps=timestamps)


get_next_jpos_vmap_plan = jumpy.vmap(get_next_jpos, include=(True, False))
get_next_jpos_vmap_ts = jumpy.vmap(get_next_jpos, include=(False, True))
InputState.push = jax.jit(InputState.push)

# Settings
num_steps = 10
horizon = 3
dt_horizon = 0.15
dt_future = 0.18
dt_planner = 0.21
num_plans = 4
num_joints = 3
joint_idx = 0

first_plan = PlannerOutput(jpos=jp.zeros((num_joints,)),
                           jvel=jp.zeros((horizon, num_joints)),
                           timestamps=dt_horizon*jp.arange(0, horizon + 1, dtype=jp.float32))

_msgs = [first_plan] * num_plans
seq = 0 * jp.arange(-num_plans, 0, dtype=jp.int32) - 1
ts_sent = 0 * jp.arange(-num_plans, 0, dtype=jp.float32)
ts_recv = 0 * jp.arange(-num_plans, 0, dtype=jp.float32)
prev_plans = InputState.from_outputs(seq, ts_sent, ts_recv, _msgs)

fig, ax_traj = plt.subplots(1, 1)
trajectory = []
rngs = jumpy.random.split(jumpy.random.PRNGKey(0), num=num_steps)
for i in range(0, num_steps):
    # i*dt_planner
    ts_next = dt_future + i*dt_planner
    timestamps = ts_next + dt_horizon * jp.arange(0, horizon + 1, dtype=jp.float32)

    # Get init plan
    init_plan = get_init_plan(prev_plans[-1].data, timestamps)

    # Get new plan
    jvel = jumpy.random.uniform(rngs[i], (horizon, num_joints), low=-1, high=1)
    new_plan = init_plan.replace(jvel=jvel)
    trajectory.append(new_plan)

    # Plot subplans
    jpos_plan = get_next_jpos_vmap_ts(new_plan, new_plan.timestamps)[:, joint_idx]
    ax_traj.plot(new_plan.timestamps, jpos_plan, color="black", marker="x")

    # Update prev_plans
    prev_plans = prev_plans.push(i, i * dt_planner + dt_future, i * dt_planner + dt_future, new_plan)

    # Get global plan
    global_plan = get_global_plan(prev_plans.data, debug=False)

    # Plotting
    # todo: jpos of init plan should be on the same line as the latest plan
    if i > 100:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.set(xlim=[global_plan.timestamps[0], global_plan.timestamps[-1]],
               ylim=[-0.5, 0.5])
        ax.vlines(timestamps[0], ymin=-999, ymax=999, color="black", alpha=0.5, linestyle="--")

        jpos_global = get_next_jpos_vmap_ts(global_plan, global_plan.timestamps)[:, joint_idx]
        ax.plot(global_plan.timestamps, jpos_global, color="black", alpha=1.0, linestyle="--", marker="x")

        # Get jpos
        jpos_next = get_next_jpos_vmap_plan(prev_plans.data, ts_next)
        ax.scatter([ts_next]*num_plans, jpos_next[:, joint_idx], color="black", alpha=1.0, marker="^")
        for j in range(num_plans):
            c = ["red", "green", "blue"][j % 3]
            jpos_plan = get_next_jpos_vmap_ts(prev_plans[j].data, prev_plans.data.timestamps[j])[:, joint_idx]
            ax.plot(prev_plans.data.timestamps[j], jpos_plan, color=c, alpha=0.5, marker="o")
            ax.hlines(jpos_next[j, joint_idx], xmin=0, xmax=10, color=c, alpha=0.5, linestyle="--")
            ax.scatter(ts_next, jpos_next[j, joint_idx], color=c, alpha=0.5, marker="^")

seq = 0 * jp.arange(-len(trajectory), 0, dtype=jp.int32) - 1
ts_sent = 0 * jp.arange(-len(trajectory), 0, dtype=jp.float32)
ts_recv = 0 * jp.arange(-len(trajectory), 0, dtype=jp.float32)
history = InputState.from_outputs(seq, ts_sent, ts_recv, trajectory)

global_plan = get_global_plan(history.data, debug=False)
jpos_global = get_next_jpos_vmap_ts(global_plan, global_plan.timestamps)[:, joint_idx]
ax_traj.plot(global_plan.timestamps, jpos_global, color="red", alpha=1.0, linestyle="--", marker="x")
plt.show()
print("done")

# jpos_idx = 0
# get_next_jpos_vmap = jumpy.vmap(get_next_jpos, include=(False, True))
# timestamps_plot = jp.linspace(plan_global.timestamps[0], plan_global.timestamps[-1], 100)
# jpos_plot = get_next_jpos_vmap(plan_global, timestamps_plot)[:, jpos_idx]
#
# jpos_global = get_next_jpos_vmap(plan_global, plan_global.timestamps)[:, jpos_idx]
#
# # self._ax.plot(timestamps_plot, jpos_plot, color="black", alpha=1.0, linestyle="--")
# self._ax.plot(plan_global.timestamps, jpos_global, color="black", alpha=1.0, linestyle="--", marker="x")
#
# plans = inputs["planner"].data
# for i in range(plans.timestamps.shape[0]):
# 	c = ["red", "green", "blue"][i % 3]
# 	jpos_plan = get_next_jpos_vmap(plan_global, plans.timestamps[i])[:, jpos_idx]
# 	self._ax.plot(plans.timestamps[i], jpos_plan, color=c, alpha=0.5, marker="o")
