import os
import matplotlib.pyplot as plt

import jumpy
import jax
import jumpy.numpy as jp
import rex.jax_utils as rjax
import jax.numpy as jnp
import numpy as onp

import rex
from rex.base import InputState
from rex.proto import log_pb2

from envs.vx300s.env import get_global_plan, get_next_jpos, get_init_plan
import envs.vx300s as vx300s
import experiments as exp


PATH_VX300S = os.path.dirname(vx300s.__file__)
LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/"
CONFIG = {"mjx": {"xml_path": f"{PATH_VX300S}/assets/vx300s_mjx.xml"},
          "brax": {"xml_path": f"{PATH_VX300S}/assets/vx300s_brax.xml"},
          "real": {"cam_trans": [0.589, 0.598, 0.355], "cam_rot": [0.252,  0.855, -0.436, -0.125], "cam_idx": 2,
                   "z_fixed": 0.051,
                   "cam_intrinsics": f"{PATH_VX300S}/assets/logitech_c170.yaml"},
          "planner": {"type": "rex",
	                  "brax_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_brax.xml",
                      "rex_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_brax.xml",
                      "mjx_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_mjx.xml",
                      # "rex_graph_path": "/home/r2ci/rex/logs/real_0.35umax_vx300s_2023-11-20-1759/record_pre.pb",
                      "rex_graph_path": "/home/r2ci/rex/logs/real_3winplanner_2eps_vx300s_2023-11-27-1708/record_pre.pb",
                      # "rex_graph_path": "/home/r2ci/rex/logs/vx300s_3winplanner_vx300s_2023-11-23-1722/record_pre.pb",
                      "supergraph_mode": "MCS",
                      "horizon": 2, "u_max": 0.35*3.14, "dt": 0.15, "dt_substeps": 0.015, "num_samples": 75, "max_iter": 2},
          "viewer": {"xml_path": f"{PATH_VX300S}/assets/vx300s_cem_mjx.xml"}}


get_next_jpos_vmap_plan = jumpy.vmap(get_next_jpos, include=(True, False))
get_next_jpos_vmap_ts = jumpy.vmap(get_next_jpos, include=(False, True))
InputState.push = jax.jit(InputState.push)

if __name__ == "__main__":
    DIST_FILE = "real_3winplanner_vx300s_2023-11-27-1647.pkl"  # "real_faster_nomovement_vx300s_2023-11-20-1548.pkl"  # f"vx300s_vx300s_2023-10-19-1629.pkl"
    RATES = dict(world=80, supervisor=8, planner=5.0, controller=20, armactuator=20, armsensor=80, boxsensor=10, viewer=20)
    WIN_PLANNER = 3
    USE_DELAYS = True
    ENV_FN = vx300s.brax.build_vx300s
    DELAY_FN = lambda d: d.quantile(0.99) * int(USE_DELAYS)  # todo: this is slow (takes 3 seconds).
    delays_sim = exp.load_distributions(DIST_FILE, module=vx300s.dists) if DIST_FILE is not None else vx300s.get_default_distributions()
    env = vx300s.make_env(delays_sim, DELAY_FN, RATES, CONFIG, win_planner=WIN_PLANNER, scheduling=0, jitter=0,
                          env_fn=ENV_FN, name="test", clock=0, real_time_factor=0,
                          max_steps=800, use_delays=USE_DELAYS, viewer=False)
    planner = env.nodes["planner"]

    # Load record
    LOG_DIRS = [f"{LOG_DIR}/bug_RexCEMPlanner_vx300s_2023-12-06-1352",
                f"{LOG_DIR}/bug_BraxCEMPlanner_vx300s_2023-12-06-1720"]
    outputs_planner = []
    ss_planner = []
    for LOG_DIR in LOG_DIRS:
        record_eps = log_pb2.ExperimentRecord()
        with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
            record_eps.ParseFromString(f.read())

        # Get data
        helper = exp.RecordHelper(record_eps)
        data = helper._data_stacked
        data = jax.tree_util.tree_map(lambda x: x[0], data)  # Remove episode dimension

        # Get planner data
        planner_data = data["planner"]
        ss_planner.append(data["planner"]["step_states"])
        outputs_planner.append(data["planner"]["outputs"])

    # Plot plans
    fig, axes_traj = plt.subplots(2, 1)

    # Select experiment
    outputs_rex, ss_rex = outputs_planner[0], ss_planner[0]
    outputs_brax, ss_brax = outputs_planner[1], ss_planner[1]

    num_steps = ss_rex.seq.shape[0]
    # todo: make brax planner use new cem implementation.
    # todo: evaluate cost of rex plan vs brax plan.
    # todo: print actions applied to world.
    # todo: evaluate cost of brax plan vs rex plan
    jit_planner_step = jax.jit(planner.step)
    jit_planner_debug = jax.jit(planner._debug)
    for i in range(5, num_steps-1):
        # Get step states
        ss_init_brax = rjax.tree_take(ss_brax, i)
        new_ss_brax = rjax.tree_take(ss_brax, i+1)

        ss_init_rex = rjax.tree_take(ss_rex, i)
        new_ss_rex = rjax.tree_take(ss_rex, i+1)

        # Get outputs
        new_plan_brax = rjax.tree_take(outputs_brax, i)
        new_plan_rex = rjax.tree_take(outputs_rex, i)

        # Prepare rex step_state
        ss_init = ss_init_brax.replace(params=ss_init_rex.params)

        # Compare plans
        cost_new_plan_brax = jit_planner_debug(ss_init, new_plan_brax)  # Compute rex plan from ss_init
        _, new_plan = jit_planner_step(ss_init)  # Evaluate cost of brax plan
        cost_new_plan = jit_planner_debug(ss_init, new_plan)  # Evaluate cost of rex plan
        print(f"Cost of brax plan: {cost_new_plan_brax}, cost of rex plan: {cost_new_plan}")

    # Plot trajectories
    # todo: Run brax_plan
    trajectory = []
    joint_trajectory = []
    joint_timestamps = []
    joint_idx = 0
    ax_rex, ax_brax = axes_traj[0], axes_traj[1]
    for i in range(5, num_steps - 1):
        # Get step states
        ss_init_brax = rjax.tree_take(ss_brax, i)
        new_ss_brax = rjax.tree_take(ss_brax, i + 1)

        ss_init_rex = rjax.tree_take(ss_rex, i)
        new_ss_rex = rjax.tree_take(ss_rex, i + 1)

        # Get outputs
        new_plan_brax = rjax.tree_take(outputs_brax, i)
        new_plan_rex = rjax.tree_take(outputs_rex, i)
        # Get joint position
        jpos_now = ss_init_rex.inputs["armsensor"].data.jpos[-1, joint_idx]
        joint_timestamps.append(ss_init_rex.inputs["armsensor"].ts_recv[-1])
        joint_trajectory.append(jpos_now)

        # Get plan
        trajectory.append(new_plan_rex)

        # Plot subplans
        jpos_plan = get_next_jpos_vmap_ts(new_plan_rex, new_plan_rex.timestamps)[:, joint_idx]
        ax_rex.plot(new_plan_rex.timestamps, jpos_plan, color="black", marker="x")

    # Combine trajectory
    seq = 0 * jnp.arange(-len(trajectory), 0, dtype=jnp.int32) - 1
    ts_sent = 0 * jnp.arange(-len(trajectory), 0, dtype=jnp.float32)
    ts_recv = 0 * jnp.arange(-len(trajectory), 0, dtype=jnp.float32)
    history = InputState.from_outputs(seq, ts_sent, ts_recv, trajectory)

    global_plan = get_global_plan(history.data, debug=False)
    jpos_global = get_next_jpos_vmap_ts(global_plan, global_plan.timestamps)[:, joint_idx]
    ax_rex.plot(global_plan.timestamps, jpos_global, color="red", alpha=1.0, linestyle="--", marker="x")

    ax_rex.plot(jnp.array(joint_timestamps), jnp.array(joint_trajectory), color="blue", alpha=0.5, linestyle="--", marker="x")

    plt.show()

