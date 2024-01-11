import os
import matplotlib.pyplot as plt

import jax
import jumpy.numpy as jp
import jax.numpy as jnp
import numpy as onp

from flax import struct

from brax.base import Transform, System
from brax.io import mjcf
from brax.spring import pipeline as s_pipeline

import rex
import rex.open_colors as oc
from rex.proto import log_pb2

from envs.vx300s.env import get_global_plan, get_next_jpos, get_init_plan
import envs.vx300s as vx300s
import experiments as exp


@struct.dataclass
class EEPose:
    eepos: jax.typing.ArrayLike
    eeorn: jax.typing.ArrayLike


def get_ee_pose(sys: System, jpos: jax.typing.ArrayLike) -> EEPose:
    # Set
    qpos = jnp.concatenate([sys.init_q[:6], jpos, jnp.array([0])])
    pipeline_state = s_pipeline.init(sys, qpos, jnp.zeros_like(sys.init_q))
    x_i = pipeline_state.x.vmap().do(
        Transform.create(pos=sys.link.inertia.transform.pos)
    )

    # Get position
    ee_arm_idx = sys.link_names.index("ee_link")
    eepos = x_i.pos[ee_arm_idx]

    # Get orientation
    quat = x_i.rot[ee_arm_idx]
    eeorn = jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype="float32")
    return EEPose(eepos, eeorn)


def plot_box_pushing_experiment(record: log_pb2.EpisodeRecord, xml_path: str, plot_ee: bool = True, plot_jpos: bool = True, plot_cost: bool = True):
    CSCHEME = {"planner": "indigo", "controller": "violet", "armactuator": "grape", "armsensor": "pink"}
    CSCHEME.update({"cm": "blue", "cost": "red", "cost_orn": "pink", "cost_down": "grape", "cost_align": "violet",
                    "cost_height": "indigo", "cost_near": "blue", "cost_dist": "cyan"})
    ECOLOR, FCOLOR = oc.cscheme_fn(CSCHEME)

    # Get data
    helper = exp.RecordHelper(record)
    timestamps = helper._timestamps[EPS_IDX]
    data = helper._data[EPS_IDX]
    data = jax.tree_util.tree_map(lambda x: x.tree if hasattr(x, "tree") else None, data)

    # Get jit functions
    jit_vmap_get_ee_pose = jax.jit(jax.vmap(get_ee_pose, in_axes=(None, 0)))
    jit_vmap_get_global_plan = jax.jit(get_global_plan)
    jit_vmap_get_next_jpos = jax.jit(jax.vmap(get_next_jpos, in_axes=(None, 0)))
    jit_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
    jit_vmap_cost_fn = jax.jit(jax.vmap(vx300s.planner.cost.box_pushing_cost, in_axes=(None, 0, 0, None, 0)))

    # Get global plan
    global_plan = jit_vmap_get_global_plan(data["planner"]["outputs"])
    planner_jpos = jit_vmap_get_next_jpos(global_plan, global_plan.timestamps)

    # Load system
    sys = mjcf.load(xml_path)

    # Interpolate jpos
    timestamps_interp = timestamps["controller"]["ts_output"]
    planner_jpos_interp = jit_vmap_interp(timestamps_interp, global_plan.timestamps, planner_jpos)
    controller_jpos_interp = jit_vmap_interp(timestamps_interp, timestamps["controller"]["ts_output"], data["controller"]["outputs"].jpos)
    armactuator_jpos_interp = jit_vmap_interp(timestamps_interp, timestamps["armactuator"]["ts_output"], data["armactuator"]["outputs"].jpos)
    armsensor_jpos_interp = jit_vmap_interp(timestamps_interp, timestamps["armsensor"]["ts_output"], data["armsensor"]["outputs"].jpos)

    # Interpolate boxpos
    boxpos_interp = jit_vmap_interp(timestamps_interp, timestamps["boxsensor"]["ts_output"], data["boxsensor"]["outputs"].boxpos)

    # Get joint errors
    planner_jpos_error = planner_jpos_interp - armsensor_jpos_interp
    controller_jpos_error = controller_jpos_interp - armsensor_jpos_interp
    armactuator_jpos_error = armactuator_jpos_interp - armsensor_jpos_interp

    # Get ee positions
    planner_eepose = jit_vmap_get_ee_pose(sys, planner_jpos_interp)
    controller_eepose = jit_vmap_get_ee_pose(sys, controller_jpos_interp)
    armactuator_eepose = jit_vmap_get_ee_pose(sys, armactuator_jpos_interp)
    armsensor_eepose = jit_vmap_get_ee_pose(sys, armsensor_jpos_interp)

    # Get Euclidean distance between sensor and other eepos
    planner_ee_error = planner_eepose.eepos - armsensor_eepose.eepos
    controller_ee_error = controller_eepose.eepos - armsensor_eepose.eepos
    armactuator_ee_error = armactuator_eepose.eepos - armsensor_eepose.eepos

    # Get cost
    cost_params = jax.tree_util.tree_map(lambda x: x[0], data["planner"]["step_states"].params.cost_params)
    goalpos = jax.tree_util.tree_map(lambda x: x[0], data["planner"]["step_states"].state.goalpos)
    _, cost_info = jit_vmap_cost_fn(cost_params, boxpos_interp, armsensor_eepose.eepos, goalpos, armsensor_eepose.eeorn)
    cost = cost_info.pop("cost")
    cm = cost_info.pop("cm")
    _ = cost_info.pop("alpha")

    # Plot cost
    fig_cost, axes_cost = plt.subplots(1, 3, figsize=(14, 5))
    axes_cost[0].plot(timestamps_interp, cm, label="cm", color=ECOLOR["cm"])  # Distance plot
    axes_cost[0].set_title("Distance")
    axes_cost[0].set_ylabel("cm")
    axes_cost[0].set_xlabel("time (s)")

    for key, c in cost_info.items():
        if key not in CSCHEME: continue
        axes_cost[1].plot(timestamps_interp, 100*c / cost, label=key, color=ECOLOR[key])  # Percentage of total cost
    axes_cost[1].set_title("Percentage of total cost")
    axes_cost[1].set_ylabel("%")
    axes_cost[1].set_xlabel("time (s)")

    for key, c in cost_info.items():
        if key not in CSCHEME: continue
        axes_cost[2].plot(timestamps_interp, c, label=key, color=ECOLOR[key])  # Absolute cost
    axes_cost[2].plot(timestamps_interp, cost, label="cost", color=ECOLOR["cost"])  # Absolute cost
    axes_cost[2].set_title("cost")
    axes_cost[2].set_xlabel("time (s)")
    axes_cost[-1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))

    # Plot plans
    fig_ee, axes_ee = plt.subplots(2, 4, figsize=(16, 8))
    fig_ee.delaxes(axes_ee[0, 0])
    for ee_idx in range(0, 3):
        ax_pos = axes_ee[0, ee_idx+1]
        ax_err = axes_ee[1, ee_idx+1]
        ax_pos.plot(timestamps_interp, planner_eepose.eepos[:, ee_idx]*100, label=f"planner", color=ECOLOR["planner"])
        ax_pos.plot(timestamps_interp, controller_eepose.eepos[:, ee_idx]*100, label=f"controller", color=ECOLOR["controller"])
        ax_pos.plot(timestamps_interp, armactuator_eepose.eepos[:, ee_idx]*100, label=f"armactuator", color=ECOLOR["armactuator"])
        ax_pos.plot(timestamps_interp, armsensor_eepose.eepos[:, ee_idx]*100, label=f"armsensor", color=ECOLOR["armsensor"])
        ax_pos.set_title(f"ee_pos({['x', 'y', 'z'][ee_idx]})")

        ax_err.plot(timestamps_interp, planner_ee_error[:, ee_idx]*100, label=f"planner", color=ECOLOR["planner"])
        ax_err.plot(timestamps_interp, controller_ee_error[:, ee_idx]*100, label=f"controller", color=ECOLOR["controller"])
        ax_err.plot(timestamps_interp, armactuator_ee_error[:, ee_idx]*100, label=f"armactuator", color=ECOLOR["armactuator"])
        ax_err.set_xlabel("time (s)")
    axes_ee[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    axes_ee[0, 1].set_ylabel("pos (cm)")
    axes_ee[1, 0].set_ylabel("error (cm)")
    axes_ee[1, 0].set_xlabel("time (s)")
    axes_ee[1, 0].set_title("Euclidean error")
    axes_ee[1, 0].plot(timestamps_interp, jnp.linalg.norm(planner_ee_error, axis=-1)*100, label=f"planner", color=ECOLOR["planner"])
    axes_ee[1, 0].plot(timestamps_interp, jnp.linalg.norm(controller_ee_error, axis=-1)*100, label=f"controller", color=ECOLOR["controller"])
    axes_ee[1, 0].plot(timestamps_interp, jnp.linalg.norm(armactuator_ee_error, axis=-1)*100, label=f"armactuator", color=ECOLOR["armactuator"])

    # Plot jpos
    fig_jpos, axes_jpos = plt.subplots(2, 6, figsize=(24, 8))
    for joint_idx in range(6):
        ax_jpos = axes_jpos[0, joint_idx]
        ax_err = axes_jpos[1, joint_idx]

        joint_labels = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

        # Set axis labels
        ax_jpos.set_title(f"{joint_labels[joint_idx]}")
        ax_err.set_xlabel("time (s)")

        # Plot planner output
        ax_jpos.plot(global_plan.timestamps, planner_jpos[:, joint_idx], label=f"planner", color=ECOLOR["planner"])
        ax_err.plot(timestamps_interp, planner_jpos_error[:, joint_idx], label=f"planner", color=ECOLOR["planner"])

        # Plot controller output
        controller_jpos = data["controller"]["outputs"].jpos
        controller_ts = timestamps["controller"]["ts_output"]
        ax_jpos.plot(controller_ts, controller_jpos[:, joint_idx], label=f"controller", color=ECOLOR["controller"])
        ax_err.plot(timestamps_interp, controller_jpos_error[:, joint_idx], label=f"controller", color=ECOLOR["controller"])

        # Plot controller output
        armactuator_jpos = data["armactuator"]["outputs"].jpos
        armactuator_ts = timestamps["controller"]["ts_output"]
        ax_jpos.plot(armactuator_ts, armactuator_jpos[:, joint_idx], label=f"armactuator", color=ECOLOR["armactuator"])
        ax_err.plot(timestamps_interp, armactuator_jpos_error[:, joint_idx], label=f"armactuator", color=ECOLOR["armactuator"])

        # Plot armsensor output
        armsensor_jpos = data["armsensor"]["outputs"].jpos
        armsensor_ts = timestamps["armsensor"]["ts_output"]
        ax_jpos.plot(armsensor_ts, armsensor_jpos[:, joint_idx], label=f"armsensor", color=ECOLOR["armsensor"])
    axes_jpos[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    axes_jpos[0, 0].set_ylabel("joint position (rad)")
    axes_jpos[1, 0].set_ylabel("error (rad)")
    return fig_ee, fig_jpos, fig_cost


if __name__ == "__main__":
    PATH_VX300S = os.path.dirname(vx300s.__file__)
    LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/"
    XML_PATH = f"{PATH_VX300S}/assets/vx300s_cem_brax.xml"
    # LOG_DIR = f"{LOG_DIR}/real_rex_randomeps_largeS_mock_10eps_vx300s_2023-12-08-1653"
    LOG_DIR = f"{LOG_DIR}/simulate_4horizon_smallS_vx300s_2023-12-10-1221"
    EPS_IDX = 4

    # Get data
    record_eps = log_pb2.ExperimentRecord()
    with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
        record_eps.ParseFromString(f.read())

    # Plot
    # fig_ee, fig_jpos, fig_cost = plot_box_pushing_experiment(record_eps.episode[EPS_IDX], XML_PATH, plot_ee=True, plot_jpos=True, plot_cost=True)
    fig_ee, fig_jpos, fig_cost = vx300s.show_box_pushing_experiment(record_eps.episode[EPS_IDX], XML_PATH, plot_ee=True, plot_jpos=True, plot_cost=True)

    # # Get data
    # helper = exp.RecordHelper(record_eps)
    # timestamps = helper._timestamps[EPS_IDX]
    # data = helper._data[EPS_IDX]
    # data = jax.tree_util.tree_map(lambda x: x.tree if hasattr(x, "tree") else None, data)
    #
    # # Get jit functions
    # jit_vmap_get_ee_pose = jax.jit(jax.vmap(get_ee_pose, in_axes=(None, 0)))
    # jit_vmap_get_global_plan = jax.jit(get_global_plan)
    # jit_vmap_get_next_jpos = jax.jit(jax.vmap(get_next_jpos, in_axes=(None, 0)))
    # jit_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
    # jit_vmap_cost_fn = jax.jit(jax.vmap(vx300s.planner.cost.box_pushing_cost, in_axes=(None, 0, 0, None, 0)))
    #
    # # Get global plan
    # global_plan = jit_vmap_get_global_plan(data["planner"]["outputs"])
    # planner_jpos = jit_vmap_get_next_jpos(global_plan, global_plan.timestamps)
    #
    # # Load system
    # sys = mjcf.load(XML_PATH)
    #
    # # Interpolate jpos
    # timestamps_interp = timestamps["controller"]["ts_output"]
    # planner_jpos_interp = jit_vmap_interp(timestamps_interp, global_plan.timestamps, planner_jpos)
    # controller_jpos_interp = jit_vmap_interp(timestamps_interp, timestamps["controller"]["ts_output"], data["controller"]["outputs"].jpos)
    # armactuator_jpos_interp = jit_vmap_interp(timestamps_interp, timestamps["armactuator"]["ts_output"], data["armactuator"]["outputs"].jpos)
    # armsensor_jpos_interp = jit_vmap_interp(timestamps_interp, timestamps["armsensor"]["ts_output"], data["armsensor"]["outputs"].jpos)
    #
    # # Interpolate boxpos
    # boxpos_interp = jit_vmap_interp(timestamps_interp, timestamps["boxsensor"]["ts_output"], data["boxsensor"]["outputs"].boxpos)
    #
    # # Get joint errors
    # planner_jpos_error = planner_jpos_interp - armsensor_jpos_interp
    # controller_jpos_error = controller_jpos_interp - armsensor_jpos_interp
    # armactuator_jpos_error = armactuator_jpos_interp - armsensor_jpos_interp
    #
    # # Get ee positions
    # planner_eepose = jit_vmap_get_ee_pose(sys, planner_jpos_interp)
    # controller_eepose = jit_vmap_get_ee_pose(sys, controller_jpos_interp)
    # armactuator_eepose = jit_vmap_get_ee_pose(sys, armactuator_jpos_interp)
    # armsensor_eepose = jit_vmap_get_ee_pose(sys, armsensor_jpos_interp)
    #
    # # Get Euclidean distance between sensor and other eepos
    # planner_ee_error = planner_eepose.eepos - armsensor_eepose.eepos
    # controller_ee_error = controller_eepose.eepos - armsensor_eepose.eepos
    # armactuator_ee_error = armactuator_eepose.eepos - armsensor_eepose.eepos
    #
    # # Get cost
    # cost_params = jax.tree_util.tree_map(lambda x: x[0], data["planner"]["step_states"].params.cost_params)
    # goalpos = jax.tree_util.tree_map(lambda x: x[0], data["planner"]["step_states"].state.goalpos)
    # _, cost_info = jit_vmap_cost_fn(cost_params, boxpos_interp, armsensor_eepose.eepos, goalpos, armsensor_eepose.eeorn)
    # cost = cost_info.pop("cost")
    # cm = cost_info.pop("cm")
    # alpha = cost_info.pop("alpha")
    #
    # # Plot cost
    # fig, axes_cost = plt.subplots(1, 3, figsize=(14, 5))
    # axes_cost[0].plot(timestamps_interp, cm, label="cm", color=ECOLOR["cm"])  # Distance plot
    # axes_cost[0].set_title("Distance")
    # axes_cost[0].set_ylabel("cm")
    # axes_cost[0].set_xlabel("time (s)")
    #
    # for key, c in cost_info.items():
    #     if key not in CSCHEME: continue
    #     axes_cost[1].plot(timestamps_interp, 100*c / cost, label=key, color=ECOLOR[key])  # Percentage of total cost
    # axes_cost[1].set_title("Percentage of total cost")
    # axes_cost[1].set_ylabel("%")
    # axes_cost[1].set_xlabel("time (s)")
    #
    # for key, c in cost_info.items():
    #     if key not in CSCHEME: continue
    #     axes_cost[2].plot(timestamps_interp, c, label=key, color=ECOLOR[key])  # Absolute cost
    # axes_cost[2].plot(timestamps_interp, cost, label="cost", color=ECOLOR["cost"])  # Absolute cost
    # axes_cost[2].set_title("cost")
    # axes_cost[2].set_xlabel("time (s)")
    # axes_cost[-1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    #
    # # Plot plans
    # fig, axes_ee = plt.subplots(2, 4, figsize=(16, 8))
    # fig.delaxes(axes_ee[0, 0])
    # for ee_idx in range(0, 3):
    #     ax_pos = axes_ee[0, ee_idx+1]
    #     ax_err = axes_ee[1, ee_idx+1]
    #     ax_pos.plot(timestamps_interp, planner_eepose.eepos[:, ee_idx]*100, label=f"planner", color=ECOLOR["planner"])
    #     ax_pos.plot(timestamps_interp, controller_eepose.eepos[:, ee_idx]*100, label=f"controller", color=ECOLOR["controller"])
    #     ax_pos.plot(timestamps_interp, armactuator_eepose.eepos[:, ee_idx]*100, label=f"armactuator", color=ECOLOR["armactuator"])
    #     ax_pos.plot(timestamps_interp, armsensor_eepose.eepos[:, ee_idx]*100, label=f"armsensor", color=ECOLOR["armsensor"])
    #     ax_pos.set_title(f"ee_pos({['x', 'y', 'z'][ee_idx]})")
    #
    #     ax_err.plot(timestamps_interp, planner_ee_error[:, ee_idx]*100, label=f"planner", color=ECOLOR["planner"])
    #     ax_err.plot(timestamps_interp, controller_ee_error[:, ee_idx]*100, label=f"controller", color=ECOLOR["controller"])
    #     ax_err.plot(timestamps_interp, armactuator_ee_error[:, ee_idx]*100, label=f"armactuator", color=ECOLOR["armactuator"])
    #     ax_err.set_xlabel("time (s)")
    # axes_ee[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    # axes_ee[0, 1].set_ylabel("pos (cm)")
    # axes_ee[1, 0].set_ylabel("error (cm)")
    # axes_ee[1, 0].set_xlabel("time (s)")
    # axes_ee[1, 0].set_title("Euclidean error")
    # axes_ee[1, 0].plot(timestamps_interp, jnp.linalg.norm(planner_ee_error, axis=-1)*100, label=f"planner", color=ECOLOR["planner"])
    # axes_ee[1, 0].plot(timestamps_interp, jnp.linalg.norm(controller_ee_error, axis=-1)*100, label=f"controller", color=ECOLOR["controller"])
    # axes_ee[1, 0].plot(timestamps_interp, jnp.linalg.norm(armactuator_ee_error, axis=-1)*100, label=f"armactuator", color=ECOLOR["armactuator"])
    #
    # # Plot jpos
    # fig, axes_jpos = plt.subplots(2, 6, figsize=(24, 8))
    # for joint_idx in range(6):
    #     ax_jpos = axes_jpos[0, joint_idx]
    #     ax_err = axes_jpos[1, joint_idx]
    #
    #     joint_labels = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    #
    #     # Set axis labels
    #     ax_jpos.set_title(f"{joint_labels[joint_idx]}")
    #     ax_err.set_xlabel("time (s)")
    #
    #     # Plot planner output
    #     ax_jpos.plot(global_plan.timestamps, planner_jpos[:, joint_idx], label=f"planner", color=ECOLOR["planner"])
    #     ax_err.plot(timestamps_interp, planner_jpos_error[:, joint_idx], label=f"planner", color=ECOLOR["planner"])
    #
    #     # Plot controller output
    #     controller_jpos = data["controller"]["outputs"].jpos
    #     controller_ts = timestamps["controller"]["ts_output"]
    #     ax_jpos.plot(controller_ts, controller_jpos[:, joint_idx], label=f"controller", color=ECOLOR["controller"])
    #     ax_err.plot(timestamps_interp, controller_jpos_error[:, joint_idx], label=f"controller", color=ECOLOR["controller"])
    #
    #     # Plot controller output
    #     armactuator_jpos = data["armactuator"]["outputs"].jpos
    #     armactuator_ts = timestamps["controller"]["ts_output"]
    #     ax_jpos.plot(armactuator_ts, armactuator_jpos[:, joint_idx], label=f"armactuator", color=ECOLOR["armactuator"])
    #     ax_err.plot(timestamps_interp, armactuator_jpos_error[:, joint_idx], label=f"armactuator", color=ECOLOR["armactuator"])
    #
    #     # Plot armsensor output
    #     armsensor_jpos = data["armsensor"]["outputs"].jpos
    #     armsensor_ts = timestamps["armsensor"]["ts_output"]
    #     ax_jpos.plot(armsensor_ts, armsensor_jpos[:, joint_idx], label=f"armsensor", color=ECOLOR["armsensor"])
    # axes_jpos[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    # axes_jpos[0, 0].set_ylabel("joint position (rad)")
    # axes_jpos[1, 0].set_ylabel("error (rad)")

    plt.show()

