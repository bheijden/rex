import pandas as pd
import itertools
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns

import jax
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


if __name__ == "__main__":
    PATH_VX300S = os.path.dirname(vx300s.__file__)
    LOG_PATH = "/home/r2ci/rex/paper/logs"
    XML_PATH = f"{PATH_VX300S}/assets/vx300s_cem_brax.xml"
    # Variable frequency 1.2*mean delay
    EXP_DIRS = {
        "mcs": f"{LOG_PATH}/2023-12-12-1636_real_2ndcalib_rex_randomeps_MCS_recorded_VarHz_3iter_vx300s",
        "topo": f"{LOG_PATH}/2023-12-12-1718_real_2ndcalib_rex_randomeps_topological_recorded_VarHz_3iter_vx300s",
        "gen": f"{LOG_PATH}/2023-12-12-1708_real_2ndcalib_rex_randomeps_generational_recorded_VarHz_3iter_vx300s",
        "seq": f"{LOG_PATH}/2023-12-12-1734_real_2ndcalib_brax_VarHz_3iter_vx300s",
    }
    RECORDS = {}
    for name, LOG_DIR in EXP_DIRS.items():
        RECORDS[name] = log_pb2.ExperimentRecord()
        with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
            RECORDS[name].ParseFromString(f.read())
    fig_perf = vx300s.show_box_pushing_performance(RECORDS, XML_PATH)

    # Fixed frequency 3 Hz
    EXP_DIRS = {
        "mcs": f"{LOG_PATH}/2023-12-12-1746_real_2ndcalib_rex_randomeps_MCS_recorded_3Hz_3iter_vx300s",
        "topo": f"{LOG_PATH}/2023-12-12-1824_real_2ndcalib_rex_randomeps_topological_recorded_3Hz_3iter_vx300s",
        "gen": f"{LOG_PATH}/2023-12-12-1814_real_2ndcalib_rex_randomeps_generational_recorded_3Hz_3iter_vx300s",
        "seq": f"{LOG_PATH}/2023-12-12-1834_real_2ndcalib_brax_3Hz_3iter_vx300s",
    }
    RECORDS = {}
    for name, LOG_DIR in EXP_DIRS.items():
        RECORDS[name] = log_pb2.ExperimentRecord()
        with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
            RECORDS[name].ParseFromString(f.read())
    fig_perf = vx300s.show_box_pushing_performance(RECORDS, XML_PATH)

    plt.show()
    exit()
    # Prepare data
    # PARAMS = {}
    DATA = {}
    TIMESTAMPS = {}
    ts = []
    for name, LOG_DIR in EXP_DIRS.items():
        # # Load params
        # with open(f"{LOG_DIR}/params.yaml", "rb") as f:
        #     PARAMS[name] = yaml.safe_load(f)

        # Get data
        record_eps = log_pb2.ExperimentRecord()
        with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
            record_eps.ParseFromString(f.read())

        # Get data
        helper = exp.RecordHelper(record_eps, method="truncated")

        # Interpolate all data to the same timestamps.
        DATA[name] = helper._data_stacked[:8]
        TIMESTAMPS[name] = helper._timestamps_stacked[:8]

        # Get max timestamps
        ts.append(min(helper._timestamps_stacked["armsensor"]["ts_output"].max(),
                      helper._timestamps_stacked["boxsensor"]["ts_output"].max()))

    # Get jit functions
    jit_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
    jit_vmap_get_ee_pose = jax.jit(jax.vmap(get_ee_pose, in_axes=(None, 0)))
    jit_vmap_cost_fn = jax.jit(jax.vmap(vx300s.planner.cost.box_pushing_cost, in_axes=(None, 0, 0, None, 0)))

    # Load system
    m = mjcf.load(XML_PATH)

    # Interpolate all data to timestamps of the shortest experiment
    print(f"Interpolating to {min(ts)} seconds: {ts}")
    ts = min(ts)
    timestamps_interp = onp.arange(0, ts, 0.05)

    # Interpolate
    DATA_INTERP = {}
    for name, data in DATA.items():
        timestamps = TIMESTAMPS[name]
        num_eps = timestamps["armsensor"]["ts_output"].shape[0]

        # Store all of the below in a dict
        eps_data = []
        for eps_idx in range(num_eps):
            jpos_target = jit_vmap_interp(timestamps_interp, timestamps["armactuator"]["ts_output"][eps_idx], data["armactuator"]["outputs"].jpos[eps_idx])
            jpos = jit_vmap_interp(timestamps_interp, timestamps["armsensor"]["ts_output"][eps_idx], data["armsensor"]["outputs"].jpos[eps_idx])
            boxpos = jit_vmap_interp(timestamps_interp, timestamps["boxsensor"]["ts_output"][eps_idx], data["boxsensor"]["outputs"].boxpos[eps_idx])
            ee_pose = jit_vmap_get_ee_pose(m, jpos)
            ee_pose_target = jit_vmap_get_ee_pose(m, jpos_target)
            cost_params = jax.tree_util.tree_map(lambda x: x[eps_idx, 0], data["planner"]["step_states"].params.cost_params)
            goalpos = jax.tree_util.tree_map(lambda x: x[eps_idx, 0], data["planner"]["step_states"].state.goalpos)

            # Get cost
            _, cost_info = jit_vmap_cost_fn(cost_params, boxpos, ee_pose.eepos, goalpos, ee_pose.eeorn)
            cost = cost_info.pop("cost")
            cm = cost_info.pop("cm")
            _ = cost_info.pop("alpha")

            # Get error
            jpos_err_abs = jnp.abs(jpos_target - jpos)
            ee_error = (ee_pose_target.eepos - ee_pose.eepos) * 100
            ee_error_abs = jnp.abs(ee_error)
            ee_error_norm = jnp.linalg.norm(ee_error, axis=-1)

            # Store all of the above in eps_data
            eps_data.append({
                "jpos_target": jpos_target,
                "jpos": jpos,
                "boxpos": boxpos,
                "ee_pose": ee_pose,
                "ee_pose_target": ee_pose_target,
                "cost_params": cost_params,
                "goalpos": goalpos,
                "cost": cost,
                "cm": cm,
                "jpos_err_abs": jpos_err_abs,
                "ee_error": ee_error,
                "ee_error_abs": ee_error_abs,
                "ee_error_norm": ee_error_norm,
            })

        # stack eps_data
        DATA_INTERP[name] = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *eps_data)

    # Get colors
    CWHEEL = itertools.cycle({c for c in oc.CWHEEL.keys() if c not in ["white", "black", "gray"]})
    CSCHEME = {k: next(CWHEEL) for k in DATA_INTERP.keys()}
    ECOLOR, FCOLOR = oc.cscheme_fn(CSCHEME)

    # Prepare dfs
    perf_df = []
    for k, v in DATA_INTERP.items():
        timestamps_tiled = onp.tile(timestamps_interp, v["cost"].shape[0])
        for t, cost, cm in zip(timestamps_tiled, onp.array(v["cost"]).flatten(), onp.array(v["cm"]).flatten()):
            # For each cost, create a dictionary with 'key' and 'cost'
            perf_df.append({"experiment": k, "time": t, "cost": cost, "cm": cm})

    # Convert the list of dictionaries to a DataFrame
    perf_df = pd.DataFrame(perf_df)

    # Plot cost
    fig, axes_perf = plt.subplots(1, 2, figsize=(10, 5))
    sns.lineplot(ax=axes_perf[0], data=perf_df, x="time", y="cost", palette=ECOLOR, hue="experiment", errorbar="sd")  # Distance plot
    sns.lineplot(ax=axes_perf[1], data=perf_df, x="time", y="cm", palette=ECOLOR, hue="experiment", errorbar="sd")  # Distance plot
    axes_perf[0].set_title("Distance")
    axes_perf[0].set_ylabel("cm")
    axes_perf[0].set_xlabel("time (s)")
    axes_perf[1].set_title("Cost")
    axes_perf[1].set_ylabel("cost")
    axes_perf[1].set_xlabel("time (s)")
    plt.show()
