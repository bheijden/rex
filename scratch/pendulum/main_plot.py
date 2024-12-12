from typing import Tuple, List
import functools
import dill as pickle
import os
import json
import hashlib
import pandas as pd
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

import rexv2.open_colors as oc
from rexv2 import base
from rexv2.gmm_estimator import GMMEstimator, plot_component_norm_pdfs

# Setup sns plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.5)

# Setup matplotlib
scaling = 5
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 6 * scaling
plt.rcParams['legend.fontsize'] = 5 * scaling
plt.rcParams['font.size'] = 7 * scaling
plt.rcParams['xtick.labelsize'] = 5 * scaling
plt.rcParams['ytick.labelsize'] = 5 * scaling
plt.rcParams['xtick.major.pad'] = -0.0 * scaling
plt.rcParams['ytick.major.pad'] = -0.0 * scaling
plt.rcParams['lines.linewidth'] = 0.65 * scaling
plt.rcParams['lines.markersize'] = 4.0 * scaling
plt.rcParams['axes.xmargin'] = 0.0
plt.rcParams['axes.ymargin'] = 0.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Rescale figures
width_points = 245.71811 * scaling
width_inches = width_points / plt.rcParams['figure.dpi']
default_figsize = plt.rcParams['figure.figsize']
rescaled_width = width_inches
rescaled_height = width_inches * default_figsize[1] / default_figsize[0]
rescaled_figsize = [rescaled_width, rescaled_height]
fullwidth_figsize = [c * s for c, s in zip([1, 0.52], rescaled_figsize)]
halfwidth_figsize = [c * s for c, s in zip([0.5, 0.50], rescaled_figsize)]
thirdwidth_figsize = [c * s for c, s in zip([1 / 3, 0.5], rescaled_figsize)]
fourthwidth_figsize = [c * s for c, s in zip([1 / 4, 0.5], rescaled_figsize)]
fifthwidth_figsize = [c * s for c, s in zip([1 / 5, 0.5], rescaled_figsize)]
twothirdwidth_figsize = [c * s for c, s in zip([2 / 3, 0.5], rescaled_figsize)]
sixthwidth_figsize = [c * s for c, s in zip([1 / 6, 0.5], rescaled_figsize)]
print("Default figsize:", default_figsize)
print("Rescaled figsize:", rescaled_figsize)
print("Fullwidth figsize:", fullwidth_figsize)
print("Halfwidth figsize:", halfwidth_figsize)
print("Thirdwidth figsize:", thirdwidth_figsize)
print("Twothirdwidth figsize:", twothirdwidth_figsize)
print("Sixthwidth figsize:", sixthwidth_figsize)
LABELS = {
    # Delay
    "step_delay": "step_delay",
    "inputs_delay": "input_delay",
    "component_delay": "Gaussian comp. ",
    "gmm_delay": "gmm",
    "data_step_delay": "computation delays",
    "data_inputs_delay": "communication delays",
    # Crazyflie
    "mocap": "mocap",
    "mocap_recon": "mocap (recon)",
    # "mocap_recon": "delays+dynamics",
    "mocap_nodelay": "dynamics",
    "cf_delay": "delay sim.",
    "cf_nodelay": "no-delay sim.",
    # RL policies
    "delay_estimator": "estimator + delay sim.",
    "nodelay_fullstate": "full state",
    "delay_stacked": "stack + delay sim.",
    "nodelay_stacked": "stack",
    # Setup
    "predictive": "pred.",
    "filtered": "filt.",
    "encoder": "enc.",
    "cam": "cam",
    # Real world evaluation
    "delay_estimator2estimator_cam_pred": "estim. (delay)",
    "nodelay_fullstate2estimator_cam_pred": "state",
    "nodelay_fullstate2estimator_nocam_nopred": "state (no cam)",
    "nodelay_fullstate2estimator_cam_nopred": "state (no pred)",
    "delay_stacked2stacked_cam": "stack (delay)",
    "nodelay_stacked2stacked_cam": "stack",
    "nodelay_stacked2stacked_nocam": "stack (no cam)",
    # Components
    "sensor": "encoder",
    "actuator": "actuator",
    "camera": "camera",
    "estimator": "estimator",
    "est_future": "predictive",
    "est_meas": "filtered (corrected)",
    "est_nopred": "filtered",
    "controller": "controller",
    "world": "brax",
    # SYSID_PARAMS
    "delay": "delays",
    "delay_dynamics": "delays+dynamics",
    # TOPOLOGY
    "tree": "tree",
    "chain": "chain",
    "sparse": "sparse",
    # MASK
    "leaf": "leafs",
    "all": "all",
}
CSCHEME = {
    # Computation
    1: "red",
    2: "red",
    4: "pink",
    8: "grape",
    16: "violet",
    32: "indigo",
    64: "blue",
    128: "cyan",
    256: "cyan",
    512: "teal",
    1024: "green",
    2048: "lime",
    4096: "yellow",
    8192: "orange",
    16384: "orange",
    # Systems
    "pendulum": "indigo",
    "quadrotor": "grape",
    # Delay
    "step_delay": "blue",
    "inputs_delay": "cyan",
    "component_delay": "orange",
    "gmm_delay": "gray",
    # Crazyflie
    "mocap": "red",
    "mocap_recon": "indigo",
    "mocap_nodelay": "teal",
    "cf_delay": "grape",
    "cf_nodelay": "orange",
    # RL policies
    "delay_estimator": "blue",
    "nodelay_fullstate": "teal",
    "delay_stacked": "grape",
    "nodelay_stacked": "orange",
    # Real world evaluation
    "delay_estimator2estimator_cam_pred": "blue",
    "nodelay_fullstate2estimator_cam_pred": "green",
    "nodelay_fullstate2estimator_nocam_nopred": "red",
    "nodelay_fullstate2estimator_cam_nopred": "violet",
    "delay_stacked2stacked_cam": "grape",
    "nodelay_stacked2stacked_cam": "orange",
    "nodelay_stacked2stacked_nocam": "indigo",
    # Nodes
    "sensor": "red",
    "actuator": "pink",
    "camera": "violet",
    "estimator": "violet",
    "est_future": "green",
    "est_meas": "violet",
    "est_nopred": "yellow",
    "controller": "indigo",
    "world": "blue",
    # line
    "zoomed_frame": "gray",
    # SYSID_PARAMS
    "delay": "indigo",
    "delay_dynamics": "red",
    # TOPOLOGY
    # "tree": "tree",
    # "chain": "chain",
    # "sparse": "sparse",
    # MASK
    "leaf": "pink",
    "all": "violet",
}
CSCHEME.update({LABELS[k]: v for k, v in CSCHEME.items() if k in LABELS.keys()})
ECOLOR, FCOLOR = oc.cscheme_fn(CSCHEME)


@jax.jit
def wrap_unwrap(x):
    _wrap_unwrap = lambda o: jnp.unwrap((x + onp.pi + o) % (2 * onp.pi) - onp.pi, discont=onp.pi) - o

    x_map = jax.vmap(_wrap_unwrap)(jnp.array([0.1, 0.0, -0.1]))
    # take i where the first x_map[i,0] is closest to onp.pi
    i = jnp.argmin(jnp.abs(x_map[:, 0] - onp.pi))
    return x_map[i]


def create_dataframe(**kwargs):
    length = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 for v in kwargs.values())

    data = {}
    for k, v in kwargs.items():
        if hasattr(v, '__len__') and not isinstance(v, str):
            if len(v) != length:
                raise ValueError(f"All arrays must have the same length, but '{k}' does not.")
            data[k] = v
        else:
            data[k] = [v] * length

    return pd.DataFrame(data)


def export_legend(fig, legend, expand=None):
    expand = [-5, -5, 5, 5] if expand is None else expand
    # fig = legend.figure
    # fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    return bbox
    # fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def index_experiment_files(exp_dir, debug: bool = False):
    configs = []
    data = {}
    for file in os.listdir(exp_dir):
        if file.startswith("run_") and file.endswith(".pkl"):
            file_path = os.path.join(exp_dir, file)
            with open(file_path, "rb") as f:
                content = pickle.load(f)
                config = content.get("config", {})
                config["file_name"] = file
                data[file] = content
                configs.append(config)
        if debug:
            break
    return pd.DataFrame(configs), data


def load_experiment_files(df, exp_dir):
    file_paths = [os.path.join(exp_dir, file) for file in df["file_name"]]
    experiments = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            experiments.append(pickle.load(f))
    df["experiment"] = experiments
    return df


def plot_crazyflie_pathfollowing(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")
    TIMINGS_FILE = os.path.join(exp_dir, "delay_elapsed.pkl")
    # Original
    # SUFFIX_EXP = ""
    # DELAY_R1_FILE = os.path.join(exp_dir, "delay_1.0R/sysid_data.pkl")
    # DELAY_R075_FILE = os.path.join(exp_dir, "delay_0.75R/sysid_data.pkl")
    # DELAY_R05_FILE = os.path.join(exp_dir, "delay_0.5R/sysid_data.pkl")
    # NODELAY_R1_FILE = os.path.join(exp_dir, "nodelay_1.0R/sysid_data.pkl")
    # NODELAY_R075_FILE = os.path.join(exp_dir, "nodelay_0.75R/sysid_data.pkl")
    # NODELAY_R05_FILE = os.path.join(exp_dir, "nodelay_crash_0.5R/sysid_data.pkl")
    # Light - Rebuttal
    # SUFFIX_EXP = "_light"
    # DELAY_R1_FILE = os.path.join(exp_dir, "delay_1.0R_125_Test01/sysid_data.pkl")
    # DELAY_R075_FILE = os.path.join(exp_dir, "delay_0.75R_125_Test02/sysid_data.pkl")
    # DELAY_R05_FILE = os.path.join(exp_dir, "delay_0.5R_125_Test03/sysid_data.pkl")
    # NODELAY_R1_FILE = os.path.join(exp_dir, "nodelay_1.0R_125_Test04/sysid_data.pkl")
    # NODELAY_R075_FILE = os.path.join(exp_dir, "nodelay_0.75R_125_Test05/sysid_data.pkl")
    # NODELAY_R05_FILE = os.path.join(exp_dir, "nodelay_0.5R_125_Test06/sysid_data.pkl")
    # Dark - Rebuttal
    SUFFIX_EXP = "_dark"
    DELAY_R1_FILE = os.path.join(exp_dir, "delay_1.0R_124_Test04/sysid_data.pkl")
    DELAY_R075_FILE = os.path.join(exp_dir, "delay_0.75R_123_Test01/sysid_data.pkl")
    DELAY_R05_FILE = os.path.join(exp_dir, "delay_0.5R_123_Test02/sysid_data.pkl")
    NODELAY_R1_FILE = os.path.join(exp_dir, "nodelay_1.0R_124_Test03/sysid_data.pkl")
    NODELAY_R075_FILE = os.path.join(exp_dir, "nodelay_0.75R_124_Test02/sysid_data.pkl")
    NODELAY_R05_FILE = os.path.join(exp_dir, "nodelay_0.5R_123_Test05/sysid_data.pkl")
    # Simulation
    NODELAY_SIM_FILE = os.path.join(exp_dir, "nodelay_sim_radii_rollout.pkl")
    DELAY_SIM_FILE = os.path.join(exp_dir, "delay_sim_radii_rollout.pkl")
    CACHE_FILE = f"{cache_dir}/plot_crazyflie_pathfollowing.pkl"
    EPS_IDX = -1
    SKIP_SEC = 3.0
    FILES = {
        "delay_1.0R": DELAY_R1_FILE,
        "delay_0.75R": DELAY_R075_FILE,
        "delay_0.5R": DELAY_R05_FILE,
        "nodelay_1.0R": NODELAY_R1_FILE,
        "nodelay_0.75R": NODELAY_R075_FILE,
        "nodelay_0.5R": NODELAY_R05_FILE,
        "nodelay_sim": NODELAY_SIM_FILE,
        "delay_sim": DELAY_SIM_FILE,
    }

    if regenerate_cache or not os.path.exists(CACHE_FILE):
        from envs.crazyflie.ode import MoCapOutput, metrics
        jv_in_agent_frame = jax.jit(jax.vmap(MoCapOutput.static_in_agent_frame, in_axes=(0, None)))
        jv_metrics = jax.jit(jax.vmap(metrics, in_axes=(0, None, None)))
        EVAL_SIM = {}
        EVAL_REAL = {}
        for name, file in FILES.items():
            delay = False if "nodelay" in name else True
            delay_str = "delay" if delay else "nodelay"
            with open(file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, base.EpisodeRecord):
                    # real-world
                    radius = float(data.nodes["agent"].params.fixed_radius[0])
                    center = data.nodes["agent"].params.center[0]
                    mocap = data.nodes["mocap"].steps.output[0]
                    vel_on, vel_off, pos_on, pos_off = jv_metrics(mocap, radius, center)
                    entry = {
                        "ts": data.nodes["mocap"].steps.output.ts[0],
                        "radius": radius,
                        "center": onp.array(center),
                        "mocap": mocap,
                        "mocap_ia": jv_in_agent_frame(mocap, center),
                        "delay": delay,
                        "vel_on": vel_on,
                        "vel_off": vel_off,
                        "pos_on": pos_on,
                        "pos_off": pos_off,
                    }
                    EVAL_REAL[radius] = EVAL_REAL.get(radius, [])
                    EVAL_REAL[radius].append(entry)
                    print(f"real | {delay_str} | R={radius:.2f} | Loaded {name} from {file}.")
                elif isinstance(data, base.GraphState):
                    # simulation
                    num_radius = data.params["agent"][:, 0].fixed_radius.shape[0]
                    for i in range(num_radius):
                        radius = float(data.params["agent"].fixed_radius[i, 0])
                        center = data.params["agent"].center[i, 0]
                        mocap = data.inputs["estimator"]["mocap"].data[i, :, -1]
                        vel_on, vel_off, pos_on, pos_off = jv_metrics(mocap, radius, center)
                        entry = {
                            "ts": mocap.ts,
                            "radius": radius,
                            "center": center,
                            "mocap": mocap,
                            "mocap_ia": jv_in_agent_frame(mocap, center),
                            "delay": delay,
                            "vel_on": vel_on,
                            "vel_off": vel_off,
                            "pos_on": pos_on,
                            "pos_off": pos_off,
                        }
                        EVAL_SIM[radius] = EVAL_SIM.get(radius, [])
                        EVAL_SIM[radius].append(entry)
                        print(f"real | {delay_str} | R={radius:.2f} | Loaded {name} from {file}.")
                else:
                    raise ValueError(f"Unknown data type {type(data)}")

        CACHE_DATA = {"real": EVAL_REAL, "sim": EVAL_SIM}
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(CACHE_DATA, f)
            print(f"Saved cache to {CACHE_FILE}")
    else:
        print(f"Loading cache from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            CACHE_DATA = pickle.load(f)
        EVAL_REAL = CACHE_DATA["real"]
        EVAL_SIM = CACHE_DATA["sim"]

    # Get timings
    with open(TIMINGS_FILE, "rb") as f:
        timings = pickle.load(f)
    print(f"Quadrotor | run time: {timings['train']:.2f} s | JIT time: {timings['train_jit']:.2f} s")

    # Plot
    figures = {}
    figsize = [c * s for c, s in zip([1 / 5, 0.5], rescaled_figsize)]

    for key, data in CACHE_DATA.items():
        for radius, entries in data.items():
            figname = f"{key}_xy_R{radius}"
            fig, ax = plt.subplots(figsize=figsize)
            figures[figname] = (fig, ax)
            for entry in entries:
                delay_str = "cf_delay" if entry["delay"] else "cf_nodelay"
                # Where either pos[:, 0] or pos[:, 1] is > abs 2.0, replace subsequent values with NaN
                pos = entry["mocap"].pos
                # Clip to truncate crashed cases
                bounds_x = onp.abs(pos[:, 0]) < 2.0
                bounds_y = onp.abs(pos[:, 1]) < 2.0
                bounds = onp.logical_and(bounds_x, bounds_y)
                max_idx = onp.argwhere(~bounds).flatten()
                max_idx = max_idx[0] if len(max_idx) > 0 else len(bounds)
                pos_clip = pos[:max_idx]
                # Determine min_index
                min_index = min(np.argmax(entry["ts"] > SKIP_SEC), max_idx)
                vel_on = float(entry["vel_on"][min_index:max_idx].mean())
                vel_off = float(entry["vel_off"][min_index:max_idx].mean())
                pos_on = float(entry["pos_on"][min_index:max_idx].mean())
                pos_off = float(entry["pos_off"][min_index:max_idx].mean())
                print(f"{key} | R={radius:.2f} m | {delay_str} | vel_on={vel_on:.2f} m/s, pos_off={pos_off:.2f} m")
                # Plot
                ax.plot(pos_clip[:, 0], pos_clip[:, 1], label=LABELS[delay_str], color=ECOLOR[delay_str], linewidth=3)
            ax.set(xlabel="x (m)",
                   ylabel="y (m)")
            # ax.set_aspect("equal")
            ax.set(
                xlabel="x (m)",
                ylabel="y (m)",
                xlim=(-1.5, 1.5),
                ylim=(-1.5, 1.5),
                xticks=[-1, 0, 1],
                yticks=[-1, 0, 1],
            )
            ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                      bbox_to_anchor=(2, 2),
                      ncol=2, loc='lower right', fancybox=True, shadow=True)

    ###############################################################
    # SAVING
    ###############################################################
    for plot_name, (fig, ax) in figures.items():
        fig.savefig(f"{fig_dir}/cf_pf_{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        ax.set_aspect("equal")
        fig.savefig(f"{fig_dir}/cf_pf_{plot_name}{SUFFIX_EXP}.pdf", bbox_inches='tight')
        print(f"Saved {plot_name} to {fig_dir}/cf_pf_{plot_name}{SUFFIX_EXP}.pdf")


def plot_crazyflie_sysid(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")
    DATA_SYSID_FILE = os.path.join(exp_dir, "sysid_data.pkl")
    EVAL_SYSID_FILE = os.path.join(exp_dir, "sysid_gs.pkl")
    TIMINGS_SYSID_FILE = os.path.join(exp_dir, "sysid_elapsed.pkl")
    SOL_SYSID_FILE = os.path.join(exp_dir, "sysid_log_state.pkl")
    EVAL_NODELAY_SYSID_FILE = os.path.join(exp_dir, "sysid_gs_nodelay.pkl")
    CACHE_FILE = f"{cache_dir}/plot_crazyflie_sysid.pkl"
    EPS_IDX = -1

    if regenerate_cache or not os.path.exists(CACHE_FILE):
        # Load TIMINGS and SOL
        with open(TIMINGS_SYSID_FILE, "rb") as f:
            timings = pickle.load(f)
        with open(SOL_SYSID_FILE, "rb") as f:
            sol = pickle.load(f)
        log_gen_1_nonan = onp.nan_to_num(sol.state["log_gen_1"], copy=True, nan=onp.inf)
        idx_min = onp.argmin(log_gen_1_nonan)
        tsolve = timings["solve"]
        tjit = timings["solve_jit"]
        tbest = tsolve*(idx_min+1) / len(sol.state["log_gen_1"])
        num_params = len(sol.state["top_params"][-1])
        print(f"Crazyflie | Best solution at generation {idx_min+1} | Solve time: {tsolve:.2f} s | JIT time: {tjit:.2f} s | Best time: {tbest:.2f} s | num_params: {num_params}")

        fig, ax = plt.subplots(figsize=thirdwidth_figsize)
        ax.plot(sol.state["log_gen_1"])
        fig, ax = sol.plot("sol_state")
        ax.set(yscale="log")

        # Load recorded real-world data
        with open(DATA_SYSID_FILE, "rb") as f:  # Load record
            data: base.EpisodeRecord = pickle.load(f)
        outputs = {name: n.steps.output for name, n in data.nodes.items()}
        outputs = jax.tree_util.tree_map(lambda x: x[EPS_IDX], outputs)
        ts = {name: n.steps.ts_start for name, n in data.nodes.items()}
        ts = jax.tree_util.tree_map(lambda x: x[EPS_IDX], ts)

        # Load evaluation data
        with open(EVAL_SYSID_FILE, "rb") as f:  # Load record
            gs: base.GraphState = pickle.load(f)
        with open(EVAL_NODELAY_SYSID_FILE, "rb") as f:  # Load record
            gs_nodelay: base.GraphState = pickle.load(f)
        params = dict(jax.tree_util.tree_map(lambda x: x[0], gs.params))
        delay_pid_to_world = params["pid"].actuator_delay.mean()
        delay_world_to_pid = params["pid"].sensor_delay.mean()
        delay_world_to_mocap = params["mocap"].sensor_delay.mean()
        print(f"Delays | PID to world: {delay_pid_to_world*1000:.0f} ms, "
              f"world to PID: {delay_world_to_pid*1000:.0f} ms, "
              f"world to mocap: {delay_world_to_mocap*1000:.0f} ms")

        from envs.crazyflie.ode import in_body_frame
        in_body_frame_jv = jax.jit(jax.vmap(in_body_frame))

        # Extract recorded sensor data (Assumed to be ground truth and not delayed)
        vel_ib = in_body_frame_jv(outputs["mocap"].att, outputs["mocap"].vel)
        mocap = dict(ts=ts["mocap"],
                     vx=vel_ib[:, 0],
                     vy=vel_ib[:, 1],
                     phi=outputs["mocap"].att[:, 0],
                     theta=outputs["mocap"].att[:, 1],
                     )

        # Extract reconstructed mocap data
        # mocap_recon_ts = gs.inputs["mocap"]["world"][:, -1].ts_recv
        # mocap_recon = gs.inputs["mocap"]["world"][:, -1].data
        mocap_recon = gs.inputs["estimator"]["mocap"][:, -1].data
        vel_ib_recon = in_body_frame_jv(mocap_recon.att, mocap_recon.vel)
        mocap_recon = dict(ts=mocap_recon.ts + delay_world_to_mocap,
        # mocap_recon = dict(ts=mocap_recon_ts,
                           vx=vel_ib_recon[:, 0],
                           vy=vel_ib_recon[:, 1],
                           phi=mocap_recon.att[:, 0],
                           theta=mocap_recon.att[:, 1],
                           )
        # Extract reconstructed mocap data
        mocap_nodelay = gs_nodelay.inputs["estimator"]["mocap"][:, -1].data
        vel_ib_nodelay = in_body_frame_jv(mocap_nodelay.att, mocap_nodelay.vel)
        mocap_nodelay = dict(ts=mocap_nodelay.ts,
                             vx=vel_ib_nodelay[:, 0],
                             vy=vel_ib_nodelay[:, 1],
                             phi=mocap_nodelay.att[:, 0],
                             theta=mocap_nodelay.att[:, 1],
                             )

        # Group measurement related data (same keys: ts, vel, att)
        recon = dict(mocap=mocap, mocap_recon=mocap_recon, mocap_nodelay=mocap_nodelay)

        # include = ["mocap", "mocap_recon", "mocap_nodelay"]
        include = ["mocap", "mocap_recon"]#, "mocap_nodelay"]

        def plot_graphs(ax, keys, interp, y_key, xlabel, ylabel):
            for key, value in interp.items():
                if key not in keys:
                    continue
                ax.plot(value["ts"], value[y_key], label=LABELS[key], color=ECOLOR[key], linewidth=3)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            # Place the legend on the right outside of the figure
            ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                      bbox_to_anchor=(2, 2),
                      ncol=6, loc='lower right', fancybox=True, shadow=True)

        def plot_zoomed_frame(ax, x_zoom, y_zoom, w, h, w_scale, h_scale):
            rect = patches.Rectangle(
                (x_zoom + w / 2 - w_scale * w / 2, y_zoom + h / 2 - h_scale * h / 2),
                w * w_scale, h * h_scale, linewidth=4,
                edgecolor=ECOLOR["zoomed_frame"], facecolor='none'
            )
            ax.add_patch(rect)

        figures = {}
        sysid_figsize = [c * s for c, s in zip([1/3, 0.5], rescaled_figsize)]
        # sysid_figsize = [c * s for c, s in zip([0.5, 0.5], rescaled_figsize)]
        # Plot attitude
        ylabels =[r"$\mathbf{\phi}$ (rad)", r"$\mathbf{\theta}$ (rad)"]
        for k, ylabel in zip(["phi", "theta"], ylabels):
            fig, ax = plt.subplots(figsize=sysid_figsize)  # Used to be twothirdwidth_figsize
            figures[k] = (fig, ax)
            plot_graphs(ax, include, recon, k, "Time (s)", ylabel)
            if k == "theta":
                plot_zoomed_frame(ax, 5.2, 0.15, 0.2, 0.1, 3.0, 1.5)
                # plot_zoomed_frame(ax, 5.0, 0.1, 2.0, 0.35, 1.0, 1.0)
            ax.set(
                ylim=[-0.5, 0.5],
                yticks=[-0.5, 0, 0.5],
            )

        fig, ax = plt.subplots(figsize=sysid_figsize)
        figures["theta_zoom"] = (fig, ax)
        plot_graphs(ax, include, recon, "theta", "Time (s)", ylabels[1])
        ax.set(
            xlim=[5.2, 5.2 + 0.2],
            xticks=[5.2, 5.3, 5.4],
            ylim=[0.15, 0.15+0.1],
            yticks=[0.16, 0.2, 0.24],
        )
        # ax.set(
        #     xlim=[5.0, 5.0 + 2],
        #     xticks=[5, 6, 7],
        #     ylim=[0.1, 0.1+0.35],
        #     yticks=[0.1, 0.2, 0.3, 0.4],
        # )

        # Plot Velocity in body frame
        ylabels = [r"$v_x$ (m/s)", r"$v_y$ (m/s)", r"$v_z$ (m/s)"]
        for k, ylabel in zip(["vx", "vy"], ylabels):
            fig, ax = plt.subplots(figsize=sysid_figsize)  # Used to be twothirdwidth_figsize
            figures[k] = (fig, ax)
            plot_graphs(ax, include, recon, k, "Time (s)", ylabel)
            if k == "vy":
                plot_zoomed_frame(ax, 6.0, -2.0, 0.2, 0.2, 4.0, 2.0)
                # plot_zoomed_frame(ax, 5.0, -2.0, 2.0, 2.0, 1.0, 1.0)
            ax.set(
                ylim=[-3, 3],
                yticks=[-2, 0, 2],
            )

        fig, ax = plt.subplots(figsize=sysid_figsize)
        figures["vy_zoom"] = (fig, ax)
        plot_graphs(ax, include, recon, "vy", "Time (s)", ylabels[1])
        ax.set(
            xlim=[6.1, 6.1 + 0.2],
            xticks=[6.1, 6.2, 6.3],
            ylim=[-1.925, -1.825],
            yticks=[-1.9, -1.85],
        )
        # ax.set(
        #     xlim=[5.0, 5.0 + 2],
        #     xticks=[5, 6, 7],
        #     ylim=[-2, 0],
        #     yticks=[-2, -1, 0],
        # )

        # ax.legend()

        ###############################################################
        # SAVING
        ###############################################################
        for plot_name, (fig, ax) in figures.items():
            fig.savefig(f"{fig_dir}/cf_sysid_{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
            ax.get_legend().remove()
            fig.savefig(f"{fig_dir}/cf_sysid_{plot_name}.pdf", bbox_inches='tight')
            print(f"Saved {plot_name} to {fig_dir}/cf_sysid_{plot_name}.pdf")
        return


def plot_crazyflie_dists(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")
    DATA_DELAY_FILE = os.path.join(exp_dir, "sysid_data.pkl")
    CACHE_FILE = f"{cache_dir}/plot_crazyflie_dists.pkl"  # todo: Nothing cached at the moment...

    if regenerate_cache or not os.path.exists(CACHE_FILE):
        # Load recorded real-world data
        with open(DATA_DELAY_FILE, "rb") as f:  # Load record
            record: base.EpisodeRecord = pickle.load(f)

        n = next(iter(record.nodes.values()))
        if n.steps.seq.ndim == 2:
            record = jax.tree_util.tree_map(lambda x: x.flatten(), record)
            # Filter -1 from record
            record = jax.tree_util.tree_map(lambda x: x[x != -1], record)

        # Get data
        data, info = dict(step={}, inputs={}), dict(step={}, inputs={})
        for name, n in record.nodes.items():
            data["step"][name] = n.steps.delay
            info["step"][name] = n.info
            data["inputs"][name] = {}
            info["inputs"][name] = {}
            for input_name, i in n.inputs.items():
                data["inputs"][name][input_name] = i.messages.delay
                info["inputs"][name][input_name] = (n.info.inputs[input_name], n.info)

        def init_estimator(x, i):
            name = i.name if not isinstance(i, tuple) else f"{i[0].output}->{i[1].name}"
            est = GMMEstimator(x, name)
            return est

        # Initialize estimators
        est = jax.tree_util.tree_map(lambda x, i: init_estimator(x, i), data, info)

        # Fit estimators
        jax.tree_util.tree_map(lambda e: e.fit(num_steps=300, num_components=2, step_size=0.05, seed=0), est)

        # Get distributions
        dist = jax.tree_util.tree_map(lambda e: e.get_dist(), est)

        # Get final_state
        w_ms = jax.tree_util.tree_map(lambda e: e._rescale(e.adam_get_params(e.final_state_norm)), est)

        CACHE_DATA = dict(data=data, info=info, w_ms=w_ms, dist=dist)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(CACHE_DATA, f)
            print(f"Saved cache to {CACHE_FILE}")
    else:
        with open(CACHE_FILE, "rb") as f:
            CACHE_DATA = pickle.load(f)
        print(f"Loading cache from {CACHE_FILE}")
        data = CACHE_DATA["data"]
        info = CACHE_DATA["info"]
        w_ms = CACHE_DATA["w_ms"]
        dist = CACHE_DATA["dist"]

    # Calculate statistics
    print("QUADROTOR DELAY STATISTICS")
    mean_data = jax.tree_util.tree_map(lambda x: f"mean: {x.mean()*1000: .1f} ms", data)
    std_data = jax.tree_util.tree_map(lambda x: f"std: {x.std()*1000: .1f} ms", data)
    max_data = jax.tree_util.tree_map(lambda x: f"max: {x.max()*1000: .1f} ms", data)
    min_data = jax.tree_util.tree_map(lambda x: f"min: {x.min()*1000: .1f} ms", data)
    stats_data = jax.tree_util.tree_map(lambda *x: ", ".join(x), mean_data, std_data, max_data, min_data)
    eqx.tree_pprint(stats_data["step"])
    # eqx.tree_pprint(stats_data["inputs"])

    # Start plotting distributions
    figures = {}
    # twothirdwidth_figsize = [c * s for c, s in zip([2 / 3, 0.5], rescaled_figsize)]
    # [c * s for c, s in zip([1, 0.52], rescaled_figsize)]
    figsize = [c * s for c, s in zip([1 / 3, 0.5], rescaled_figsize)]  # Used to be twothirdwidth_figsize

    def cycling_kwargs_generator():
        cindex = 1
        while True:
            kwargs_comp = {"color": ECOLOR["component_delay"], "linestyle": "-", "linewidth": 3}
            if cindex == 1:
                # kwargs_comp["label"] = f"{LABELS['component_delay']} {cindex}"
                kwargs_comp["label"] = f"{LABELS['component_delay']}"
            else:
                kwargs_comp["label"] = None
            yield kwargs_comp
            cindex += 1

    def plot_dist(_ax, dtype, _dist, _w_ms, _delays):
        percentile = 99.9
        xscale = 1000.0  # m to ms
        xmin = 0.0
        # xmax = _dist.quantile(percentile / 100)  # if delays is not None else dist.quantile(0.99)*1.05
        xmax = _delays.max()#_dist.quantile(percentile / 100)  # if delays is not None else dist.quantile(0.99)*1.05
        xmax *= xscale

        # Rescale and only keep below xmax
        _delays = xscale*_delays
        _delays = _delays[_delays <= xmax]  # Filter out negative delays

        # Plot histogram
        _ax.hist(_delays, bins=30, density=True, label=LABELS[f"data_{dtype}_delay"], edgecolor=ECOLOR[f"{dtype}_delay"], facecolor=FCOLOR[f"{dtype}_delay"], alpha=0.3)

        # Plot pdfs
        kwargs_gmm = {"color": ECOLOR["gmm_delay"], "linestyle": "-", "linewidth": 3}
        w, _, m, s = _w_ms
        m *= xscale
        s += onp.log(xscale)
        plot_component_norm_pdfs(w, m, s, xmin=xmin, xmax=xmax*2, ax=_ax, num_points=300,
                                 kwargs_comp=cycling_kwargs_generator, plot_comp=False,
                                 kwargs_gmm=kwargs_gmm)
        ylim = list(_ax.get_ylim())  # Get current y-limits
        ylim[0] = 1e-2
        _ax.set(xlabel="delay (ms)",
               ylabel="density",
               yscale="log",
               ylim=ylim,
               # xlim=[xmin, round(xmax, 0)],
               xlim=[xmin, xmax],
               xticks=[0, round(xmax / 2, 0), round(xmax, 0)],
               )

        # Create a rectangular patch for "communication delay"
        other_dtype = "inputs" if dtype == "step" else "step"
        other_patch = patches.Patch(label=LABELS[f"data_{other_dtype}_delay"], edgecolor=ECOLOR[f"{other_dtype}_delay"], facecolor=FCOLOR[f"{other_dtype}_delay"], alpha=0.3)

        # Add the patch to the legend
        handles, labels = ax.get_legend_handles_labels()
        handles = [other_patch] + handles
        labels = [LABELS[f"data_{other_dtype}_delay"]] + labels

        _ax.legend(handles=handles, labels=labels,
                   bbox_to_anchor=(2, 2),
                   ncol=6, loc='lower right', fancybox=True, shadow=True)

    for name, d in dist["step"].items():
        dtype = "step"
        figname = f"dist_{dtype}_{name}"
        fig, ax = plt.subplots(figsize=figsize)
        figures[figname] = (fig, ax)
        plot_dist(ax, dtype, d, w_ms[dtype][name], data[dtype][name])

    for name_in, dd in dist["inputs"].items():
        for name_out, d in dd.items():
            dtype = "inputs"
            figname = f"dist_{dtype}_{name_out}->{name_in}"
            fig, ax = plt.subplots(figsize=figsize)
            figures[figname] = (fig, ax)
            plot_dist(ax, dtype, d, w_ms[dtype][name_in][name_out], data[dtype][name_in][name_out])

    ###############################################################
    # SAVING
    ###############################################################
    for plot_name, (fig, ax) in figures.items():
        fig.savefig(f"{fig_dir}/cf_{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{fig_dir}/cf_{plot_name}.pdf", bbox_inches='tight')
        print(f"Saved {plot_name} to {fig_dir}/cf_{plot_name}.pdf")

        # Save excalidraw version
        for line in ax.lines:
            line.set_linewidth(4.5)
        for bar in ax.patches:
            bar.set_linewidth(0)  # Adjust the linewidth as needed
            bar.set_edgecolor(FCOLOR["gmm"])  # Optional: set edge color to make lines visible
            bar.set_facecolor(FCOLOR["gmm"])  # Optional: set edge color to make lines visible
            bar.set_alpha(1.0)
        # Adjust x-ticks position
        ax.tick_params(axis='x', pad=4, labelsize=35)  # Move x-ticks down (adjust `pad` value as needed)

        # Adjust figure size to ensure width is 1.3 times the height
        fig_width, fig_height = fig.get_size_inches()
        fig.set_size_inches(fig_width, fig_width / 1.3)
        # ax.set_aspect(1.35)  # Set aspect ratio to 1.3 (width = 1.3 * height)

        # Remove top and right spines (part of the frame)
        # ax.set_frame_on(True)  # Hide the axes frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Customize bottom and left spines (x and y axes)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(5)  # Adjust thickness as needed
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(5)  # Adjust thickness as needed

        # Increase x-label font size
        ax.set_xlabel("delay (ms)", fontsize=40)  # Set x-label text and increase font size (adjust `fontsize` as needed)

        # Increase gridline thickness
        ax.grid(True, linewidth=4)  # Set to desired thickness
        ax.set(# xlabel="",
               ylabel="",
               # xticks=[],
               # yticks=[]
               )  # Remove axis labels
        if "->" in plot_name:
            ax.set_xlabel("")  # Remove x-axis labels
        # Remove y-axis labels but keep ticks
        ax.set_yticklabels([])  # Remove y-axis labels
        # Save the line plot only
        fig.savefig(f"{fig_dir}/cf_{plot_name}_min.png", transparent=True, bbox_inches='tight')


def plot_pendulum_dists(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")
    DATA_DELAY_FILE = os.path.join(exp_dir, "real_data.pkl")
    CACHE_FILE = f"{cache_dir}/plot_pendulum_dists.pkl"  # todo: Nothing cached at the moment...

    if regenerate_cache or not os.path.exists(CACHE_FILE):
        # Load recorded real-world data
        with open(DATA_DELAY_FILE, "rb") as f:  # Load record
            record: base.ExperimentRecord = pickle.load(f)
        record = record.stack()

        n = next(iter(record.nodes.values()))
        if n.steps.seq.ndim == 2:
            record = jax.tree_util.tree_map(lambda x: x.flatten(), record)
            # Filter -1 from record
            record = jax.tree_util.tree_map(lambda x: x[x != -1], record)

        # Get data
        data, info = dict(step={}, inputs={}), dict(step={}, inputs={})
        for name, n in record.nodes.items():
            data["step"][name] = n.steps.delay
            info["step"][name] = n.info
            data["inputs"][name] = {}
            info["inputs"][name] = {}
            for input_name, i in n.inputs.items():
                data["inputs"][name][input_name] = i.messages.delay
                info["inputs"][name][input_name] = (n.info.inputs[input_name], n.info)

        def init_estimator(x, i):
            name = i.name if not isinstance(i, tuple) else f"{i[0].output}->{i[1].name}"
            est = GMMEstimator(x, name)
            return est

        # Initialize estimators
        est = jax.tree_util.tree_map(lambda x, i: init_estimator(x, i), data, info)

        # Fit estimators
        jax.tree_util.tree_map(lambda e: e.fit(num_steps=300, num_components=4, step_size=0.05, seed=0), est)

        # Get distributions
        dist = jax.tree_util.tree_map(lambda e: e.get_dist(), est)

        # Get final_state
        w_ms = jax.tree_util.tree_map(lambda e: e._rescale(e.adam_get_params(e.final_state_norm)), est)

        CACHE_DATA = dict(data=data, info=info, w_ms=w_ms, dist=dist)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(CACHE_DATA, f)
            print(f"Saved cache to {CACHE_FILE}")
    else:
        with open(CACHE_FILE, "rb") as f:
            CACHE_DATA = pickle.load(f)
        print(f"Loading cache from {CACHE_FILE}")
        data = CACHE_DATA["data"]
        info = CACHE_DATA["info"]
        w_ms = CACHE_DATA["w_ms"]
        dist = CACHE_DATA["dist"]

    # Calculate statistics
    print("PENDULUM DELAY STATISTICS")
    mean_data = jax.tree_util.tree_map(lambda x: f"mean: {x.mean() * 1000: .1f} ms", data)
    std_data = jax.tree_util.tree_map(lambda x: f"std: {x.std() * 1000: .1f} ms", data)
    max_data = jax.tree_util.tree_map(lambda x: f"max: {x.max() * 1000: .1f} ms", data)
    min_data = jax.tree_util.tree_map(lambda x: f"min: {x.min() * 1000: .1f} ms", data)
    stats_data = jax.tree_util.tree_map(lambda *x: ", ".join(x), mean_data, std_data, max_data, min_data)
    eqx.tree_pprint(stats_data["step"])
#     eqx.tree_pprint(stats_data["inputs"])

    # Start plotting distributions
    figures = {}
    # twothirdwidth_figsize = [c * s for c, s in zip([2 / 3, 0.5], rescaled_figsize)]
    # [c * s for c, s in zip([1, 0.52], rescaled_figsize)]
    figsize = [c * s for c, s in zip([1 / 3, 0.5], rescaled_figsize)]  # Used to be twothirdwidth_figsize

    def cycling_kwargs_generator():
        cindex = 1
        while True:
            kwargs_comp = {"color": ECOLOR["component_delay"], "linestyle": "-", "linewidth": 3}
            if cindex == 1:
                # kwargs_comp["label"] = f"{LABELS['component_delay']} {cindex}"
                kwargs_comp["label"] = f"{LABELS['component_delay']}"
            else:
                kwargs_comp["label"] = None
            yield kwargs_comp
            cindex += 1

    def plot_dist(_ax, dtype, _dist, _w_ms, _delays):
        percentile = 99.9
        xscale = 1000.0  # m to ms
        xmin = 0.0
        xmax = _dist.quantile(percentile / 100)  # if delays is not None else dist.quantile(0.99)*1.05
        xmax *= xscale

        # Rescale and only keep below xmax
        _delays = xscale*_delays
        _delays = _delays[_delays <= xmax]  # Filter out negative delays

        # Plot histogram
        _ax.hist(_delays, bins=30, density=True, label=LABELS[f"data_{dtype}_delay"], edgecolor=ECOLOR[f"{dtype}_delay"], facecolor=FCOLOR[f"{dtype}_delay"], alpha=0.3)

        # Plot pdfs
        kwargs_gmm = {"color": ECOLOR["gmm_delay"], "linestyle": "-", "linewidth": 3}
        w, _, m, s = _w_ms
        m *= xscale
        s += onp.log(xscale)
        plot_component_norm_pdfs(w, m, s, xmin=xmin, xmax=xmax*2, ax=_ax, num_points=300, #xscale=xscale,
                                 kwargs_comp=cycling_kwargs_generator, plot_comp=False,
                                 kwargs_gmm=kwargs_gmm)
        ylim = list(_ax.get_ylim())  # Get current y-limits
        ylim[0] = 1e-4
        _ax.set(xlabel="delay (ms)",
               ylabel="density",
               yscale="log",
               ylim=ylim,
               xlim=[xmin, round(xmax, 0)],
               xticks=[0, round(xmax / 2, 0), round(xmax, 0)],
               )

        # Create a rectangular patch for "communication delay"
        other_dtype = "inputs" if dtype == "step" else "step"
        other_patch = patches.Patch(label=LABELS[f"data_{other_dtype}_delay"], edgecolor=ECOLOR[f"{other_dtype}_delay"], facecolor=FCOLOR[f"{other_dtype}_delay"], alpha=0.3)

        # Add the patch to the legend
        handles, labels = ax.get_legend_handles_labels()
        handles = [other_patch] + handles
        labels = [LABELS[f"data_{other_dtype}_delay"]] + labels

        # Plot legend
        _ax.legend(handles=handles, labels=labels,
                   bbox_to_anchor=(2, 2),
                   ncol=6, loc='lower right', fancybox=True, shadow=True)

    for name, d in dist["step"].items():
        dtype = "step"
        figname = f"dist_{dtype}_{name}"
        fig, ax = plt.subplots(figsize=figsize)
        figures[figname] = (fig, ax)
        plot_dist(ax, dtype, d, w_ms[dtype][name], data[dtype][name])

    for name_in, dd in dist["inputs"].items():
        for name_out, d in dd.items():
            dtype = "inputs"
            figname = f"dist_{dtype}_{name_out}->{name_in}"
            fig, ax = plt.subplots(figsize=figsize)
            figures[figname] = (fig, ax)
            plot_dist(ax, dtype, d, w_ms[dtype][name_in][name_out], data[dtype][name_in][name_out])

    ###############################################################
    # SAVING
    ###############################################################
    for plot_name, (fig, ax) in figures.items():
        fig.savefig(f"{fig_dir}/{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{fig_dir}/{plot_name}.pdf", bbox_inches='tight')
        print(f"Saved {plot_name} to {fig_dir}/{plot_name}.pdf")

        # Save excalidraw version
        for line in ax.lines:
            line.set_linewidth(4.5)
        for bar in ax.patches:
            bar.set_linewidth(0)  # Adjust the linewidth as needed
            bar.set_edgecolor(FCOLOR["gmm"])  # Optional: set edge color to make lines visible
            bar.set_facecolor(FCOLOR["gmm"])  # Optional: set edge color to make lines visible
            bar.set_alpha(1.0)
        # Adjust x-ticks position
        ax.tick_params(axis='x', pad=4, labelsize=35)  # Move x-ticks down (adjust `pad` value as needed)

        # Adjust figure size to ensure width is 1.3 times the height
        fig_width, fig_height = fig.get_size_inches()
        fig.set_size_inches(fig_width, fig_width / 1.3)
        # ax.set_aspect(1.35)  # Set aspect ratio to 1.3 (width = 1.3 * height)

        # Remove top and right spines (part of the frame)
        # ax.set_frame_on(True)  # Hide the axes frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Customize bottom and left spines (x and y axes)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(5)  # Adjust thickness as needed
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(5)  # Adjust thickness as needed

        # Increase x-label font size
        ax.set_xlabel("delay (ms)", fontsize=40)  # Set x-label text and increase font size (adjust `fontsize` as needed)

        # Increase gridline thickness
        ax.grid(True, linewidth=4)  # Set to desired thickness
        ax.set(# xlabel="",
               ylabel="",
               # xticks=[],
               # yticks=[]
               )  # Remove axis labels
        if "->" in plot_name:
            ax.set_xlabel("")  # Remove x-axis labels
        # Remove y-axis labels but keep ticks
        ax.set_yticklabels([])  # Remove y-axis labels
        # Save the line plot only
        fig.savefig(f"{fig_dir}/{plot_name}_min.png", transparent=True, bbox_inches='tight')


def plot_pendulum_sysid(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")
    DATA_SYSID_FILE = os.path.join(exp_dir, "sysid_data.pkl")
    TIMINGS_SYSID_FILE = os.path.join(exp_dir, "sysid_elapsed.pkl")
    SOL_SYSID_FILE = os.path.join(exp_dir, "sysid_log_state.pkl")
    # STATE_SYSID_FILE = os.path.join(exp_dir, "sysid_sol_state.pkl")
    EVAL_SYSID_FILE = os.path.join(exp_dir, "sysid_gs_eval.pkl")
    CACHE_FILE = f"{cache_dir}/plot_pendulum_sysid.pkl"  # todo: Nothing cached at the moment...
    EPS_IDX = -1

    if regenerate_cache or not os.path.exists(CACHE_FILE):
        # Load SOL_STATE, SOL, and TIMINGS
        # with open(STATE_SYSID_FILE, "rb") as f:
        #     state = pickle.load(f)
        with open(TIMINGS_SYSID_FILE, "rb") as f:
            timings = pickle.load(f)
        with open(SOL_SYSID_FILE, "rb") as f:
            sol = pickle.load(f)
        log_gen_1_nonan = onp.nan_to_num(sol.state["log_gen_1"], copy=True, nan=onp.inf)
        idx_min = onp.argmin(log_gen_1_nonan)
        tsolve = timings["solve"]
        tjit = timings["solve_jit"]
        tbest = tsolve*(idx_min+1) / len(sol.state["log_gen_1"])
        num_params = len(sol.state["top_params"][-1])
        print(f"Pendulum | Best solution at generation {idx_min+1} | Solve time: {tsolve:.2f} s | JIT time: {tjit:.2f} s | Best time: {tbest:.2f} s | num_params: {num_params}")

        # Load recorded real-world data
        with open(DATA_SYSID_FILE, "rb") as f:  # Load record
            data: base.EpisodeRecord = pickle.load(f)

        # Load outputs
        outputs = {name: n.steps.output for name, n in data.nodes.items()}
        outputs = jax.tree_util.tree_map(lambda x: x[EPS_IDX], outputs)
        ts = {name: n.steps.ts_start for name, n in data.nodes.items()}
        ts = jax.tree_util.tree_map(lambda x: x[EPS_IDX], ts)

        # Load evaluation data
        with open(EVAL_SYSID_FILE, "rb") as f:  # Load record
            gs: base.GraphState = pickle.load(f)
        params = dict(jax.tree_util.tree_map(lambda x: x[0], gs.params))
        delay_act_to_world = params["actuator"].actuator_delay.mean()
        delay_world_to_sensor = params["sensor"].sensor_delay.mean()
        delay_world_to_cam = params["camera"].sensor_delay.mean()
        print(f"Delays | Actuator to world: {delay_act_to_world*1000:.0f} ms, "
              f"world to sensor: {delay_world_to_sensor*1000:.0f} ms, "
              f"world to cam: {delay_world_to_cam*1000:.0f} ms")

        # Get action
        actuator = dict(ts=ts["actuator"],
                        act=outputs["actuator"].action[..., 0])

        # Extract recorded sensor data (Assumed to be ground truth and not delayed)
        sensor = dict(ts=ts["sensor"],
                      th=outputs["sensor"].th,
                      thdot=outputs["sensor"].thdot)

        # Predicted world state
        world = dict(ts=gs.ts["world"],
                     th=gs.state["world"].th,
                     thdot=gs.state["world"].thdot)

        # Predicted camera state
        camera = dict(ts=gs.inputs["estimator"]["camera"].ts_sent[..., 0],
                      th=gs.inputs["estimator"]["camera"].data.th[..., 0],
                      thdot=gs.inputs["estimator"]["camera"].data.thdot[..., 0])

        # Estimated state up to the last measurement
        est_future = dict(ts=gs.inputs["controller"]["estimator"].data.ts[..., -1],
                          th=gs.inputs["controller"]["estimator"].data.mean.th[..., -1],
                          thdot=gs.inputs["controller"]["estimator"].data.mean.thdot[..., -1])

        # Estimated state up to the last measurement (
        est_nopred = dict(ts=gs.ts["estimator"] + params["estimator"].dt_future,
                          th=gs.state["estimator"].prior.mu[:, 0],
                          thdot=gs.state["estimator"].prior.mu[:, 1])

        # Estimated state up to the last measurement (use .data.ts[:, -1] meaning delays are included)
        est_meas = dict(ts=gs.inputs["estimator"]["camera"].data.ts[:, -1],
                        th=est_nopred["th"],
                        thdot=est_nopred["thdot"])

        # Group measurement related data (same keys: ts, th, thdot)
        recon = dict(sensor=sensor, world=world, camera=camera, est_nopred=est_nopred, est_meas=est_meas,
                     est_future=est_future)
        # Wrap
        for key, value in recon.items():
            value["th"] = wrap_unwrap(value["th"])

        # Get ground truth (sensor)
        lowest_max_ts = min([v["ts"][-1] for v in recon.values()])
        gt_ind = sensor["ts"] < lowest_max_ts
        gt = {k: v[gt_ind] for k, v in sensor.items()}
        ts_gt = gt["ts"]
        th_gt = gt["th"]
        thdot_gt = gt["thdot"]

        # Interpolate
        interp = {k: v.copy() for k, v in recon.items()}
        for key, value in interp.items():
            value["th"] = jnp.interp(ts_gt, value["ts"], value["th"])
            value["thdot"] = jnp.interp(ts_gt, value["ts"], value["thdot"])
            value["ts"] = ts_gt

        # Calculate error
        error = {k: v.copy() for k, v in interp.items()}
        for key, value in error.items():
            value["th"] = (value["th"] - th_gt) ** 2
            value["thdot"] = (value["thdot"] - thdot_gt) ** 2
            MSE_th = jnp.mean(value["th"])
            MSE_thdot = jnp.mean(value["thdot"])
            print(f"{LABELS[key]}: MSE_th={MSE_th:.3f}, MSE_thdot={MSE_thdot:.3f}")

        # PLOT ERRORS
        if False:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=twothirdwidth_figsize)
            for key, value in error.items():
                ax.plot(value["ts"], value["thdot"], label=key, color=ECOLOR[key])
            ax.legend()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=twothirdwidth_figsize)
            for key, value in error.items():
                ax.plot(value["ts"], value["th"], label=key, color=ECOLOR[key])
            ax.legend()

        # th_include = ["sensor", "world", "camera", "est_future", "est_nopred", "est_meas"]
        # thdot_include = ["sensor", "world", "est_future", "est_nopred", "est_meas"]
        th_include = ["sensor", "world", "est_future", "est_nopred"]
        thdot_include = ["sensor", "world", "est_future", "est_nopred"]

        def plot_graphs(ax, keys, interp, y_key, xlabel, ylabel):
            for key, value in interp.items():
                if key not in keys:
                    continue
                ax.plot(value["ts"], value[y_key] - onp.pi, label=LABELS[key], color=ECOLOR[key], linewidth=3)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            # Place the legend on the right outside of the figure
            ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                      bbox_to_anchor=(2, 2),
                      ncol=6, loc='lower right', fancybox=True, shadow=True)

        def plot_zoomed_frame(ax, x_zoom, y_zoom, w, h, w_scale, h_scale):
            rect = patches.Rectangle(
                (x_zoom + w / 2 - w_scale * w / 2, y_zoom + h / 2 - h_scale * h / 2),
                w * w_scale, h * h_scale, linewidth=4,
                edgecolor=ECOLOR["zoomed_frame"], facecolor='none'
            )
            ax.add_patch(rect)

        figures = {}
        # twothirdwidth_figsize = [c * s for c, s in zip([2 / 3, 0.5], rescaled_figsize)]
        # [c * s for c, s in zip([1, 0.52], rescaled_figsize)]
        sysid_figsize = [c * s for c, s in zip([3/3, 0.4], rescaled_figsize)]   # Used to be twothirdwidth_figsize
        sysid_zoom_figsize = [c * s for c, s in zip([1/3, 0.4], rescaled_figsize)]  # Used to be onethirdwidth_figsize
        # Plot th
        fig, ax = plt.subplots(figsize=sysid_figsize)  # Used to be twothirdwidth_figsize
        figures["sysid_th"] = (fig, ax)
        plot_graphs(ax, th_include, interp, "th", "Time (s)", r"$\mathbf{\theta}$ (rad)")
        plot_zoomed_frame(ax, 18.8, 7.5, 0.4, 1.5, 1.0, 1.0)
        ax.set(
            ylim=[-7.5, 10],
            yticks=[-7.5, 0, 7.5],
        )

        fig, ax = plt.subplots(figsize=sysid_zoom_figsize) # Used to be onethirdwidth_figsize
        figures["sysid_th_zoom"] = (fig, ax)
        plot_graphs(ax, th_include, interp, "th", "Time (s)", r"$\mathbf{\theta}$ (rad)")
        ax.set(
            xlim=[18.8, 18.8 + 0.4],
            xticks=[19.0, 19.2],
            ylim=[7.5, 9.0],
            yticks=[7.5, 8.0, 8.5, 9.0],
        )
        # ax.legend()

        # Plot thdot
        fig, ax = plt.subplots(figsize=sysid_figsize)
        figures["sysid_thdot"] = (fig, ax)
        plot_graphs(ax, thdot_include, interp, "thdot", "Time (s)", r"$\mathbf{\dot{\theta}}$ (rad/s)")
        plot_zoomed_frame(ax, 18.8, -4, 0.4, 13, 1.0, 1.0)
        ax.set(
            ylim=[-25, 25],
            yticks=[-20, 0, 20],
        )

        fig, ax = plt.subplots(figsize=sysid_zoom_figsize)
        figures["sysid_thdot_zoom"] = (fig, ax)
        plot_graphs(ax, thdot_include, interp, "thdot", "Time (s)", r"$\mathbf{\dot{\theta}}$ (rad/s)")
        ax.set(
            xlim=[18.8, 18.8 + 0.4],
            xticks=[19.0, 19.2],
            ylim=[-4, -4 + 13],
            yticks=[-4, 0, 4, 8],
        )
        # ax.legend()

        ###############################################################
        # SAVING
        ###############################################################
        for plot_name, (fig, ax) in figures.items():
            fig.savefig(f"{fig_dir}/{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
            ax.get_legend().remove()
            fig.savefig(f"{fig_dir}/{plot_name}.pdf", bbox_inches='tight')
            print(f"Saved {plot_name} to {fig_dir}/{plot_name}.pdf")
        return


def plot_pendulum_rl(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")

    SKIP_STEPS = 100
    COS_TH_THRESHOLD = 0.1737  # -0.1737 (~10 deg),  0.2 (strict), 0.9 (not very effective)
    THDOT_THRESHOLD = 0.5  # 0.5 (strict), 2.0 (not strict)
    TIMINGS_FILE = os.path.join(exp_dir, "ctrl_elapsed.pkl")
    EVAL_REAL = {
        "delay_estimator2estimator_cam_pred": dict(policy="delay_estimator", setup="predictive", estimator=True, cam=True, pred=True, eval="real_data.pkl", notes="Policy should have smallest sim2real gap, so should work"),
        "nodelay_fullstate2estimator_cam_pred": dict(policy="nodelay_fullstate", setup="predictive", estimator=True, cam=True, pred=True, eval="nodelay_cam_real_data.pkl", notes="Estimator in real-world compensates for sim2real gap (partial obs., delays), so simple policy works."),
        "nodelay_fullstate2estimator_nocam_nopred": dict(policy="nodelay_fullstate", setup="encoder", estimator=False, cam=False, pred=False, eval="nodelay_nocam_real_data.pkl", notes="No cam, so no delays, so simple policy should work best. Maximum performance expected."),
        "nodelay_fullstate2estimator_cam_nopred": dict(policy="nodelay_fullstate", setup="filtered", estimator=True, cam=True, pred=False, eval="nodelay_cam_nopred_real_data.pkl", notes="Estimator, that only resolve partial obs., but does not predict forward."),
        "delay_stacked2stacked_cam": dict(policy="delay_stacked", setup="cam", estimator=False, cam=True, pred=False, eval="stacked_real_data.pkl", notes="Policy trained with delays and stacking should work OK."),
        "nodelay_stacked2stacked_cam": dict(policy="nodelay_stacked", setup="cam", estimator=False, cam=True, pred=False, eval="stacked_nodelay_real_data.pkl", notes="Policy fails, because no delay compensation"),
        "nodelay_stacked2stacked_nocam": dict(policy="nodelay_stacked", setup="encoder", estimator=False, cam=False, pred=False, eval="stacked_nodelay_nocam_real_data.pkl", notes="Policy should work OK (no delays), as it has learned to infer thdot from stack"),
    }

    EVAL_SIM = {
        "delay_estimator": dict(fullstate=True, estimator=True, delay=True, eval="ctrl_gs_evals.pkl", metrics="ctrl_ppo_metrics.pkl", notes="Delays and estimator in loop. Should be best sim2real but sim problem is harder."),
        "nodelay_fullstate": dict(fullstate=True, estimator=False, delay=False, eval="nodelay_gs_evals.pkl", metrics="nodelay_ppo_metrics.pkl", notes="No delay simulation, but estimator in real-world compensates for delays."),
        "delay_stacked": dict(fullstate=False, estimator=False, delay=True, eval="stacked_gs_evals.pkl", metrics="stacked_ppo_metrics.pkl", notes="No estimator, but delay simulation forces policy to compensate for delays in real-world."),
        "nodelay_stacked": dict(fullstate=False, estimator=False, delay=False, eval="stacked_nodelay_gs_evals.pkl", metrics="stacked_nodelay_ppo_metrics.pkl", notes="No estimator, no delay simulation, so policy only solves partial observability but does not compensate for delays.")
    }

    CACHE_FILE = f"{cache_dir}/plot_pendulum_rl.pkl"
    if regenerate_cache or not os.path.exists(CACHE_FILE):

        def _get_reward_and_success(th, thdot):
            th, thdot = th[SKIP_STEPS:], thdot[SKIP_STEPS:]
            th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
            reward = (th_norm ** 2 + 0.1 * (th_norm / (1 + 10 * jnp.abs(th_norm))) ** 2).sum()
            cos_th = jnp.cos(th)
            is_upright = cos_th > COS_TH_THRESHOLD
            is_static = jnp.abs(thdot) < THDOT_THRESHOLD
            is_valid = jnp.logical_and(is_upright, is_static)
            rate = is_valid.sum() / is_valid.size
            return reward, rate

        df_rates_sim = {}
        rates_sim = {}
        rwd_sim = {}
        for name, eval in EVAL_SIM.items():
            with open(os.path.join(exp_dir, eval["eval"]), "rb") as f:
                gs: base.GraphState = pickle.load(f)
                num_eps = gs.state["world"].thdot.shape[0] * gs.state["world"].thdot.shape[1]
                reward, rate = jax.vmap(_get_reward_and_success)(gs.state["world"].th.reshape(num_eps, -1), gs.state["world"].thdot.reshape(num_eps, -1))
                rates_sim[name] = rate
                rwd_sim[name] = reward
                df_rates_sim[name] = create_dataframe(entry=name, success=rates_sim[name], reward=rwd_sim[name], environment="sim", policy=name, setup=name)

        df_rates = {}
        rates = {}
        rwd = {}
        mean_rwd = {}
        std_rwd = {}
        iqm_rwd = {}
        mean_rates = {}
        std_rates = {}
        iqm_rates = {}
        for name, eval in EVAL_REAL.items():
            # Get success rates
            with open(os.path.join(exp_dir, eval["eval"]), "rb") as f:
                data: base.ExperimentRecord = pickle.load(f)
            exp_rates = []
            exp_rwd = []
            for e in data.episodes:
                th = e.nodes["sensor"].steps.output.th
                thdot = e.nodes["sensor"].steps.output.thdot
                reward, rate = _get_reward_and_success(th, thdot)
                exp_rwd.append(reward)
                exp_rates.append(rate)
            exp_rates = onp.array(exp_rates)
            exp_rates = onp.sort(exp_rates)[::-1]  # Sort rates high to low
            exp_rwd = onp.array(exp_rwd)
            exp_rwd = onp.sort(exp_rwd)[::-1]

            rwd[name] = exp_rwd
            iqm_rwd[name] = exp_rwd[exp_rwd.size // 4:3 * exp_rwd.size // 4].mean()
            mean_rwd[name] = exp_rwd.mean()
            std_rwd[name] = exp_rwd.std()

            rates[name] = exp_rates
            iqm_rates[name] = exp_rates[exp_rates.size // 4:3 * exp_rates.size // 4].mean()
            mean_rates[name] = exp_rates.mean()
            std_rates[name] = exp_rates.std()
            num_zeros = (exp_rates == 0).sum()
            print(f"{LABELS[name]} | Mean success rate: {mean_rates[name]:.3f} ± {std_rates[name]:.3f} | IQM: {iqm_rates[name]:.3f} | Zeros: {num_zeros}")
            df_rates[name] = create_dataframe(entry=name, success=rates[name], reward=rwd[name], environment="real", policy=eval["policy"], setup=eval["setup"])

        # merge all dataframes in df_rates and df_rates_sim
        df_sim2real = pd.concat([df_rates[k] for k in df_rates.keys()] + [df_rates_sim[k] for k in df_rates_sim.keys()], ignore_index=True)

        # Get transient data
        df_trans = {}
        for name, eval in EVAL_SIM.items():
            with open(os.path.join(exp_dir, eval["metrics"]), "rb") as f:
                ppo_metrics = pickle.load(f)
            returns = np.array(ppo_metrics["eval/mean_returns"]).reshape(-1)
            success_rates = np.array(ppo_metrics["eval/success_rate"]).reshape(-1)
            steps = np.array(ppo_metrics["train/total_steps"]).reshape(-1)
            df_trans[name] = create_dataframe(returns=returns, steps=steps, success=success_rates, entry=name, environment="sim", policy=name, setup=name)
        df_trans = pd.concat(df_trans.values(), ignore_index=True)

        # Save to cache
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(dict(df_sim2real=df_sim2real, df_trans=df_trans), f)

    else:
        print(f"Loading cache from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            df_cache = pickle.load(f)
            df_sim2real = df_cache["df_sim2real"]
            df_trans = df_cache["df_trans"]

    # Get timings
    with open(TIMINGS_FILE, "rb") as f:
        timings = pickle.load(f)
    print(f"Pendulum | run time: {timings['solve']:.2f} s | JIT time: {timings['solve_jit']:.2f} s")

    figures = {}

    # Make sim2real performance plots
    df_cam = df_sim2real[~df_sim2real["policy"].isin(["nodelay_fullstate"])]
    df_cam = df_cam[~df_cam["setup"].isin(["encoder"])]
    df_cam["policy"] = df_cam["policy"].map(LABELS)  # Replace policy names
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    figures["perf_sim2real"] = (fig, ax)
    sns.barplot(x="environment", y="success", hue="policy", data=df_cam, ax=ax,
                palette=FCOLOR, hue_order=[LABELS[k] for k in ["delay_estimator", "nodelay_stacked", "delay_stacked"]],
                order=[k for k in ["sim", "real"]])
    ax.set(
        ylim=[0.0, 1.0],
        yticks=[0, 0.5, 1],
    )
    # Place the legend on the right outside of the figure
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              bbox_to_anchor=(2, 2),
              ncol=4, loc='lower right', fancybox=True, shadow=True)

    # Make estimator performance plots
    df_est = df_sim2real[df_sim2real["setup"].isin(["predictive", "filtered", "encoder"])]
    df_est = df_est[df_est["policy"].isin(["nodelay_fullstate"])]
    df_est["policy"] = df_est["policy"].map(LABELS)  # Replace policy names
    df_est["setup"] = df_est["setup"].map(LABELS)  # Replace setup names
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fourthwidth_figsize)
    figures["perf_estimator"] = (fig, ax)
    sns.barplot(x="setup", y="success", hue="policy", data=df_est, ax=ax,
                palette=FCOLOR, hue_order=[LABELS[k] for k in ["nodelay_fullstate"]],
                order=[LABELS[k] for k in ["encoder", "filtered", "predictive"]]
                )
    ax.set(
        xlabel="observation",
        ylim=[0.0, 1.0],
        yticks=[0, 0.5, 1],
    )
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              bbox_to_anchor=(2, 2),
              ncol=4, loc='lower right', fancybox=True, shadow=True)
    # plt.show()

    # Make training plots
    steps = df_trans['steps'].unique()
    num_steps = len(steps)
    new_steps = np.linspace(0, 2e6, num_steps) # Redistribute data to cover 0 to 2e6 range
    steps_map = {s: ns for s, ns in zip(steps, new_steps)}
    df_trans["steps"] = df_trans["steps"].map(steps_map)
    df_trans["policy"] = df_trans["policy"].map(LABELS)  # Replace policy names
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    figures["training_curves"] = (fig, ax)
    sns.lineplot(data=df_trans, x="steps", y="returns", hue="policy", ax=ax, palette=FCOLOR, errorbar="sd",
                 hue_order=[LABELS[k] for k in ["delay_estimator", "nodelay_stacked", "delay_stacked", "nodelay_fullstate"]],
                 )
    ax.set(
        ylabel="cum. reward",
        yscale="symlog",
        ylim=[-3500, -100],
        xlim=[0, 2.0e6],
    )
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              bbox_to_anchor=(2, 2),
              ncol=4, loc='lower right', fancybox=True, shadow=True)
    # plt.show()

    ###############################################################
    # SAVING
    ###############################################################
    for plot_name, (fig, ax) in figures.items():
        fig.savefig(f"{fig_dir}/{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{fig_dir}/{plot_name}.pdf", bbox_inches='tight')
        print(f"Saved {plot_name} to {fig_dir}/{plot_name}.pdf")
    return figures


@jax.jit
def _process_abstract(opt_params, true_params, graphs_sysid, rollout_opt, rollout):
    min_params = {k: v.min() for k, v in true_params.items()}
    max_params = {k: v.max() for k, v in true_params.items()}
    error = jax.tree_util.tree_map(
        lambda _opt, _true, _min, _max: None if _opt is None else jnp.abs(_opt - _true) / onp.linalg.norm(_max - _min), opt_params,
        true_params, min_params, max_params, is_leaf=lambda x: x is None)
    # Filter error to only include delays
    error_delays = {f"({u},{v})": delay.alpha for v, p in error.items() for u, delay in p.delays.items()}
    error_delays_leafs, _ = jax.tree_util.tree_flatten(error_delays)
    mean_error_delays = jnp.array(error_delays_leafs).mean(axis=0)

    # Determine leafs (vertices without outgoing edges)
    leafs = set(graphs_sysid.vertices.keys())
    for u, _ in graphs_sysid.edges.keys():
        if u in leafs:
            leafs.remove(u)
    assert len(leafs) < len(graphs_sysid.vertices), "All vertices should have outgoing edges"
    # Calculate MSE output reconstruction (only leafs to make comparable)
    opt_y = {k: v.y for k, v in rollout_opt.buffer.items() if k in leafs}
    opt_seq = {k: v for k, v in rollout_opt.seq.items() if k in leafs}
    true_y = {k: v.y for k, v in rollout.buffer.items() if k in leafs}
    true_seq = {k: v for k, v in rollout.seq.items() if k in leafs}

    @jax.vmap
    def _MSE(_true_y, _opt_y, _true_seq, _opt_seq):
        assert _true_y.shape == _opt_y.shape, "True and opt y must have same shape"
        index = jnp.arange(_true_y.shape[0])
        mask = jnp.logical_and(index < _true_seq, index < _opt_seq)
        se = (_true_y - _opt_y) ** 2
        mse = se.sum() / mask.sum()
        return mse

    MSE_leafs = jax.tree_util.tree_map(_MSE, true_y, opt_y, true_seq, opt_seq)
    MSE_leafs_flat, _ = jax.tree_util.tree_flatten(MSE_leafs)
    mean_MSE_leafs = jnp.array(MSE_leafs_flat).mean(axis=0)
    return error_delays, mean_error_delays, MSE_leafs, mean_MSE_leafs


def plot_abstract(exp_dir: str, cache_dir: str = None, fig_dir: str = None, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir}/cache-plots" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")

    LABELS_CONNECTION = {
        "sparse": {
            "(0,1)": "01",
            "(0,2)": "02",
            "(0,3)": "03",
            "(0,4)": "04",
            "(0,5)": "05",
            "(1,2)": "12",
            # todo: Extend for 12 nodes.
        },
        "tree": {
            "(0,1)": "01",
            "(0,2)": "02",
            "(1,3)": "13",
            "(1,4)": "14",
            "(2,5)": "25"
                     "",
            # todo: Extend for 12 nodes.
        },
        "chain": {f"({i},{i + 1})": f"{i}{i + 1}" for i in range(0, 12)}
    }
    YLIM_MSE = {
        "harmonic": [0.01, 2.0],
        "linear": [0.001, 0.1],
    }

    INDEX_CACHE_FILE = f"{cache_dir}/index.pkl"
    if regenerate_cache or not os.path.exists(INDEX_CACHE_FILE):
        i = 0
        configs = []
        for file in tqdm.tqdm(os.listdir(exp_dir), desc="Processing abstract experiments", ):
            if file.startswith("run_") and file.endswith(".pkl"):
                i += 1
                file_path = os.path.join(exp_dir, file)
                with open(file_path, "rb") as f:
                    d = pickle.load(f)
                config = d.get("config")
                config["file_name"] = file
                # Rename linear to chain
                config["TOPOLOGY"] = "chain" if config["TOPOLOGY"] == "linear" else config["TOPOLOGY"]
                # Verify that true loss is zero for all experiments with STD_JITTER=0.0 (should be zero with perfect model)
                if config["STD_JITTER"] == 0 and (d["true_loss"] > 0.1).any():
                    print(f"WARNING! True loss is not zero for all experiments with STD_JITTER=0.0")
                    print(d["config"])
                # Calculate error and MSE
                error_delays, mean_error_delays, MSE_leafs, mean_MSE_leafs = _process_abstract(d["opt_params"], d["true_params"], d["graphs_sysid"], d["rollout_opt"], d["rollout"])
                config["error_delays"] = jax.tree_util.tree_map(lambda x: onp.array(x), error_delays) #  {"(u,v)": [runs]}
                config["mean_error_delays"] = onp.array(mean_error_delays)  # [runs] --> Averaged over all connections within a run
                config["MSE_leafs"] = jax.tree_util.tree_map(lambda x: onp.array(x), MSE_leafs)
                config["mean_MSE_leafs"] = onp.array(mean_MSE_leafs)
                # Save to cache
                configs.append(config)
                # if i > 5:
                #     break
        df = pd.DataFrame(configs)
        # Save to cache
        with open(INDEX_CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
    else:
        print(f"Loading cache from {INDEX_CACHE_FILE}")
        with open(INDEX_CACHE_FILE, "rb") as f:
            df = pickle.load(f)
    # Prepare experiment data per plot
    figures = {}
    for PARAM_CLS in ["harmonic", "linear"]:
        for STD_JITTER in [0.0, 0.05]:
            for NUM_NODES in [6]: # TODO: [4, 6, 12]:
                for TOPOLOGY in ["sparse", "tree", "chain"]:
                    df_mean = df[(df["PARAM_CLS"] == PARAM_CLS) & (df["STD_JITTER"] == STD_JITTER) & (df["NUM_NODES"] == NUM_NODES) & (df["TOPOLOGY"] == TOPOLOGY)]
                    entries = []
                    for _, row in df_mean.iterrows():
                        for mse, delay in zip(row["mean_MSE_leafs"], row["mean_error_delays"]):
                            entries.append(dict(MASK=row["MASK"], SYSID_PARAMS=row["SYSID_PARAMS"], mean_error_delays=delay, mean_MSE_leafs=mse))
                    d = pd.DataFrame(entries)
                    # Apply mapping labels
                    d["SYSID_PARAMS"] = d["SYSID_PARAMS"].map(LABELS)
                    d["MASK"] = d["MASK"].map(LABELS)

                    # Plot mean_error_delays
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
                    fig_name = f"mean_error_delays_P-{PARAM_CLS}_J-{STD_JITTER}_N-{NUM_NODES}_T-{TOPOLOGY}"
                    figures[fig_name] = (fig, ax)
                    sns.barplot(x="MASK", y="mean_error_delays", hue="SYSID_PARAMS", data=d, palette=FCOLOR, ax=ax,
                                order=[LABELS[k] for k in ["all", "leaf"]],
                                hue_order=[LABELS[k] for k in ["delay_dynamics", "delay"]]
                                )
                    ax.set(xlabel="data", yticks=[0, 0.5, 1.0], ylabel="norm. error", ylim=[0.0, 1.0])
                    # Place the legend on the right outside of the figure
                    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                              bbox_to_anchor=(2, 2),
                              ncol=6, loc='lower right', fancybox=True, shadow=True)

                    # Plot mean_MSE_leafs
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
                    fig_name = f"mean_MSE_leafs_P-{PARAM_CLS}_J-{STD_JITTER}_N-{NUM_NODES}_T-{TOPOLOGY}"
                    figures[fig_name] = (fig, ax)
                    sns.barplot(x="MASK", y="mean_MSE_leafs", hue="SYSID_PARAMS", data=d, palette=FCOLOR, ax=ax,
                                order=[LABELS[k] for k in ["all", "leaf"]],
                                hue_order=[LABELS[k] for k in ["delay_dynamics", "delay"]]
                                )
                    ax.set(xlabel="data", ylabel="MSE", ylim=YLIM_MSE[PARAM_CLS], yscale="log")  # Set ylim=[..., ...] appropriately
                    # Place the legend on the right outside of the figure
                    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                              bbox_to_anchor=(2, 2),
                              ncol=6, loc='lower right', fancybox=True, shadow=True)

                    # Plot at node and connection level
                    for MASK in ["all", "leaf"]:
                        df_plot = df[(df["PARAM_CLS"] == PARAM_CLS) & (df["STD_JITTER"] == STD_JITTER) & (df["NUM_NODES"] == NUM_NODES) & (df["TOPOLOGY"] == TOPOLOGY) & (df["MASK"] == MASK)]
                        c_list = []
                        n_list = []
                        for _, row in df_plot.iterrows():

                            for k, v in row["error_delays"].items():
                                assert TOPOLOGY in LABELS_CONNECTION, f"TOPOLOGY={TOPOLOGY} not implemented"
                                try:
                                    c = LABELS_CONNECTION[TOPOLOGY][k]
                                except KeyError as e:
                                    raise KeyError(f"Connection {k} not found in LABELS_CONNECTION[{TOPOLOGY}]") from e

                                _d = create_dataframe(SYSID_PARAMS=row["SYSID_PARAMS"], CONNECTION=c, error_delays=v)
                                c_list.append(_d)
                            for k, v in row["MSE_leafs"].items():
                                _d = create_dataframe(SYSID_PARAMS=row["SYSID_PARAMS"], LEAF=k, MSE_leafs=v)
                                n_list.append(_d)
                        # Concatenate all pd's in c_list, n_list
                        df_c = pd.concat(c_list, ignore_index=True)
                        df_n = pd.concat(n_list, ignore_index=True)

                        # Apply mapping labels
                        df_c["SYSID_PARAMS"] = df_c["SYSID_PARAMS"].map(LABELS)
                        df_n["SYSID_PARAMS"] = df_n["SYSID_PARAMS"].map(LABELS)

                        # Plot error_delays
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
                        fig_name = f"error_delays_P-{PARAM_CLS}_J-{STD_JITTER}_N-{NUM_NODES}_T-{TOPOLOGY}_M-{MASK}"
                        figures[fig_name] = (fig, ax)
                        sns.barplot(x="CONNECTION", y="error_delays", hue="SYSID_PARAMS", data=df_c, palette=FCOLOR, ax=ax,
                                    hue_order=[LABELS[k] for k in ["delay_dynamics", "delay"]]
                                    )
                        ax.set(xlabel="edge", yticks=[0, 0.5, 1.0], ylabel="norm. error", ylim=[0.0, 1.0])
                        # Place the legend on the right outside of the figure
                        ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                                  bbox_to_anchor=(2, 2),
                                  ncol=6, loc='lower right', fancybox=True, shadow=True)

                        # Plot MSE_leafs
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
                        fig_name = f"MSE_leafs_P-{PARAM_CLS}_J-{STD_JITTER}_N-{NUM_NODES}_T-{TOPOLOGY}_M-{MASK}"
                        figures[fig_name] = (fig, ax)
                        sns.barplot(x="LEAF", y="MSE_leafs", hue="SYSID_PARAMS", data=df_n, palette=FCOLOR, ax=ax,
                                    hue_order=[LABELS[k] for k in ["delay_dynamics", "delay"]]
                                    )
                        ax.set(xlabel="node", ylabel="MSE", ylim=YLIM_MSE[PARAM_CLS], yscale="log")  # Set ylim=[..., ...] appropriately
                        # Place the legend on the right outside of the figure
                        ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                                  bbox_to_anchor=(2, 2),
                                  ncol=6, loc='lower right', fancybox=True, shadow=True)
                        # plt.show()
    ###############################################################
    # SAVING
    ###############################################################
    for plot_name, (fig, ax) in figures.items():
        fig.savefig(f"{fig_dir}/legend_{plot_name}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{fig_dir}/{plot_name}.pdf", bbox_inches='tight')
        print(f"Saved {plot_name} to {fig_dir}/{plot_name}.pdf")


def plot_computation(exp_dir_pend: str, exp_dir_cf: str, fig_dir: str, regenerate_cache: bool = True):
    # Create cache directory
    cache_dir = f"{exp_dir_pend}/cache-plots"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    # Create figure directory
    fig_dir = f"{exp_dir_pend}/figs" if fig_dir is None else fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Figure directory: {fig_dir}")

    PEND_FILE = f"{exp_dir_pend}/rollout_stats.pkl"
    CF_FILE = f"{exp_dir_cf}/delay_rollout_stats.pkl"

    INDEX_CACHE_FILE = f"{cache_dir}/index.pkl"
    if regenerate_cache or not os.path.exists(INDEX_CACHE_FILE):
        # No caching used.
        pass

    # Load data
    with open(PEND_FILE, "rb") as f:
        pend_stats = pickle.load(f)
    with open(CF_FILE, "rb") as f:
        cf_stats = pickle.load(f)

    # Create dataframe
    df_pend = pd.DataFrame(pend_stats).transpose()
    df_cf = pd.DataFrame(cf_stats).transpose()

    # Add experiment name as a row
    df_pend["system"] = "pendulum"
    df_cf["system"] = "quadrotor"
    df_stats = pd.concat([df_pend, df_cf])
    df_stats["num_rollouts"] = df_stats["num_rollouts"].astype(int)
    df_stats = df_stats.sort_values(by='num_rollouts')
    # df_power_of_4 = df_stats[df_stats['num_rollouts'].isin([2 ** (2*i) for i in range(15)])]

    # Plot rollout stats
    figsize = [c * s for c, s in zip([2/3, 0.5], rescaled_figsize)]
    figures = {}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    figures["rollout_stats"] = (fig, ax)
    sns.barplot(x="num_rollouts", y="fps", hue="system", data=df_stats, ax=ax,
                palette=FCOLOR,
                # hue_order=[LABELS[k] for k in ["delay_estimator", "nodelay_stacked", "delay_stacked"]],
                # order=[k for k in ["sim", "real"]]
    )

    # Get current x-ticks
    xticks = df_stats["num_rollouts"].unique()
    current_xticks = ax.get_xticks()

    # # Create new labels with smaller "2" and smaller overall text
    # new_xticklabels = [
    #     f"$\\mathregular{{2^{{\\scriptscriptstyle{{{int(np.log2(tick))}}}}}}}$" if tick > 0 else ""
    #     for tick in xticks
    # ]

    # # Create new labels for the current x-ticks
    new_xticklabels = [f"$2^{{{int(np.log2(tick))}}}$" if i % 2 == 0 else "" for i, tick in enumerate(xticks)]

    # Set the new x-tick labels
    ax.set_xticks(current_xticks)
    ax.set_xticklabels(new_xticklabels)
    ax.set_yscale("log")#, base=2)
    ax.set(
        xlabel="parallel envs",
        ylabel="speed (steps/s)",
        # yscale="log",
        ylim=[1e3, 1e8],
    )
    # Place the legend on the right outside of the figure
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              bbox_to_anchor=(2, 2),
              ncol=4, loc='lower right', fancybox=True, shadow=True)
    # plt.show()

    ###############################################################
    # SAVING
    ###############################################################
    for plot_name, (fig, ax) in figures.items():
        fig.savefig(f"{fig_dir}/{plot_name}_legend.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{fig_dir}/{plot_name}.pdf", bbox_inches='tight')
        print(f"Saved {plot_name} to {fig_dir}/{plot_name}.pdf")

    return []


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True)
    onp.set_printoptions(precision=3, suppress=True)
    # todo: plot results
    #   [D] Delay distributions
    #   [D] System identification
    #   - detection pipeline.
    #   - pixels
    #   - Brax setup
    #   [D] action
    #   [D] th, thdot (+ zoomed)
    #   - CMA-ES loss
    #   [D] MSE
    #   - timings
    #   [D] Reinforcement learning (rwd learning curves, sim2real evaluation, timings)
    #   [D] Videos of experiments
    #   - Appendix: computation graph with jittery camera (with and without world)
    #   -

    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v2_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v3_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_longerstack_dark"
    EXP_DIR_PENDULUM = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v4_dark"
    EXP_DIR_ABSTRACT = "/home/r2ci/rex/scratch/abstract/logs/main_4-6-12Nodes_0.05Jitter_0sup"  # todo: _cem
    # EXP_DIR_CRAZYFLIE = "/home/r2ci/rex/scratch/crazyflie/logs/20240816_path_following_inclined_landing_experiments"
    EXP_DIR_CRAZYFLIE_SYSID = "/home/r2ci/rex/scratch/crazyflie/logs/20240816_path_following_inclined_landing_experiments/sysid_redo_with_nodelay"
    EXP_DIR_CRAZYFLIE_DISTS = "/home/r2ci/rex/scratch/crazyflie/logs/20240816_path_following_inclined_landing_experiments/delay_1.0R"
    EXP_DIR_CRAZYFLIE_PF = "/home/r2ci/rex/scratch/crazyflie/logs/20241206_path_following_inclined_landing_experiments_rebuttal"
    REGENERATE_CACHE = False
    print(f"Pendulum experiment directory: {EXP_DIR_PENDULUM}")
    print(f"Abstract experiment directory: {EXP_DIR_ABSTRACT}")

    # Create figure directory
    FIG_DIR = f"/home/r2ci/Documents/project/thesis/REX/figures/python"
    # os.makedirs(FIG_DIR, exist_ok=True)
    # print(f"Figure directory: {FIG_DIR}")
    # Create cache directory
    # CACHE_DIR = f"{EXP_DIR}/cache-plots"
    # os.makedirs(CACHE_DIR, exist_ok=True)
    # print(f"Cache directory: {CACHE_DIR}")

    # Plot
    # figs = plot_crazyflie_dists(exp_dir=EXP_DIR_CRAZYFLIE_DISTS, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    # figs = plot_crazyflie_sysid(exp_dir=EXP_DIR_CRAZYFLIE_SYSID, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE or True)
    figs = plot_crazyflie_pathfollowing(exp_dir=EXP_DIR_CRAZYFLIE_PF, fig_dir=FIG_DIR, regenerate_cache=True)
    # figs = plot_abstract(exp_dir=EXP_DIR_ABSTRACT, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    # figs = plot_pendulum_sysid(exp_dir=EXP_DIR_PENDULUM, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    # figs = plot_pendulum_dists(exp_dir=EXP_DIR_PENDULUM, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    # figs = plot_pendulum_rl(exp_dir=EXP_DIR_PENDULUM, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    # plot_delay_distributions(...)
    # figs = plot_computation(exp_dir_pend=EXP_DIR_PENDULUM, exp_dir_cf=EXP_DIR_CRAZYFLIE_PF, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE) # todo: not working yet.
    # plt.show()
