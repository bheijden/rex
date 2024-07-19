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

# Setup sns plotting
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
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
    "camera": "detector",
    "estimator": "estimator",
    "est_future": "predictive",
    "est_meas": "filtered (corrected)",
    "est_nopred": "filtered",
    "controller": "controller",
    "world": "brax",
}
CSCHEME = {
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
    "actuator": "cyan",
    "camera": "pink",
    "estimator": "violet",
    "est_future": "green",
    "est_meas": "violet",
    "est_nopred": "yellow",
    "controller": "indigo",
    "world": "blue",
    # line
    "zoomed_frame": "gray",
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


def plot_system_identification(exp_dir: str, cache_dir: str, fig_dir: str, regenerate_cache: bool = True):
    DATA_SYSID_FILE = os.path.join(exp_dir, "sysid_data.pkl")
    EVAL_SYSID_FILE = os.path.join(exp_dir, "sysid_gs_eval.pkl")
    CACHE_FILE = f"{cache_dir}/plot_system_identification.pkl"
    EPS_IDX = -1

    if regenerate_cache or not os.path.exists(CACHE_FILE):
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
        params = dict(jax.tree_util.tree_map(lambda x: x[0], gs.params))

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

        th_include = ["sensor", "world", "camera", "est_future", "est_nopred", "est_meas"]
        thdot_include = ["sensor", "world", "est_future", "est_nopred", "est_meas"]

        def plot_graphs(ax, keys, interp, y_key, xlabel, ylabel):
            for key, value in interp.items():
                if key not in keys:
                    continue
                ax.plot(value["ts"], value[y_key], label=LABELS[key], color=ECOLOR[key])
            ax.set(xlabel=xlabel, ylabel=ylabel)
            # Place the legend on the right outside of the figure
            ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                      bbox_to_anchor=(2, 2),
                      ncol=6, loc='lower right', fancybox=True, shadow=True)

        def plot_zoomed_frame(ax, x_zoom, y_zoom, w, h, w_scale, h_scale):
            rect = patches.Rectangle(
                (x_zoom + w / 2 - w_scale * w / 2, y_zoom + h / 2 - h_scale * h / 2),
                w * w_scale, h * h_scale, linewidth=2,
                edgecolor=ECOLOR["zoomed_frame"], facecolor='none'
            )
            ax.add_patch(rect)

        figures = {}
        # Plot th
        fig, ax = plt.subplots(figsize=twothirdwidth_figsize)
        figures["sysid_th"] = (fig, ax)
        plot_graphs(ax, th_include, interp, "th", "Time (s)", "Angle (rad)")
        plot_zoomed_frame(ax, 18.8, 11.0, 0.4, 1.1, 1.0, 1.0)

        fig, ax = plt.subplots(figsize=thirdwidth_figsize)
        figures["sysid_th_zoom"] = (fig, ax)
        plot_graphs(ax, th_include, interp, "th", "Time (s)", "Angle (rad)")
        ax.set(xlim=[18.8, 18.8 + 0.4], ylim=[11.0, 11.0 + 1.1])
        # ax.legend()

        # Plot thdot
        fig, ax = plt.subplots(figsize=twothirdwidth_figsize)
        figures["sysid_thdot"] = (fig, ax)
        plot_graphs(ax, thdot_include, interp, "thdot", "Time (s)", "Ang. rate (rad/s)")
        plot_zoomed_frame(ax, 18.8, -4, 0.4, 13, 1.0, 1.0)

        fig, ax = plt.subplots(figsize=thirdwidth_figsize)
        figures["sysid_thdot_zoom"] = (fig, ax)
        plot_graphs(ax, thdot_include, interp, "thdot", "Time (s)", "Ang. rate (rad/s)")
        ax.set(xlim=[18.8, 18.8 + 0.4], ylim=[-4, -4 + 13])
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


def plot_reinforcement_learning(exp_dir: str, cache_dir: str, fig_dir: str, regenerate_cache: bool = True):
    SKIP_STEPS = 75
    COS_TH_THRESHOLD = 0.9
    THDOT_THRESHOLD = 2.0

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

    CACHE_FILE = f"{cache_dir}/plot_reinforcement_learning.pkl"
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    figures["perf_estimator"] = (fig, ax)
    sns.barplot(x="setup", y="success", hue="policy", data=df_est, ax=ax,
                palette=FCOLOR, hue_order=[LABELS[k] for k in ["nodelay_fullstate"]],
                order=[LABELS[k] for k in ["encoder", "filtered", "predictive"]]
                )
    ax.set(
        ylim=[0.0, 1.0],
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
    sns.lineplot(data=df_trans, x="steps", y="returns", hue="policy", ax=ax, palette=FCOLOR, errorbar="sd")
    ax.set(
        ylim=[-3500, 0],
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


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True)
    onp.set_printoptions(precision=3, suppress=True)
    # todo: plot results
    #  - Delay distributions
    #  - System identification
    #   - detection pipeline.
    #   - pixels
    #   - Brax setup
    #   [D] action
    #   [D] th, thdot (+ zoomed)
    #   - CMA-ES loss
    #   - MSE
    #   - timings
    #  - Reinforcement learning (rwd learning curves, sim2real evaluation, timings)
    #  - Appendix: computation graph with jittery camera (with and without world)
    #  - Videos of experiments
    #  -

    EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v2_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v3_dark"
    EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v4_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_longerstack_dark"
    CACHE_DIR = f"{EXP_DIR}/cache-plots"
    FIG_DIR = f"{EXP_DIR}/figs"
    REGENERATE_CACHE = False
    print(f"Experiment directory: {EXP_DIR}")

    # Create figure directory
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Figure directory: {FIG_DIR}")

    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")

    figs = plot_system_identification(exp_dir=EXP_DIR, cache_dir=CACHE_DIR, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    figs = plot_reinforcement_learning(exp_dir=EXP_DIR, cache_dir=CACHE_DIR, fig_dir=FIG_DIR, regenerate_cache=REGENERATE_CACHE)
    # plot_delay_distributions(...)
    # plot_reinforcement_learning(...)
    plt.show()

