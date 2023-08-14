# HACK: https://github.com/DLR-RM/stable-baselines3/pull/780
import sys
import gymnasium
sys.modules["gym"] = gymnasium

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tempfile
import datetime
import yaml
from typing import Dict, List, Tuple, Union, Callable, Any, Type
from types import ModuleType
import dill as pickle
import time
import jax
import numpy as onp
import jumpy
import jumpy.numpy as jp
from stable_baselines3.common.vec_env import VecMonitor

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

import sbx
import stable_baselines3 as sb3
import time
import rex
from rex.supergraph import create_graph
import rex.constants as rc
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
import experiments as exp
import envs.pendulum as pend


def plot_dists(dist, data=None, info=None, est=None):
    HAS_DATA = False if data is None else True
    HAS_INFO = False if info is None else True
    HAS_EST = False if est is None else True
    if data is None:
        data = jax.tree_map(lambda x: None, dist)
    if info is None:
        class Dummy:
            def __init__(self, name, output=None, cls=None):
                self.name = name
                self.output = output
                if cls is not None:
                    self.cls = cls

        info = {"step": {k: Dummy(k, cls=True) for k in dist["step"].keys()},
                "inputs": {k: {v: (Dummy(k, cls=True),Dummy(v, output=v)) for v in dist["inputs"][k].keys()} for k in dist["step"].keys()}}
    if est is None:
        est = jax.tree_map(lambda x: None, dist)

    # First shallow copy of arguments
    dist = jax.tree_map(lambda x: x, dist)
    data = jax.tree_map(lambda x: x, data)
    info = jax.tree_map(lambda x: x, info)
    est = jax.tree_map(lambda x: x, est)

    # Pop world from
    [_d["inputs"]["agent"].pop("last_action", None) for _d in [data, info, est, dist]]
    [_d["inputs"]["sensor"].pop("world", None) for _d in [data, info, est, dist]]
    [_d["inputs"].pop("world", None) for _d in [data, info, est, dist]]
    [_d["step"].pop("world", None) for _d in [data, info, est, dist]]

    # Split
    est_inputs, est_step = est["inputs"], est["step"]
    data_inputs, data_step = data["inputs"], data["step"]
    info_inputs, info_step = info["inputs"], info["step"]
    dist_inputs, dist_step = dist["inputs"], dist["step"]

    # Plot gmm
    from matplotlib.ticker import FormatStrFormatter
    import numpy as onp

    def plot_gmm(ax, dist, delays, i, edgecolor):
        m = onp.max(delays) if delays is not None else dist.high
        x = onp.linspace(0, m, 1000)
        y = dist.pdf(x)
        # if isinstance(i, tuple):
        #     output, input = i
        #     if output is None:
        #         ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
        #         ax.set_title(f"{input}")
        #     else:
        #         ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
        #         ax.set_title(f"{input} -> {output}")
        if hasattr(i, "cls"):
            ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
            ax.set_title(f"{i.name}")
        else:
            node_info, input_info = i
            ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
            ax.set_title(f"{input_info.output} -> {node_info.name} ({input_info.name})")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set_xlabel('delay (s)', fontsize=10)
        ax.set_ylabel('density', fontsize=10)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend()

    # Plot distributions
    from rex.plot import get_subplots

    fig_step, axes_step = get_subplots(dist_step, figsize=(10, 10), sharex=False, sharey=False, major="row")
    fig_inputs, axes_inputs = get_subplots(dist_inputs, figsize=(10, 10), sharex=False, sharey=False, major="row")

    # Plot measured delays
    from rex.open_colors import ecolor, fcolor

    if HAS_EST:
        jax.tree_map(
            lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.computation, facecolor=fcolor.computation, plot_dist=False),
            axes_step, est_step)
        jax.tree_map(
            lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.communication, facecolor=fcolor.communication, plot_dist=False),
            axes_inputs, est_inputs)

    # Plot gmm
    from functools import partial

    jax.tree_map(partial(plot_gmm, edgecolor=ecolor.computation), axes_step, dist_step, data_step, info_step)
    jax.tree_map(partial(plot_gmm, edgecolor=ecolor.communication), axes_inputs, dist_inputs, data_inputs, info_inputs)

    return fig_step, fig_inputs


if __name__ == "__main__":
    # todo: record deterministic and asynchronous episodes
    # Environment
    ENV = "disc-pendulum"  # "disc_pendulum"
    DIST_FILE = "21eps_pretrained_sbx_sac_gmms_2comps.pkl"  # todo: absolute path "/"
    JITTER = rc.LATEST
    SCHEDULING = rc.FREQUENCY
    WIN_ACTION = 2
    WIN_OBS = 3
    BLOCKING = False
    ADVANCE = False
    ENV_FN = pend.ode.build_pendulum  #  dpend.ode.build_double_pendulum  # pend.ode.build_pendulum
    ENV_CLS = pend.env.PendulumEnv  # dpend.env.DoublePendulumEnv  # pend.env.PendulumEnv
    CLOCK = rc.WALL_CLOCK  # todo: change
    RTF = rc.REAL_TIME  # todo: change
    MAX_STEPS = 5 * 20
    RATES = dict(world=100, agent=20, actuator=20, sensor=20, render=20)
    USE_DELAYS = True
    QUANTILE = 0.5
    delay_fn = lambda d: d.quantile(QUANTILE)*int(USE_DELAYS)  # todo: this is slow (takes 3 seconds).

    # Training
    SEED = 0
    NUM_EVAL_PRE = 2

    # Load models. PPO=[64, 64], SAC=[256, 256]
    MODEL_CLS = sbx.SAC  # sbx.SAC  sb3.SAC
    MODEL_MODULE = pend.models
    MODEL_PRELOAD = "sbx_ppo_pendulum"  # sbx_sac_pendulum

    # Logging
    NAME = f"test-{ENV}-{ENV_FN.__module__}-{rc.CLOCK_MODES[CLOCK]}-{rc.RTF_MODES[RTF]}-{rc.SCHEDULING_MODES[SCHEDULING]}-{rc.JITTER_MODES[JITTER]}"
    LOG_DIR = f"/home/r2ci/rex/paper/logs/{NAME}-{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"
    MUST_LOG = True
    MUST_PLOT = True
    MUST_DIST = True
    SHOW_PLOT = True
    RECORD_SETTINGS = {"agent": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
                       "world": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
                       "actuator": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
                       "sensor": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False),
                       "render": dict(node=True, outputs=False, rngs=False, states=False, params=False, step_states=False)}

    # Load distributions
    delays_sim = exp.load_distributions(DIST_FILE, module=pend.dists)
    fig_sim_step, fig_sim_inputs = plot_dists(delays_sim) if MUST_DIST and MUST_PLOT else (None, None)

    # Prepare environment
    env = exp.make_env(delays_sim, delay_fn, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
                       scheduling=SCHEDULING, jitter=JITTER, env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True,
                       clock=CLOCK, real_time_factor=RTF, max_steps=MAX_STEPS, use_delays=USE_DELAYS)
    gym_env = GymWrapper(env)

    # Load model
    model: MODEL_CLS = exp.load_model(MODEL_PRELOAD, MODEL_CLS, env=gym_env, seed=SEED, module=MODEL_MODULE)
    policy = exp.make_policy(model, constant_action=1.0)

    # Evaluate model
    record_pre = exp.eval_env(gym_env, policy, n_eval_episodes=NUM_EVAL_PRE, verbose=True, seed=SEED, record_settings=RECORD_SETTINGS)

    # Compile env
    cenv = exp.make_compiled_env(env, record_pre.episode, max_steps=MAX_STEPS, eval_env=False)

    # Plot
    graph = create_graph(record_pre.episode[-1])
    fig_gr, _ = exp.show_graph(record_pre.episode[-1]) if MUST_PLOT else (None, None)
    fig_cg, _ = exp.show_computation_graph(graph, cenv.graph.S, root="agent", plot_type="computation") if MUST_PLOT else (None, None)
    fig_com, _ = exp.show_communication(record_pre.episode[-1]) if MUST_PLOT else (None, None)
    fig_grp, _ = exp.show_grouped(record_pre.episode[-1].node[-1], "state") if MUST_PLOT else (None, None)

    # Fit distributions
    data, info, est, dist = exp.make_delay_distributions(record_pre, num_steps=500, num_components=8, step_size=0.05, seed=SEED) if MUST_DIST else (None, None, None, None)
    fig_step, fig_inputs = plot_dists(dist, data, info, est) if MUST_DIST and MUST_PLOT else (None, None)

    # Only show
    plt.show() if SHOW_PLOT else None

    # Log
    if MUST_LOG:
        os.mkdir(LOG_DIR)
        # Identify all capitalized variables & save them to file
        capitalized_vars = {k: v for k, v in globals().items() if k.isupper()}
        with open(f"{LOG_DIR}/params.yaml", 'w') as file:
            yaml.dump(capitalized_vars, file)
        # Save envs
        env.unwrapped.save(f"{LOG_DIR}/{env.unwrapped.name}.pkl")
        cenv.unwrapped.save(f"{LOG_DIR}/{cenv.unwrapped.name}.pkl")
        # Save pre-train record to file
        with open(LOG_DIR + "/record_pre.pb", "wb") as f:
            f.write(record_pre.SerializeToString())
        # Save plots
        fig_gr.savefig(LOG_DIR + "/robotic_system.png") if fig_gr is not None else None
        fig_cg.savefig(LOG_DIR + "/computation_graph.png") if fig_gr is not None else None
        fig_com.savefig(LOG_DIR + "/communication.png") if fig_gr is not None else None
        fig_grp.savefig(LOG_DIR + "/grouped_agent_sensor.png") if fig_gr is not None else None
        fig_sim_step.savefig(LOG_DIR + "/delay_sim_step.png") if fig_gr is not None else None
        fig_sim_inputs.savefig(LOG_DIR + "/delay_sim_inputs.png") if fig_gr is not None else None
        fig_step.savefig(LOG_DIR + "/delay_step.png") if fig_gr is not None else None
        fig_inputs.savefig(LOG_DIR + "/delay_inputs.png") if fig_gr is not None else None
        # Save to file
        import dill as pickle
        if MUST_DIST:
            with open(LOG_DIR + "/distributions.pkl", "wb") as f:
                pickle.dump(dist, f)
        # Save model used to evaluate
        model.save(LOG_DIR + "/eval_model.zip")
