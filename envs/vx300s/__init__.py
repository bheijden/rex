import envs.vx300s.brax as brax
import envs.vx300s.mjx as mjx
import envs.vx300s.real as real
import envs.vx300s.env as env
import envs.vx300s.models as models
import envs.vx300s.dists as dists
import envs.vx300s.planner


# From experiments.__init__.py
import itertools
import pandas as pd

import tqdm
import os
from functools import partial
from typing import Dict, List, Tuple, Union, Callable, Any, Type, Optional
from types import ModuleType
from pickle import UnpicklingError
from google.protobuf.pyext._message import RepeatedCompositeContainer
import dill as pickle
import time
import jax
import flax.serialization as serialization
import networkx as nx
from jax.tree_util import tree_map
import numpy as onp
import jumpy
import jumpy.numpy as jp
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from flax import struct
from brax.base import Transform, System
from brax.io import mjcf
from brax.spring import pipeline as s_pipeline


import rex.utils as utils
from rex.compiled import CompiledGraph
import rex.supergraph as supergraph
from rex.plot import plot_graph, plot_computation_graph, plot_grouped, plot_input_thread, plot_event_thread
from rex.gmm_estimator import GMMEstimator
from rex.env import BaseEnv
from rex.node import Node
from rex.asynchronous import AsyncGraph
from rex.proto import log_pb2
from rex.distributions import Distribution, Gaussian
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
    ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, INFO, LATEST
import rex.open_colors as oc

from envs.vx300s.env import Vx300sEnv, Controller, Supervisor, get_global_plan, get_next_jpos
import envs.vx300s.dists
import envs.vx300s.models
import experiments as exp


def make_env(delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
             delay_fn: Callable[[Distribution], float],
             rates: Dict[str, float],
             config: Dict[str, Dict[str, Any]],
             scheduling: int = FREQUENCY,
             win_planner: int = 2,
             delay_planner: float = None,
             jitter: int = LATEST,
             env_fn: Callable = brax.build_vx300s,
             name: str = "vx300s",
             max_steps: int = 20,
             clock: int = SIMULATED,
             real_time_factor: int = REAL_TIME,
             use_delays: bool = True,
             viewer: bool = True,
             ) -> Vx300sEnv:
    # Override delays
    delays_sim["step"] = delays_sim.get("step", {})
    delays_sim["inputs"] = delays_sim.get("inputs", {})
    for n in ["supervisor", "world", "planner", "controller", "armactuator", "armsensor", "boxsensor", "viewer"]:
        delays_sim["step"][n] = delays_sim["step"].get(n, Gaussian(0.))
        delays_sim["inputs"][n] = delays_sim["inputs"].get(n, {})
    delays_sim["step"]["world"] = Gaussian(0.)
    delays_sim["step"]["armactuator"] = Gaussian(0.)
    delays_sim["step"]["armsensor"] = Gaussian(0.)
    # delays_sim["step"]["boxsensor"] = Gaussian(0.)
    # delays_sim["inputs"]["world"]["armactuator"] = Gaussian(0.)
    delays_sim["inputs"]["armactuator"]["controller"] = delays_sim["inputs"]["armactuator"].get("controller", Gaussian(0.))
    delays_sim["inputs"]["controller"]["planner"] = delays_sim["inputs"]["controller"].get("planner", Gaussian(0.))
    # delays_sim["inputs"]["armsensor"]["world"] = Gaussian(0.)
    # delays_sim["inputs"]["boxsensor"]["world"] = Gaussian(0.)
    delays_sim["inputs"]["planner"]["armsensor"] = delays_sim["inputs"]["planner"].get("armsensor", Gaussian(0.))
    delays_sim["inputs"]["planner"]["boxsensor"] = delays_sim["inputs"]["planner"].get("boxsensor", Gaussian(0.))
    delays_sim["inputs"]["planner"]["supervisor"] = delays_sim["inputs"]["planner"].get("boxsensor", Gaussian(0.))
    delays_sim["inputs"]["supervisor"]["armsensor"] = delays_sim["inputs"]["supervisor"].get("armsensor", Gaussian(0.))
    delays_sim["inputs"]["supervisor"]["boxsensor"] = delays_sim["inputs"]["supervisor"].get("boxsensor", Gaussian(0.))
    delays_sim["inputs"]["viewer"]["armsensor"] = delays_sim["inputs"]["viewer"].get("armsensor", Gaussian(0.))
    delays_sim["inputs"]["viewer"]["boxsensor"] = delays_sim["inputs"]["viewer"].get("boxsensor", Gaussian(0.))
    delays_sim["inputs"]["viewer"]["supervisor"] = delays_sim["inputs"]["viewer"].get("supervisor", Gaussian(0.))
    delays_sim = tree_map(lambda d: Gaussian(0), delays_sim) if not use_delays else delays_sim
    delays = jax.tree_map(delay_fn, delays_sim)

    # Overwrite delay if
    delays["step"]["planner"] = delays["step"]["planner"] if delay_planner is None else delay_planner

    # Determine whether we use the real system
    real = True if "real" in env_fn.__module__ else False
    if real:
        assert clock == WALL_CLOCK, "Real system must be run with wall clock"
        assert real_time_factor == REAL_TIME, "Real system must be run in real time"

    # Define all nodes
    nodes = {}

    # Build nodes
    nodes.update(env_fn(rates, delays_sim, delays, config, scheduling=scheduling))
    world, armactuator, armsensor, boxsensor = nodes["world"], nodes["armactuator"], nodes["armsensor"], nodes["boxsensor"]

    # Define supervisor
    supervisor = Supervisor("supervisor", rate=rates["supervisor"], advance=True, scheduling=scheduling,
                            delay_sim=delays_sim["step"]["supervisor"], delay=delays["step"]["supervisor"])
    nodes["supervisor"] = supervisor

    # Define controller
    controller = Controller("controller", rate=rates["controller"], advance=False, scheduling=scheduling,
                            delay_sim=delays_sim["step"]["controller"], delay=delays["step"]["controller"])
    nodes["controller"] = controller

    # Define planner
    if config["planner"]["type"] == "brax":
        planner = envs.vx300s.planner.BraxCEMPlanner(name="planner", rate=rates["planner"], advance=True, scheduling=scheduling,
                                                     delay_sim=delays_sim["step"]["planner"], delay=delays["step"]["planner"],
                                                     mj_path=config["planner"]["brax_xml_path"], pipeline="generalized",
                                                     z_fixed=config["planner"]["z_fixed"],
                                                     horizon=config["planner"]["horizon"], u_max=config["planner"]["u_max"],
                                                     dt=config["planner"]["dt"], dt_substeps=config["planner"]["dt_substeps"],
                                                     num_samples=config["planner"]["num_samples"],
                                                     max_iter=config["planner"]["max_iter"])
    elif config["planner"]["type"] == "rex":
        planner = envs.vx300s.planner.RexCEMPlanner(name="planner", rate=rates["planner"], advance=True, scheduling=scheduling,
                                                    nodes=nodes,
                                                    num_cost_est=config["planner"]["num_cost_est"],
                                                    num_cost_mpc=config["planner"]["num_cost_mpc"],
                                                    use_estimator=config["planner"]["use_estimator"],
                                                    randomize_eps=config["planner"]["randomize_eps"],
                                                    graph_path=config["planner"]["rex_graph_path"],
                                                    supergraph_mode=config["planner"]["supergraph_mode"],
                                                    delay_sim=delays_sim["step"]["planner"], delay=delays["step"]["planner"],
                                                    mj_path=config["planner"]["rex_xml_path"], pipeline="generalized",
                                                    z_fixed=config["planner"]["z_fixed"],
                                                    horizon=config["planner"]["horizon"], u_max=config["planner"]["u_max"],
                                                    dt=config["planner"]["dt"], dt_substeps=config["planner"]["dt_substeps"],
                                                    num_samples=config["planner"]["num_samples"],
                                                    max_iter=config["planner"]["max_iter"])
    else:
        raise ValueError(f"Unknown planner type {config['planner']['type']}")
    nodes["planner"] = planner

    # Connect
    supervisor.connect(armsensor, name="armsensor", window=1, blocking=False, jitter=jitter,
                       delay_sim=delays_sim["inputs"]["supervisor"]["armsensor"], delay=delays["inputs"]["supervisor"]["armsensor"])
    supervisor.connect(boxsensor, name="boxsensor", window=1, blocking=False, jitter=jitter,
                       delay_sim=delays_sim["inputs"]["supervisor"]["boxsensor"], delay=delays["inputs"]["supervisor"]["boxsensor"])
    planner.connect(armsensor, name="armsensor", window=4, blocking=False, jitter=jitter,
                    delay_sim=delays_sim["inputs"]["planner"]["armsensor"], delay=delays["inputs"]["planner"]["armsensor"])
    planner.connect(boxsensor, name="boxsensor", window=1, blocking=False, jitter=jitter,
                    delay_sim=delays_sim["inputs"]["planner"]["boxsensor"], delay=delays["inputs"]["planner"]["boxsensor"])
    controller.connect(planner, name="planner", window=win_planner, blocking=False, jitter=jitter, skip=False,
                       delay_sim=delays_sim["inputs"]["controller"]["planner"], delay=delays["inputs"]["controller"]["planner"])
    armactuator.connect(controller, name="controller", window=1, blocking=True, jitter=jitter,
                        delay_sim=delays_sim["inputs"]["armactuator"]["controller"], delay=delays["inputs"]["armactuator"]["controller"])

    # Define viewer
    if viewer:
        viewer = mjx.Viewer(xml_path=config["viewer"]["xml_path"], name="viewer", rate=rates["viewer"], scheduling=scheduling, advance=False,
                            delay=delays["step"]["viewer"], delay_sim=delays_sim["step"]["viewer"])
        nodes["viewer"] = viewer

        viewer.connect(supervisor, name="supervisor", window=1, blocking=False, jitter=jitter,
                       delay_sim=delays_sim["inputs"]["viewer"]["supervisor"], delay=delays["inputs"]["viewer"]["supervisor"])
        viewer.connect(armsensor, name="armsensor", window=1, blocking=False, jitter=jitter,
                       delay_sim=delays_sim["inputs"]["viewer"]["armsensor"], delay=delays["inputs"]["viewer"]["armsensor"])
        viewer.connect(boxsensor, name="boxsensor", window=1, blocking=False, jitter=jitter,
                       delay_sim=delays_sim["inputs"]["viewer"]["boxsensor"], delay=delays["inputs"]["viewer"]["boxsensor"])

    # Create environment
    env_name = [name]
    env_name.append(f"{'real' if real else 'sim'}")
    env_name.append(JITTER_MODES[jitter])
    env_name.append(SCHEDULING_MODES[scheduling])
    env_name = "_".join(env_name)
    graph = AsyncGraph(nodes, root=supervisor, clock=clock, real_time_factor=real_time_factor)
    env = Vx300sEnv(graph, max_steps=max_steps, name=env_name)
    return env


def eval_env(env: BaseEnv,
             policy: Callable,
             n_eval_episodes: int = 10,
             record_settings: Dict[str, Dict[str, bool]] = None,
             seed: int = None,
             progress_bar: bool = True) -> log_pb2.ExperimentRecord:
    """
    Evaluate an environment using a model.

    :param env: The environment
    :param policy: A policy to evaluate
    :param n_eval_episodes: Number of episodes to evaluate the root
    :param verbose: Whether to print the evaluation results
    :param record_settings: What to record. If None, all is recorded, except the step_states.
    :return: (mean episode reward, std of the reward)
    """

    # Update record settings
    # episode_records = log_pb2.ExperimentRecord(environment=pickle.dumps(env.unwrapped))
    episode_records = log_pb2.ExperimentRecord()
    record_settings = record_settings or {}
    _record_settings = {}
    for name in env.graph.nodes_and_root.keys():
        # Set default settings
        _record_settings[name] = dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False)
        # Get user provided settings
        node_settings = record_settings.setdefault(name, _record_settings[name])
        # Update default settings with user provided settings
        _record_settings[name].update(**node_settings)

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

    rng = jumpy.random.PRNGKey(seed) if seed is not None else jumpy.random.PRNGKey(0)

    episode_rewards = []
    for i in range(n_eval_episodes):
        rng, rng_eps = jumpy.random.split(rng)
        episode_rewards.append(0.0)
        graph_state, obs, info = env.reset(rng=rng_eps)
        steps, done = 0, False
        pbar = tqdm.tqdm(total=env.max_steps, desc=f"Episode {i+1}/{n_eval_episodes}", disable=not progress_bar)
        info["eps_rwd"] = episode_rewards[-1]
        pbar.set_postfix_str(format_info(info))
        pbar.refresh()
        while not done:
            action = policy(obs)
            graph_state, obs, reward, terminated, truncated, info = env.step(graph_state, action)
            done = terminated or truncated
            steps += 1
            episode_rewards[-1] += reward
            info["eps_rwd"] = episode_rewards[-1]
            pbar.set_postfix_str(format_info(info))
            pbar.update(1)
            if done:
                # Stop environment
                env.stop()

                # Save record
                node_records = [node.record(**_record_settings[name]) for name, node in env.graph.nodes_and_root.items()]
                episode_records.episode.append(log_pb2.EpisodeRecord(node=node_records))
                pbar.close()
                break
    # Compute mean reward over the last episodes
    mean_reward = onp.mean(episode_rewards)
    std_reward = onp.std(episode_rewards)
    if progress_bar:
        print(f"{env.name} | mean reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return episode_records, episode_rewards


def make_delay_distributions(record: Union[log_pb2.ExperimentRecord],
                             num_steps: int = 100,
                             num_components: int = 2,
                             step_size: float = 0.05,
                             seed: int = 0):
    # Prepare data
    if isinstance(record, log_pb2.ExperimentRecord):
        data, info = utils.get_delay_data(record, concatenate=True)
    # data = tree_map(lambda *x: jp.concatenate(x, axis=0), *record._delays)
    else:
        raise NotImplementedError

    def init_estimator(x, i):
        name = i.name if not isinstance(i, tuple) else f"{i[0].name}.input({i[1].name})"
        est = GMMEstimator(x, name)
        return est

    # Initialize estimators
    est = jax.tree_map(lambda x, i: init_estimator(x, i), data, info)

    # Fit estimators
    jax.tree_map(lambda e: e.fit(num_steps=num_steps, num_components=num_components, step_size=step_size, seed=seed), est)

    # Get distributions
    dist = jax.tree_map(lambda e: e.get_dist(include_data=True), est)
    return data, info, est, dist


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

    # # Pop world from
    # [_d["inputs"]["agent"].pop("last_action", None) for _d in [data, info, est, dist]]
    # [_d["inputs"]["sensor"].pop("world", None) for _d in [data, info, est, dist]]
    # [_d["inputs"].pop("world", None) for _d in [data, info, est, dist]]
    # [_d["step"].pop("world", None) for _d in [data, info, est, dist]]

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


def show_computation_graph(G: nx.DiGraph, root: str = None, xmax: float = 2.0, draw_pruned: bool = False, ax=None) -> Tuple[
    plt.Figure, plt.Axes]:
    order = ["planner", "controller", "armactuator", "world", "armsensor", "boxsensor", "supervisor", "viewer", "cost"]
    cscheme = {"world": "gray", "armsensor": "grape", "boxsensor": "grape", "supervisor": "teal", "viewer": "teal",
               "planner": "indigo", "controller": "orange", "armactuator": "orange", "cost": "yellow"}

    # Create new plot
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 5)
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, xmax])
    else:
        fig = ax.get_figure()

    plot_computation_graph(ax, G, root=root, order=order, cscheme=cscheme, xmax=xmax, node_size=200,
                           draw_pruned=draw_pruned, draw_nodelabels=True, node_labeltype="seq", connectionstyle="arc3,rad=0.1")

    # Plot legend
    if ax is None:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        by_label = dict(sorted(by_label.items()))
        ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
                  bbox_to_anchor=(1.0, 0.50))

    return fig, ax


def show_graph(episode_record: log_pb2.EpisodeRecord, pos: Dict[str, Tuple[float]] = None, cscheme: Dict[str, str] = None) -> \
        Tuple[plt.Figure, plt.Axes]:
    cscheme = cscheme or {"world": "gray", "armsensor": "grape", "boxsensor": "grape", "planner": "indigo", "controller": "orange",
               "armactuator": "orange", "supervisor": "teal", "viewer": "teal"}
    pos = pos or {"world": (0, 0), "armsensor": (1.5, 1.5), "boxsensor": (1.5, -1.5), "supervisor": (3, 3), "viewer": (4.5, 3),
                  "planner": (3, 0), "controller": (4.5, 0), "armactuator": (6.0, 0)}

    # Create new plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax.set(facecolor=oc.ccolor("gray"))

    # Draw graph
    plot_graph(ax, episode_record, cscheme=cscheme, pos=pos)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))
    return fig, ax


def show_communication(record: log_pb2.EpisodeRecord) -> Tuple[plt.Figure, plt.Axes]:
    # Reformat record
    d = {n.info.name: n for n in record.node}

    # Create new plots
    fig, ax = plt.subplots()
    xlim = [-0.001, 1.0]
    ax.set(ylim=[-18, 95], xlim=xlim, yticks=[], facecolor=oc.ccolor("gray"))
    ystart, dy, margin = 90, -10, 4

    # Plot all thread traces
    # ystart = plot_input_thread(ax, d["world"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["world"], ystart=ystart, dy=dy)

    # ystart = plot_input_thread(ax, d["sensor"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["armsensor"], ystart=ystart, dy=dy)
    ystart = plot_event_thread(ax, d["boxsensor"], ystart=ystart, dy=dy)

    # idx = len(d["agent"].inputs) - 1
    # ystart = plot_input_thread(ax, d["agent"].inputs[idx], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["supervisor"], ystart=ystart, dy=dy)
    ystart = plot_event_thread(ax, d["planner"], ystart=ystart, dy=dy)
    ystart = plot_event_thread(ax, d["controller"], ystart=ystart, dy=dy)

    # ystart = plot_input_thread(ax, d["actuator"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["armactuator"], ystart=ystart, dy=dy)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))
    return fig, ax


def get_default_distributions() -> Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]]:
    delays_sim = dict(step={}, inputs={})
    for n in ["world", "supervisor", "planner", "controller", "armactuator", "armsensor", "boxsensor", "viewer"]:
        delays_sim["step"][n] = Gaussian(0.)
        delays_sim["inputs"][n] = {}
    delays_sim["step"]["world"] = Gaussian(0.)
    delays_sim["step"]["armactuator"] = Gaussian(0.)
    delays_sim["step"]["armsensor"] = Gaussian(0.005)
    delays_sim["step"]["boxsensor"] = Gaussian(0.05)
    delays_sim["step"]["controller"] = Gaussian(0.03)
    delays_sim["step"]["planner"] = Gaussian(0.18)
    delays_sim["step"]["supervisor"] = Gaussian(0.01)
    delays_sim["step"]["viewer"] = Gaussian(0.01)
    delays_sim["inputs"]["world"]["armactuator"] = Gaussian(0.01)
    delays_sim["inputs"]["armactuator"]["controller"] = Gaussian(0.002)
    delays_sim["inputs"]["controller"]["planner"] = Gaussian(0.002)
    delays_sim["inputs"]["armsensor"]["world"] = Gaussian(0.01)
    delays_sim["inputs"]["boxsensor"]["world"] = Gaussian(0.05)
    delays_sim["inputs"]["supervisor"]["armsensor"] = Gaussian(0.002)
    delays_sim["inputs"]["supervisor"]["boxsensor"] = Gaussian(0.002)
    delays_sim["inputs"]["viewer"]["armsensor"] = Gaussian(0.002)
    delays_sim["inputs"]["viewer"]["boxsensor"] = Gaussian(0.002)
    delays_sim["inputs"]["viewer"]["supervisor"] = Gaussian(0.002)
    delays_sim["inputs"]["planner"]["armsensor"] = Gaussian(0.002)
    delays_sim["inputs"]["planner"]["boxsensor"] = Gaussian(0.002)
    return delays_sim


def get_nodelay_distributions() -> Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]]:
    delays_sim = dict(step={}, inputs={})
    for n in ["world", "supervisor", "planner", "controller", "armactuator", "armsensor", "boxsensor", "viewer"]:
        delays_sim["step"][n] = Gaussian(0.)
        delays_sim["inputs"][n] = {}
    delays_sim["step"]["world"] = Gaussian(0.)
    delays_sim["step"]["armactuator"] = Gaussian(0.)
    delays_sim["step"]["armsensor"] = Gaussian(0.0)
    delays_sim["step"]["boxsensor"] = Gaussian(0.0)
    delays_sim["step"]["controller"] = Gaussian(0.0)
    delays_sim["step"]["planner"] = Gaussian(0.0)
    delays_sim["step"]["supervisor"] = Gaussian(0.0)
    delays_sim["step"]["viewer"] = Gaussian(0.0)
    delays_sim["inputs"]["world"]["armactuator"] = Gaussian(0.0)
    delays_sim["inputs"]["armactuator"]["controller"] = Gaussian(0.0)
    delays_sim["inputs"]["controller"]["planner"] = Gaussian(0.0)
    delays_sim["inputs"]["armsensor"]["world"] = Gaussian(0.0)
    delays_sim["inputs"]["boxsensor"]["world"] = Gaussian(0.0)
    delays_sim["inputs"]["supervisor"]["armsensor"] = Gaussian(0.0)
    delays_sim["inputs"]["supervisor"]["boxsensor"] = Gaussian(0.0)
    delays_sim["inputs"]["viewer"]["armsensor"] = Gaussian(0.0)
    delays_sim["inputs"]["viewer"]["boxsensor"] = Gaussian(0.0)
    delays_sim["inputs"]["viewer"]["supervisor"] = Gaussian(0.0)
    delays_sim["inputs"]["planner"]["armsensor"] = Gaussian(0.0)
    delays_sim["inputs"]["planner"]["boxsensor"] = Gaussian(0.0)
    return delays_sim


def show_box_pushing_experiment(record: log_pb2.EpisodeRecord, xml_path: str, plot_ee: bool = True, plot_jpos: bool = True, plot_cost: bool = True):

    @struct.dataclass
    class EEPose:
        eepos: jp.ndarray
        eeorn: jp.ndarray

    def get_ee_pose(sys: System, jpos: jp.ndarray) -> EEPose:
        # Set
        qpos = jp.concatenate([sys.init_q[:6], jpos, jp.array([0])])
        pipeline_state = s_pipeline.init(sys, qpos, jp.zeros_like(sys.init_q))
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

    CSCHEME = {"planner": "indigo", "controller": "violet", "armactuator": "grape", "armsensor": "pink"}
    CSCHEME.update({"cm": "blue", "cost": "red", "cost_orn": "pink", "cost_down": "grape", "cost_align": "violet",
                    "cost_height": "indigo", "cost_near": "blue", "cost_dist": "cyan"})
    ECOLOR, FCOLOR = oc.cscheme_fn(CSCHEME)

    # Get data
    helper = exp.RecordHelper(record)
    timestamps = helper._timestamps[0]
    data = helper._data[0]
    data = jax.tree_util.tree_map(lambda x: x.tree if hasattr(x, "tree") else None, data)

    # Get jit functions
    jit_vmap_get_ee_pose = jax.jit(jax.vmap(get_ee_pose, in_axes=(None, 0)))
    jit_vmap_get_global_plan = jax.jit(get_global_plan)
    jit_vmap_get_next_jpos = jax.jit(jax.vmap(get_next_jpos, in_axes=(None, 0)))
    jit_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
    jit_vmap_cost_fn = jax.jit(jax.vmap(envs.vx300s.planner.cost.box_pushing_cost, in_axes=(None, 0, 0, None, 0)))

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

        ax_err.plot(timestamps_interp, jnp.abs(planner_ee_error[:, ee_idx])*100, label=f"planner", color=ECOLOR["planner"])
        ax_err.plot(timestamps_interp, jnp.abs(controller_ee_error[:, ee_idx])*100, label=f"controller", color=ECOLOR["controller"])
        ax_err.plot(timestamps_interp, jnp.abs(armactuator_ee_error[:, ee_idx])*100, label=f"armactuator", color=ECOLOR["armactuator"])
        ax_err.set_ylim([0, 7])
        ax_err.set_xlabel("time (s)")
    axes_ee[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    axes_ee[0, 1].set_ylabel("pos (cm)")
    axes_ee[1, 0].set_ylabel("error (cm)")
    axes_ee[1, 0].set_xlabel("time (s)")
    axes_ee[1, 0].set_ylim([0, 7])
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
        ax_err.set_ylim([0, 0.3])
        ax_err.set_xlabel("time (s)")

        # Plot planner output
        ax_jpos.plot(global_plan.timestamps, planner_jpos[:, joint_idx], label=f"planner", color=ECOLOR["planner"])
        ax_err.plot(timestamps_interp, jnp.abs(planner_jpos_error[:, joint_idx]), label=f"planner", color=ECOLOR["planner"])

        # Plot controller output
        controller_jpos = data["controller"]["outputs"].jpos
        controller_ts = timestamps["controller"]["ts_output"]
        ax_jpos.plot(controller_ts, controller_jpos[:, joint_idx], label=f"controller", color=ECOLOR["controller"])
        ax_err.plot(timestamps_interp, jnp.abs(controller_jpos_error[:, joint_idx]), label=f"controller", color=ECOLOR["controller"])

        # Plot controller output
        armactuator_jpos = data["armactuator"]["outputs"].jpos
        armactuator_ts = timestamps["controller"]["ts_output"]
        ax_jpos.plot(armactuator_ts, armactuator_jpos[:, joint_idx], label=f"armactuator", color=ECOLOR["armactuator"])
        ax_err.plot(timestamps_interp, jnp.abs(armactuator_jpos_error[:, joint_idx]), label=f"armactuator", color=ECOLOR["armactuator"])

        # Plot armsensor output
        armsensor_jpos = data["armsensor"]["outputs"].jpos
        armsensor_ts = timestamps["armsensor"]["ts_output"]
        ax_jpos.plot(armsensor_ts, armsensor_jpos[:, joint_idx], label=f"armsensor", color=ECOLOR["armsensor"])
    axes_jpos[0, -1].legend(ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))
    axes_jpos[0, 0].set_ylabel("joint position (rad)")
    axes_jpos[1, 0].set_ylabel("error (rad)")
    return fig_ee, fig_jpos, fig_cost


def show_box_pushing_performance(records: Dict[str, log_pb2.ExperimentRecord], xml_path: str):
    DATA, TIMESTAMPS, DELAYS, DATA_INTERP, TIMESTAMPS_INTERP = process_box_pushing_performance_data(records, xml_path)

    # Get colors
    CWHEEL = itertools.cycle({c for c in oc.CWHEEL.keys() if c not in ["white", "black", "gray"]})
    CSCHEME = {k: next(CWHEEL) for k in DATA_INTERP.keys()}
    ECOLOR, FCOLOR = oc.cscheme_fn(CSCHEME)

    # Prepare dfs
    perf_df = []
    for k, v in DATA_INTERP.items():
        timestamps_tiled = onp.tile(TIMESTAMPS_INTERP, v["cost"].shape[0])
        for t, cost, cm in zip(timestamps_tiled, onp.array(v["cost"]).flatten(), onp.array(v["cm"]).flatten()):
            # For each cost, create a dictionary with 'key' and 'cost'
            perf_df.append({"experiment": k, "time": t, "cost": cost, "cm": cm})

    # Convert the list of dictionaries to a DataFrame
    perf_df = pd.DataFrame(perf_df)

    # Plot cost
    fig_perf, axes_perf = plt.subplots(1, 2, figsize=(10, 5))
    sns.lineplot(ax=axes_perf[0], data=perf_df, x="time", y="cm", palette=ECOLOR, hue="experiment",
                 errorbar="sd")  # Distance plot
    sns.lineplot(ax=axes_perf[1], data=perf_df, x="time", y="cost", palette=ECOLOR, hue="experiment",
                 errorbar="sd")  # Distance plot
    axes_perf[0].set_title("Distance")
    axes_perf[0].set_ylabel("cm")
    axes_perf[0].set_xlabel("time (s)")
    axes_perf[1].set_title("Cost")
    axes_perf[1].set_ylabel("cost")
    axes_perf[1].set_xlabel("time (s)")
    return fig_perf


def process_box_pushing_performance_data(records: Dict[str, log_pb2.ExperimentRecord], xml_path: str, step_size: float = 0.05, max_time: float = 100.0):
    records = records if isinstance(records, dict) else {"1": records}

    @struct.dataclass
    class EEPose:
        eepos: jp.ndarray
        eeorn: jp.ndarray

    def get_ee_pose(sys: System, jpos: jp.ndarray) -> EEPose:
        # Set
        qpos = jp.concatenate([sys.init_q[:6], jpos, jp.array([0])])
        pipeline_state = s_pipeline.init(sys, qpos, jp.zeros_like(sys.init_q))
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

    DATA = {}
    DELAYS = {}
    TIMESTAMPS = {}
    ts = []
    for name, record in records.items():
        # Get data
        helper = exp.RecordHelper(record, method="truncated")

        # Interpolate all data to the same timestamps.
        DATA[name] = helper._data_stacked
        TIMESTAMPS[name] = helper._timestamps_stacked
        DELAYS[name] = helper._delays_stacked

        # Get max timestamps
        ts.append(min(helper._timestamps_stacked["armsensor"]["ts_output"].max(),
                      helper._timestamps_stacked["boxsensor"]["ts_output"].max()))

    # Get jit functions
    jit_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
    jit_vmap_get_ee_pose = jax.jit(jax.vmap(get_ee_pose, in_axes=(None, 0)))
    jit_vmap_cost_fn = jax.jit(jax.vmap(planner.cost.box_pushing_cost, in_axes=(None, 0, 0, None, 0)))

    # Load system
    m = mjcf.load(xml_path)

    # Interpolate all data to timestamps of the shortest experiment
    # print(f"Interpolating to {min(ts)} seconds: {ts}")
    ts = min(ts)
    ts = min(ts, max_time)
    TIMESTAMPS_INTERP = onp.arange(0, ts, step_size)

    # Interpolate
    DATA_INTERP = {}
    for name, data in DATA.items():
        timestamps = TIMESTAMPS[name]
        num_eps = timestamps["armsensor"]["ts_output"].shape[0]

        # Store all of the below in a dict
        eps_data = []
        for eps_idx in range(num_eps):
            jpos_target = jit_vmap_interp(TIMESTAMPS_INTERP, timestamps["armactuator"]["ts_output"][eps_idx],
                                          data["armactuator"]["outputs"].jpos[eps_idx])
            jpos = jit_vmap_interp(TIMESTAMPS_INTERP, timestamps["armsensor"]["ts_output"][eps_idx],
                                   data["armsensor"]["outputs"].jpos[eps_idx])
            boxpos = jit_vmap_interp(TIMESTAMPS_INTERP, timestamps["boxsensor"]["ts_output"][eps_idx],
                                     data["boxsensor"]["outputs"].boxpos[eps_idx])
            ee_pose = jit_vmap_get_ee_pose(m, jpos)
            ee_pose_target = jit_vmap_get_ee_pose(m, jpos_target)
            cost_params = jax.tree_util.tree_map(lambda x: x[eps_idx, 0],
                                                 data["planner"]["step_states"].params.cost_params)
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

    # Convert all data to numpy
    DATA = jax.tree_util.tree_map(lambda x: onp.array(x), DATA)
    DATA_INTERP = jax.tree_util.tree_map(lambda x: onp.array(x), DATA_INTERP)
    return DATA, TIMESTAMPS, DELAYS, DATA_INTERP, TIMESTAMPS_INTERP
