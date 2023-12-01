import envs.vx300s.brax as brax
import envs.vx300s.mjx as mjx
import envs.vx300s.real as real
import envs.vx300s.env as env
import envs.vx300s.models as models
import envs.vx300s.dists as dists
import envs.vx300s.planner


# From experiments.__init__.py
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

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm

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
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper

from envs.vx300s.env import Vx300sEnv, Controller, Supervisor
import envs.vx300s.dists
import envs.vx300s.models


def make_env(delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
             delay_fn: Callable[[Distribution], float],
             rates: Dict[str, float],
             config: Dict[str, Dict[str, Any]],
             scheduling: int = PHASE,
             win_planner: int = 2,
             jitter: int = BUFFER,
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
    # delays_sim["step"]["world"] = Gaussian(0.)
    # delays_sim["step"]["armactuator"] = Gaussian(0.)
    # delays_sim["step"]["armsensor"] = Gaussian(0.)
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
    supervisor = Supervisor("supervisor", rate=rates["supervisor"], advance=True,
                            delay_sim=delays_sim["step"]["supervisor"], delay=delays["step"]["supervisor"])
    nodes["supervisor"] = supervisor

    # Define controller
    controller = Controller("controller", rate=rates["controller"], advance=False,
                            delay_sim=delays_sim["step"]["controller"], delay=delays["step"]["controller"])
    nodes["controller"] = controller

    # Define planner
    if config["planner"]["type"] == "brax":
        planner = envs.vx300s.planner.BraxCEMPlanner(name="planner", rate=rates["planner"], advance=True,
                                                     delay_sim=delays_sim["step"]["planner"], delay=delays["step"]["planner"],
                                                     mj_path=config["planner"]["brax_xml_path"], pipeline="generalized",
                                                     horizon=config["planner"]["horizon"], u_max=config["planner"]["u_max"],
                                                     dt=config["planner"]["dt"], dt_substeps=config["planner"]["dt_substeps"],
                                                     num_samples=config["planner"]["num_samples"],
                                                     max_iter=config["planner"]["max_iter"])
    elif config["planner"]["type"] == "rex":
        planner = envs.vx300s.planner.RexCEMPlanner(name="planner", rate=rates["planner"], advance=True,
                                                    nodes=nodes,
                                                    graph_path=config["planner"]["rex_graph_path"],
                                                    supergraph_mode=config["planner"]["supergraph_mode"],
                                                    delay_sim=delays_sim["step"]["planner"], delay=delays["step"]["planner"],
                                                    mj_path=config["planner"]["rex_xml_path"], pipeline="generalized",
                                                    horizon=config["planner"]["horizon"], u_max=config["planner"]["u_max"],
                                                    dt=config["planner"]["dt"], dt_substeps=config["planner"]["dt_substeps"],
                                                    num_samples=config["planner"]["num_samples"],
                                                    max_iter=config["planner"]["max_iter"])
    else:
        raise ValueError(f"Unknown planner type {config['planner']['type']}")
    nodes["planner"] = planner

    # Connect
    supervisor.connect(armsensor, name="armsensor", window=1, blocking=False, jitter=LATEST,
                       delay_sim=delays_sim["inputs"]["supervisor"]["armsensor"], delay=delays["inputs"]["supervisor"]["armsensor"])
    supervisor.connect(boxsensor, name="boxsensor", window=1, blocking=False, jitter=LATEST,
                       delay_sim=delays_sim["inputs"]["supervisor"]["boxsensor"], delay=delays["inputs"]["supervisor"]["boxsensor"])
    planner.connect(armsensor, name="armsensor", window=4, blocking=False, jitter=LATEST,
                    delay_sim=delays_sim["inputs"]["planner"]["armsensor"], delay=delays["inputs"]["planner"]["armsensor"])
    planner.connect(boxsensor, name="boxsensor", window=1, blocking=False, jitter=LATEST,
                    delay_sim=delays_sim["inputs"]["planner"]["boxsensor"], delay=delays["inputs"]["planner"]["boxsensor"])
    controller.connect(planner, name="planner", window=win_planner, blocking=False, jitter=LATEST, skip=False,
                       delay_sim=delays_sim["inputs"]["controller"]["planner"], delay=delays["inputs"]["controller"]["planner"])
    armactuator.connect(controller, name="controller", window=1, blocking=True, jitter=LATEST,
                        delay_sim=delays_sim["inputs"]["armactuator"]["controller"], delay=delays["inputs"]["armactuator"]["controller"])

    # Define viewer
    if viewer:
        viewer = mjx.Viewer(xml_path=config["viewer"]["xml_path"], name="viewer", rate=rates["viewer"], scheduling=scheduling, advance=False,
                            delay=delays["step"]["viewer"], delay_sim=delays_sim["step"]["viewer"])
        nodes["viewer"] = viewer

        viewer.connect(supervisor, name="supervisor", window=1, blocking=False, jitter=LATEST,
                       delay_sim=delays_sim["inputs"]["viewer"]["supervisor"], delay=delays["inputs"]["viewer"]["supervisor"])
        viewer.connect(armsensor, name="armsensor", window=1, blocking=False, jitter=LATEST,
                       delay_sim=delays_sim["inputs"]["viewer"]["armsensor"], delay=delays["inputs"]["viewer"]["armsensor"])
        viewer.connect(boxsensor, name="boxsensor", window=1, blocking=False, jitter=LATEST,
                       delay_sim=delays_sim["inputs"]["viewer"]["boxsensor"], delay=delays["inputs"]["viewer"]["boxsensor"])

    # Create environment
    env_name = [name]
    env_name.append(f"{'real' if real else 'brax'}")
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