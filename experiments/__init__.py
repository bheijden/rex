# Script to define environments from
# Automatically register all environments in the envs folder
# Script to fit delays --> Save as proto for either ode or real
# Script to evaluate performance
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
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.utils import obs_as_tensor as sb3_obs_as_tensor
    import stable_baselines3.common.on_policy_algorithm

    def obs_as_tensor(obs, device):
        """Monkeypatch needed to convert jax arrays to pytorch tensors in stable-baselines3."""
        if isinstance(obs, jax.numpy.ndarray):
            np_obs = onp.asarray(obs)
            return sb3_obs_as_tensor(np_obs, device)
        else:
            return sb3_obs_as_tensor(obs, device)

    stable_baselines3.common.on_policy_algorithm.obs_as_tensor = obs_as_tensor
except ImportError:
    print(f"Could not import stable_baselines3. Some functionality will be missing.")
    BaseAlgorithm = None
    Monitor = None
    VecMonitor = None
    sb3_obs_as_tensor = None

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
    ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, INFO
import rex.open_colors as oc
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper

from envs.pendulum.env import PendulumEnv, Agent
import envs.pendulum.dists
import envs.pendulum.models
import envs.pendulum.ode as ode


def make_env(delays_sim: Dict[str, Dict[str, Union[Distribution, Dict[str, Distribution]]]],
             delay_fn: Callable[[Distribution], float],
             rates: Dict[str, float],
             blocking: bool = True,
             advance: bool = False,
             win_action: int = 1,
             win_obs: int = 2,
             scheduling: int = PHASE,
             jitter: int = BUFFER,
             env_fn: Callable = ode.build_pendulum,
             env_cls: Type[BaseEnv] = BaseEnv,
             name: str = "disc-pendulum",
             eval_env: bool = False,
             max_steps: int = 100,
             clock: int = WALL_CLOCK,
             real_time_factor: int = REAL_TIME,
             use_delays: bool = True) -> BaseEnv:
    # Override delays
    delays_sim["step"]["world"] = Gaussian(0.)
    delays_sim["inputs"]["world"]["actuator"] = Gaussian(0.)
    delays_sim["inputs"]["sensor"]["world"] = Gaussian(0.)
    delays_sim = tree_map(lambda d: Gaussian(0), delays_sim) if not use_delays else delays_sim
    delays = jax.tree_map(delay_fn, delays_sim)

    # Determine whether we use the real system
    real = True if "real" in env_fn.__module__ else False

    # Build nodes
    nodes = env_fn(rates, delays_sim, delays, scheduling=scheduling, advance=advance)
    world, actuator, sensor = nodes["world"], nodes["actuator"], nodes["sensor"]

    # Set eval
    assert hasattr(world, "eval_env"), "World node must have an eval_env attribute"
    world.eval_env = eval_env

    # Define root
    agent = env_cls.root_cls("agent", rate=rates["agent"], advance=True,
                             delay_sim=delays_sim["step"]["agent"], delay=delays["step"]["agent"])
    nodes["agent"] = agent

    # Connect
    if win_action > 0:
        agent.connect(agent, name="last_action", window=win_action, blocking=True, jitter=LATEST, skip=True)
    agent.connect(sensor, name="state", window=win_obs, blocking=blocking, jitter=jitter,
                  delay_sim=delays_sim["inputs"]["agent"]["state"], delay=delays["inputs"]["agent"]["state"])
    actuator.connect(agent, name="action", window=1, blocking=blocking, jitter=jitter,
                     delay_sim=delays_sim["inputs"]["actuator"]["action"], delay=delays["inputs"]["actuator"]["action"])

    # Create environment
    env_name = [name]
    env_name.append(f"{'real' if real else 'ode'}")
    env_name.append("eval" if eval_env else "train")
    env_name.append(JITTER_MODES[jitter])
    env_name.append(SCHEDULING_MODES[scheduling])
    env_name.append(f"awin{win_action}")
    env_name.append(f"owin{win_obs}")
    env_name.append("blocking" if blocking else "nonblocking")
    env_name.append("advance" if advance else "noadvance")
    env_name = "_".join(env_name)
    graph = AsyncGraph(nodes, root=agent, clock=clock, real_time_factor=real_time_factor)
    env = env_cls(graph, max_steps=max_steps, name=env_name)
    return env


def make_compiled_env(env: PendulumEnv,
                      records: Union[log_pb2.EpisodeRecord, List[log_pb2.EpisodeRecord]],
                      eval_env: bool = False,
                      max_steps: int = 100,
                      supergraph_mode: str = "MCS",
                      progress_bar: bool = False,
                      nodes_from: str = "records",
                      ) -> PendulumEnv:
    records = records if isinstance(records, (list, RepeatedCompositeContainer)) else [records]

    # Re-initialize nodes
    if nodes_from == "records":
        nodes: Dict[str, Union[Node, Agent]] = {}
        for n in records[-1].node:
            nodes[n.info.name] = pickle.loads(n.info.state)
        [n.unpickle(nodes) for n in nodes.values()]
    elif nodes_from == "env":
        nodes = env.nodes_world_and_agent
    else:
        raise ValueError(f"Unknown nodes_from {nodes_from}")

    # Grab root
    agent: Agent = nodes["agent"]

    # Grab world
    world: Node = nodes["world"]
    assert hasattr(world, "eval_env"), "World node must have an eval_env attribute"
    world.eval_env = eval_env

    # Trace record
    record_network, S, _, Gs, Gs_monomorphism = supergraph.get_network_record(records, "agent", supergraph_mode=supergraph_mode, progress_bar=progress_bar)
    timings = supergraph.get_timings_from_network_record(record_network, Gs, Gs_monomorphism)

    # Create env
    _name = env.name
    _name = _name.replace("train", "eval" if eval_env else "train")
    _name = _name.replace("eval", "eval" if eval_env else "train")
    env_name = [_name]
    env_name.append("compiled")
    env_name = "_".join(env_name)

    graph = CompiledGraph(nodes, agent, S, default_timings=timings)
    env = env.__class__(graph, max_steps, name=env_name)
    return env


def show_grouped(record: log_pb2.NodeRecord, input_name) -> Tuple[plt.Figure, plt.Axes]:
    # Create new plot
    xlim = [-0.001, 0.3]
    fig, ax = plt.subplots()
    ax.set(ylim=xlim, xlim=xlim, yticks=[], facecolor=oc.ccolor("gray"))

    # Function arguments
    plot_grouped(ax, record, input_name)

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
    xlim = [-0.001, 0.3]
    ax.set(ylim=[-18, 95], xlim=xlim, yticks=[], facecolor=oc.ccolor("gray"))
    ystart, dy, margin = 90, -10, 4

    # Plot all thread traces
    ystart = plot_input_thread(ax, d["world"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["world"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["sensor"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["sensor"], ystart=ystart, dy=dy)

    idx = len(d["agent"].inputs) - 1
    ystart = plot_input_thread(ax, d["agent"].inputs[idx], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["agent"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["actuator"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["actuator"], ystart=ystart, dy=dy)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))
    return fig, ax


def show_computation_graph(G: nx.DiGraph, S: nx.DiGraph, root: str, plot_type: str = "computation", xmax: float = 0.6, supergraph_mode="MCS") -> Tuple[
    plt.Figure, plt.Axes]:
    order = ["world", "sensor", "agent", "actuator"]
    cscheme = {"world": "gray", "sensor": "grape", "agent": "teal", "actuator": "indigo", "render": "yellow", "estimator": "orange"}

    # Create new plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)

    if plot_type == "computation":
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
        plot_computation_graph(ax, G, root=root, order=order, cscheme=cscheme, xmax=xmax, node_size=200,
                               draw_pruned=True, draw_nodelabels=True, node_labeltype="seq", connectionstyle="arc3,rad=0.1")
    elif plot_type == "topological":
        raise NotImplementedError("Not refactored since supergraph")
        # Create new plot
        # ax.set(facecolor=oc.ccolor("gray"), xlabel="Topological order", yticks=[], xlim=[-1, 20])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plot_topological_order(ax, G, root=root, xmax=xmax, cscheme=cscheme, node_labeltype="seq", draw_excess=True, draw_root_excess=False)
    elif plot_type == "depth":
        raise NotImplementedError("Not refactored since supergraph")
        # ax.set(facecolor=oc.ccolor("gray"), xlabel="Depth order", yticks=[], xlim=[-1, 10])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plot_depth_order(ax, G, root=root, MCS=MCS, xmax=xmax, cscheme=cscheme, split_mode=split_mode, supergraph_mode=supergraph_mode, node_labeltype="seq", draw_excess=True)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))

    return fig, ax


def show_graph(episode_record: log_pb2.EpisodeRecord, pos: Dict[str, Tuple[float]] = None, cscheme: Dict[str, str] = None) -> \
        Tuple[plt.Figure, plt.Axes]:
    cscheme = cscheme or {"world": "gray", "sensor": "grape", "agent": "teal", "actuator": "indigo"}
    pos = pos or {"world": (0, 0), "sensor": (1.5, 0), "agent": (3, 0), "actuator": (4.5, 0), "render": (1.5, 1.5)}

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


def eval_env(env: BaseEnv,
             policy: Callable[[jax.typing.ArrayLike], jax.Array],
             n_eval_episodes: int = 10,
             verbose: bool = False,
             record_settings: Dict[str, Dict[str, bool]] = None,
             seed: int = None,
             return_rewards: bool = False) -> log_pb2.ExperimentRecord:
    """
    Evaluate an environment using a model.

    :param env: The environment
    :param policy: A policy to evaluate
    :param n_eval_episodes: Number of episodes to evaluate the root
    :param verbose: Whether to print the evaluation results
    :param record_settings: What to record. If None, all is recorded, except the step_states.
    :return: (mean episode reward, std of the reward)
    """
    # Wrap environment in gym wrapper.
    if not env.env_is_wrapped(GymWrapper):
        env = GymWrapper(env)
    env: GymWrapper

    # Set seed
    env.seed(seed)

    # Update record settings
    episode_records = log_pb2.ExperimentRecord(environment=pickle.dumps(env.unwrapped))
    record_settings = record_settings or {}
    _record_settings = {}
    for name in env.graph.nodes_and_root.keys():
        # Set default settings
        _record_settings[name] = dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True)
        # Get user provided settings
        node_settings = record_settings.setdefault(name, _record_settings[name])
        # Update default settings with user provided settings
        _record_settings[name].update(**node_settings)

    episode_rewards = []
    for _ in range(n_eval_episodes):
        episode_rewards.append(0.0)
        steps, done, (obs, info) = 0, False, env.reset()
        tstart = time.time()
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_rewards[-1] += reward
            if done:
                # Stop environment
                tend = time.time()
                env.stop()

                # Save record
                node_records = [node.record(**_record_settings[name]) for name, node in env.graph.nodes_and_root.items()]
                episode_records.episode.append(log_pb2.EpisodeRecord(node=node_records))

                # Print evaluation
                if verbose:
                    print(f"{env.name} | steps={steps} | fps={steps / (tend - tstart): 2.4f} | reward={episode_rewards[-1]}")
                break
    # Compute mean reward for the last 10 episodes
    mean_reward = onp.mean(episode_rewards)
    std_reward = onp.std(episode_rewards)
    if verbose:
        print(f"{env.name} | mean reward={mean_reward:.2f} +/- {std_reward:.2f}")
    if return_rewards:
        return episode_records, episode_rewards
    else:
        return episode_records


def make_policy(model: BaseAlgorithm, constant_action: float = None):
    """
    :param model: The RL model
    :param constant_action: Whether to overwrite the predicted action with a constant action.
    :return: A function that takes an observation (numpy array) and returns the action (numpy array)
    """

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        if constant_action is not None:
            action = jnp.ones(action.shape) * constant_action
        return action

    return policy


def make_delay_distributions(record: Union[log_pb2.ExperimentRecord],
                             num_steps: int = 100,
                             num_components: int = 2,
                             step_size: float = 0.05,
                             seed: int = 0):
    # Prepare data
    if isinstance(record, log_pb2.ExperimentRecord):
        data, info = utils.get_delay_data(record, concatenate=True)
    # data = tree_map(lambda *x: jnp.concatenate(x, axis=0), *record._delays)
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


def load_distributions(file: str, module: ModuleType = envs.pendulum.dists) -> Dict[
    str, Dict[str, Union[Distribution, Dict[str, Distribution]]]]:
    # Append pkl extension
    if not file.endswith(".pkl"):
        file = file + ".pkl"

    # If not an absolute path, load from within module
    if not file.startswith("/"):
        module_path = os.path.dirname(os.path.abspath(module.__file__))
        with open(f"{module_path}/{file}", "rb") as f:
            dists = pickle.load(f)
    else:
        with open(f"{file}", "rb") as f:
            dists = pickle.load(f)
    return dists


def load_model(file, model_cls: Type[BaseAlgorithm], module: ModuleType = envs.pendulum.models, fail: bool = False,
               **kwargs) -> BaseAlgorithm:
    # If not an absolute path, load from within module
    try:
        # Append zip extension
        if not file.endswith(".zip"):
            file = file + ".zip"

        if not file.startswith("/"):
            module_path = os.path.dirname(os.path.abspath(module.__file__))
            model = model_cls.load(f"{module_path}/{file}", **kwargs)
        else:
            model = model_cls.load(file, **kwargs)
    except (FileNotFoundError, ValueError, AttributeError) as e:
        if fail:
            raise e
        else:
            if "spaces" in e.__str__():
                print("Spaces don't match (action and/or observation). Loading a random model instead.")
            elif isinstance(e, FileNotFoundError):
                print(f"Could not load model from {file}. Loading a random model instead.")
            elif isinstance(e, AttributeError):
                print(
                    f"Could not load model from {file}. Are you loading an sb3 model into an sbx algo? Loading a random model instead.")
            model = model_cls("MlpPolicy", verbose=1, **kwargs)
    return model


class SysIdPolicy:
    def __init__(self, rate: float, duration: float = 5.0, min: float = -8, max: float = 8, seed: int = 0, model=None,
                 use_ros: bool = False):
        if use_ros:
            import rospy
            from std_msgs.msg import Float32
            rospy.init_node("pendulum_client", anonymous=True)
            self._msg_cls = Float32
            self._pub_action = rospy.Publisher("/sysid/action", Float32, queue_size=1)
        self._use_ros = use_ros
        self._rate = rate
        self._steps = 0
        self._model = model
        self._duration = duration
        self._last_time = 0
        self._max = max
        self._min = min
        self._action = jnp.array([max], dtype=jnp.float32)
        self._use_model = True
        self._rng = jax.random.PRNGKey(seed)

    def _wall_clock_switch(self):
        tstep = time.time()
        switch = tstep - self._last_time > self._duration
        if switch:
            self._last_time = tstep
        return switch

    def _step_switch(self):
        self._steps += 1
        switch = self._steps % (self._rate * self._duration) == 0
        return switch

    def predict(self, obs, deterministic: bool = True):
        # switch = self._wall_clock_switch()
        switch = self._step_switch()
        if switch:
            if self._model is not None and not self._use_model:
                self._use_model = True
            else:
                self._use_model = False
                self._rng, rng_action = jax.random.split(self._rng)
                action = jax.random.uniform(rng_action, minval=self._min, maxval=self._max)
                self._action = jnp.array([action], dtype=jnp.float32)
        if self._use_model and self._model is not None:
            self._action, _ = self._model.predict(obs, deterministic=deterministic)
            # policy_action, _ = self._model.predict(obs, deterministic=deterministic)
            # onp.array(policy_action)
        else:
            if self._model is not None:
                policy_action, _ = self._model.predict(obs, deterministic=deterministic)
                onp.array(policy_action)
        if self._use_ros:
            msg = self._msg_cls(self._action[0])
            self._pub_action.publish(msg)
        return self._action, None


import gymnasium as gym


class RecordEnv:
    id = "RecordEnv-v0"

    def __new__(cls, file: str, clock: int = None, real_time_factor: int = None, graph_type: int = None):
        # todo: re-apply wrappers (gymwrapper, vectorwrapper, jit, mp, etc...)
        # todo: overwrite params (CLOCK, RTF, GRAPH_TYPE)
        return BaseEnv.load(file)


# if RecordEnv.id not in gym.envs.registration.registry.env_specs:
if RecordEnv.id not in gym.envs.registration.registry:
    gym.envs.register(id=RecordEnv.id,
                      entry_point="experiments:RecordEnv",
                      order_enforce=False)


class RexVecEnv:
    def __new__(cls,
                env_id: Union[str, Callable[..., gym.Env]],
                n_envs: int = 1,
                seed: Optional[int] = None,
                start_index: int = 0,
                monitor_dir: Optional[str] = None,
                wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
                env_kwargs: Optional[Dict[str, Any]] = None,
                vec_env_cls: Optional[Type["RexVecEnv"]] = None,
                vec_env_kwargs: Optional[Dict[str, Any]] = None,
                monitor_kwargs: Optional[Dict[str, Any]] = None,
                wrapper_kwargs: Optional[Dict[str, Any]] = None, ):
        # Make env
        env = env_id(**env_kwargs)

        # Wrap model
        if env.graph_type in [INTERPRETED]:
            env = GymWrapper(env)

            # Monitor
            monitor_cls = Monitor
        else:  # compiled graph
            assert env._cgraph is not None, "Compiled graph is None"
            env = AutoResetWrapper(env)  # Wrap into auto reset wrapper
            env = VecGymWrapper(env, num_envs=n_envs)  # Wrap into vectorized environment

            # Jit
            env.jit()

            # Monitor
            monitor_cls = VecMonitor

        # Wrap the env in a Monitor wrapper
        # to have additional training information
        monitor_path = monitor_dir if monitor_dir is not None else None
        # Create the monitor folder if needed
        if monitor_path is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = monitor_cls(env, filename=monitor_path, **monitor_kwargs)  # Wrap into vectorized monitor

        # Seed
        if seed is not None:
            env.seed(seed)

        return env


# from stable_baselines3.common.utils import obs_as_tensor as sb3_obs_as_tensor





class _NoValue: pass


class _HasValue:
    def __init__(self, tree):
        self.tree = tree


def stack_padding(it):

    def resize(row, size):
        new = onp.array(row)
        new.resize(size, refcheck=False)
        return new

    # find longest row length
    row_length = max(it, key=len).shape
    mat = onp.array([resize(row, row_length) for row in it])

    return mat


def _padded_stack(*data: jax.typing.ArrayLike) -> Optional[jax.Array]:
    empties = [isinstance(d, _NoValue) for d in data]
    has_empty = any(empties)
    if has_empty:
        if all(empties):
            return None
        else:
            raise ValueError("Cannot stack partially empty data")
    elif all([isinstance(d, _HasValue) for d in data]):
        return tree_map(_padded_stack, *[d.tree for d in data])
    assert all([not isinstance(d, _HasValue) for d in data])

    # Stack data
    data_stacked = stack_padding(data)
    return data_stacked


def _truncated_stack(*data: jax.typing.ArrayLike) -> Optional[jax.Array]:
    has_empty = any([isinstance(d, _NoValue) for d in data])
    if has_empty:
        return None
    elif all([isinstance(d, _HasValue) for d in data]):
        return tree_map(_truncated_stack, *[d.tree for d in data])
    assert all([not isinstance(d, _HasValue) for d in data])

    # Determine min_length
    min_length = min([x.shape[0] for x in data if x.ndim > 0])

    # Truncate data to min_length
    data = tree_map(lambda d: d[:min_length], data)

    # Stack data
    data_stacked = tree_map(lambda *d: jnp.stack(d), *data)
    return data_stacked


class RecordHelper:
    def __init__(self, record: Union[log_pb2.ExperimentRecord, log_pb2.EpisodeRecord],
                 validate: bool = True, stack: bool = True, method: str = "padded"):
        self.record = record
        self.validate = validate
        self.stack = stack

        # Convert to experiment record
        if isinstance(record, log_pb2.EpisodeRecord):
            self._record = log_pb2.ExperimentRecord(episode=[record])
        else:
            self._record = record
        assert isinstance(self._record, log_pb2.ExperimentRecord), "Record must be an ExperimentRecord or EpisodeRecord"

        # Store preprocessed data in convenient format
        self._delays: List[Dict[str, Dict[str, Union[jax.typing.ArrayLike, Dict[str, jax.typing.ArrayLike]]]]]
        self._delays_stacked: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Dict[str, jax.typing.ArrayLike]]]]
        self._data: List[Dict[str, Dict[str, Any]]]
        self._data_stacked: Dict[str, Dict[str, Any]]
        self._nodes: List[Dict[str, Union[str, Node, Agent]]] = []

        # Pre-process record data
        self._preprocess_data()

        # Validate record
        if self.validate:
            self._validate_data()

        # Stack data
        if self.stack:
            assert method in ["truncated", "padded"], "Stacking method must be either 'truncated' or 'padded'"
            assert self.validate, "Stacking requires validation. Set validate=True."
            self._stack_data(method)

    def get_nodes(self, episode: int = -1) -> Dict[str, Union[Node, Agent]]:
        nodes = {}
        for name, node_bytes in self._nodes[episode].items():
            assert len(node_bytes) > 0, "Node state must be non-empty."
            nodes[name] = pickle.loads(node_bytes)

        # Fully restore node by unpickling (re-connects to other nodes, execute custom unpickling routines if any)
        for n in nodes.values():
            n.unpickle(nodes)
        return nodes

    def _unpickle_data(self, record: log_pb2.Serialization):
        if len(record.encoded_bytes) == 0:
            return _NoValue()
        encoded_bytes = record.encoded_bytes
        try:
            target = pickle.loads(record.target)
            data = [serialization.from_bytes(target, b) for b in encoded_bytes]
        except UnpicklingError as e:
            print(f"Failed to load target. Unpickling to state_dict instead: {e}")
            data = [serialization.msgpack_restore(b) for b in encoded_bytes]
        return _HasValue(tree_map(lambda *x: jnp.stack(x), *data))

    def _preprocess_data(self):
        # Get delays
        self._delays, _ = utils.get_delay_data(self._record, concatenate=False)

        # Get timestamps
        self._timestamps = utils.get_timestamps(self._record)

        # Get data
        self._data = []
        self._nodes = []
        for i, e in enumerate(self._record.episode):
            # Store nodes
            eps_nodes = {}
            self._nodes.append(eps_nodes)
            for n in e.node:
                eps_nodes[n.info.name] = n.info.state

            # Store data
            eps_data = {n.info.name: dict(outputs=None, rngs=None, states=None, params=None, step_states=None) for n in e.node}
            self._data.append(eps_data)
            for n in e.node:
                # Store outputs
                eps_data[n.info.name]["outputs"] = self._unpickle_data(n.outputs)
                eps_data[n.info.name]["rngs"] = self._unpickle_data(n.rngs)
                eps_data[n.info.name]["states"] = self._unpickle_data(n.states)
                eps_data[n.info.name]["params"] = self._unpickle_data(n.params)
                eps_data[n.info.name]["step_states"] = self._unpickle_data(n.step_states)

    def _validate_data(self):
        # todo: Do not raise errors, but rather set flags. Check flags in stack and truncate.
        # todo: check if all data is present
        # todo: Check if data is of same length?
        # todo: Check if computation graph is the same
        # todo: Check if all nodes are present
        # todo: Check if step_states correspond to logged inputs
        # Stack
        # self._lengths = tree_map(lambda *l: list(l), *self._lengths)
        # self._max_lengths = tree_map(lambda *l: max(l), *self._lengths
        pass

    def _stack_data(self, method: str):
        # Stack data
        if method == "truncated":
            self._data_stacked = tree_map(_truncated_stack, *self._data)
            self._delays_stacked = tree_map(_truncated_stack, *self._delays)
            self._timestamps_stacked = tree_map(_truncated_stack, *self._timestamps)
        elif method == "padded":
            self._data_stacked = tree_map(_padded_stack, *self._data)
            self._delays_stacked = tree_map(_padded_stack, *self._delays)
            self._timestamps_stacked = tree_map(_padded_stack, *self._timestamps)
        else:
            raise NotImplementedError


class RolloutWrapper(object):
    def __init__(self, env: BaseEnv, model_forward=None):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env = env
        self.model_forward = model_forward
        self.num_env_steps = self.env.max_steps

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout(self, rng_eval):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0,))
        return batch_rollout(rng_eval)

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout(self, rng_input):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_policy, rng_episode = jax.random.split(rng_input, num=3)
        state, obs, info = self.env.reset(rng_reset)

        if self.model_forward is not None:
            policy_state = self.model_forward.policy.actor_state
            # policy_state = self.model_forward.reset(rng_policy)
        else:
            policy_state = None

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_state, rng, cum_reward, valid_mask = state_input
            rng, rng_net = jax.random.split(rng, 2)
            # rng, rng_step = jax.random.split(rng, 2)
            # new_step = jax.random.choice(rng_step, self.num_env_steps, shape=(), replace=False)
            # state = state.replace(step=new_step)
            if self.model_forward is not None:
                scaled_action = self.model_forward.policy._predict(obs, deterministic=True)
                action = self.model_forward.policy.unscale_action(scaled_action)
            else:
                action = self.env.action_space().sample(rng_net)
            next_state, next_obs, reward, terminated, truncated, info = self.env.step(state, action)
            done = jnp.logical_or(terminated, truncated)
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_state,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by root in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state, info = self.env.reset(rng, self.env_params)
        return obs.shape
