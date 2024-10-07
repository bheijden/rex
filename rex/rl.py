from typing import Optional, Tuple, Union, Any, Sequence, Dict, Callable
from jax._src.typing import Array, ArrayLike, DTypeLike
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict
from functools import partial
# from gymnax.environments import environment, spaces
# from brax import envs
# from brax.envs.wrappers.training import EpisodeWrapper

from rex.graph import Graph
from rex import base


class Space:
    """
    Minimal jittable class for abstract space.
    """

    def sample(self, rng: ArrayLike) -> jax.Array:
        raise NotImplementedError

    def contains(self, x: Union[int, ArrayLike]) -> Union[bool, jax.Array]:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = int

    def sample(self, rng: ArrayLike) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n).astype(self.dtype)

    def contains(self, x: Union[int, ArrayLike]) -> Union[bool, jax.Array]:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """Minimal jittable class for array-shaped spaces."""

    def __init__(
        self,
        low: ArrayLike,
        high: ArrayLike,
        shape: Sequence[int] = None,
        dtype: DTypeLike = float,
    ):
        self.low = low
        self.high = high
        self.shape = shape if shape is not None else low.shape
        self.dtype = dtype

    def sample(self, rng: ArrayLike) -> jax.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)

    def contains(self, x: ArrayLike) -> Union[bool, jax.Array]:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return jnp.all(range_cond)


EnvState = base.GraphState
# Tuple of (graph_state, observation, info)
ResetReturn = Tuple[EnvState, jax.Array, Dict[str, Any]]
# Tuple of (graph_state, observation, reward, terminated, truncated, info)
StepReturn = Tuple[EnvState, jax.Array, Union[float, jax.Array], Union[bool, jax.Array], Union[bool, jax.Array], Dict[str, Any]]


class BaseEnv:
    def __init__(self, graph: Graph):
        self.graph = graph

    @property
    def max_steps(self) -> Union[int, jax.typing.ArrayLike]:
        """The maximum number of steps in the environment.

        Per default, this is the maximum number of steps the supervisor (i.e. agent) is stepped in the provided computation graph.
        You can override this property to provide a custom value (smaller than the default).
        This value is used as the episode length when evaluating the environment during training.
        """
        return self.graph.max_steps

    def observation_space(self, graph_state: base.GraphState) -> Box:
        raise NotImplementedError("Subclasses must implement this method.")

    def action_space(self, graph_state: base.GraphState) -> Box:
        raise NotImplementedError("Subclasses must implement this method.")

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        raise NotImplementedError("Subclasses must implement this method.")

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        raise NotImplementedError("Subclasses must implement this method.")


class Environment:
    def __init__(self, graph: Graph, params: Dict[str, base.Params] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        self.graph = graph
        self.params = params
        self.only_init = only_init
        self.starting_eps = starting_eps
        self.randomize_eps = randomize_eps
        self.order = order

    @property
    def max_steps(self) -> Union[int, jax.typing.ArrayLike]:
        return self.graph.max_steps

    def observation_space(self, graph_state: base.GraphState) -> Box:
        raise NotImplementedError("Subclasses must implement this method.")

    def action_space(self, graph_state: base.GraphState) -> Box:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_step_state(self, graph_state: base.GraphState, name: str = None) -> base.StepState:
        name = name if name is not None else self.graph.supervisor.name
        return graph_state.step_state.get(name, None)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        return {}

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")

    def update_graph_state_pre_step(self, graph_state: base.GraphState, action: jax.Array) -> base.GraphState:
        """Override this method if you want to update the graph state before graph.step(...).

        Note: This method is called before the graph is stepped, so after an action is provided to .step().
        :param graph_state: The current graph state.
        :param action: The action taken. Is always provided.
        """
        return graph_state

    def update_graph_state_post_step(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        """Override this method if you want to update the graph state after graph.step(...).

        Note: This method is called after the graph has been stepped (before returning from .step()),
              or before returning the initial observation from .reset().
        :param graph_state: The current graph state.
        :param action: The action taken. Is None when called from .reset().
        """
        return graph_state

    def init(self, rng: jax.Array = None) -> base.GraphState:
        """
        Initializes the graph state.
        Note: If only_init is True, the graph will only be initialized and starting_step will be 1 (instead of 0).
        This means that the first partition before the first supervisor will *not* be run.
        This may result in some initial messages not being correctly passed, as the first partition is skipped, hence
        the messages are not buffered.

        The advantage of this is that the first partition is not run, which avoids doubling the number of partitions
        that need to run when using auto resetting at every step without a fixed set of initial states.

        :param rng: Random number generator.
        :return: The initial graph state.
        """

        if self.only_init:
            gs = self.graph.init(rng, params=self.params, starting_step=1,  # Avoids running first partition
                                 starting_eps=self.starting_eps, randomize_eps=self.randomize_eps, order=self.order)
        else:
            gs = self.graph.init(rng, params=self.params, starting_step=0,
                                 starting_eps=self.starting_eps, randomize_eps=self.randomize_eps, order=self.order)
            gs, _ = self.graph.reset(gs)  # Run the first partition (excluding the supervisor)
        return gs

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        """
        Reset the environment.
        Can be overridden to provide custom reset behavior.

        :param rng: Random number generator. Used to initialize a new graph state.
        :return: Tuple of (graph_state, observation, info)
        """
        gs = self.init(rng)
        # Post-step update of graph_state
        gs_post = self.update_graph_state_post_step(gs, action=None)
        # Get observation
        obs = self.get_observation(gs_post)
        # Get info
        info = self.get_info(gs_post)
        return gs, obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        """
        Step the environment.
        Can be overridden to provide custom step behavior.

        :param graph_state: The current graph state.
        :param action: The action to take.
        :return: Tuple of (graph_state, observation, reward, terminated, truncated, info)
        """
        # Convert action to output
        output = self.get_output(graph_state, action)
        # Pre-step update of graph_state
        gs_pre = self.update_graph_state_pre_step(graph_state, action)
        # Step the graph
        gs_step, _ = self.graph.step(gs_pre, self.get_step_state(gs_pre), output)
        # Get reward
        reward = self.get_reward(gs_step, action)
        # Get done flags
        truncated = self.get_truncated(gs_step)
        terminated = self.get_terminated(gs_step)
        # Post-step update of graph_state
        gs_post = self.update_graph_state_post_step(gs_step, action)
        # Get info
        info = self.get_info(gs_post, action)
        # Get observation
        obs = self.get_observation(gs_post)
        return gs_post, obs, reward, terminated, truncated, info


class BaseWrapper(object):
    """Base class for wrappers."""

    def __init__(self, env: Union[Environment, "BaseWrapper"]):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class InitialState:
    graph_state: base.GraphState
    obs: jax.Array
    info: Dict[str, Any]


class AutoResetWrapper(BaseWrapper):
    def __init__(self, env: Union[Environment, "BaseWrapper"], fixed_init: bool = True):
        self.fixed_init = fixed_init
        super().__init__(env)

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        gs, obs, info = self._env.reset(rng)
        if self.fixed_init:
            init_state = InitialState(graph_state=gs, obs=obs, info=info)
            aux_gs = gs.replace_aux({"init": init_state})
        else:
            aux_gs = gs
        return aux_gs, obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        """
        We step the environment and reset the state if the episode is done.
        If so, we use the initial state stored in the aux to reset the environment.

        """
        # Step the environment
        gs, obs, reward, terminated, truncated, info = self._env.step(graph_state, action)
        done = jnp.logical_or(terminated, truncated)

        if self.fixed_init:
            # Pull out the initial state
            init = gs.aux["init"]
            init = init.replace(graph_state=init.graph_state.replace(rng=gs.rng))
        else:
            # todo: Why can't the node be in self._env.params? What if all params in self._env.params are preset?
            names = [name for name in list(gs.rng.keys()) if self._env.params is not None and name not in self._env.params]
            if len(names) == 0:
                names = list(gs.rng.keys())
                print(f"rl.AutoResetWrapper.step(...) | Warning: No node found in graph state. May not actually be fully randomizing...")
            name = names[0]  # Grab arbitrary node not in preset params
            new_rng, rng_init = jax.random.split(gs.rng[name])
            gs = gs.replace(rng=gs.rng.copy({name: new_rng}))
            init_gs, init_obs, init_info = self._env.reset(rng_init)
            init = InitialState(graph_state=init_gs, obs=init_obs, info=init_info)
            # jax.debug.print("x={x}, rng_init={rng_init}", x=init_obs[0], rng_init=rng_init)

        # Define the two branches of the conditional
        def is_done(*args):
            # Add aux to the graph state to match shapes
            _gs = init.graph_state.replace(aux=gs.aux)
            return _gs, init.obs, init.info

        def not_done(*args):
            return gs, obs, info

        next_gs, next_obs, next_info = jax.lax.cond(done, is_done, not_done)

        # next_info["final_gs"] = final_gs

        # Note that the reward, terminated, and truncated flags are not reset
        # (i.e. they are from the previous episode).
        return next_gs, next_obs, reward, terminated, truncated, next_info


@struct.dataclass
class LogState:
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(BaseWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: Union[Environment, "BaseWrapper"]):
        super().__init__(env)

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        gs, obs, info = self._env.reset(rng)

        log_state = LogState(
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            timestep=0,
        )
        log_gs = gs.replace_aux({"log": log_state})
        return log_gs, obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        gs, obs, reward, terminated, truncated, info = self._env.step(graph_state, action)
        done = jnp.logical_or(terminated, truncated)
        log_state = gs.aux["log"]
        new_episode_return = log_state.episode_returns + reward
        new_episode_length = log_state.episode_lengths + 1
        log_state = log_state.replace(
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=log_state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=log_state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=log_state.timestep + 1,
        )
        info["returned_episode_returns"] = log_state.returned_episode_returns
        info["returned_episode_lengths"] = log_state.returned_episode_lengths
        info["timestep"] = log_state.timestep
        info["returned_episode"] = done
        log_gs = gs.replace_aux({"log": log_state})
        return log_gs, obs, reward, terminated, truncated, info


@struct.dataclass
class SquashState:
    low: jax.Array
    high: jax.Array
    squash: bool = struct.field(pytree_node=False)

    def scale(self, x) -> jax.Array:
        """Scales the input to [-1, 1] and unsquashes."""
        if self.squash:
            x = 2.0 * (x - self.low) / (self.high - self.low) - 1.0
            # use the opposite of tanh to unsquash
            x = jnp.arctanh(x)
        return x

    def unsquash(self, x) -> jax.Array:
        """
        Squashes x to [-1, 1] and then unscales to the original range [low, high].

        else x is clipped to the range of the action space.

        """
        if self.squash:
            x = jnp.tanh(x)
            x = 0.5 * (x + 1.0) * (self.high - self.low) + self.low
        else:
            x = jnp.clip(x, self.low, self.high)
        return x

    @property
    def action_space(self) -> Box:
        if self.squash:
            return Box(low=-1.0, high=1.0, shape=self.low.shape, dtype=self.low.dtype)
        else:
            return Box(low=self.low, high=self.high, shape=self.low.shape, dtype=self.low.dtype)


class SquashAction(BaseWrapper):
    def __init__(self, env: Union[Environment, "BaseWrapper"], squash: bool = True):
        super().__init__(env)
        self.squash = squash

    def action_space(self, graph_state: base.GraphState) -> Box:
        act_space = self._env.action_space(graph_state)
        act_scaling = SquashState(low=act_space.low, high=act_space.high, squash=self.squash)
        return act_scaling.action_space

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        gs, obs, info = self._env.reset(rng)
        act_space = self._env.action_space(gs)
        act_scaling = SquashState(low=act_space.low, high=act_space.high, squash=self.squash)
        transform_gs = gs.replace_aux({"act_scaling": act_scaling})
        return transform_gs, obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        act_scaling = graph_state.aux["act_scaling"]
        action = act_scaling.unsquash(action)
        return self._env.step(graph_state, action)


class ClipAction(BaseWrapper):
    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        act_space = self._env.action_space(graph_state)
        action = jnp.clip(action, act_space.low, act_space.high)
        return self._env.step(graph_state, action)


class VecEnv(BaseWrapper):
    def __init__(self, env, in_axes: Union[int, None, Sequence[Any]] = 0):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=in_axes)
        self.step = jax.vmap(self._env.step, in_axes=in_axes)


@struct.dataclass
class NormalizeVec:
    mean: jax.Array
    var: jax.Array
    count: Union[float, jax.typing.ArrayLike]
    return_val: jax.Array
    clip: Union[float, jax.typing.ArrayLike]

    def normalize(self, x, clip=True, subtract_mean=True):
        """Normalize x"""
        if subtract_mean:
            x = x - self.mean
        x = x / jnp.sqrt(self.var + 1e-8)
        if clip:
            x = jnp.clip(x, -self.clip, self.clip)
        return x

    def unnormalize(self, x, add_mean=True):
        """Unnormalize x with variance."""
        x = x * jnp.sqrt(self.var + 1e-8)
        if add_mean:
            x = x + self.mean
        return x


class NormalizeVecObservation(BaseWrapper):
    def __init__(self, env: Union[Environment, "BaseWrapper"], clip_obs: float = 10.0):
        super().__init__(env)
        self.clip_obs = clip_obs

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        gs, obs, info = self._env.reset(rng)
        norm_state = NormalizeVec(
            mean=jnp.zeros_like(obs[0]),
            var=jnp.ones_like(obs[0]),
            count=1e-4,
            return_val=None,
            clip=self.clip_obs
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - norm_state.mean
        tot_count = norm_state.count + batch_count

        new_mean = norm_state.mean + delta * batch_count / tot_count
        m_a = norm_state.var * norm_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * norm_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        norm_state = NormalizeVec(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=None,
            clip=self.clip_obs
        )
        norm_gs = gs.replace_aux({"norm_obs": norm_state})
        norm_obs = norm_state.normalize(obs, clip=True, subtract_mean=True)
        return norm_gs, norm_obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        gs_excl_norm = graph_state.replace_aux({"norm_obs": None})  # Exclude the normalization state (cannot be vmapped)
        gs, obs, reward, terminated, truncated, info = self._env.step(gs_excl_norm, action)

        norm_state = graph_state.aux["norm_obs"]
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - norm_state.mean
        tot_count = norm_state.count + batch_count

        new_mean = norm_state.mean + delta * batch_count / tot_count
        m_a = norm_state.var * norm_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * norm_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        norm_state = NormalizeVec(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=None,
            clip=self.clip_obs
        )
        norm_gs = gs.replace_aux({"norm_obs": norm_state})
        norm_obs = norm_state.normalize(obs, clip=True, subtract_mean=True)
        return norm_gs, norm_obs, reward, terminated, truncated, info


class NormalizeVecReward(BaseWrapper):
    def __init__(self, env: Union[Environment, "BaseWrapper"], gamma: Union[float, jax.typing.ArrayLike], clip_reward: float = 10.0):
        super().__init__(env)
        self.gamma = gamma
        self.clip_reward = clip_reward

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        gs, obs, info = self._env.reset(rng)

        batch_count = obs.shape[0]
        norm_state = NormalizeVec(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            clip=self.clip_reward
        )
        norm_gs = gs.replace_aux({"norm_reward": norm_state})
        return norm_gs, obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        gs_excl_norm = graph_state.replace_aux({"norm_reward": None})  # Exclude the normalization state (cannot be vmapped)
        gs, obs, reward, terminated, truncated, info = self._env.step(gs_excl_norm, action)
        done = jnp.logical_or(terminated, truncated)
        norm_state = graph_state.aux["norm_reward"]
        return_val = norm_state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - norm_state.mean
        tot_count = norm_state.count + batch_count

        new_mean = norm_state.mean + delta * batch_count / tot_count
        m_a = norm_state.var * norm_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * norm_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        norm_state = NormalizeVec(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            clip=self.clip_reward
        )
        norm_gs = gs.replace_aux({"norm_reward": norm_state})
        norm_reward = norm_state.normalize(reward, clip=True, subtract_mean=False)
        # norm_reward = jnp.clip(reward / jnp.sqrt(norm_state.var + 1e-8), -self.clip_reward, self.clip_reward)
        return norm_gs, obs, norm_reward, terminated, truncated, info


@struct.dataclass
class Transition:
    gs: base.GraphState
    action: jax.typing.ArrayLike
    obs: jax.typing.ArrayLike
    next_obs: jax.typing.ArrayLike
    reward: Union[float, jax.typing.ArrayLike]
    terminated: Union[bool, jax.typing.ArrayLike]
    truncated: Union[bool, jax.typing.ArrayLike]
    info: Dict[str, jax.typing.ArrayLike]


def rollout(env: Environment, get_action: Callable[[jax.Array], jax.Array], num_steps: int, rng: jax.Array = None):
    """Rollout a single episode of the environment."""
    rng = jax.random.PRNGKey(0) if rng is None else rng

    def _step(carry, _):
        _gs, _obs, _is_done = carry
        action = get_action(_obs)
        next_gs, next_obs, reward, terminated, truncated, info = env.step(_gs, action)
        reward = reward * (1 - _is_done)  # If previous step was done, reward is 0
        done = jnp.logical_or(terminated, truncated)
        next_is_done = jnp.logical_or(_is_done, done)
        transition = Transition(gs=_gs, action=action, obs=_obs, next_obs=next_obs, reward=reward, terminated=terminated, truncated=truncated, info=info)
        return (next_gs, next_obs, next_is_done), transition

    # Reset the environment
    gs, obs, info = env.reset(rng)

    # Step
    _, transition = jax.lax.scan(_step, (gs, obs, jnp.array(False)), jnp.arange(num_steps))
    return transition

####################################################################################################
# Not used yet
####################################################################################################
# class BraxGymnaxWrapper:
#     def __init__(self, env_name, backend="positional"):
#         env = envs.get_environment(env_name=env_name, backend=backend)
#         env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
#         env = AutoResetWrapper(env)
#         self._env = env
#         self.action_size = env.action_size
#         self.observation_size = (env.observation_size,)
#
#     def reset(self, key, params=None):
#         state = self._env.reset(key)
#         return state.obs, state
#
#     def step(self, key, state, action, params=None):
#         next_state = self._env.step(state, action)
#         return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}
#
#     def observation_space(self, params):
#         return Box(
#             low=-jnp.inf,
#             high=jnp.inf,
#             shape=(self._env.observation_size,),
#         )
#
#     def action_space(self, params):
#         return Box(
#             low=-1.0,
#             high=1.0,
#             shape=(self._env.action_size,),
#         )

# class TransformObservation(BaseWrapper):
#     def __init__(self, env, transform_obs):
#         super().__init__(env)
#         self.transform_obs = transform_obs
#
#     def reset(self, key, params=None):
#         obs, state = self._env.reset(key, params)
#         return self.transform_obs(obs), state
#
#     def step(self, key, state, action, params=None):
#         obs, state, reward, done, info = self._env.step(key, state, action, params)
#         return self.transform_obs(obs), state, reward, done, info
#
#
# class TransformReward(BaseWrapper):
#     def __init__(self, env, transform_reward):
#         super().__init__(env)
#         self.transform_reward = transform_reward
#
#     def step(self, key, state, action, params=None):
#         obs, state, reward, done, info = self._env.step(key, state, action, params)
#         return obs, state, self.transform_reward(reward), done, info
#
#
# class FlattenObservationWrapper(BaseWrapper):
#     """Flatten the observations of the environment."""
#
#     def __init__(self, env: Union[Environment, "BaseWrapper"]):
#         super().__init__(env)
#
#     def observation_space(self, params) -> Box:
#         obs_space = self._env.observation_space(params)
#         assert isinstance(obs_space, Box), "Only Box spaces are supported for now."
#         return Box(
#             low=obs_space.low,
#             high=obs_space.high,
#             shape=(onp.prod(obs_space.shape),),
#             dtype=obs_space.dtype,
#         )
#
#     @partial(jax.jit, static_argnums=(0,))
#     def reset(
#         self, key: jax.Array, params = None
#     ) -> Tuple[jax.Array, environment.EnvState]:
#         obs, state = self._env.reset(key, params)
#         obs = jnp.reshape(obs, (-1,))
#         return obs, state
#
#     @partial(jax.jit, static_argnums=(0,))
#     def step(
#         self,
#         key: jax.Array,
#         state: environment.EnvState,
#         action: Union[int, float],
#         params: Optional[environment.EnvParams] = None,
#     ) -> Tuple[jax.Array, environment.EnvState, float, bool, dict]:
#         obs, state, reward, done, info = self._env.step(key, state, action, params)
#         obs = jnp.reshape(obs, (-1,))
#         return obs, state, reward, done, info