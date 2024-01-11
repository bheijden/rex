from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import gymnasium.spaces as gs
import jax
import jax.numpy as jnp
import numpy as onp

from rex.asynchronous import AsyncGraph
from rex.spaces import Space, Discrete, Box
from rex.base import GraphState, RexStepReturn


class Wrapper:
    """Wraps the environment to allow modular transformations."""

    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, item):
        if item == "__setstate__":
            raise AttributeError(item)
        if item in ["save", "load"]:
            raise AttributeError("Wrapper does not support save/load. Please use the wrapped environment instead.")
        return getattr(self.env, item)


class AutoResetWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.unwrapped.graph, AsyncGraph):
            raise TypeError("AutoResetWrapper is only compatible with AsyncGraph environments.")

    def step(self, graph_state: GraphState, action: Any) -> Tuple[GraphState, Any, float, bool, bool, Dict]:
        # Step environment
        graph_state, obs, reward, terminated, truncated, info = self.env.step(graph_state, action)
        done = jnp.logical_or(terminated, truncated)

        # Store last_obs in info (so that it can be used as terminal_observation in case of a reset).
        info["last_observation"] = obs

        # Auto-reset environment
        assert isinstance(done, bool) or done.ndim < 2, "done must be a scalar or a vector of booleans."
        rng_re = self.env.graph.root.get_step_state(graph_state).rng
        graph_state_re, obs_re, info_re = self.env.reset(rng_re, graph_state.replace(step=0))

        def where_done(x, y):
            x = jnp.array(x)
            y = jnp.array(y)
            _done = jnp.array(done)
            _done = jnp.reshape(_done, list(done.shape) + [1] * (len(x.shape) - len(done.shape)))  # type: ignore
            return jnp.where(_done, x, y)

        next_graph_state, next_obs = jax.tree_map(where_done, (graph_state_re, obs_re), (graph_state, obs))
        return next_graph_state, next_obs, reward, terminated, truncated, info


class GymWrapper(Wrapper, gym.Env):
    def __init__(self, env):
        super().__init__(env)
        self._name = self.env.graph.root.name
        self._graph_state: GraphState = None
        self._seed: int = None
        self._rng: jax.random.KeyArray = None

    @property
    def action_space(self) -> gym.Space:
        if self._graph_state is None:
            space = self.env.action_space()
        else:
            params = self._graph_state.nodes[self._name].params
            space = self.env.action_space(params)
        return rex_space_to_gym_space(space)

    @property
    def observation_space(self) -> gym.Space:
        if self._graph_state is None:
            space = self.env.observation_space()
        else:
            params = self._graph_state.nodes[self._name].params
            space = self.env.observation_space(params)
        return rex_space_to_gym_space(space)

    def jit(self):
        self._step = jax.jit(self._step)
        self._reset = jax.jit(self._reset)

    def _step(self, graph_state: GraphState, action: jax.typing.ArrayLike) -> RexStepReturn:
        graph_state, obs, reward, terminated, truncated, info = self.env.step(graph_state, action)
        return (
            jax.lax.stop_gradient(graph_state),
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(reward),
            jax.lax.stop_gradient(terminated),
            jax.lax.stop_gradient(truncated),
            info,
        )

    def step(self, action: jax.typing.ArrayLike) -> Tuple[jax.Array, float, bool, bool, Dict]:
        self._graph_state, obs, reward, terminated, truncated, info = self._step(self._graph_state, action)
        return obs, reward, terminated, truncated, info

    def _reset(self, rng: jax.random.KeyArray) -> Tuple[jax.Array, GraphState, jax.Array, Dict]:
        new_rng, rng_reset = jax.random.split(rng, num=2)
        graph_state, obs, info = self.env.reset(rng_reset)
        return new_rng, jax.lax.stop_gradient(graph_state), jax.lax.stop_gradient(obs), info

    def reset(self) -> Tuple[jax.Array, Dict]:
        if self._rng is None:
            self.seed()
        self._rng, self._graph_state, obs, info = self._reset(self._rng)
        return obs, info

    def seed(self, seed=None) -> List[int]:
        if seed is None:
            seed = onp.random.randint(0, 2**32 - 1)
        self._seed = seed
        self._rng = jax.random.PRNGKey(self._seed)
        return [seed]

    def close(self):
        self.env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if isinstance(self, wrapper_class) or isinstance(self.env, wrapper_class):
            return True
        else:
            return self.env.env_is_wrapped(wrapper_class, indices)


try:
    from stable_baselines3.common.vec_env import VecEnv as sb3VecEnv
except ImportError:  # pragma: no cover
    print("stable_baselines3 not installed. Using a proxy for DummyVecEnv.")

    class sb3VecEnv:
        def __init__(self, num_envs: int, observation_space: gs.Space, action_space: gs.Space):
            self.num_envs = num_envs
            self.observation_space = observation_space
            self.action_space = action_space

        def step(self, actions):
            """
            Step the environments with the given action

            :param actions: the action
            :return: observation, reward, terminated, truncated, information
            """
            self.step_async(actions)
            return self.step_wait()


class VecGymWrapper(Wrapper, sb3VecEnv):
    def __init__(self, env, num_envs: int = 1):
        assert not isinstance(env, GymWrapper), "VecGymWrapper cannot accept an env that is wrapped with a GymWrapper."
        Wrapper.__init__(self, env)
        self._name = self.unwrapped.graph.root.name
        self._graph_state: GraphState = None
        self._seed: int = None
        self._rng: jax.random.KeyArray = None

        # Vectorize environments
        self._env_step = jax.vmap(self.env.step)
        self._env_reset = jax.vmap(self.env.reset)

        # Call baseclass constructor
        self.num_envs = num_envs
        sb3VecEnv.__init__(self, num_envs, self._observation_space, self._action_space)

    @property
    def _action_space(self) -> gym.Space:
        if self._graph_state is None:
            space = self.env.action_space()
        else:
            params = self._graph_state.nodes[self._name].params
            single_params = jax.tree_map(lambda x: x[0], params)
            space = self.env.action_space(single_params)
        return rex_space_to_gym_space(space)

    @property
    def _observation_space(self) -> gym.Space:
        if self._graph_state is None:
            space = self.env.observation_space()
        else:
            params = self._graph_state.nodes[self._name].params
            single_params = jax.tree_map(lambda x: x[0], params)
            space = self.env.observation_space(single_params)
        return rex_space_to_gym_space(space)

    def jit(self):
        self._step = jax.jit(self._step)
        self._reset = jax.jit(self._reset)

    def _step(
        self, graph_state: GraphState, action: jax.typing.ArrayLike
    ) -> Tuple[GraphState, jax.Array, float, bool, bool, List[Dict]]:
        graph_state, obs, reward, terminated, truncated, info = self._env_step(graph_state, action)
        new_infos = self._transpose_infos(info)
        return (
            jax.lax.stop_gradient(graph_state),
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(reward),
            jax.lax.stop_gradient(terminated),
            jax.lax.stop_gradient(truncated),
            new_infos,
        )

    def _reset(self, rng: jax.random.KeyArray) -> Tuple[jax.Array, GraphState, jax.Array, List[Dict]]:
        new_rng, *rng_envs = jax.random.split(rng, num=self.num_envs + 1)
        graph_state, obs, info = self._env_reset(jnp.array(rng_envs))
        new_infos = self._transpose_infos(info)
        return new_rng, graph_state, obs, new_infos

    # todo: VecEnv seems to not be able to handle the info dict (i.e., new gymnasium API)
    def reset(self) -> Tuple[jax.Array]:
        # def reset(self) -> Tuple[jp.ndarray, Dict]:
        if self._rng is None:
            self.seed()
        self._rng, self._graph_state, obs, info = self._reset(self._rng)
        return jax.lax.stop_gradient(obs)

    def close(self):
        self.env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if isinstance(self, wrapper_class) or isinstance(self.env, wrapper_class):
            return self.num_envs * [True]
        else:
            return self.num_envs * [self.env.env_is_wrapped(wrapper_class, indices)]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError
        # return self.num_envs*[getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def seed(self, seed=None) -> List[int]:
        if seed is None:
            seed = onp.random.randint(0, 2**32 - 1)
        self._seed = seed
        self._rng = jax.random.PRNGKey(seed)
        return self.num_envs * [seed]

    def get_attr(self, attr_name, indices=None):
        # raise NotImplementedError
        return self.num_envs * [getattr(self.env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError
        # return self.num_envs*[setattr(self.env, attr_name, value)]

    def step_wait(self):
        self._graph_state, obs, rewards, terminateds, truncateds, infos = self._step(self._graph_state, self._actions)
        dones = jnp.logical_or(terminateds, truncateds)

        # Add terminal infos
        if "last_observation" in infos[0]:
            for i, done in enumerate(dones):
                if done:
                    # save final observation where user can get it, then reset
                    infos[i]["terminal_observation"] = onp.array(infos[i]["last_observation"])

        # return onp.array(obs), onp.array(rewards), terminateds, truncateds, infos
        return onp.array(obs), onp.array(rewards), dones, infos

    def step_async(self, actions):
        self._actions = actions

    def _transpose_infos(self, infos):
        flattened, pytreedef = jax.tree_util.tree_flatten(infos)
        new_infos = self.num_envs * [len(flattened) * [None]]
        for idx_tree, leaf in enumerate(flattened):
            for idx_env, val in enumerate(leaf):
                new_infos[idx_env][idx_tree] = val
        new_infos = [jax.tree_util.tree_unflatten(pytreedef, i) for i in new_infos]
        return new_infos


def rex_space_to_gym_space(space: Space) -> gs.Space:
    """Convert Gymnax space to equivalent Gym space

    This implementation was inspired by gymnax:
    https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/spaces.py
    """
    if isinstance(space, Discrete):
        return gs.Discrete(space.n)
    elif isinstance(space, Box):
        low = float(space.low) if (onp.isscalar(space.low) or space.low.size == 1) else onp.array(space.low)
        high = float(space.high) if (onp.isscalar(space.high) or space.low.size == 1) else onp.array(space.high)
        return gs.Box(low, high, space.shape, space.dtype)
    # elif isinstance(space, Dict):
    #     return gs.Dict({k: gymnax_space_to_gym_space(v) for k, v in space.spaces})
    # elif isinstance(space, Tuple):
    #     return gs.Tuple(space.spaces)
    else:
        raise NotImplementedError(f"Conversion of {space.__class__.__name__} not supported")
