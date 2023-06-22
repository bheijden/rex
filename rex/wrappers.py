from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import gymnasium.spaces as gs
import jax
import jumpy
import jumpy.numpy as jp
import numpy as onp

from rex.graph import Graph
from rex.spaces import Space, Discrete, Box
from rex.base import GraphState, RexStepReturn
import rex.jumpy as rjp


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
        if isinstance(env.unwrapped.graph, Graph):
            raise TypeError("AutoResetWrapper is only compatible with Graph environments.")

    def step(self, graph_state: GraphState, action: Any) -> Tuple[GraphState, Any, float, bool, bool, Dict]:
        # Step environment
        graph_state, obs, reward, truncated, done, info = self.env.step(graph_state, action)

        # Store last_obs in info (so that it can be used as terminal_observation in case of a reset).
        info["last_observation"] = obs

        # Auto-reset environment
        assert isinstance(done, bool) or done.ndim < 2, "done must be a scalar or a vector of booleans."
        rng_re = self.env.graph.root.get_step_state(graph_state).rng
        graph_state_re, obs_re, info_re = self.env.reset(rng_re, graph_state)

        def where_done(x, y):
            x = jp.array(x)
            y = jp.array(y)
            _done = jp.array(done)
            _done = jp.reshape(_done, list(done.shape) + [1] * (len(x.shape) - len(done.shape)))  # type: ignore
            return jp.where(_done, x, y)

        next_graph_state, next_obs = jax.tree_map(where_done, (graph_state_re, obs_re), (graph_state, obs))
        return next_graph_state, next_obs, reward, truncated, done, info


class GymWrapper(Wrapper, gym.Env):
    def __init__(self, env):
        super().__init__(env)
        self._name = self.env.graph.root.name
        self._graph_state: GraphState = None
        self._seed: int = None
        self._rng: jp.ndarray = None

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

    def _step(self, graph_state: GraphState, action: jp.ndarray) -> RexStepReturn:
        graph_state, obs, reward, truncated, done, info = self.env.step(graph_state, action)
        return (
            jumpy.lax.stop_gradient(graph_state),
            jumpy.lax.stop_gradient(obs),
            jumpy.lax.stop_gradient(reward),
            jumpy.lax.stop_gradient(truncated),
            jumpy.lax.stop_gradient(done),
            info,
        )

    def step(self, action: jp.ndarray) -> Tuple[jp.ndarray, float, bool, bool, Dict]:
        self._graph_state, obs, reward, truncated, done, info = self._step(self._graph_state, action)
        return obs, reward, truncated, done, info

    def _reset(self, rng: jp.ndarray) -> Tuple[jp.ndarray, GraphState, jp.ndarray, Dict]:
        new_rng, rng_reset = jumpy.random.split(rng, num=2)
        graph_state, obs, info = self.env.reset(rng_reset)
        return new_rng, jumpy.lax.stop_gradient(graph_state), jumpy.lax.stop_gradient(obs), info

    def reset(self) -> Tuple[jp.ndarray, Dict]:
        if self._rng is None:
            self.seed()
        self._rng, self._graph_state, obs, info = self._reset(self._rng)
        return obs, info

    def seed(self, seed=None) -> List[int]:
        if seed is None:
            seed = onp.random.randint(0, 2**32 - 1)
        self._seed = seed
        self._rng = jumpy.random.PRNGKey(self._seed)
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
            :return: observation, reward, done, information
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
        self._rng: jp.ndarray = None

        # Vectorize environments
        self._env_step = rjp.vmap(self.env.step)
        self._env_reset = rjp.vmap(self.env.reset)

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
        self, graph_state: GraphState, action: jp.ndarray
    ) -> Tuple[GraphState, jp.ndarray, float, bool, bool, List[Dict]]:
        graph_state, obs, reward, truncated, done, info = self._env_step(graph_state, action)
        new_infos = self._transpose_infos(info)
        return (
            jumpy.lax.stop_gradient(graph_state),
            jumpy.lax.stop_gradient(obs),
            jumpy.lax.stop_gradient(reward),
            jumpy.lax.stop_gradient(truncated),
            jumpy.lax.stop_gradient(done),
            new_infos,
        )

    def _reset(self, rng: jp.ndarray) -> Tuple[jp.ndarray, GraphState, jp.ndarray, List[Dict]]:
        new_rng, *rng_envs = jumpy.random.split(rng, num=self.num_envs + 1)
        graph_state, obs, info = self._env_reset(jp.array(rng_envs))
        new_infos = self._transpose_infos(info)
        return new_rng, graph_state, obs, new_infos

    def reset(self) -> Tuple[jp.ndarray, Dict]:
        if self._rng is None:
            self.seed()
        self._rng, self._graph_state, obs, info = self._reset(self._rng)
        return jumpy.lax.stop_gradient(obs), info

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
        self._rng = jumpy.random.PRNGKey(seed)
        return self.num_envs * [seed]

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError
        # return self.num_envs*[getattr(self.env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError
        # return self.num_envs*[setattr(self.env, attr_name, value)]

    def step_wait(self):
        self._graph_state, obs, rewards, truncateds, dones, infos = self._step(self._graph_state, self._actions)

        # Add terminal infos
        if "last_observation" in infos[0]:
            for i, done in enumerate(dones):
                if done:
                    # save final observation where user can get it, then reset
                    infos[i]["terminal_observation"] = onp.array(infos[i]["last_observation"])

        return onp.array(obs), onp.array(rewards), truncateds, dones, infos

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
