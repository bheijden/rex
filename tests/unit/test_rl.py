import jax
import jax.numpy as jnp
import pytest

import rex.rl as rl
from rex.graph import Graph
from tests.unit.test_utils import Env


def test_box_sample():
    rng = jax.random.PRNGKey(0)
    low = jnp.array([0.0, 0.0])
    high = jnp.array([1.0, 1.0])
    box = rl.Box(low, high)
    sample = box.sample(rng)
    assert sample.dtype == jnp.float32 or sample.dtype == jnp.float64
    assert sample.shape == box.shape
    assert jnp.all(sample >= low) and jnp.all(sample <= high)


def test_box_contains():
    low = jnp.array([0.0, 0.0])
    high = jnp.array([1.0, 1.0])
    box = rl.Box(low, high)
    assert box.contains(jnp.array([0.5, 0.5]))
    assert not box.contains(jnp.array([-0.1, 0.5]))
    assert not box.contains(jnp.array([1.1, 0.5]))
    assert not box.contains(jnp.array([0.5, 1.5]))


def test_base_env(graph: Graph):
    # Create base env
    env = rl.BaseEnv(graph)

    # Test API
    _ = env.max_steps


@pytest.mark.parametrize("only_init", [True, False])
def test_environment(only_init: bool, graph: Graph):
    # Create base env
    env = Env(graph, only_init=only_init)

    # Test API
    _ = env.max_steps
    gs, _obs, _info = env.reset()
    action = env.action_space(gs).sample(jax.random.PRNGKey(0))
    _gs, _obs, _reward, _terminated, _truncated, _info = env.step(gs, action)


@pytest.mark.parametrize("fixed_init", [True, False])
def test_auto_reset_wrapper(fixed_init: bool, graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.AutoResetWrapper(env, fixed_init=fixed_init)

    # Test API
    gs, _obs, _info = wrapped_env.reset()
    action = wrapped_env.action_space(gs).sample(jax.random.PRNGKey(0))
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, action)


def test_log_wrapper(graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.LogWrapper(env)

    # Test API
    gs, _obs, _info = wrapped_env.reset()
    action = wrapped_env.action_space(gs).sample(jax.random.PRNGKey(0))
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, action)


@pytest.mark.parametrize("squash", [True, False])
def test_squash_action(squash: bool, graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.SquashAction(env, squash=squash)

    # Test API
    gs, _obs, _info = wrapped_env.reset()
    action = wrapped_env.action_space(gs).sample(jax.random.PRNGKey(0))
    squashed = gs.aux["act_scaling"].scale(action)
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, squashed)


def test_clip_action(graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.ClipAction(env)

    # Test API
    gs, _obs, _info = wrapped_env.reset()
    action = wrapped_env.action_space(gs).sample(jax.random.PRNGKey(0))
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, action)


def test_vec_env(graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.VecEnv(env)

    # Prepare vectorized input
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)

    # Test API
    gs, _obs, _info = wrapped_env.reset(rngs)
    action = jax.vmap(wrapped_env.action_space(gs).sample)(rngs)
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, action)


def test_normalize_vec_observation(graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.VecEnv(env)
    wrapped_env = rl.NormalizeVecObservation(wrapped_env)

    # Prepare vectorized input
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)

    # Test API
    gs, _obs, _info = wrapped_env.reset(rngs)
    action = jax.vmap(wrapped_env.action_space(gs).sample)(rngs)
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, action)

    # Test denormalize
    _gs.aux.get("norm_obs").denormalize(_obs)


def test_normalize_vec_reward(graph: Graph):
    # Create env
    env = Env(graph)

    # Wrap environment
    wrapped_env = rl.VecEnv(env)
    wrapped_env = rl.NormalizeVecReward(wrapped_env, gamma=0.99)

    # Prepare vectorized input
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)

    # Test API
    gs, _obs, _info = wrapped_env.reset(rngs)
    action = jax.vmap(wrapped_env.action_space(gs).sample)(rngs)
    _gs, _obs, _reward, _terminated, _truncated, _info = wrapped_env.step(gs, action)


def test_rollout(graph: Graph):
    # Create env
    env = Env(graph)

    def get_action(_obs: jax.Array) -> jax.Array:
        return jnp.array([0.5])

    rollout = rl.rollout(env, get_action, num_steps=10)  # noqa: F841
