from rex.spaces import Box, Discrete
from rex.wrappers import rex_space_to_gym_space
import jax
import jax.numpy as jnp


def test_box_space():
	low = jnp.array([0, 1, 2])
	high = jnp.array([1, 2, 3])
	b = Box(low=low, high=high)
	assert b.contains(b.sample(jax.random.PRNGKey(0)))
	assert b.contains(b.sample(jax.random.PRNGKey(1)))
	gym_b = rex_space_to_gym_space(b)


def test_discrete_space():
	d = Discrete(3)
	assert d.contains(d.sample(jax.random.PRNGKey(0)))
	assert d.contains(d.sample(jax.random.PRNGKey(1)))
	gym_d = rex_space_to_gym_space(d)
