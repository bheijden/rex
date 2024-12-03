import functools

import jax
import jax.numpy as jnp

from rex.base import Identity
from rex.cem import cem, CEMSolver, CEMState


def dummy_loss(params, transform, rng):
    """A simple dummy loss function for testing."""
    abs_params = jax.tree_util.tree_map(lambda x: jnp.abs(x), params)
    leaves, _ = jax.tree_util.tree_flatten(abs_params)
    cum_loss = functools.reduce(jnp.add, leaves).sum()
    return cum_loss


def test_cem():
    """Test the `cem` method."""
    transform = Identity()
    max_steps = 10

    u_min = {"param": jnp.array([-1.0, -1.0])}
    u_max = {"param": jnp.array([1.0, 1.0])}
    solver = CEMSolver.init(u_min=u_min, u_max=u_max, num_samples=50, evolution_smoothing=0.1, elite_portion=0.2)
    state = solver.init_state(mean={"param": jnp.array([0.0, 0.0])})

    final_state, losses = cem(dummy_loss, solver, state, transform, max_steps=max_steps, verbose=True)

    assert isinstance(final_state, CEMState)
    assert isinstance(losses, jnp.ndarray)
    assert losses.shape[0] == max_steps
    assert final_state.bestsofar_loss <= jnp.inf  # Check if best loss is updated
    assert jnp.isclose(final_state.bestsofar["param"], jnp.zeros(2), atol=1e-1).all()
    assert jnp.all(jax.tree_util.tree_map(lambda x: x.shape, final_state.mean) == {"param": (2,)})
    assert jnp.all(jax.tree_util.tree_map(lambda x: x.shape, final_state.stdev) == {"param": (2,)})
