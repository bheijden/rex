import functools

import jax
import jax.numpy as jnp

from rex.base import Identity
from rex.evo import evo, EvoSolver


def dummy_loss(params, transform, rng):
    """A simple dummy loss function for testing."""
    abs_params = jax.tree_util.tree_map(lambda x: jnp.abs(x), params)
    leaves, _ = jax.tree_util.tree_flatten(abs_params)
    cum_loss = functools.reduce(jnp.add, leaves).sum()
    return cum_loss


def test_evo():
    """Test the `evo` method."""
    transform = Identity()
    max_steps = 10

    u_min = {"param": jnp.array([-1.0, -1.0])}
    u_max = {"param": jnp.array([1.0, 1.0])}
    strategy_kwargs = dict(popsize=50, elite_ratio=0.1, sigma_init=0.4, mean_decay=0.0)
    solver = EvoSolver.init(u_min=u_min, u_max=u_max, strategy="CMA_ES", strategy_kwargs=strategy_kwargs)
    state = solver.init_state(mean={"param": jnp.array([0.0, 0.0])})

    logger = solver.init_logger(num_generations=max_steps)
    final_state, final_logger, losses = evo(
        dummy_loss, solver, state, transform, max_steps=max_steps, verbose=True, logger=logger
    )

    assert isinstance(losses, jnp.ndarray)
    assert losses.shape[0] == max_steps
    assert final_state.best_fitness <= jnp.inf  # Check if best loss is updated
    assert jnp.isclose(solver.unflatten(final_state.best_member)["param"], jnp.zeros(2), atol=1e-1).all()

    # Test plotting
    final_logger.plot("Training Loss")
