import jax.numpy as jnp
import evosax as evx
import equinox as eqx

# Quadrotor
max_steps = 100
init_state = jnp.zeros(8)
strategy_kwargs = dict(popsize=200, elite_ratio=0.1, sigma_init=0.4, mean_decay=0.)
strategy = evx.Strategies["CMA_ES"](pholder_params=init_state, **strategy_kwargs)
strategy_params = strategy.default_params
eqx.tree_pprint(strategy_params, short_arrays=False)


# Pendulum
max_steps = 40
init_state = jnp.zeros(27)
strategy_kwargs = dict(popsize=200, elite_ratio=0.1, sigma_init=0.4, mean_decay=0.)
strategy = evx.Strategies["CMA_ES"](pholder_params=init_state, **strategy_kwargs)
strategy_params = strategy.default_params
eqx.tree_pprint(strategy_params, short_arrays=False)