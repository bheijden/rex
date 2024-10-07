from typing import Union, Tuple, Callable, Any, Dict
import jax
import jax.numpy as jnp
import jax.random as rnd
import equinox as eqx
from flax import struct

from rex.base import Params, Loss, Transform

try:
    import evosax as evx
except ImportError as e:
    raise ImportError(f"Failed to import evosax: {e}. Please install it with `pip install evosax`.")


ESLog = evx.ESLog
EvoState = evx.strategy.EvoState


@struct.dataclass
class LogState:
    state: Dict
    logger: ESLog = struct.field(pytree_node=False)

    def update(self, x: jnp.ndarray, fitness: jnp.ndarray) -> 'LogState':
        new_log_state = self.logger.update(self.state, x, fitness)
        return self.replace(state=new_log_state)

    def save(self, filename: str):
        return self.logger.save(self.state, filename)

    def load(self, filename: str) -> 'LogState':
        new_state = self.logger.load(filename)
        return self.replace(state=new_state)

    def plot(self, title, ylims=None, fig=None, ax=None, no_legend=False):
        return self.logger.plot(self.state, title, ylims, fig, ax, no_legend)


@struct.dataclass
class EvoSolver:
    strategy_params: evx.strategy.EvoParams
    strategy: evx.strategy.Strategy = struct.field(pytree_node=False)
    strategy_name: str = struct.field(pytree_node=False)

    @classmethod
    def init(cls, u_min: Dict[str, Params], u_max: Dict[str, Params], strategy: str, strategy_kwargs: Dict = None, fitness_kwargs: Dict = None):
        strategy_name = strategy
        strategy_kwargs = strategy_kwargs or dict()
        fitness_kwargs = fitness_kwargs or dict()
        assert "num_dim" not in strategy_kwargs, "u_min is used as `pholder_params`, so `num_dim` cannot be provided in strategy_kwargs."
        assert "pholder_params" not in strategy_kwargs, "u_min is used as `pholder_params`, so `pholder_params` it cannot be provided in strategy_kwargs."
        strategy_cls = evx.Strategies[strategy]
        strategy = strategy_cls(pholder_params=u_min, **strategy_kwargs, **fitness_kwargs)
        strategy_params = strategy.default_params
        clip_min = strategy.param_reshaper.flatten_single(u_min)
        clip_max = strategy.param_reshaper.flatten_single(u_max)
        strategy_params = strategy_params.replace(clip_min=clip_min, clip_max=clip_max)
        return cls(strategy_params=strategy_params, strategy=strategy, strategy_name=strategy_name)

    def init_state(self, mean: Dict[str, Params], rng: jax.Array = None) -> EvoState:
        if rng is None:
            rng = rnd.PRNGKey(0)
        state = self.strategy.initialize(rng, self.strategy_params, mean)
        return state

    def init_logger(self, num_generations: int, top_k: int = 5, maximize: bool = False) -> LogState:
        logger = evx.ESLog(pholder_params=self.strategy.param_reshaper.placeholder_params,
                           num_generations=num_generations, top_k=top_k, maximize=maximize)
        log_state = logger.initialize()
        return LogState(state=log_state, logger=logger)

    def flatten(self, tree: Any):
        return self.strategy.param_reshaper.flatten_single(tree)

    def unflatten(self, x: jax.typing.ArrayLike):
        return self.strategy.param_reshaper.reshape_single(x)


def evo(loss: Loss, solver: EvoSolver, init_state: evx.strategy.EvoState, transform: Transform,
        max_steps: int = 100, rng: jax.Array = None, verbose: bool = True, logger: LogState = None):
    if rng is None:
        rng = rnd.PRNGKey(0)
    rngs = jax.random.split(rng, num=max_steps).reshape(max_steps, 2)

    def _evo_step(_state, xs):
        i, _rngs = xs
        _evo_state, _logger = _state
        new_state, losses = evo_step(loss, solver, _evo_state, transform, _rngs, _logger)
        new_evo_state, new_logger = new_state
        if verbose:
            max_loss = jnp.max(losses)
            loss_nonan = jnp.where(jnp.isnan(losses), jnp.inf, losses)
            min_loss = jnp.min(loss_nonan)
            mean_loss = jnp.mean(loss_nonan)
            total_samples = (i + 1) * solver.strategy.popsize
            jax.debug.print(
                "step: {step} | min_loss: {min_loss} | mean_loss: {mean_loss} | max_loss: {max_loss} | bestsofar_loss: {bestsofar_loss} | total_samples: {total_samples}",
                step=i, min_loss=min_loss, mean_loss=mean_loss, max_loss=max_loss, bestsofar_loss=new_evo_state.best_fitness,
                total_samples=total_samples)
        return new_state, losses

    final_state, losses = jax.lax.scan(_evo_step, (init_state, logger), (jnp.arange(max_steps), rngs))
    return *final_state, losses


def evo_step(loss: Loss, solver: EvoSolver, state: evx.strategy.EvoState, transform: Transform, rng: jax.Array = None, logger: LogState = None):
    if rng is None:
        rng = rnd.PRNGKey(0)

    # Split the rng
    rngs = jax.random.split(rng, num=1+solver.strategy.popsize)

    # Generate the population
    x, state = solver.strategy.ask(rngs[0], state, solver.strategy_params)
    # Evaluate the population members
    losses = eqx.filter_vmap(loss, in_axes=(0, None, 0))(x, transform, rngs[1:])
    loss_nonan = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    # Update the evolution strategy
    new_state = solver.strategy.tell(x, loss_nonan, state, solver.strategy_params)
    # Update the log
    if logger is not None:
        new_logger = logger.update(x, losses)
        return (new_state, new_logger), losses
    else:
        return (new_state, None), losses
