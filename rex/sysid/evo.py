from typing import Union, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import jax.random as rnd
import equinox as eqx
import evosax as evx
from flax import struct

from rex.sysid.base import Params, Loss, LossArgs


ESLog = evx.ESLog


@struct.dataclass
class EvoSolver:
    strategy_params: evx.strategy.EvoParams
    strategy: evx.strategy.Strategy = struct.field(pytree_node=False)

    @classmethod
    def init(cls, *args, **kwargs):
        raise NotImplementedError

    def init_state(self, *args, **kwargs):
        raise NotImplementedError


@struct.dataclass
class CMA_ESSolver(EvoSolver):
    strategy_params: evx.strategies.cma_es.EvoParams
    strategy: evx.strategies.cma_es.CMA_ES = struct.field(pytree_node=False)

    @classmethod
    def init(self, u_min: Params, u_max: Params, num_samples: int = 100, elite_ratio: float = 0.1,
             sigma_init: float = 1.0, mean_decay: float = 0., n_devices=1, **fitness_kwargs):
        strategy = evx.CMA_ES(popsize=num_samples, pholder_params=u_min, elite_ratio=elite_ratio, sigma_init=sigma_init,
                              mean_decay=mean_decay, n_devices=n_devices, **fitness_kwargs)
        strategy_params = strategy.default_params
        clip_min = strategy.param_reshaper.flatten_single(u_min)
        clip_max = strategy.param_reshaper.flatten_single(u_max)
        strategy_params = strategy_params.replace(clip_min=clip_min, clip_max=clip_max)
        return self(strategy_params=strategy_params, strategy=strategy)

    def init_state(self, mean: Params, stdev: Params = None, rng: jax.Array = None) -> evx.strategies.cma_es.EvoState:
        if rng is None:
            rng = rnd.PRNGKey(0)

        if stdev is None:
            stdev = (self.strategy_params.clip_max - self.strategy_params.clip_min) / 2.
        else:
            stdev = self.strategy.param_reshaper.flatten(stdev)

        state = self.strategy.initialize(rng, self.strategy_params, mean)

        # Replace the stdev
        C = jnp.diag(stdev ** 2)
        state = state.replace(C=C)
        return state

    def flatten(self, tree: Any):
        return self.strategy.param_reshaper.flatten_single(tree)

    def unflatten(self, x: jax.typing.ArrayLike):
        return self.strategy.param_reshaper.reshape_single(x)


def evo(loss: Loss, solver: EvoSolver, init_state: evx.strategy.EvoState, args: LossArgs,
        max_steps: int = 100, rng: jax.Array = None, verbose: bool = True, log: evx.utils.ESLog = None):
    if rng is None:
        rng = rnd.PRNGKey(0)
    rngs = jax.random.split(rng, num=max_steps).reshape(max_steps, 2)

    # Jittable logging helper
    log_state = log.initialize() if log is not None else None

    def _evo_step(_state, xs):
        i, _rngs = xs
        _evo_state, _log_state = _state
        new_state, losses = evo_step(loss, solver, _evo_state, args, _rngs, log_state=_log_state, log=log)
        new_evo_state, new_log_state = new_state
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

    final_state, losses = jax.lax.scan(_evo_step, (init_state, log_state), (jnp.arange(max_steps), rngs))
    return final_state, losses


def evo_step(loss: Loss, solver: EvoSolver, state: evx.strategy.EvoState, args: LossArgs, rng: jax.Array = None,
             log_state=None, log: evx.utils.ESLog = None):
    if rng is None:
        rng = rnd.PRNGKey(0)

    # Split the rng
    rngs = jax.random.split(rng, num=1+solver.strategy.popsize)

    # Generate the population
    x, state = solver.strategy.ask(rngs[0], state, solver.strategy_params)
    # Evaluate the population members
    losses = eqx.filter_vmap(loss, in_axes=(0, None, 0))(x, args, rngs[1:])
    loss_nonan = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    # Update the evolution strategy
    new_state = solver.strategy.tell(x, loss_nonan, state, solver.strategy_params)
    # Update the log
    if log is not None:
        if log_state is None:
            log_state = log.initialize()
        new_log = log.update(log_state, x, losses)
        return (new_state, new_log), losses
    else:
        return (new_state, None), losses
