from typing import Union, Dict
import jax
import jax.numpy as jnp
import jax.random as rnd
import equinox as eqx
from flax import struct

from rex.base import Loss, Transform, Params


@struct.dataclass
class CEMState:
    mean: Dict[str, Params]
    stdev: Dict[str, Params]
    bestsofar: Dict[str, Params]
    bestsofar_loss: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class CEMSolver:
    """See https://arxiv.org/pdf/1907.03613.pdf for details on CEM"""
    u_min: Dict[str, Params]
    u_max: Dict[str, Params]
    evolution_smoothing: Union[float, jax.typing.ArrayLike]
    num_samples: int = struct.field(pytree_node=False)
    elite_portion: float = struct.field(pytree_node=False)

    @classmethod
    def init(cls, u_min: Dict[str, Params], u_max: Dict[str, Params], num_samples: int = 100,
             evolution_smoothing: Union[float, jax.typing.ArrayLike] = 0.1, elite_portion: float = 0.1):
        return cls(u_min=u_min, u_max=u_max, evolution_smoothing=evolution_smoothing, elite_portion=elite_portion,
                   num_samples=num_samples)

    def init_state(self, mean: Dict[str, Params], stdev: Dict[str, Params] = None) -> CEMState:
        if stdev is None:
            stdev = jax.tree_util.tree_map(lambda _x_min, _x_max: (_x_max - _x_min) / 2., self.u_min, self.u_max)
        u_mean = jax.tree_util.tree_map(lambda x: jnp.array(x), mean)
        u_stdev = jax.tree_util.tree_map(lambda x: jnp.array(x), stdev)
        state = CEMState(mean=u_mean, stdev=u_stdev, bestsofar=u_mean, bestsofar_loss=jnp.inf)
        return state


def gaussian_samples(solver: CEMSolver, state: CEMState, rng: jax.Array) -> Dict[str, Params]:
    def sample(rng, mean, stdev, u_min, u_max):
        noises = jax.random.normal(rng, mean.shape)
        samples = mean + stdev * noises
        clipped_samples = jnp.clip(samples, u_min, u_max)
        return clipped_samples

    flat_mean, treedef_mean = jax.tree_util.tree_flatten(state.mean)
    flat_rngs = jax.random.split(rng, num=len(flat_mean))
    rngs = jax.tree_util.tree_unflatten(treedef_mean, flat_rngs)
    samples = jax.tree_util.tree_map(lambda _rng, _mean, _stdev, _u_min, _u_max: sample(_rng, _mean, _stdev, _u_min, _u_max),
                                     rngs, state.mean, state.stdev, solver.u_min, solver.u_max)
    return samples


def cem_update_mean_stdev(solver: CEMSolver, state: CEMState, samples: Dict[str, Params], losses: jax.typing.ArrayLike) -> CEMState:
    evolution_smoothing = solver.evolution_smoothing
    num_samples = solver.num_samples
    num_elites = int(num_samples * solver.elite_portion)

    # Replace nan with max loss
    # max_loss = jnp.max(losses)
    losses = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    elite_indices = jnp.argsort(losses)[:num_elites]
    elite_samples = jax.tree_util.tree_map(lambda x: x[elite_indices], samples)  # todo: why is this possible without a gather?

    # Update mean & stdev
    new_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), elite_samples)
    new_stdev = jax.tree_util.tree_map(lambda x: jnp.std(x, axis=0), elite_samples)
    updated_mean = jax.tree_util.tree_map(lambda x, y: evolution_smoothing * x + (1 - evolution_smoothing) * y, state.mean,
                                          new_mean)
    updated_stdev = jax.tree_util.tree_map(lambda x, y: evolution_smoothing * x + (1 - evolution_smoothing) * y, state.stdev,
                                           new_stdev)

    # Update bestsofar
    best_index = elite_indices[0]
    best_loss = losses[best_index]
    best_sample = jax.tree_util.tree_map(lambda x: x[best_index], samples)
    updated_bestsofar = jax.tree_util.tree_map(lambda x, y: jnp.where(state.bestsofar_loss < best_loss, x, y), state.bestsofar,
                                               best_sample)
    updated_bestsofar_loss = jnp.where(state.bestsofar_loss < best_loss, state.bestsofar_loss, best_loss)
    updated_state = state.replace(mean=updated_mean, stdev=updated_stdev, bestsofar=updated_bestsofar,
                                  bestsofar_loss=updated_bestsofar_loss)
    return updated_state


def cem(loss: Loss, solver: CEMSolver, init_state: CEMState, transform: Transform,
        max_steps: int = 100, rng: jax.Array = None, verbose: bool = True):
    if rng is None:
        rng = rnd.PRNGKey(0)
    rngs = jax.random.split(rng, num=max_steps).reshape(max_steps, 2)

    def _cem_step(_state, xs):
        i, _rngs = xs
        new_state, losses = cem_step(loss, solver, _state, transform, _rngs)
        if verbose:
            max_loss = jnp.max(losses)
            loss_nonan = jnp.where(jnp.isnan(losses), jnp.inf, losses)
            min_loss = jnp.min(loss_nonan)
            mean_loss = jnp.mean(loss_nonan)
            total_samples = (i + 1) * solver.num_samples
            jax.debug.print(
                "step: {step} | min_loss: {min_loss} | mean_loss: {mean_loss} | max_loss: {max_loss} | bestsofar_loss: {bestsofar_loss} | total_samples: {total_samples}",
                step=i, min_loss=min_loss, mean_loss=mean_loss, max_loss=max_loss, bestsofar_loss=new_state.bestsofar_loss,
                total_samples=total_samples)
        return new_state, losses

    final_state, losses = jax.lax.scan(_cem_step, init_state, (jnp.arange(max_steps), rngs))
    return final_state, losses


def cem_step(loss: Loss, solver: CEMSolver, state: CEMState, transform: Transform, rng: jax.Array = None):
    if rng is None:
        rng = rnd.PRNGKey(0)
    rngs = jax.random.split(rng, num=solver.num_samples*2)

    samples = eqx.filter_vmap(gaussian_samples, in_axes=(None, None, 0))(solver, state, rngs[:solver.num_samples])
    losses = eqx.filter_vmap(loss, in_axes=(0, None, 0))(samples, transform, rngs[solver.num_samples:])
    new_state = cem_update_mean_stdev(solver, state, samples, losses)
    return new_state, losses
