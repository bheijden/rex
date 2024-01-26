from typing import Union, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import jax.random as rnd
import optimistix as optx
import equinox as eqx

from sysid.utils import Params, Residual, ResidualArgs
from flax import struct


@struct.dataclass
class CEMSolver:
	"""See https://arxiv.org/pdf/1907.03613.pdf for details on CEM"""
	u_min: Params
	u_max: Params
	evolution_smoothing: Union[float, jax.typing.ArrayLike]
	num_samples: int = struct.field(pytree_node=False)
	elite_portion: float = struct.field(pytree_node=False)

	@classmethod
	def init(cls, u_min: Params, u_max: Params, u_stdev: Params = None,
	         evolution_smoothing: Union[float, jax.typing.ArrayLike] = 0.1, elite_portion: float = 0.1,
	         num_samples: int = 100):
		return cls(u_min=u_min, u_max=u_max, evolution_smoothing=evolution_smoothing, elite_portion=elite_portion,
		           num_samples=num_samples)


@struct.dataclass
class CEMState:
	mean: Params
	stdev: Params
	bestsofar: Params
	bestsofar_loss: Union[float, jax.typing.ArrayLike]


def gaussian_samples(solver: CEMSolver, state: CEMState, rng: jax.Array) -> Params:
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


def cem_update_mean_stdev(solver: CEMSolver, state: CEMState, samples: Params, losses: jax.typing.ArrayLike) -> CEMState:
	evolution_smoothing = solver.evolution_smoothing
	num_samples = solver.num_samples
	num_elites = int(num_samples * solver.elite_portion)

	# Replace nan with max loss
	max_loss = jnp.max(losses)
	losses = jnp.where(jnp.isnan(losses), max_loss, losses)
	elite_indices = jnp.argsort(losses)[:num_elites]
	elite_samples = jax.tree_util.tree_map(lambda x: x[elite_indices], samples)

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


def cem(residual: Residual, solver: CEMSolver, opt_params: Params, args: ResidualArgs, opt_params_stdev: Params = None,
        max_steps: int = 100, rng: jax.Array = None, verbose: bool = True):
	if rng is None:
		rng = rnd.PRNGKey(0)
	rngs = jax.random.split(rng, num=max_steps * solver.num_samples).reshape(max_steps, solver.num_samples, 2)

	# Initialize state
	u_mean = opt_params
	if opt_params_stdev is None:
		u_stdev = jax.tree_util.tree_map(lambda _x_min, _x_max: (_x_max - _x_min) / 2., solver.u_min, solver.u_max)
	init_state = CEMState(mean=u_mean, stdev=u_stdev, bestsofar=u_mean, bestsofar_loss=jnp.inf)

	def loss(sample: Params) -> Union[float, jax.Array]:
		res = residual(sample, args)
		# loss = optx.rms_norm(res)
		return 0.5*optx._misc.sum_squares(res)

	def cem_step(_state, xs):
		i, _rngs = xs
		samples = eqx.filter_vmap(gaussian_samples, in_axes=(None, None, 0))(solver, _state, _rngs)
		losses = eqx.filter_vmap(loss, in_axes=(0,))(samples)
		new_state = cem_update_mean_stdev(solver, _state, samples, losses)
		if verbose:
			max_loss = jnp.max(losses)
			loss_nonan = jnp.where(jnp.isnan(losses), max_loss, losses)
			min_loss = jnp.min(loss_nonan)
			mean_loss = jnp.mean(loss_nonan)
			total_samples = (i + 1) * solver.num_samples
			jax.debug.print(
				"step: {step} | min_loss: {min_loss} | mean_loss: {mean_loss} | bestsofar_loss: {bestsofar_loss} | total_samples: {total_samples}",
				step=i, min_loss=min_loss, mean_loss=mean_loss, bestsofar_loss=new_state.bestsofar_loss,
				total_samples=total_samples)
		return new_state, losses

	final_state, losses = jax.lax.scan(cem_step, init_state, (jnp.arange(max_steps), rngs))
	return final_state, losses
