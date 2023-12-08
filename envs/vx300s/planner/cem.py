from functools import partial
import jax
import jumpy

import numpy as onp
import jax.numpy as jnp
import jumpy.numpy as jp
import rex.jumpy as rjp

from flax import struct

from trajax.optimizers import cem as cem_planner

import trajax.optimizers as opt


@struct.dataclass
class CEMParams:
    """See https://arxiv.org/pdf/1907.03613.pdf for details on CEM"""
    u_min: jp.ndarray
    u_max: jp.ndarray
    sampling_smoothing: jp.float32
    evolution_smoothing: jp.float32


def cem_rex(objective,
            init_state,
            init_controls,
            control_low,
            control_high,
            random_key=None,
            hyperparams=None):
    """Cross Entropy Method (CEM).
  
    CEM is a sampling-based optimization algorithm. At each iteration, CEM samples
    a batch of candidate actions and computes the mean and standard deviation of
    top-performing samples, which are used to sample from in the next iteration.
  
    Args:
      objective:
      cost: cost(x, u, t) returns a scalar
      dynamics: dynamics(x, u, t) returns next state
      init_state: initial state
      init_controls: initial controls, of the shape (horizon, dim_control)
      control_low: lower bound of control space
      control_high: upper bound of control space
      random_key: jax.random.PRNGKey() that serves as a random seed
      hyperparams: a dictionary of algorithm hyperparameters with following keys
        sampling_smoothing -- amount of smoothing in action sampling. Refer to
                            eq. 3-4 in https://arxiv.org/pdf/1907.03613.pdf for
                              more details. evolution_smoothing -- amount of
                              smoothing in updating mean and standard deviation
                              elite_portion -- proportion of samples that is
                              considered elites max_iter -- maximum number of
                              iterations num_samples -- number of action sequences
                              sampled
  
    Returns:
      X: Optimal state trajectory.
      U: Optimized control sequence, an array of shape (horizon, dim_control)
      obj: scalar objective achieved.
    """
    # cost = partial(cost, params)
    # dynamics = partial(dynamics, params)

    if random_key is None:
        random_key = jumpy.random.PRNGKey(0)
    if hyperparams is None:
        hyperparams = opt.default_cem_hyperparams()
    mean = jp.array(init_controls)
    stdev = jp.array([(control_high - control_low) / 2.] * init_controls.shape[0])
    # obj_fn = partial(objective, cost, dynamics)

    def loop_body(_, args):
        mean, stdev, random_key = args
        random_key, rng_ctrl = jumpy.random.split(random_key, num=2)
        controls = opt.gaussian_samples(rng_ctrl, mean, stdev, control_low, control_high, hyperparams)
        indices = jp.arange(0, hyperparams['num_samples'])
        costs = jumpy.vmap(objective, include=(True, True, False))(indices, controls, init_state)
        mean, stdev = opt.cem_update_mean_stdev(mean, stdev, controls, costs, hyperparams)
        return mean, stdev, random_key

    # TODO(sindhwani): swap with lax.scan to make this optimizer differentiable.
    mean, stdev, random_key = jumpy.lax.fori_loop(0, hyperparams['max_iter'], loop_body,
                                                  (mean, stdev, random_key))

    # X = rollout(dynamics, mean, init_state)
    # obj = objective(0, mean, init_state)
    return mean, jp.array(0.)
