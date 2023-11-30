import jax
import jumpy

import numpy as onp
import jax.numpy as jnp
import jumpy.numpy as jp
import rex.jumpy as rjp

from flax import struct

from trajax.optimizers import cem as cem_planner


@struct.dataclass
class CEMParams:
    """See https://arxiv.org/pdf/1907.03613.pdf for details on CEM"""
    u_min: jp.ndarray
    u_max: jp.ndarray
    sampling_smoothing: jp.float32
    evolution_smoothing: jp.float32
