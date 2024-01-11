"""
This implementation was inspired by gymnax:

https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/spaces.py

"""

from typing import Sequence, Union
import jax
import jax.numpy as jnp
from jax._src.typing import Array, ArrayLike, DTypeLike
# import jumpy
# import jumpy.numpy as jp


class Space:
    """
    Minimal jittable class for abstract space.
    """

    def sample(self, rng: ArrayLike) -> jax.Array:
        raise NotImplementedError

    def contains(self, x: Union[int, ArrayLike]) -> Union[bool, jax.Array]:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = int

    def sample(self, rng: ArrayLike) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n).astype(self.dtype)

    def contains(self, x: Union[int, ArrayLike]) -> Union[bool, jax.Array]:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """Minimal jittable class for array-shaped spaces."""

    def __init__(
        self,
        low: ArrayLike,
        high: ArrayLike,
        shape: Sequence[int] = None,
        dtype: DTypeLike = float,
    ):
        self.low = low
        self.high = high
        self.shape = shape if shape is not None else low.shape
        self.dtype = dtype

    def sample(self, rng: ArrayLike) -> jax.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)

    def contains(self, x: ArrayLike) -> Union[bool, jax.Array]:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return jnp.all(range_cond)
