"""
This implementation was inspired by gymnax:

https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/spaces.py

"""

from typing import Tuple
import jumpy as jp


class Space:
    """
    Minimal jittable class for abstract space.
    """

    def sample(self, rng: jp.ndarray) -> jp.ndarray:
        raise NotImplementedError

    def contains(self, x: jp.int32) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jp.int32

    def sample(self, rng: jp.ndarray) -> jp.ndarray:
        """Sample random action uniformly from set of categorical choices."""
        return jp.randint(rng, shape=self.shape, low=0, high=self.n).astype(self.dtype)

    def contains(self, x: jp.int32) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """Minimal jittable class for array-shaped spaces."""

    def __init__(
        self,
        low: jp.ndarray,
        high: jp.ndarray,
        shape: Tuple[jp.int32] = None,
        dtype: jp.dtype = jp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape if shape is not None else low.shape
        self.dtype = dtype

    def sample(self, rng: jp.ndarray) -> jp.ndarray:
        """Sample random action uniformly from 1D continuous range."""
        return jp.random_uniform(rng, shape=self.shape, low=self.low, high=self.high).astype(self.dtype)

    def contains(self, x: jp.int32) -> bool:
        """Check whether specific object is within space."""
        range_cond = jp.logical_and(jp.all(x >= self.low), jp.all(x <= self.high))
        return jp.all(range_cond)
