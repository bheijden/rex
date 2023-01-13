import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import rex.jumpy as rjp


def test_jumpy():

    with rjp.use(backend="jax"):
        arr = jp.array([1, 2, 3], dtype=jp.float32)
        assert isinstance(arr, jnp.ndarray), "Expected jax array"

    with rjp.use(backend="numpy"):
        arr = jp.array([1, 2, 3], dtype=jp.float32)
        assert isinstance(arr, onp.ndarray), "Expected numpy array"


