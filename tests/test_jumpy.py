import pytest
from functools import partial
import jax
import jumpy
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp

import rex.jumpy as rjp


@pytest.mark.parametrize("backend, jit", [("numpy", False), ("jax", False), ("jax", True)])
def test_jumpy(backend: str, jit: bool):

    with rjp.use(backend=backend):
        # Test array creation
        arr = jp.array([1, 2, 3], dtype=jp.float32)
        arr_type = jnp.ndarray if backend == "jax" else onp.ndarray
        assert isinstance(arr, arr_type), f"Expected {backend} array"

        # Test simple functions
        rng = jumpy.random.PRNGKey(jp.int32(0))
        arr = jp.array([1, 2, 3], dtype=jp.float32)
        _ = rjp.normal(rng, arr.shape, arr.dtype)
        _ = rjp.log(arr)

        # Test index update
        arr = rjp.index_update(arr, 0, jp.float32(99))

        # Test tree take
        tree = (arr, arr)
        tree = rjp.tree_take(tree, 0)
        try:
            rjp.tree_take(tree, 3)
        except (IndexError, ValueError) as e:
            print(f"Should fail: {e}")

        # Test scn
        def _f(carry, x):
            return carry*2, x

        scan_fn = partial(rjp.scan, f=_f, length=arr.shape[0])
        scan_fn = jax.jit(scan_fn) if jit else scan_fn
        carry_out, scan_out = scan_fn(init=arr, xs=None)


