from typing import Sequence, Callable, Any
from jumpy import _in_jit
import jax
import jumpy as jp
import numpy as onp
import jax.numpy as jnp


class use_numpy:
    def __init__(self):
        self._has_jax = jp._has_jax
        self._float32 = jp.float32
        self._int32 = jp.int32

    def __enter__(self):
        jp._has_jax = False
        jp.float32 = onp.float32
        jp.int32 = onp.int32

    def __exit__(self, exc_type, exc_val, exc_tb):
        jp._has_jax = self._has_jax
        jp.float32 = self._float32
        jp.int32 = self._int32


class use_jax:
    def __init__(self):
        self._has_jax = jp._has_jax
        self._float32 = jp.float32
        self._int32 = jp.int32

    def __enter__(self):
        jp._has_jax = True
        jp.float32 = jnp.float32
        jp.int32 = jnp.int32

    def __exit__(self, exc_type, exc_val, exc_tb):
        jp._has_jax = self._has_jax
        jp.float32 = self._float32
        jp.int32 = self._int32


class use:
    def __init__(self, backend: str = "jax"):
        if backend == "jax":
            self._context = use_jax()
        elif backend == "numpy":
            self._context = use_numpy()

    def __enter__(self):
        self._context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)


def switch(index, branches: Sequence[Callable], *operands: Any):
    """Conditionally apply exactly one of ``branches`` given by ``index`` operands.

    Has the semantics of the following Python::

        def switch(index, branches, *operands):
          index = clamp(0, index, len(branches) - 1)
          return branches[index](*operands)
    """
    if _in_jit():
        return jax.lax.switch(index, branches, *operands)
    else:
        # if True and _has_jax:
        #     return jax.lax.switch(index, branches, *operands)
        # else:
            # index = onp.clip(index, 0, len(branches) - 1)
        return branches[index](*operands)