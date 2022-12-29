from typing import Sequence, Callable, Any, TypeVar, Tuple
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


def select(pred, on_true, on_false):
    """Conditionally select between ``on_true`` and ``on_false`` given ``pred``.

    Has the semantics of the following Python::

        def select(pred, on_true, on_false):
          return on_true if pred else on_false
    """
    if _in_jit():
        return jax.numpy.select(pred, on_true, on_false)
    else:
        if jp._has_jax:
            return jax.numpy.select(pred, on_true, on_false)
        else:
            return onp.select(pred, on_true, on_false)


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int = None,
    reverse: bool = False,
    unroll: int = 1,
) -> Tuple[Carry, Y]:
    """Scan a function over leading array axes while carrying along state."""
    if _in_jit():
        return jax.lax.scan(f, init, xs, length, reverse, unroll)
    else:
        # raise NotImplementedError("Must infer length correctly here.")
        xs_flat, xs_tree = jax.tree_util.tree_flatten(xs)
        carry = init
        ys = []
        maybe_reversed = reversed if reverse else lambda x: x
        for i in maybe_reversed(range(length)):
            xs_slice = [x[i] for x in xs_flat]
            carry, y = f(carry, jax.tree_util.tree_unflatten(xs_tree, xs_slice))
            ys.append(y)
        stacked_y = jax.tree_util.tree_map(lambda *y: onp.stack(y), *maybe_reversed(ys))
        return carry, stacked_y


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, X], X], init_val: X) -> X:
    """Call body_fun over range from lower to upper, starting with init_val."""
    if _in_jit():
        return jax.lax.fori_loop(lower, upper, body_fun, init_val)
    else:
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val


def dynamic_slice(
    operand: X, start_indices: Sequence[int], slice_sizes: Sequence[int]
) -> X:
    """Dynamic slice of ``operand`` with per-dimension ``start_indices`` and ``slice_sizes``.

    Has the semantics of the following Python::

        def dynamic_slice(operand, start_indices, slice_sizes):
          return operand[tuple(slice(start, start + size) for start, size in zip(start_indices, slice_sizes))]
    """
    if _in_jit():
        return jax.lax.dynamic_slice(operand, start_indices, slice_sizes)
    else:
        # if jp._has_jax:
        #     return jax.lax.dynamic_slice(operand, start_indices, slice_sizes)
        # else:
        slices = tuple(
            slice(start, start + size) for start, size in zip(start_indices, slice_sizes)
        )
        return operand[slices]


def cond(
    pred, true_fun: Callable[..., bool], false_fun: Callable[..., bool], *operands: Any
):
    """Conditionally apply true_fun or false_fun to operands."""
    if _in_jit():
        return jax.lax.cond(pred, true_fun, false_fun, *operands)
    else:
        if pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)



# def nonzero(x, size=None, fill_value=None):
#     # NOTE! not identical behavior between jax.numpy.nonzero and onp.nonzero
#     """Return the indices of the elements that are non-zero."""
#     if _in_jit():
#         return jax.numpy.nonzero(x, size, fill_value)
#     else:
#         if jp._has_jax:
#             return jax.numpy.nonzero(x, size, fill_value)
#         else:
#             return onp.nonzero(x)