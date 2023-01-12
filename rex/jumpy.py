from typing import Sequence, Callable, Any, TypeVar, Tuple, Union
from jumpy import _in_jit
import jax
import jumpy as jp
import numpy as onp
import jax.numpy as jnp

int32 = Union[jnp.int32, onp.int32]
float32 = Union[jnp.float32, onp.float32]


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


# def select(pred, on_true, on_false):
#     """Conditionally select between ``on_true`` and ``on_false`` given ``pred``.
#
#     Has the semantics of the following Python::
#
#         def select(pred, on_true, on_false):
#           return on_true if pred else on_false
#     """
#     if _in_jit():
#         return jax.numpy.select(pred, on_true, on_false)
#     else:
#         if jp._has_jax:
#             return jax.numpy.select(pred, on_true, on_false)
#         else:
#             return onp.select(pred, on_true, on_false)


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")
F = TypeVar("F", bound=Callable)


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


def random_prngkey(seed: jp.int32) -> jp.ndarray:
    """Returns a PRNG key given a seed."""
    # NOTE: selects backend based on seed type.
    if jp._which_np(seed) is jnp:
        return jax.random.PRNGKey(seed)
    else:
        rng = onp.random.default_rng(seed)
        return rng.integers(low=0, high=2 ** 32, dtype="uint32", size=2)


def index_update(x: jp.ndarray, idx: jp.ndarray, y: jp.ndarray, copy: bool = True) -> jp.ndarray:
    """Pure equivalent of x[idx] = y."""
    if jp._which_np(x, idx, y) is jnp:
        return jnp.array(x).at[idx].set(jnp.array(y))
    else:
        if copy:
            x = onp.copy(x)
        x[idx] = y
        return x


def take(tree: Any, i: Union[jp.ndarray, Sequence[int], int], axis: int = 0) -> Any:
    """Returns tree sliced by i."""
    np = jp._which_np(i)
    if isinstance(i, (list, tuple)):
        i = np.array(i, dtype=int)
    return jax.tree_util.tree_map(lambda x: np.take(x, i, axis=axis, mode="clip"), tree)


def vmap(fun: F, include: Sequence[bool] = None) -> F:
    """Creates a function which maps ``fun`` over argument axes.

    :param fun: Function to be mapped.
    :param include: A boolean array of the same length as the number of arguments to ``fun``.
                    If ``include[i]`` is ``True``, then the ``i``th argument to ``fun`` is mapped over.
                    If ``include`` is ``None``, then all arguments are mapped over.
    """
    # Prepare jittable version of fun.
    in_axes = 0
    if include:
        in_axes = [0 if inc else None for inc in include]
    fun_jit = jax.vmap(fun, in_axes=in_axes)

    def _batched(*args, **kwargs):
        # If we're in a jit, just call the jitted version.
        if _in_jit():
            return fun_jit(*args, **kwargs)

        # Otherwise, we need to do the batching ourselves.
        if include is not None and len(include) != len(args):
            raise RuntimeError("Len of `args` list must match length of `include`.")

        # by default, vectorize over every arg
        _include = [True for _ in args] if include is None else include

        # determine number of parallel evaluations to unroll into serial evals
        batch_size = None
        for a, inc in zip(args, _include):
            if inc:
                flat_args, _ = jax.tree_util.tree_flatten(a)
                batch_size = flat_args[0].shape[0]
                break

        # rebuild b_args for each serial evaluation
        rets = []
        for b_idx in range(batch_size):
            b_args = []
            for a, inc in zip(args, _include):
                if inc:
                    b_args.append(take(a, b_idx))
                else:
                    b_args.append(a)
            rets.append(fun(*b_args))

        return jax.tree_util.tree_map(lambda *x: onp.stack(x), *rets)

    return _batched


def stop_gradient(x: X) -> X:
    """Returns x with zero gradient."""
    if jp._which_np(x) is jnp:
        return jax.lax.stop_gradient(x)
    else:
        return x