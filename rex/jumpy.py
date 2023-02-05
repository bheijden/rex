from typing import Sequence, Callable, Any, TypeVar, Tuple, Union
import jumpy
import jax
import jumpy.numpy as jp
import jax.numpy as jnp
import numpy as onp

int32 = Union[jnp.int32, onp.int32]
float32 = Union[jnp.float32, onp.float32]


class use_numpy:
    def __init__(self):
        self.is_jax_installed = jumpy.is_jax_installed
        self._float32 = jp.float32
        self._int32 = jp.int32
        self._uint8 = jp.uint8

    def __enter__(self):
        jumpy.is_jax_installed = False
        jp.float32 = onp.float32
        jp.int32 = onp.int32
        jp.uint8 = onp.uint8

    def __exit__(self, exc_type, exc_val, exc_tb):
        jumpy.is_jax_installed = self.is_jax_installed
        jp.float32 = self._float32
        jp.int32 = self._int32
        jp.uint8 = self._uint8


class use_jax:
    def __init__(self):
        self.is_jax_installed = jumpy.is_jax_installed
        self._float32 = jp.float32
        self._int32 = jp.int32
        self._uint8 = jp.uint8

    def __enter__(self):
        jumpy.is_jax_installed = True
        jp.float32 = jnp.float32
        jp.int32 = jnp.int32
        jp.uint8 = jnp.uint8

    def __exit__(self, exc_type, exc_val, exc_tb):
        jumpy.is_jax_installed = self.is_jax_installed
        jp.float32 = self._float32
        jp.int32 = self._int32
        jp.uint8 = self._uint8


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


# def select(pred, on_true, on_false):
#     """Conditionally select between ``on_true`` and ``on_false`` given ``pred``.
#
#     Has the semantics of the following Python::
#
#         def select(pred, on_true, on_false):
#           return on_true if pred else on_false
#     """
#     if jumpy.core.is_jitted():
#         return jax.numpy.select(pred, on_true, on_false)
#     else:
#         if jumpy.is_jax_installed:
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
    if jumpy.core.is_jitted():
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


def dynamic_slice(
        operand: X, start_indices: Sequence[int], slice_sizes: Sequence[int]
) -> X:
    """Dynamic slice of ``operand`` with per-dimension ``start_indices`` and ``slice_sizes``.

    Has the semantics of the following Python::

        def dynamic_slice(operand, start_indices, slice_sizes):
          return operand[tuple(slice(start, start + size) for start, size in zip(start_indices, slice_sizes))]
    """
    if jumpy.core.is_jitted():
        return jax.lax.dynamic_slice(operand, start_indices, slice_sizes)
    else:
        # if jumpy.is_jax_installed:
        #     return jax.lax.dynamic_slice(operand, start_indices, slice_sizes)
        # else:
        slices = tuple(
            slice(start, start + size) for start, size in zip(start_indices, slice_sizes)
        )
        return operand[slices]


def index_update(x: jp.ndarray, idx: jp.ndarray, y: jp.ndarray, copy: bool = True) -> jp.ndarray:
    """Pure equivalent of x[idx] = y."""
    if jumpy.core.which_np(x, idx, y) is jnp:
        return jnp.array(x).at[idx].set(jnp.array(y))
    else:
        if copy:
            x = onp.copy(x)
        x[idx] = y
        return x


def tree_take(tree: Any, i: Union[jp.ndarray, Sequence[int], int], axis: int = 0) -> Any:
    """Returns tree sliced by i."""
    np = jumpy.core.which_np(i)
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
        if jumpy.core.is_jitted():
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
                    b_args.append(tree_take(a, b_idx))
                else:
                    b_args.append(a)
            rets.append(fun(*b_args))

        return jax.tree_util.tree_map(lambda *x: onp.stack(x), *rets)

    return _batched


def normal(key, shape=(), dtype=onp.float32):
    """Draw random samples from a normal (Gaussian) distribution."""
    if jumpy.core.which_np(key, shape, dtype) is jnp:
        return jax.random.normal(key, shape, dtype)
    else:
        return onp.random.normal(size=shape).astype(dtype)
