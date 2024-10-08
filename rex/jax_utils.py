import functools
from typing import Any, Union, Sequence
import numpy as onp
import jax
from jax.interpreters import xla
from jax._src.api_util import flatten_axes
import jax.numpy as jnp


def tree_dot(tree1: Any, tree2: Any) -> Union[float, jax.Array]:
    """Compute the dot product of two pytrees of arrays with the same pytree
    structure."""
    leaves1, treedef1 = jax.tree_util.tree_flatten(tree1)
    leaves2, treedef2 = jax.tree_util.tree_flatten(tree2)
    if treedef1 != treedef2:
        raise ValueError("trees must have the same structure")
    assert len(leaves1) == len(leaves2)
    dots = []
    for leaf1, leaf2 in zip(leaves1, leaves2):
        dots.append(
            jnp.dot(
                jnp.reshape(leaf1, -1),
                jnp.conj(leaf2).reshape(-1),
                precision=jax.lax.Precision.HIGHEST,  # pyright: ignore
            )
        )
    if len(dots) == 0:
        return jnp.array(0.0)
    else:
        return functools.reduce(jnp.add, dots)


def tree_take(
    tree: Any,
    i: Union[int, jax.typing.ArrayLike],
    axis: int = 0,
    mode: str = None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None,
) -> Any:
    """Returns tree sliced by i."""
    return jax.tree_util.tree_map(
        lambda x: jnp.take(
            x,
            i,
            axis=axis,
            mode=mode,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            fill_value=fill_value,
        ),
        tree,
    )


def tree_dynamic_slice(tree: Any, start_indices: Union[int, jax.typing.ArrayLike], slice_sizes: Sequence[int] = None) -> Any:
    slice_sizes = slice_sizes if slice_sizes is not None else [1] * len(start_indices)

    # Slice the input state
    num_dims = len(slice_sizes)
    tree_slice_sizes = jax.tree_map(lambda _x: slice_sizes + list(_x.shape[num_dims:]), tree)

    # Convert start_indices
    start_indices = (
        jnp.array([start_indices]) if isinstance(start_indices, int) or start_indices.shape == () else start_indices
    )
    tree_start_indices = jax.tree_map(
        lambda _x: jnp.concatenate([start_indices, onp.zeros_like(_x.shape[num_dims:]).astype(int)]), tree
    )

    # Slice the tree
    res = jax.tree_map(
        lambda x, start, size: jax.lax.dynamic_slice(x, start, size)[0, 0], tree, tree_start_indices, tree_slice_sizes
    )
    return res


def tree_extend(tree_template, tree, is_leaf=None):
    """Extend tree to match tree_template."""
    # NOTE! Static data of tree_template and tree must be equal (i.e. tree.node_data())
    tree_template_flat, tree_template_treedef = jax.tree_util.tree_flatten(tree_template, is_leaf=is_leaf)
    try:
        tree_flat = flatten_axes("tree_match", tree_template_treedef, tree)
    except ValueError as e:
        # Extend to this error message that Static data of tree_template and tree must be equal (i.e. tree.node_data())
        # More info: https://github.com/google/jax/issues/19729
        raise ValueError(
            f"Hint: ensure that tree_template.node_data() == tree.node_data() when extending a tree. "
            f"This means all static fields (e.g. marked with pytree_node=False) must be equal. "
            f"Best is to derive tree from tree_template to ensure they share the static fields. "
        ) from e
    tree_extended = jax.tree_util.tree_unflatten(tree_template_treedef, tree_flat)
    return tree_extended


def same_structure(x1, x2, tag: str = None, raise_on_mismatch: bool = True):
    # https://jax.readthedocs.io/en/latest/type_promotion.html#weak-types
    # How to detect recompilation reason? (https://github.com/google/jax/issues/4274)
    # todo: investigate ros logging and jax logging. It seems that jax logging is silenced by ros logging.
    # Once you initialize the host as a ROS node, it's problematic to use jax.log_compiles (it silences all jax logging)
    # import logging
    # logging.getLogger("jax").setLevel(logging.INFO)  (maybe logging.DEBUG is required)
    # Use with jax.log_compiles:  # Or alternatively, use os.environ["JAX_LOG_COMPILES"] = "true"
    # Possible solution: use jnp.promote_dtypes(x, x) on itself to promote to no weak type.
    # You can wrap the function with @no_weaktype to promote all outputs to no weak type.
    def assert_same_aval(leaf1, leaf2):
        xla_leaf1 = xla.abstractify(leaf1)
        xla_leaf2 = xla.abstractify(leaf2)
        if not xla_leaf1 == xla_leaf2:
            if xla_leaf1.shape != xla_leaf2.shape:
                raise ValueError(f"Shape mismatch: {xla_leaf1.shape} != {xla_leaf2.shape}")
            elif xla_leaf1.dtype != xla_leaf2.dtype:
                raise ValueError(f"Dtype mismatch: {xla_leaf1.dtype} != {xla_leaf2.dtype}")
            elif xla_leaf1.weak_type != xla_leaf2.weak_type:
                raise ValueError(f"Weak type mismatch: {xla_leaf1.weak_type} != {xla_leaf2.weak_type}")
            else:
                raise ValueError(f"Leaf mismatch: {xla_leaf1} != {xla_leaf2}")
        else:
            return True

    try:
        jax.tree_util.tree_map(assert_same_aval, x1, x2)
    except ValueError as e:
        if raise_on_mismatch:
            msg = f"Structure mismatch: {tag}" if tag else ""
            raise ValueError(msg) from e
        else:
            return False
    else:
        return True


def promote_to_no_weak_type(_x):
    # Applies jnp.promote_types to itself to promote to no weak type
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.promote_types.html#jax.numpy.promote_types
    # https://github.com/google/jax/issues/4490   <--- not yet looked at, but perhaps jax.dtypes.canonicalize_dtype(onp.int64) is useful?
    _y = jnp.array(_x)
    _z = _y.astype(jnp.promote_types(_y.dtype, _y.dtype))
    return _z


def no_weaktype(identifier: str = None):
    def _no_weaktype(fn):
        def no_weaktype_wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            return jax.tree_util.tree_map(lambda x: promote_to_no_weak_type(x), res)

        no_weaktype_wrapper = functools.wraps(fn)(no_weaktype_wrapper)
        if identifier is not None:
            # functools.update_wrapper(no_weaktype_wrapper, fn)
            no_weaktype_wrapper.__name__ = identifier
        return no_weaktype_wrapper

    return _no_weaktype


# def make_tqdm(*args, **kwargs):
#     from tqdm import tqdm
#     pbar = tqdm(*args, **kwargs)
#
#     def _update_tqdm(fmt, *args, **kwargs):
#         """
#
#         :param fmt: format string, e.g. ``"hello {x}"``, that will be used to format
#                     input arguments, like ``str.format``. See the Python docs on
#                     `string formatting <https://docs.python.org/3/library/stdtypes.html#str.format>`_
#                     and `format string syntax <https://docs.python.org/3/library/string.html#formatstrings>`_.
#         :param *args: A list of positional arguments to be formatted, as if passed to ``fmt.format``.
#         :param **kwargs: Additional keyword arguments to be formatted, as if passed to ``fmt.format``.
#         :return:
#         """
#         # todo: This fn is called repeatedly. how to detect when to close and when to open a new pbar?
#         # todo: Probably need to define a pbar state and update it accordingly.
#         pbar.set_postfix_str(fmt.format(*args, **kwargs))
#         pbar.update(n=1)  # Updates the progress bar (when is it finished?)
#         pbar.close()  # When to close?