from typing import Any, Union
import jax
from jax._src.api_util import flatten_axes
import jax.numpy as jnp


def tree_take(tree: Any, i: Union[int, jax.typing.ArrayLike], axis: int = 0, mode: str = None,
              unique_indices=False, indices_are_sorted=False, fill_value=None) -> Any:
    """Returns tree sliced by i."""
    return jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=axis,
                                                     mode=mode,
                                                     unique_indices=unique_indices,
                                                     indices_are_sorted=indices_are_sorted,
                                                     fill_value=fill_value), tree)


def tree_extend(tree_template, tree, is_leaf=None):
    """Extend tree to match tree_template."""
    tree_template_flat, tree_template_treedef = jax.tree_util.tree_flatten(tree_template, is_leaf=is_leaf)
    tree_flat = flatten_axes("tree_match", tree_template_treedef, tree)
    tree_extended = jax.tree_util.tree_unflatten(tree_template_treedef, tree_flat)
    return tree_extended