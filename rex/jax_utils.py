from typing import Any, Union
import jax
import jax.numpy as jnp


def tree_take(tree: Any, i: Union[int, jax.typing.ArrayLike], axis: int = 0, mode: str = None,
              unique_indices=False, indices_are_sorted=False, fill_value=None) -> Any:
    """Returns tree sliced by i."""
    return jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=axis,
                                                     mode=mode,
                                                     unique_indices=unique_indices,
                                                     indices_are_sorted=indices_are_sorted,
                                                     fill_value=fill_value), tree)
