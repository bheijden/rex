import jax
import jax.numpy as jnp
from rex.jax_utils import tree_dynamic_slice


tree = {'a': jnp.zeros((3, 4)), 'b': jnp.zeros((3, 4, 5)), 'c': jnp.zeros((3, 4, 5, 6))}
start_indices = jnp.array([0, 1])  # Start from index 0 in axis 0, and index 1 in axis 1
slice_sizes = [1, 2]  # Take 1 element from axis 0, and 2 elements from axis 1
result = tree_dynamic_slice(tree, start_indices=start_indices, slice_sizes=slice_sizes)
shapes = jax.tree_util.tree_map(lambda x: x.shape, result)
print(shapes)
assert result['a'].shape == (1, 2)
assert result['b'].shape == (1, 2, 5)
assert result['c'].shape == (1, 2, 5, 6)