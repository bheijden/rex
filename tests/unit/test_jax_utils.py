import pytest
import jax
import equinox as eqx
import jax.numpy as jnp
from rex.jax_utils import tree_dot, tree_take, tree_dynamic_slice, tree_extend, same_structure, promote_to_no_weak_type, no_weaktype


def test_tree_dot():
    tree1 = {'a': jnp.array([1, 2]), 'b': jnp.array([3, 4])}
    tree2 = {'a': jnp.array([5, 6]), 'b': jnp.array([7, 8])}
    result = tree_dot(tree1, tree2)
    assert result == 70  # (1*5 + 2*6) + (3*7 + 4*8)

    tree3 = {'a': jnp.array([1, 2]), 'b': jnp.array([3, 3, 5])}
    tree4 = {'a': jnp.array([5, 6]), 'b': jnp.array([7])}
    with pytest.raises(TypeError):
        tree_dot(tree3, tree4)  # Trees with incompatible structures

    assert tree_dot([], []) == 0  # Empty trees should return 0


def test_tree_take():
    tree = {'a': jnp.array([[1, 2], [3, 4]]), 'b': jnp.array([[5, 6], [7, 8]])}
    indices = jnp.array([0])
    result = tree_take(tree, indices, axis=0)
    assert jnp.array_equal(result['a'], jnp.array([[1, 2]]))
    assert jnp.array_equal(result['b'], jnp.array([[5, 6]]))


def test_tree_dynamic_slice():
    tree = {'a': jnp.zeros((3, 4)), 'b': jnp.zeros((3, 4, 5)), 'c': jnp.zeros((3, 4, 5, 6))}
    start_indices = jnp.array([0, 1])  # Start from index 0 in axis 0, and index 1 in axis 1
    slice_sizes = [1, 2]  # Take 1 element from axis 0, and 2 elements from axis 1
    result = tree_dynamic_slice(tree, start_indices=start_indices, slice_sizes=slice_sizes)
    shapes = jax.tree_util.tree_map(lambda x: x.shape, result)
    print(shapes)
    assert result['a'].shape == (1, 2)
    assert result['b'].shape == (1, 2, 5)
    assert result['c'].shape == (1, 2, 5, 6)


def test_tree_extend():
    tree_template = {'a': jnp.zeros((2,)), 'b': jnp.zeros((3,)), 'c': jnp.zeros((4,))}
    tree = {'a': jnp.array([1, 2]), 'b': jnp.array([3, 4, 5]), "c": None}
    tree_extended_structure = tree_extend(tree_template, tree)  # Has same structure as tree_template, but with None values
    tree_extended = jax.tree_util.tree_map(lambda base_x, ex_x: base_x if ex_x is None else ex_x,
                                             tree_template, tree_extended_structure)   # Fill in None values with base values
    assert jnp.array_equal(tree_extended['a'], jnp.array([1, 2]))
    assert jnp.array_equal(tree_extended['b'], jnp.array([3, 4, 5]))
    assert jnp.array_equal(tree_extended['c'], jnp.zeros((4,)))  # "c" was None in tree, so it should be filled with value from tree_template


def test_same_structure():
    tree1 = {'a': jnp.zeros((2,)), 'b': jnp.zeros((3,))}
    tree2 = {'a': jnp.ones((2,)), 'b': jnp.ones((3,))}
    assert same_structure(tree1, tree2) is True

    tree3 = {'a': jnp.zeros((2,)), 'b': jnp.zeros((4,))}
    with pytest.raises(ValueError):
        same_structure(tree1, tree3)  # Mismatched structure


def test_promote_to_no_weak_type():
    x = 3  # Weakly typed integer
    result = promote_to_no_weak_type(x)
    assert result.dtype == jnp.int32 or result.dtype == jnp.int64


def test_no_weaktype():
    @no_weaktype()
    def sample_fn(x):
        return {'a': x, 'b': x + 1}

    result = sample_fn(3)  # Weakly typed input
    assert result['a'].dtype == jnp.int32 or result['a'].dtype == jnp.int64
    assert result['b'].dtype == jnp.int32 or result['b'].dtype == jnp.int64
