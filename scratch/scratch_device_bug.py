import jax
from flax import struct


@struct.dataclass
class Foo:
    # Error has to do with this field when set as static.
    bar: jax.Array = struct.field(pytree_node=False)  # If True, the error will not occur


def step(_params):
    return _params


params = Foo(bar=jax.numpy.arange(4))
for i in range(2):
    print(f"i: {i}")  # Fails on second iteration

    # When we re-instantiate, but with a scalar, the error will not occur
    # params = Foo(bar=jax.numpy.array(1))  # Works, because foo.bar.all() is valid I guess

    # Jit the step function
    jit_step = jax.jit(step).lower(params).compile()
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    # Only when re-instantiating the dataclass, the error will occur
    params = Foo(bar=jax.numpy.arange(4))
