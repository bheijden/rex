import jax
from flax import struct

@struct.dataclass
class BarNested:
    c: jax.Array
    d: jax.Array


@struct.dataclass
class Bar:
    b: BarNested

    def __hash__(self):
        return hash((self.b.c, self))

    def __eq__(self, other):
        return self.b.c == other.b.c

@struct.dataclass
class Foo:
    # Error has to do with this field when set as static.
    a: Bar = struct.field(pytree_node=False)  # If True, the error will not occur


def step(_params):
    return _params


params = Foo(a=Bar(b=BarNested(c=jax.numpy.array(1), d=jax.numpy.arange(4))))
# params2 = Foo(a=Bar(b=BarNested(c=jax.numpy.array(1), d=jax.numpy.arange(4))))
# params.__eq__(params2)
# params == params2
for i in range(2):
    print(f"i: {i}")  # Fails on second iteration

    # Jit the step function
    jit_step = jax.jit(step).lower(params).compile()
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    # Only when re-instantiating the dataclass, the error will occur
    params = Foo(a=Bar(b=BarNested(c=jax.numpy.array(1), d=jax.numpy.arange(4))))
