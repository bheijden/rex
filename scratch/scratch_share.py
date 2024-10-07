from typing import Dict, Tuple, Callable, Union, Sequence, Any
import functools
import jax
import jax.numpy as jnp
import flax.struct as struct
import rex.base as base
import equinox as eqx


@struct.dataclass
class Foo:
    a: jnp.ndarray
    bar: 'Bar'


@struct.dataclass
class Bar:
    b: jnp.ndarray
    c: jnp.ndarray


@struct.dataclass
class BarSubclass(Bar):
    d: jnp.ndarray


@struct.dataclass
class Shared(base.Transform):
    where: Callable[[Any], Union[Any, Sequence[Any]]] = struct.field(pytree_node=False)
    replace_fn: Callable[[Any], Union[Any, Sequence[Any]]] = struct.field(pytree_node=False)
    inverse_fn: Callable[[Any], Union[Any, Sequence[Any]]] = struct.field(pytree_node=False)

    @classmethod
    def init(cls,
             where: Callable[[Any], Union[Any, Sequence[Any]]],
             replace_fn: Callable[[Any], Union[Any, Sequence[Any]]],
             inverse_fn: Callable[[Any], Union[Any, Sequence[Any]]] = lambda _tree: None
    ) -> 'Shared':
        return cls(where=where, replace_fn=replace_fn, inverse_fn=inverse_fn)

    def apply(self, params: base.Params) -> base.Params:
        new = self.replace_fn(params)
        return eqx.tree_at(self.where, params, new)

    def inv(self, params: base.Params) -> base.Params:
        new = self.inverse_fn(params)
        return eqx.tree_at(self.where, params, new)



if __name__ == "__main__":

    foo1 = Foo(a=1,
               bar=Bar(b=2, c=3))
    foo2 = Foo(a=11,
               bar=BarSubclass(b=22, c=33, d=44))

    # Define base params
    base_params = {"foo1": foo1,
                   "foo2": foo2}

    # Define opt params
    opt_params = {k: v for k, v in base_params.items()}
    opt_params = eqx.tree_at(lambda _tree: _tree["foo2"].bar, opt_params, replace_fn=lambda _node: BarSubclass(b=None, c=None, d=opt_params["foo2"].bar.d))

    where = lambda _tree: _tree["foo2"].bar
    replace_fn = lambda _tree: where(_tree).replace(**_tree["foo1"].bar.__dict__)
    inverse_fn = lambda _tree: where(_tree).replace(**{k: None for k in _tree["foo1"].bar.__dict__.keys()})
    shared = Shared.init(where=where,
                         replace_fn=replace_fn,
                         inverse_fn=inverse_fn)
    print(opt_params)
    shared_params = shared.apply(base_params)
    print(shared_params)
    inv_params = shared.inv(shared_params)
    print(inv_params)
    inv_shared_params = shared.apply(inv_params)
    print(inv_shared_params)

    extend = base.Extend.init(base_params, opt_params)
