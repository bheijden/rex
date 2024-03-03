from typing import TYPE_CHECKING, TypeVar, Tuple, Callable, Any, Union, Dict, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
from flax import struct
from rex.jax_utils import tree_extend

from rex.sysid.base import Params, Transform


@struct.dataclass
class Identity(Transform):
    @classmethod
    def init(cls):
        return cls()

    def apply(self, params: Params) -> Params:
        return params

    def inv(self, params: Params) -> Params:
        return params


@struct.dataclass
class Chain(Transform):
    transforms: Sequence[Transform]

    @classmethod
    def init(cls, *transforms):
        return cls(transforms=transforms)

    def apply(self, params: Params) -> Params:
        _intermediate = params
        for t in self.transforms:
            _intermediate = t.apply(_intermediate)
        return _intermediate

    def inv(self, params: Params) -> Params:
        _intermediate = params
        for t in self.transforms[::-1]:
            _intermediate = t.inv(_intermediate)
        return _intermediate


@struct.dataclass
class Extend(Transform):
    base_params: Params
    mask: Params

    @classmethod
    def init(cls, base_params: Params, opt_params: Params = None):
        mask = jax.tree_util.tree_map(lambda ex_x: ex_x is not None, opt_params)
        ret = cls(base_params=base_params, mask=mask)
        _ = ret.apply(opt_params)  # Test structure
        return ret

    def extend(self, params: Params) -> Params:
        params_extended_pytree = tree_extend(self.base_params, params)
        params_extended = jax.tree_util.tree_map(lambda base_x, ex_x: base_x if ex_x is None else ex_x,
                                                 self.base_params, params_extended_pytree)
        return params_extended

    def filter(self, params_extended: Params) -> Params:
        mask_ex = tree_extend(self.base_params, self.mask)
        filtered_ex = eqx.filter(params_extended, mask_ex)
        filtered_ex_flat, _ = jax.tree_util.tree_flatten(filtered_ex)
        _, mask_filt_treedef = jax.tree_util.tree_flatten(self.mask)
        filtered = jax.tree_util.tree_unflatten(mask_filt_treedef, filtered_ex_flat)
        return filtered

    def apply(self, params: Params) -> Params:
        return self.extend(params)

    def inv(self, params: Params) -> Params:
        return self.filter(params)


@struct.dataclass
class Denormalize(Transform):
    scale: Params
    offset: Params

    @classmethod
    def init(cls, min_params: Params, max_params: Params):
        offset = jax.tree_util.tree_map(lambda _min, _max: (_min + _max) / 2., min_params, max_params)
        scale = jax.tree_util.tree_map(lambda _min, _max: (_max - _min) / 2, min_params, max_params)
        # assert that the scale is not zero
        zero_filter = jax.tree_util.tree_map(lambda _scale: _scale == 0., scale)
        if jax.tree_util.tree_reduce(jnp.logical_or, zero_filter):
            raise ValueError("The scale cannot be zero. Hint: Check if there are leafs with 'True' in the following zero_filter: "
                             f"{zero_filter}")
        return cls(scale=scale, offset=offset)

    def normalize(self, params: Params) -> Params:
        params_norm = jax.tree_util.tree_map(lambda _params, _offset, _scale: (_params - _offset) / _scale, params,
                                             self.offset, self.scale)
        return params_norm

    def denormalize(self, params: Params) -> Params:
        params_unnorm = jax.tree_util.tree_map(lambda _params, _offset, _scale: _params * _scale + _offset, params,
                                               self.offset, self.scale)
        return params_unnorm

    def apply(self, params: Params) -> Params:
        return self.denormalize(params)

    def inv(self, params: Params) -> Params:
        return self.normalize(params)


@struct.dataclass
class ExpTransform(Transform):
    @classmethod
    def init(cls):
        return cls()

    def apply(self, params: Params) -> Params:
        return jax.tree_util.tree_map(lambda x: jnp.exp(x), params)

    def inv(self, params: Params) -> Params:
        return jax.tree_util.tree_map(lambda x: jnp.log(x), params)
