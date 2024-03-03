from typing import TYPE_CHECKING, TypeVar, Tuple, Callable, Any, Union, Dict, Sequence
import jax
import jax.numpy as jnp
from flax import struct
from rex.base import PyTree


Params = Dict[str, PyTree]
Filter = Dict[str, PyTree]


@struct.dataclass
class Transform:
    @classmethod
    def init(cls, *args, **kwargs):
        raise NotImplementedError

    def apply(self, params: Params) -> Params:
        raise NotImplementedError

    def inv(self, params: Params) -> Params:
        raise NotImplementedError


LossArgs = Tuple[Transform]
Loss = Callable[[Params, LossArgs, jax.Array], Union[float, jax.Array]]

