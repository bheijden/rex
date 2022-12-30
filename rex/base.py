from functools import partial
import jax
from typing import Any, Union, List, TypeVar
from flax import struct
from flax.core import FrozenDict
import jumpy as jp
import rex.jumpy as rjp


Output = TypeVar('Output')
State = TypeVar('State')
Params = TypeVar('Params')


@struct.dataclass
class InputState:
    """A ring buffer that holds the inputs for a node's input channel."""
    seq: jp.ndarray
    ts_sent: jp.ndarray
    ts_recv: jp.ndarray
    data: Output  # --> must be a pytree where the shape of every leaf will become (size, *leafs.shape)

    @classmethod
    def from_outputs(cls, seq: jp.ndarray, ts_sent: jp.ndarray, ts_recv: jp.ndarray, outputs: List[Any]) -> "InputState":
        """Create an InputState from a list of outputs.

        The oldest message should be the first in the list.
        """
        data = jp.tree_map(lambda *o: jp.stack(o, axis=0), *outputs)
        return cls(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=data)

    def _shift(self, a: jp.ndarray, new: jp.ndarray):
        rolled_a = jp.roll(a, -1, axis=0)
        new_a = rjp.index_update(rolled_a, -1, new, copy=True)
        return new_a

    # @partial(jax.jit, static_argnums=(0,))
    def push(self, seq: int, ts_sent: float, ts_recv: float, data: Any) -> "InputState":
        # todo: in-place update when we use numpy.
        size = self.seq.shape[0]
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        new_t = [seq, ts_sent, ts_recv, data]

        # get new values
        if size > 1:
            new = jp.tree_map(lambda tb, t: self._shift(tb, t), tb, new_t)
        else:
            new = jp.tree_map(lambda _tb, _t: rjp.index_update(_tb, jp.int32(0), _t, copy=True), tb, new_t)
        return InputState(*new)

    def __getitem__(self, val):
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        return InputState(*jp.tree_map(lambda _tb: _tb[val], tb))


@struct.dataclass
class StepState:
    rng: jp.ndarray
    inputs: FrozenDict[str, InputState]
    state: State
    params: Params


@struct.dataclass
class GraphState:
    nodes: FrozenDict[str, StepState]
    step: rjp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.int32(0))
    outputs: FrozenDict[str, Output] = struct.field(pytree_node=True, default_factory=lambda: FrozenDict({}))
