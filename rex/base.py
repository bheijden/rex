import jax
import jumpy
from typing import Any, Union, List, TypeVar, Dict, Union
from flax import struct
from flax.core import FrozenDict
import jumpy.numpy as jp
import rex.jumpy as rjp


Output = TypeVar('Output')
State = TypeVar('State')
Params = TypeVar('Params')
SeqsMapping = Dict[str, jp.ndarray]
BufferSizes = Dict[str, List[int]]
NodeTimings = Dict[str, Dict[str, Union[jp.ndarray, Dict[str, Dict[str, jp.ndarray]]]]]
Timings = List[NodeTimings]


@struct.dataclass
class Empty: pass


@struct.dataclass
class InputState:
    """A ring buffer that holds the inputs for a node's input channel."""
    seq: jp.ndarray
    ts_sent: jp.ndarray
    ts_recv: jp.ndarray
    data: Output  # --> must be a pytree where the shape of every leaf will become (size, *leafs.shape)

    @classmethod
    def from_outputs(cls, seq: jp.ndarray, ts_sent: jp.ndarray, ts_recv: jp.ndarray, outputs: List[Any], is_data: bool = False) -> "InputState":
        """Create an InputState from a list of outputs.

        The oldest message should be first in the list.
        """

        data = jax.tree_map(lambda *o: jp.stack(o, axis=0), *outputs) if not is_data else outputs
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
            new = jax.tree_map(lambda tb, t: self._shift(tb, t), tb, new_t)
        else:
            new = jax.tree_map(lambda _tb, _t: rjp.index_update(_tb, jp.int32(0), _t, copy=True), tb, new_t)
        return InputState(*new)

    def __getitem__(self, val):
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        return InputState(*jax.tree_map(lambda _tb: _tb[val], tb))


@struct.dataclass
class StepState:
    rng: jp.ndarray
    state: State
    params: Params
    inputs: FrozenDict[str, InputState] = struct.field(pytree_node=True, default_factory=lambda: None)
    eps: rjp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.int32(0))
    seq: rjp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.int32(0))
    ts: rjp.float32 = struct.field(pytree_node=True, default_factory=lambda: jp.float32(0.))


@struct.dataclass
class GraphBuffer:
    """
    outputs: A ring buffer that holds the outputs for every node's output channel.
    timings: The timings for a given episode (i.e. GraphState.timings[eps]).
    """
    outputs: FrozenDict[str, Output]
    timings: Timings


@struct.dataclass
class GraphState:
    step: rjp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.int32(0))
    eps: rjp.int32 = struct.field(pytree_node=True, default_factory=lambda: jp.int32(0))
    nodes: FrozenDict[str, StepState] = struct.field(pytree_node=True, default_factory=lambda: None)
    timings: Timings = struct.field(pytree_node=True, default_factory=lambda: None)
    buffer: GraphBuffer = struct.field(pytree_node=True, default_factory=lambda: None)


