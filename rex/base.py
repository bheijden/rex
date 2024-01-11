import jax
from jax.typing import ArrayLike
from typing import Any, Tuple, List, TypeVar, Dict, Union
from flax import struct
from flax.core import FrozenDict
import jax.numpy as jnp
import rex.jax_utils as rjax
import numpy as onp


Output = TypeVar("Output")
State = TypeVar("State")
Params = TypeVar("Params")
SeqsMapping = Dict[str, onp.ndarray]
BufferSizes = Dict[str, List[int]]
NodeTimings = Dict[str, Dict[str, Union[onp.ndarray, Dict[str, Dict[str, onp.ndarray]]]]]
Timings = List[NodeTimings]
GraphBuffer = FrozenDict[str, Output]


@struct.dataclass
class Empty:
    pass


@struct.dataclass
class InputState:
    """A ring buffer that holds the inputs for a node's input channel."""

    seq: ArrayLike
    ts_sent: ArrayLike
    ts_recv: ArrayLike
    data: Output  # --> must be a pytree where the shape of every leaf will become (size, *leafs.shape)

    @classmethod
    def from_outputs(
        cls, seq: ArrayLike, ts_sent: ArrayLike, ts_recv: ArrayLike, outputs: List[Any], is_data: bool = False
    ) -> "InputState":
        """Create an InputState from a list of outputs.

        The oldest message should be first in the list.
        """

        data = jax.tree_map(lambda *o: jnp.stack(o, axis=0), *outputs) if not is_data else outputs
        return cls(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=data)

    def _shift(self, a: ArrayLike, new: ArrayLike):
        rolled_a = jnp.roll(a, -1, axis=0)
        new_a = jnp.array(rolled_a).at[-1].set(jnp.array(new))
        return new_a

    # @partial(jax.jit, static_argnums=(0,))
    def push(self, seq: int, ts_sent: float, ts_recv: float, data: Any) -> "InputState":
        size = self.seq.shape[0]
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        new_t = [seq, ts_sent, ts_recv, data]

        # get new values
        if size > 1:
            new = jax.tree_map(lambda tb, t: self._shift(tb, t), tb, new_t)
        else:
            new = jax.tree_map(lambda _tb, _t: jnp.array(_tb).at[0].set(_t), tb, new_t)
        return InputState(*new)

    def __getitem__(self, val):
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        return InputState(*jax.tree_map(lambda _tb: _tb[val], tb))


@struct.dataclass
class StepState:
    rng: jax.random.KeyArray
    state: State
    params: Params
    inputs: FrozenDict[str, InputState] = struct.field(pytree_node=True, default_factory=lambda: None)
    eps: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    seq: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    ts: Union[float, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.float32(0.0))


@struct.dataclass
class GraphState:
    eps: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    nodes: FrozenDict[str, StepState] = struct.field(pytree_node=True, default_factory=lambda: None)

    def replace_eps(self, eps: Union[int, ArrayLike]):
        eps = jnp.clip(eps, onp.int32(0), onp.int32(0))
        nodes = FrozenDict({n: ss.replace(eps=eps) for n, ss in self.nodes.items()})
        return self.replace(eps=eps, nodes=nodes)

    def replace_nodes(self, nodes: Union[Dict[str, StepState], FrozenDict[str, StepState]]):
        return self.replace(nodes=self.nodes.copy(nodes))


@struct.dataclass
class CompiledGraphState(GraphState):
    step: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    timings: Timings = struct.field(pytree_node=True, default_factory=lambda: None)
    # The timings for a single episode (i.e. GraphState.timings[eps]).
    timings_eps: Timings = struct.field(pytree_node=True, default_factory=lambda: None)
    # A ring buffer that holds the outputs for every node's output channel.
    buffer: FrozenDict[str, Output] = struct.field(pytree_node=True, default_factory=lambda: None)

    def replace_buffer(self, outputs: Union[Dict[str, Output], FrozenDict[str, Output]]):
        return self.replace(buffer=self.buffer.copy(outputs))

    def replace_eps(self, eps: Union[int, ArrayLike]):
        # Next(iter()) is a bit hacky, but it simply takes the first node in the final (i.e. [-1]) generations (i.e. the root).
        max_eps = next(iter(self.timings[-1].values()))["run"].shape[-2]
        eps = jnp.clip(eps, onp.int32(0), max_eps - 1)
        nodes = FrozenDict({n: ss.replace(eps=eps) for n, ss in self.nodes.items()})
        timings_eps = rjax.tree_take(self.timings, eps)
        return self.replace(eps=eps, nodes=nodes, timings_eps=timings_eps)

    def replace_step(self, step: Union[int, ArrayLike]):
        # Next(iter()) is a bit hacky, but it simply takes the first node of the final generation (i.e. [-1] = root_slot).
        max_step = next(iter(self.timings[-1].values()))["run"].shape[-1]
        step = jnp.clip(step, onp.int32(0), max_step - 1)
        return self.replace(step=step)


RexObs = Union[Dict[str, Any], ArrayLike]
RexResetReturn = Tuple[GraphState, RexObs, Dict]
RexStepReturn = Tuple[GraphState, RexObs, float, bool, bool, Dict]
StepStates = Union[Dict[str, StepState], FrozenDict[str, StepState]]
