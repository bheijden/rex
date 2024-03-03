import jax
from jax.typing import ArrayLike
from typing import Any, Tuple, List, TypeVar, Dict, Union, Callable
from flax import struct
from flax.core import FrozenDict
import jax.numpy as jnp
from rex.proto import log_pb2
from rex.distributions import Gaussian
import rex.jax_utils as rjax
import numpy as onp


PyTree = Any
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


# def linear_activation(y: Union[float, jax.typing.ArrayLike], yp: jax.typing.ArrayLike) -> jax.Array:
#     """Linear activation function.
#
#     :param y: Value to interpolate
#     :param yp: Array of values to interpolate (monotonically increasing)
#     :return: Activation array
#     """
#     activation = jnp.ones_like(yp)
#     y1 = yp[1:]
#     y0 = yp[:-1]
#     a = jnp.clip((y - y0) / (y1 - y0), 0., 1.)
#     activation = activation.at[1:].set(a)
#     activation = activation.at[:-1].add(-a)
#     return activation


@struct.dataclass
class Delay:
    alpha: Union[float, jax.typing.ArrayLike] = struct.field(default=0.5)  # Value between [0, 1]
    min: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=0.0)  # Minimum expected delay
    max: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=0.0)  # Maximum expected delay
    # method: Callable = struct.field(pytree_node=False, default=linear_activation)  # Defines interpolation method: "linear"
    # discrete: bool = struct.field(pytree_node=False, default=True)  # Defines if discrete or continuous interpolation
    active: bool = struct.field(pytree_node=False, default=True)  # Defines if the delay is active

    def delay_window(self, rate_out: Union[float, jax.typing.ArrayLike]) -> int:
        if not self.active:
            return 0
        return int(onp.ceil(rate_out * (self.max - self.min)).astype(int))

    @property
    def value(self) -> Union[float, jax.Array]:
        """Value of delay."""
        if not self.active:
            raise ValueError("Delay is not active.")
        return self.min + self.alpha * (self.max - self.min)

    def min_dist(self) -> Gaussian:
        """Returns a Gaussian distribution that with the minimum delay as mean and 0.0 as variance."""
        if not self.active:
            raise ValueError("Delay is not active.")
        return Gaussian(self.min, 0.0)

    @classmethod
    def from_info(cls, info: log_pb2.Delay):
        return cls(alpha=info.alpha, min=info.min, max=info.max,
                   # discrete=info.discrete,
                   active=info.active)

    @property
    def info(self) -> log_pb2.Delay:
        return log_pb2.Delay(alpha=self.alpha, min=self.min, max=self.max,
                             # discrete=self.discrete,
                             active=self.active)

    def static_equal(self, other: "Delay") -> bool:
        return self.min == other.min and self.max == other.max and self.active == other.active

    # def activation(self, delays: jax.typing.ArrayLike) -> jax.Array:
    #     """Activation function.
    #     :param delays: Array of delays (monotonically decreasing)
    #     :return: Activation array
    #     """
    #     if not self.active:
    #         raise ValueError("Delay is not active.")
    #     activation = self.method(self.value, delays)
    #     return activation
    #     # if not self.active:
    #     #     raise ValueError("Delay is not active.")
    #     # activation = self.method(self.value, jnp.flip(delays))
    #     # return jnp.flip(activation)
    #
    # def interpolate(self, delays: jax.typing.ArrayLike, x: PyTree, discrete: bool = None) -> PyTree:
    #     """Interpolate tree of values. The interpolation method is defined by the method attribute.
    #
    #     Note: - The shape of the input x must be (window, *x.shape)
    #           - The shape of the output x_interp will be (*x.shape)
    #           - The shape of the delays must be (window)
    #           - discrete (true/false) defines if the interpolation is discrete or continuous
    #
    #     :param delays: Array of delays (monotonically decreasing)
    #     :param x: PyTree of values to interpolate
    #     :param discrete: Defines if discrete or continuous interpolation (overrides self.discrete)
    #     """
    #     if not self.active:
    #         return x
    #     activation = self.activation(delays)
    #
    #     # Overwrite discrete if it is not None
    #     discrete = self.discrete if discrete is None else discrete
    #
    #     if discrete:
    #         idx_max = jnp.argmax(activation)
    #         _interpolate = lambda _x: _x[idx_max]
    #     else:
    #         def _interpolate(_x):
    #             reshaped_activation = activation.reshape(activation.shape + (1,) * (len(_x.shape) - 1))
    #             return jnp.sum(reshaped_activation * _x, axis=0)
    #     x_interp = jax.tree_util.tree_map(_interpolate, x)
    #     return x_interp


@struct.dataclass
class InputState:
    """A ring buffer that holds the inputs for a node's input channel."""

    seq: ArrayLike
    ts_sent: ArrayLike
    ts_recv: ArrayLike
    data: Output  # --> must be a pytree where the shape of every leaf will become (size, *leafs.shape)
    delay_sysid: Delay = struct.field(default=None)
    rate: float = struct.field(pytree_node=False, default=None)  # Rate of output

    # @jax.jit
    def apply_delay(self, ts_step: float, delay: Delay = None) -> "InputState":
        """Apply the delay to the input state."""
        assert delay is None or (self.delay_sysid is not None and self.delay_sysid.static_equal(delay)), "The static parameters of the InputState and provided `delay` must be be equal."
        if self.delay_sysid is None or not self.delay_sysid.active:
            return self  # NOOP
        assert self.rate is not None, "Output rate must be defined to apply delay."
        window_delayed = self.delay_sysid.delay_window(self.rate)
        if window_delayed == 0:  # Return the input state if the only possible shift is 0.
            return self  # NOOP
        cum_window = self.seq.shape[0]
        window = cum_window - window_delayed
        d = delay.value if delay is not None else self.delay_sysid.value
        ts_recv = self.ts_sent + d
        idx_max = jnp.argwhere(ts_recv > ts_step, size=1, fill_value=cum_window)[0, 0]
        idx_min = idx_max - window

        # Slice the input state
        tb = [self.seq, self.ts_sent, ts_recv, self.data]
        slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), tb)
        tb_delayed = jax.tree_map(lambda _tb, _size: jax.lax.dynamic_slice(_tb, [idx_min] + [0 * s for s in _size], [window] + _size), tb, slice_sizes)
        delayed_input_state = InputState(*tb_delayed, delay_sysid=self.delay_sysid.replace(active=False), rate=self.rate)
        return delayed_input_state

    @classmethod
    def from_outputs(
        cls, seq: ArrayLike, ts_sent: ArrayLike, ts_recv: ArrayLike, outputs: List[Any], delay_sysid: Delay = None, rate: float = None, is_data: bool = False
    ) -> "InputState":
        """Create an InputState from a list of outputs.

        The oldest message should be first in the list.
        """

        data = jax.tree_map(lambda *o: jnp.stack(o, axis=0), *outputs) if not is_data else outputs
        return cls(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=data, delay_sysid=delay_sysid, rate=rate)

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
        return InputState(*new, delay_sysid=self.delay_sysid, rate=self.rate)

    def __getitem__(self, val):
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        return InputState(*jax.tree_map(lambda _tb: _tb[val], tb), delay_sysid=self.delay_sysid, rate=self.rate)


@struct.dataclass
class StepState:
    rng: jax.Array
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

    def try_get_node(self, node_name: str) -> Union[StepState, None]:
        return self.nodes.get(node_name, None)


@struct.dataclass
class CompiledGraphState(GraphState):
    step: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    timings: Timings = struct.field(pytree_node=False, default_factory=lambda: None)
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

