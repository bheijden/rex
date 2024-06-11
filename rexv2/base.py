from typing import Any, Tuple, List, TypeVar, Dict, Union, TYPE_CHECKING, Sequence
import functools
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
import numpy as onp
from numpy import ma as ma
from flax import struct
from flax.core import FrozenDict

import rexv2.jax_utils as rjax

if TYPE_CHECKING:
    from rexv2.node import BaseNode

PyTree = Any
Output = TypeVar("Output")
State = TypeVar("State")
Params = TypeVar("Params")
GraphBuffer = FrozenDict[str, Output]


@struct.dataclass
class Timestamps:
    """A timestamps data structure that holds the sequence numbers and timestamps of a connection.

    Used to artificially generate graphs.

    Meant for internal use only.
    """

    seq: Union[int, jax.Array]
    ts_start: Union[float, jax.Array]
    ts_end: Union[float, jax.Array]
    ts_recv: Dict[str, Union[float, jax.Array]] = struct.field(default=None)


@struct.dataclass
class Edge:
    """And edge data structure that holds the sequence numbers and timestamps of a connection.

    This data structure may be batched and hold data for multiple episodes.
    The last dimension represent the data during the episode.

    Given a message from  node_out to node_in, the sequence number of the send message is seq_out. The message is received
    at node_in at time ts_recv. Seq_in is the sequence number of the call that node_in processes the message.

    When there are outputs that were never received, set the seq_in to -1.

    In case the received timestamps are not available, set ts_recv to a dummy value (e.g. 0.0).

    :param seq_out: The sequence number of the message. Must be monotonically increasing.
    :param seq_in: The sequence number of the call that the message is processed. Must be monotonically increasing.
    :param ts_recv: The time the message is received at the input node. Must be monotonically increasing.
    """

    seq_out: Union[int, jax.Array]
    seq_in: Union[int, jax.Array]
    ts_recv: Union[float, jax.Array]


@struct.dataclass
class Vertex:
    """A vertex data structure that holds the sequence numbers and timestamps of a node.

    This data structure may be batched and hold data for multiple episodes.
    The last dimension represent the sequence numbers during the episode.

    In case the timestamps are not available, set ts_start and ts_end to a dummy value (e.g. 0.0).

    Ideally, for every vertex seq[i] there should be an edge with seq_out[i] for every connected node in the graph.

    :param seq: The sequence number of the node. Should start at 0 and increase by 1 every step (no gaps).
    :param ts_start: The start time of the computation of the node (i.e. when the node starts processing step 'seq').
    :param ts_end: The end time of the computation of the node (i.e. when the node finishes processing step 'seq').
    """

    seq: Union[int, jax.Array]
    ts_start: Union[float, jax.Array]
    ts_end: Union[float, jax.Array]


@struct.dataclass
class Graph:
    """A computation graph data structure that holds the vertices and edges of a computation graph.

    This data structure is used to represent the computation graph of a system. It holds the vertices and edges of the
    graph. The vertices represent consecutive step calls of nodes, and the edges represent the data flow between connected
    nodes.

    Stateful edges must not be included in the edges, but are implicitly assumed. In other words, consecutive sequence numbers
    of the same node are assumed to be connected.

    The graph should be directed and acyclic. Cycles are not allowed.

    :param vertices: A dictionary of vertices. The keys are the unique names of the node type, and the values are the vertices.
    :param edges: A dictionary of edges. The keys are of the form (n1, n2), where n1 and n2 are the unique names of the
                  output and input nodes, respectively. The values are the edges.
    """

    vertices: Dict[str, Vertex]
    edges: Dict[Tuple[str, str], Edge]

    def __len__(self):
        """Return the number of episodes."""
        shape = next(iter(self.vertices.values())).seq.shape
        if len(shape) == 0:
            return 1
        else:
            return shape[0]

    def __getitem__(self, val):
        """In case the graph is batched, and holds the graphs of multiple episodes,
        this function returns the graph of a specific episode.
        """
        shape = next(iter(self.vertices.values())).seq.shape
        if len(shape) == 0:
            return self
        else:
            return jax.tree_util.tree_map(lambda v: v[val], self)


@struct.dataclass
class Window:
    """A window buffer that holds the sequence numbers and timestamps of a connection.

    Internal use only.
    """

    seq: Union[int, jax.Array]  # seq_out
    ts_sent: Union[float, jax.Array]  # ts_end[seq_out]
    ts_recv: Union[float, jax.Array]

    def __getitem__(self, val):
        return jax.tree_util.tree_map(lambda v: v[val], self)

    def _shift(self, a: jax.typing.ArrayLike, new: jax.typing.ArrayLike):
        rolled_a = jnp.roll(a, -1, axis=0)
        new_a = jnp.array(rolled_a).at[-1].set(jnp.array(new))
        return new_a

    def push(self, seq, ts_sent, ts_recv) -> "Window":
        seq = self._shift(self.seq, seq)
        ts_sent = self._shift(self.ts_sent, ts_sent)
        ts_recv = self._shift(self.ts_recv, ts_recv)
        return Window(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv)


@struct.dataclass
class WindowedVertex(Vertex):
    """A vertex with windows.

    Internal use only.
    """

    windows: Dict[str, Window]


@struct.dataclass
class WindowedGraph:
    """A graph with windows.

    Internal use only.
    """

    vertices: Dict[str, WindowedVertex]

    def __getitem__(self, val):
        return jax.tree_util.tree_map(lambda v: v[val], self)

    def to_graph(self) -> Graph:
        num_graphs = next(iter(self.vertices.values())).seq.shape[0]
        vertices = {n: Vertex(seq=v.seq, ts_start=v.ts_start, ts_end=v.ts_end) for n, v in self.vertices.items()}
        edges = dict()
        for n2, v2 in self.vertices.items():
            for n1, w in v2.windows.items():
                # Repeat seq_in to match the shape of seq_out
                seq_in = jnp.repeat(v2.seq, w.seq.shape[-1], axis=-1).reshape(num_graphs, -1, w.seq.shape[-1])

                # Flatten
                seq_out = w.seq.reshape(num_graphs, -1)
                seq_in = seq_in.reshape(num_graphs, -1)
                ts_recv = w.ts_recv.reshape(num_graphs, -1)
                edges[(n1, n2)] = Edge(seq_out=seq_out, seq_in=seq_in, ts_recv=ts_recv)
        return Graph(vertices=vertices, edges=edges)


@struct.dataclass
class SlotVertex(WindowedVertex):
    """A vertex with slots.

    Internal use only.
    """

    # seq: Union[int, jax.Array]
    # ts_start: Union[float, jax.Array]
    # ts_end: Union[float, jax.Array]
    # windows: Dict[str, Window]
    run: Union[bool, jax.Array]
    kind: str = struct.field(pytree_node=False)
    generation: int = struct.field(pytree_node=False)


@struct.dataclass
class Timings:
    """A data structure that holds the timings of the execution of a graph.

    Can be retrieved from the graph with graph.timings.

    Internal use only.
    """

    slots: Dict[str, SlotVertex]

    def to_generation(self) -> List[Dict[str, SlotVertex]]:
        generations = {}
        for n, s in self.slots.items():
            if s.generation not in generations:
                generations[s.generation] = {}
            generations[s.generation][n] = s
        return [generations[i] for i in range(len(generations))]

    def get_masked_timings(self):
        np_timings = jax.tree_util.tree_map(lambda v: onp.array(v), self)

        # Get node names
        node_kinds = set([s.kind for key, s in np_timings.slots.items()])

        # Convert timings to list of generations
        timings = {}
        for n, v in np_timings.slots.items():
            if v.generation not in timings:
                timings[v.generation] = {}
            timings[v.generation][n] = v
        timings = [timings[i] for i in range(len(timings))]

        # Get output buffer sizes
        masked_timings_slot = []
        for i_gen, gen in enumerate(timings):
            # t_flat = {slot: t for slot, t in gen.items()}
            slots = {k: [] for k in node_kinds}
            [slots[v.kind].append(v) for k, v in gen.items()]
            [slots.pop(k) for k in list(slots.keys()) if len(slots[k]) == 0]
            # slots:= [eps, step, slot_idx, window=optional]
            slots = {k: jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=2), *v) for k, v in slots.items()}

            def _mask(mask, arr):
                # Repeat mask in extra dimensions of arr (for inputs)
                if arr.ndim > mask.ndim:
                    extra_dim = tuple([mask.ndim + a for a in range(arr.ndim - mask.ndim)])
                    new_mask = onp.expand_dims(mask, axis=extra_dim)
                    for i in extra_dim:
                        new_mask = onp.repeat(new_mask, arr.shape[i], axis=-1)
                else:
                    new_mask = mask
                # print(mask.shape, arr.shape, new_mask.shape)
                masked_arr = ma.masked_array(arr, mask=new_mask)
                return masked_arr

            masked_slots = {k: jax.tree_util.tree_map(functools.partial(_mask, ~v.run), v) for k, v in slots.items()}
            masked_timings_slot.append(masked_slots)

        def _update_mask(j, arr):
            arr.mask[:, :, :, j] = True
            return arr

        def _concat_arr(a, b):
            return ma.concatenate((a, b), axis=2)

        # Combine timings for each slot. masked_timings := [eps, step, slot_idx, gen_idx, window=optional]
        masked_timings = {}
        for i_gen, gen in enumerate(masked_timings_slot):
            for key, t in gen.items():
                # Repeat mask in extra dimensions of arr (for number of gens, and mask all but the current i_gen)
                t = jax.tree_util.tree_map(lambda x: onp.repeat(x[:, :, :, None], len(timings), axis=3), t)

                # Update mask to be True for all other gens
                for j in range(len(timings)):
                    if j == i_gen:
                        continue
                    jax.tree_util.tree_map(functools.partial(_update_mask, j), t)

                # Add to masked_timings
                if key not in masked_timings:
                    # Add as new entry
                    masked_timings[key] = t.replace(generation=None)
                else:
                    # Concatenate with existing entry
                    t = t.replace(generation=masked_timings[key].generation)  # Ensures that static fields are the same
                    masked_timings[key] = jax.tree_util.tree_map(_concat_arr, masked_timings[key], t)
        return masked_timings

    def get_buffer_sizes(self):
        # Get masked timings:= [eps, step, slot_idx, gen_idx, window=optional]
        masked_timings = self.get_masked_timings()

        # Get min buffer size for each node
        name_mapping = {n: {o: o for o in s.windows} for n, s in masked_timings.items()}
        min_buffer_sizes = {
            k: {input_name: output_name for input_name, output_name in inputs.items()} for k, inputs in name_mapping.items()
        }
        node_buffer_sizes = {n: [] for n in masked_timings.keys()}
        for n, inputs in name_mapping.items():
            t = masked_timings[n]
            for input_name, output_name in inputs.items():
                # Determine min input sequence per generation (i.e. we reduce over all slots within a generation & window)
                seq_in = onp.amin(t.windows[input_name].seq, axis=(2, 4))
                seq_in = seq_in.reshape(
                    *seq_in.shape[:-2], -1
                )  # flatten over generation & step dimension (i.e. [s1g1, s1g2, ..], [s2g1, s2g2, ..], ..)
                # NOTE: fill masked steps with max value (to not influence buffer size)
                ma.set_fill_value(
                    seq_in, onp.iinfo(onp.int32).max
                )  # Fill with max value, because it will not influence the min
                filled_seq_in = seq_in.filled()
                max_seq_in = onp.minimum.accumulate(filled_seq_in[:, ::-1], axis=-1)[:, ::-1]

                # Determine max output sequence per generation
                seq_out = onp.amax(
                    masked_timings[output_name].seq, axis=(2,)
                )  # (i.e. we reduce over all slots within a generation)
                seq_out = seq_out.reshape(
                    *seq_out.shape[:-2], -1
                )  # flatten over generation & step dimension (i.e. [s1g1, s1g2, ..], [s2g1, s2g2, ..], ..)
                ma.set_fill_value(
                    seq_out, onp.iinfo(onp.int32).min
                )  # todo: CHECK! changed from -1 to onp.iinfo(onp.int32).min to deal with negative seq numbers
                filled_seq_out = seq_out.filled()
                max_seq_out = onp.maximum.accumulate(filled_seq_out, axis=-1)

                # Calculate difference to determine buffer size
                # NOTE: Offset output sequence by +1, because the output is written to the buffer AFTER the buffer is read
                offset_max_seq_out = onp.roll(max_seq_out, shift=1, axis=1)
                offset_max_seq_out[:, 0] = onp.iinfo(
                    onp.int32
                ).min  # todo: CHANGED to min value compared to --> NOTE: First step is always -1, because no node has run at this point.
                s = offset_max_seq_out - max_seq_in

                # NOTE! +1, because, for example, when offset_max_seq_out = 0, and max_seq_in = 0, we need to buffer 1 step.
                max_s = s.max() + 1

                # Store min buffer size
                min_buffer_sizes[n][input_name] = max_s
                node_buffer_sizes[output_name].append(max_s)

        return node_buffer_sizes

    def get_output_buffer(
        self, nodes: Dict[str, "BaseNode"], sizes=None, extra_padding: int = 0, graph_state=None, rng: jax.Array = None
    ):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        # if graph_state is None:
        #     raise ValueError("graph_state is required to get the output buffer.")

        # Get buffer sizes if not provided
        if sizes is None:
            sizes = self.get_buffer_sizes()

        # Create output buffers
        buffers = {}
        stack_fn = lambda *x: jnp.stack(x, axis=0)
        rngs = jax.random.split(rng, num=len(nodes))
        for idx, (n, s) in enumerate(sizes.items()):
            assert n in nodes, f"Node `{n}` not found in nodes."
            buffer_size = max(s) + extra_padding if len(s) > 0 else max(1, extra_padding)
            assert buffer_size > 0, f"Buffer size for node `{n}` is 0."
            b = jax.tree_util.tree_map(stack_fn, *[nodes[n].init_output(rngs[idx], graph_state=graph_state)] * buffer_size)
            buffers[n] = b
        return FrozenDict(buffers)


@struct.dataclass
class InputState:
    """A ring buffer that holds the inputs for a node's input channel.

    The size of the buffer is determined by the window size of the corresponding connection
    (i.e. node.connect(..., window=...)).

    :param seq: The sequence number of the received message.
    :param ts_sent: The time the message was sent.
    :param ts_recv: The time the message was received.
    :param data: The message of the connection (arbitrary pytree structure).
    """

    seq: ArrayLike
    ts_sent: ArrayLike
    ts_recv: ArrayLike
    data: Output  # --> must be a pytree where the shape of every leaf will become (size, *leafs.shape)

    @classmethod
    def from_outputs(
        cls, seq: ArrayLike, ts_sent: ArrayLike, ts_recv: ArrayLike, outputs: List[Any], is_data: bool = False
    ) -> "InputState":
        """Create an InputState from a list of messages, timestamps, and sequence numbers.

        The oldest message should be first in the list.
        :param seq: The sequence number of the received message.
        :param ts_sent: The timestamps of when the messages were sent.
        :param ts_recv: The timestamps of when the messages were received.
        :param outputs: The messages of the connection (arbitrary pytree structure).
        :param is_data: If True, the outputs are already a stacked pytree structure.
        :return: An InputState object, that holds the messages in a ring buffer.
        """

        data = jax.tree_map(lambda *o: jnp.stack(o, axis=0), *outputs) if not is_data else outputs
        return cls(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=data)

    def _shift(self, a: ArrayLike, new: ArrayLike):
        rolled_a = jnp.roll(a, -1, axis=0)
        new_a = jnp.array(rolled_a).at[-1].set(jnp.array(new))
        return new_a

    def push(self, seq: int, ts_sent: float, ts_recv: float, data: Any) -> "InputState":
        """Push a new message into the ring buffer."""
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
        """Get the value of the ring buffer at a specific index.

        This is useful for indexing all the values of the ring buffer at a specific index.
        """
        tb = [self.seq, self.ts_sent, self.ts_recv, self.data]
        return InputState(*jax.tree_map(lambda _tb: _tb[val], tb))


@struct.dataclass
class StepState:
    """Step state definition.

    It holds all the information that is required to step a node.

    :param rng: The random number generator. Used for sampling random processes. If used, it should be updated.
    :param state: The state of the node. Usually dynamic during an episode.
    :param params: The parameters of the node. Usually static during an episode.
    :param inputs: The inputs of the node. See InputState.
    :param eps: The current episode number.
    :param seq: The current step number. Automatically increases by 1 every step.
    :param ts: The current time step at the start of the step. Determined by the computation graph.
    """

    rng: jax.Array
    state: State
    params: Params
    inputs: FrozenDict[str, InputState] = struct.field(pytree_node=True, default_factory=lambda: None)
    eps: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    seq: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    ts: Union[float, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.float32(0.0))


@struct.dataclass
class GraphState:
    """Graph state definition.

    It holds all the information that is required to step a graph.

    :param step: The current step number. Automatically increases by 1 every step.
    :param eps: The current episode number. To update the episode, use GraphState.replace_eps.
    :param nodes: The step states of all nodes in the graph. To update the step state of a node, use GraphState.replace_nodes.
    :param timings_eps: The timings data structure that describes the execution and partitioning of the graph.
    :param buffer: The output buffer of the graph. It holds the outputs of nodes during the execution. Input buffers are
                   automatically filled with the outputs of previously executed step calls of other nodes.
    """
    # The number of partitions (excl. supervisor) have run in the current episode.
    step: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    eps: Union[int, ArrayLike] = struct.field(pytree_node=True, default_factory=lambda: onp.int32(0))
    nodes: FrozenDict[str, StepState] = struct.field(pytree_node=True, default_factory=lambda: None)
    # timings: Timings = struct.field(pytree_node=False, default_factory=lambda: None)
    # The timings for a single episode (i.e. GraphState.timings[eps]).
    timings_eps: Timings = struct.field(pytree_node=True, default_factory=lambda: None)
    # A ring buffer that holds the outputs for every node's output channel.
    buffer: FrozenDict[str, Output] = struct.field(pytree_node=True, default_factory=lambda: None)
    # Some auxillary data that can be used to store additional information (e.g. wrappers
    aux: FrozenDict[str, Any] = struct.field(pytree_node=True, default_factory=lambda: FrozenDict({}))

    def replace_buffer(self, outputs: Union[Dict[str, Output], FrozenDict[str, Output]]):
        """Replace the buffer with new outputs.

        Generally not used by the user, but by the graph itself.
        """
        return self.replace(buffer=self.buffer.copy(outputs))

    def replace_eps(self, timings: Timings, eps: Union[int, ArrayLike]):
        """Replace the current episode number and corresponding timings corresponding to the episode.

        :param timings: The timings data structure that contains all timings for all episodes. Can be retrieved from the graph
                        with graph.timings.
        :param eps: The new episode number.
        :return: A new GraphState with the updated episode number and timings.
        """
        # Next(iter()) is a bit hacky, but it simply takes a node and counts the number of eps.
        max_eps = next(iter(timings.slots.values())).run.shape[-2]
        eps = jnp.clip(eps, onp.int32(0), max_eps - 1)
        nodes = FrozenDict({n: ss.replace(eps=eps) for n, ss in self.nodes.items()})
        timings_eps = rjax.tree_take(timings, eps)
        return self.replace(eps=eps, nodes=nodes, timings_eps=timings_eps)

    def replace_step(self, timings: Timings, step: Union[int, ArrayLike]):
        """Replace the current step number.

        :param timings: The timings data structure that contains all timings for all episodes. Can be retrieved from the graph
                        with graph.timings.
        :param step: The new step number.
        :return: A new GraphState with the updated step number.
        """
        # Next(iter()) is a bit hacky, but it simply takes a node and counts the number of steps.
        max_step = next(iter(timings.slots.values())).run.shape[-1]
        step = jnp.clip(step, onp.int32(0), max_step - 1)
        return self.replace(step=step)

    def replace_nodes(self, nodes: Union[Dict[str, StepState], FrozenDict[str, StepState]]):
        """Replace the step states of the graph.

        :param nodes: The new step states per node (can be an incomplete set).
        :return: A new GraphState with the updated step states.
        """
        return self.replace(nodes=self.nodes.copy(nodes))

    def replace_aux(self, aux: Union[Dict[str, Any], FrozenDict[str, Any]]):
        """Replace the auxillary data of the graph.

        :param aux: The new auxillary data.
        :return: A new GraphState with the updated auxillary data.
        """
        return self.replace(aux=self.aux.copy(aux))

    def try_get_node(self, node_name: str) -> Union[StepState, None]:
        """Try to get the step state of a node if it exists.

        :param node_name: The name of the node.
        :return: The step state of the node if it exists, else None.
        """
        return self.nodes.get(node_name, None)

    def try_get_aux(self, aux_name: str) -> Union[Any, None]:
        """Try to get auxillary data of the graph if it exists.

        :param aux_name: The name of the aux.
        :return: The aux of the node if it exists, else None.
        """
        return self.aux.get(aux_name, None)


StepStates = Union[Dict[str, StepState], FrozenDict[str, StepState]]


@struct.dataclass
class Base:
    """Base functionality extending all dataclasses.

    These methods allow for dataclasses to be operated like arrays/matrices.
    """

    def __add__(self, o: Any) -> Any:
        try:
            # If o is a pytree, element-wise addition
            return jax.tree_util.tree_map(lambda x, y: x + y, self, o)
        except ValueError:
            # If o is a scalar, element-wise addition
            return jax.tree_util.tree_map(lambda x: x + o, self)

    def __sub__(self, o: Any) -> Any:
        try:
            # If o is a pytree, element-wise subtraction
            return jax.tree_util.tree_map(lambda x, y: x - y, self, o)
        except ValueError:
            # If o is a scalar, element-wise subtraction
            return jax.tree_util.tree_map(lambda x: x - o, self)

    def __mul__(self, o: Any) -> Any:
        try:
            # If o is a pytree, element-wise multiplication
            return jax.tree_util.tree_map(lambda x, y: x * y, self, o)
        except ValueError:
            # If o is a scalar, element-wise multiplication
            return jax.tree_util.tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return jax.tree_util.tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        try:
            # If o is a pytree, element-wise division
            return jax.tree_util.tree_map(lambda x, y: x / y, self, o)
        except ValueError:
            # If o is a scalar, element-wise division
            return jax.tree_util.tree_map(lambda x: x / o, self)

    def __getitem__(self, val):
        return jax.tree_util.tree_map(lambda x: x[val], self)

    def reshape(self, shape: Sequence[int]) -> Any:
        return jax.tree_util.tree_map(lambda x: x.reshape(shape), self)

    def select(self, o: Any, cond: jax.Array) -> Any:
        return jax.tree_util.tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

    def slice(self, beg: int, end: int) -> Any:
        return jax.tree_util.tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=axis, mode='wrap'), self)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def index_set(
            self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
    ) -> Any:
        return jax.tree_util.tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(
            self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
    ) -> Any:
        return jax.tree_util.tree_map(lambda x, y: x.at[idx].add(y), self, o)


@struct.dataclass
class Empty(Base):
    """Empty class."""
    pass