from typing import Any, Dict, List, Tuple, Callable, Union
import functools
import networkx as nx
from flax.core import FrozenDict
import jumpy
import jax
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import rex.jumpy as rjp

from rex.utils import deprecation_warning
from rex.node import Node
from rex.graph import BaseGraph
from rex.base import InputState, StepState, StepStates, CompiledGraphState, GraphState, Output, Timings, GraphBuffer
from rex.supergraph import get_node_data, get_graph_buffer


int32 = Union[jnp.int32, onp.int32]
float32 = Union[jnp.float32, onp.float32]


def get_buffer_size(buffer: GraphBuffer) -> jp.int32:
    leaves = jax.tree_util.tree_leaves(buffer)
    size = leaves[0].shape[0] if len(leaves) > 0 else 1
    return size


def update_output(buffer: GraphBuffer, output: Output, seq: int32) -> Output:
    size = get_buffer_size(buffer)
    mod_seq = seq % size
    # todo: copy=True needed? --> `False` would lead to faster execution with numpy backend
    new_buffer = jax.tree_map(lambda _b, _o: rjp.index_update(_b, mod_seq, _o, copy=True), buffer, output)
    return new_buffer


def make_update_state(name: str):
    def _update_state(graph_state: CompiledGraphState, timing: Dict, step_state: StepState, output: Any) -> CompiledGraphState:
        # Define node's step state update
        new_nodes = dict()
        new_outputs = dict()

        # Increment sequence number
        new_ss = step_state.replace(seq=step_state.seq + 1)

        # Add node's step state update
        new_nodes[name] = new_ss
        new_outputs[name] = update_output(graph_state.buffer[name], output, timing["seq"])

        graph_state = graph_state.replace_buffer(new_outputs)
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
        return new_graph_state

    return _update_state


def make_update_inputs(name: str, inputs_data: Dict[str, Dict[str, str]]):
    def _update_inputs(graph_state: CompiledGraphState, timings_node: Dict) -> StepState:
        ss = graph_state.nodes[name]
        ts_step = timings_node["ts_step"]
        eps = graph_state.eps
        seq = timings_node["seq"]
        new_inputs = dict()
        for node_name, data in inputs_data.items():
            input_name = data["input_name"]
            t = timings_node["inputs"][input_name]
            buffer = graph_state.buffer[node_name]
            size = get_buffer_size(buffer)
            mod_seq = t["seq"] % size
            inputs = rjp.tree_take(buffer, mod_seq)
            _new = InputState.from_outputs(t["seq"], t["ts_sent"], t["ts_recv"], inputs, is_data=True)
            new_inputs[input_name] = _new

        return ss.replace(eps=eps, seq=seq, ts=ts_step, inputs=FrozenDict(new_inputs))

    return _update_inputs


def old_make_run_S(nodes: Dict[str, "Node"], S: nx.DiGraph, generations: List[List[str]], skip: List[str] = None):
    # todo: Buffer size may not be big enough, when updating graph_state during generational loop.
    #       Specifically, get_buffer_sizes_from_timings must be adapted to account for this.
    #       Or, alternatively, we can construct S such that it has a single node per generation (i.e. linear graph).
    #       This is currently not yet enforced.
    INTERMEDIATE_UPDATE = False

    # Determine which slots to skip
    skip_slots = [n for n, data in S.nodes(data=True) if data["kind"] in skip] if skip is not None else []

    # Define update function
    root_slot = generations[-1][0]
    root = S.nodes[root_slot]["kind"]
    node_data = get_node_data(S)
    update_input_fns = {name: make_update_inputs(name, data["inputs"]) for name, data in node_data.items()}

    def _run_node(kind: str, graph_state: CompiledGraphState, timings_node: Dict):
        # Update inputs
        ss = update_input_fns[kind](graph_state, timings_node)
        # ss = _old_ss

        # Run node step
        _new_ss, output = nodes[kind].step(ss)

        # Increment sequence number
        _new_seq_ss = _new_ss.replace(seq=_new_ss.seq + 1)

        # Update output buffer
        _new_output = update_output(graph_state.buffer[kind], output, timings_node["seq"])
        # _new_output = graph_state.outputs[kind]
        return _new_seq_ss, _new_output

    node_step_fns = {kind: functools.partial(_run_node, kind) for kind in nodes.keys()}

    def _run_generation(graph_state: CompiledGraphState, timings_gen: Dict):
        new_nodes = dict()
        new_outputs = dict()
        for slot_kind, timings_node in timings_gen.items():
            # Skip slots
            if slot_kind in skip_slots:
                continue

            if INTERMEDIATE_UPDATE:
                new_nodes = dict()
                new_outputs = dict()
            kind = S.nodes[slot_kind]["kind"]  # Node kind to run
            pred = timings_gen[slot_kind]["run"]  # Predicate for running node step

            # Prepare old states
            _old_ss = graph_state.nodes[kind]
            _old_output = graph_state.buffer[kind]

            # Add dummy inputs to old step_state (else jax complains about structural mismatch)
            if _old_ss.inputs is None:
                raise DeprecationWarning("Inputs should not be None, but pre-filled via graph.init")
                # _old_ss = update_input_fns[kind](graph_state, timings_node)

            # Run node step
            no_op = lambda *args: (_old_ss, _old_output)
            # no_op = jax.checkpoint(no_op) # todo: apply jax.checkpoint to no_op?
            try:
                new_ss, new_output = jumpy.lax.cond(pred, node_step_fns[kind], no_op, graph_state, timings_node)
            except TypeError as e:
                new_ss, new_output = node_step_fns[kind](graph_state, timings_node)
                print(f"TypeError: kind={kind}")
                raise e

            # Store new state
            new_nodes[kind] = new_ss
            new_outputs[kind] = new_output

            # Update buffer
            if INTERMEDIATE_UPDATE:
                graph_state = graph_state.replace_buffer(new_outputs)
                # new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy(new_outputs))
                graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))

        if INTERMEDIATE_UPDATE:
            new_graph_state = graph_state
        else:
            graph_state = graph_state.replace_buffer(new_outputs)
            # new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy(new_outputs))
            new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
        return new_graph_state

    def _run_S(graph_state: CompiledGraphState) -> CompiledGraphState:
        # Get eps & step  (used to index timings)
        graph_state = graph_state.replace_step(step=graph_state.step)  # Make sure step is clipped to max_step size
        step = graph_state.step

        # Determine slice sizes (depends on window size)
        # [1:] because first dimension is step.
        timings = graph_state.timings_eps
        slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), timings)

        # Slice timings
        timings_mcs = jax.tree_map(
            lambda _tb, _size: rjp.dynamic_slice(_tb, [step] + [0 * s for s in _size], [1] + _size)[0], timings, slice_sizes
        )

        # Run generations
        # NOTE! len(generations)+1 = len(timings_mcs) --> last generation is the root.
        for gen, timings_gen in zip(generations[:-1], timings_mcs):
            assert all([node in gen for node in timings_gen.keys()]), f"Missing nodes in timings: {gen}"
            graph_state = _run_generation(graph_state, timings_gen)

        # Run root input update
        new_ss_root = update_input_fns[root](graph_state, timings_mcs[-1][root_slot])
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy({root: new_ss_root}))

        # Increment step (new step may exceed max_step) --> clipping is done at the start of run_S.
        graph_state = graph_state.replace(step=graph_state.step + 1)
        return graph_state

    return _run_S


def new_make_run_S(nodes: Dict[str, "Node"], S: nx.DiGraph, generations: List[List[str]], skip: List[str] = None):
    # Determine which slots to skip
    skip_slots = [n for n, data in S.nodes(data=True) if data["kind"] in skip] if skip is not None else []

    # Define update function
    root_slot = generations[-1][0]
    root = S.nodes[root_slot]["kind"]
    node_data = get_node_data(S)
    update_input_fns = {name: make_update_inputs(name, data["inputs"]) for name, data in node_data.items()}

    def _run_node(kind: str, graph_state: CompiledGraphState, timings_node: Dict):
        # Update inputs
        ss = update_input_fns[kind](graph_state, timings_node)

        # Run node step
        _new_ss, output = nodes[kind].step(ss)

        # Increment sequence number
        _new_seq_ss = _new_ss.replace(seq=_new_ss.seq + 1)

        # Update output buffer
        _new_output = update_output(graph_state.buffer[kind], output, timings_node["seq"])

        # Update buffer
        # todo: Somehow, updating buffer inside this function compiles much slower than updating it outside (for large number of nodes)...
        # new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy({kind: _new_output}))
        graph_state = graph_state.replace_buffer({kind: _new_output})
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy({kind: _new_seq_ss}))
        return new_graph_state

    node_step_fns = {kind: functools.partial(_run_node, kind) for kind in nodes.keys()}

    def _run_generation(graph_state: CompiledGraphState, timings_gen: Dict):
        for slot_kind, timings_node in timings_gen.items():
            # Skip slots todo: not tested yet
            if slot_kind in skip_slots:
                continue
            # todo: Buffer size may not be big enough, when updating graph_state during generational loop.
            #       Specifically, get_buffer_sizes_from_timings must be adapted to account for this.
            #       Or, alternatively, we can construct S such that it has a single node per generation (i.e. linear graph).
            #       This is currently not yet enforced.
            kind = S.nodes[slot_kind]["kind"]
            pred = timings_gen[slot_kind]["run"]

            # Add dummy inputs to old step_state (else jax complains about structural mismatch)
            if graph_state.nodes[kind].inputs is None:
                raise DeprecationWarning("Inputs should not be None, but pre-filled via graph.init")
                # graph_state = graph_state.replace(
                #     nodes=graph_state.nodes.copy({kind: update_input_fns[kind](graph_state, timings_node)})
                # )

            # Run node step
            # todo: apply jax.checkpoint to no_op?
            no_op = lambda *args: graph_state
            graph_state = jumpy.lax.cond(pred, node_step_fns[kind], no_op, graph_state, timings_node)
        return graph_state

    def _run_S(graph_state: CompiledGraphState) -> CompiledGraphState:
        # Get eps & step  (used to index timings)
        graph_state = graph_state.replace_step(step=graph_state.step)  # Make sure step is clipped to max_step size
        step = graph_state.step

        # Determine slice sizes (depends on window size)
        # [1:] because first dimension is step.
        timings = graph_state.timings_eps
        slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), timings)

        # Slice timings
        timings_mcs = jax.tree_map(
            lambda _tb, _size: rjp.dynamic_slice(_tb, [step] + [0 * s for s in _size], [1] + _size)[0], timings, slice_sizes
        )

        # Run generations
        # NOTE! len(generations)+1 = len(timings_mcs) --> last generation is the root.
        for gen, timings_gen in zip(generations[:-1], timings_mcs):
            assert all([node in gen for node in timings_gen.keys()]), f"Missing nodes in timings: {gen}"
            graph_state = _run_generation(graph_state, timings_gen)

        # Run root input update
        new_ss_root = update_input_fns[root](graph_state, timings_mcs[-1][root_slot])
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy({root: new_ss_root}))

        # Increment step (new step may exceed max_step) --> clipping is done at the start of run_S.
        graph_state = graph_state.replace(step=graph_state.step + 1)
        return graph_state

    return _run_S


make_run_S = old_make_run_S


# def make_graph_reset(
#     S: nx.DiGraph, generations: List[List[str]], run_S: Callable, default_timings: Timings, default_buffer: GraphBuffer
# ):
#     # Determine root node (always in the last generation)
#     root_slot = generations[-1][0]
#     root = S.nodes[root_slot]["kind"]
#
#     # Define update function
#     node_data = get_node_data(S)
#     update_input_fns = {name: make_update_inputs(name, data["inputs"]) for name, data in node_data.items()}
#
#     def _graph_reset(graph_state: CompiledGraphState) -> Tuple[CompiledGraphState, StepState]:
#         # Get buffer
#         buffer = graph_state.buffer  # if graph_state.buffer is not None else default_buffer
#         assert buffer is not None, "The graph_state.buffer is None and no default_buffer was provided."
#
#         # Get timings
#         timings = graph_state.timings  # if graph_state.timings is not None else default_timings
#         assert timings is not None, "The graph_state.timings is None and no default_timings were provided."
#
#         # Determine episode timings
#         # NOTE! The graph_state.step indexes into the timings and indicates what subgraphs have run.
#         #       All subgraphs up until (and including) the graph.state'th subgraph have run.
#         #       I.E., graph_state.step=0 means that we have run the first subgraph AFTER this function has run.
#         #       We do not increment step+1 here, because we increment before running a chunk in graph_step.
#         max_eps, max_step = timings[-1][root_slot]["run"].shape
#         eps = jp.clip(graph_state.eps, jp.int32(0), max_eps - 1)
#         step = jp.clip(graph_state.step, jp.int32(0), max_step - 1)
#
#         # Replace buffer
#         graph_state = graph_state.replace_eps(eps)  # replaces timings according to eps
#         graph_state = graph_state.replace(step=step)  # replaces step
#
#         # Run initial chunk.
#         _next_graph_state = run_S(graph_state)
#
#         # Update input
#         next_timing = rjp.tree_take(graph_state.timings_eps[-1][root_slot], i=step)
#         next_ss = update_input_fns[root](_next_graph_state, next_timing)
#         next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({root: next_ss}))
#         return next_graph_state, next_ss
#
#     return _graph_reset
#
#
# def make_graph_step(S: nx.DiGraph, generations: List[List[str]], run_S: Callable):
#     # Determine root node (always in the last generation)
#     root_slot = generations[-1][0]
#     root = S.nodes[root_slot]["kind"]
#
#     # Define update function
#     node_data = get_node_data(S)
#     update_state = make_update_state(root)
#     update_input = make_update_inputs(root, node_data[root]["inputs"])
#
#     def _graph_step(graph_state: CompiledGraphState, step_state: StepState, action: Any) -> Tuple[CompiledGraphState, StepState]:
#         # Update graph_state with action
#         timing = rjp.tree_take(graph_state.timings_eps[-1][root_slot], i=graph_state.step)
#         new_graph_state = update_state(graph_state, timing, step_state, action)
#
#         # Grab step
#         # NOTE! The graph_state.step is used to index into the timings.
#         #       Therefore, we increment it before running the subgraph so that we index into the timings of the next step.
#         #       Hence, graph_state.step indicates what subgraphs have run.
#         max_step = new_graph_state.timings_eps[-1][root_slot]["run"].shape[0]
#         next_step = jp.clip(new_graph_state.step + 1, jp.int32(0), max_step - 1)
#         graph_state = new_graph_state.replace(step=next_step)
#
#         # Run chunk of next step.
#         _next_graph_state = run_S(graph_state)
#
#         # Update input
#         next_timing = rjp.tree_take(graph_state.timings_eps[-1][root_slot], i=next_step)
#         next_ss = update_input(_next_graph_state, next_timing)
#         next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({root: next_ss}))
#         return next_graph_state, next_ss
#
#     return _graph_step
#

class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"], root: Node, S: nx.DiGraph, default_timings: Timings = None, skip: List[str] = None):
        super().__init__(root=root, nodes=nodes)
        self._S = S
        self._default_timings = default_timings
        self._skip = skip if isinstance(skip, list) else [skip]

        if default_timings is None:
            deprecation_warning("default_timings is None. This means that the graph will not be able to run without a buffer.", stacklevel=2)

        # Get generations
        self._generations = list(nx.topological_generations(S))
        self._root_slot = self._generations[-1][0]
        self._root_kind = S.nodes[self._root_slot]["kind"]
        self._node_data = get_node_data(S)

        # Make chunk runner
        self.__run_until_root = make_run_S(self.nodes_and_root, S, self._generations, skip=skip)

    def __getstate__(self):
        args, kwargs = (), dict(nodes=self.nodes, root=self.root, S=self.S, default_timings=self._default_timings)
        return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state
        super().__setstate__((args, kwargs))

    @property
    def S(self):
        return self._S

    def init(self, rng: jp.ndarray = None, step_states: StepStates = None, starting_step: jp.int32 = 0,
             starting_eps: jp.int32 = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None) -> CompiledGraphState:
        new_gs: GraphState = super().init(rng=rng, step_states=step_states, starting_eps=starting_eps,
                                          randomize_eps=randomize_eps, order=order)

        # Convert BaseGraphState to CompiledGraphState
        assert self._default_timings is not None, "No default timings provided (not implemented yet). Cannot initialize graph."
        timings = self._default_timings

        # Get buffer & episode timings (i.e. timings[eps]) # todo: what if buffer already provided?
        buffer = get_graph_buffer(self._S, timings, self.nodes_and_root, graph_state=new_gs)

        # Prepare new CompiledGraphState
        new_cgs = CompiledGraphState(step=None, eps=None, nodes=new_gs.nodes, timings=timings, timings_eps=None, buffer=buffer)
        new_cgs = new_cgs.replace_step(step=starting_step)  # (Clips step to valid value)
        new_cgs = new_cgs.replace_eps(eps=new_gs.eps)  # (Clips eps to valid value & updates timings_eps)
        return new_cgs

    def run_until_root(self, graph_state: CompiledGraphState) -> CompiledGraphState:
        # Run supergraph (except for root)
        graph_state = self.__run_until_root(graph_state)
        return graph_state

    def run_root(self, graph_state: CompiledGraphState, step_state: StepState = None, output: Output = None) -> CompiledGraphState:
        """Runs root node if step_state and output are not provided. Otherwise, overrides step_state and output with provided values."""
        assert (step_state is None) == (output is None), "Either both step_state and output must be None or both must be not None."
        # Make update state function
        update_state = make_update_state(self._root_kind)
        root_slot = self._root_slot
        root = self.root

        def _run_root_step() -> CompiledGraphState:
            # Get next step state and output from root node
            if step_state is None and output is None:  # Run root node
                # ss = graph_state.nodes[root_kind]
                ss = root.get_step_state(graph_state)
                new_ss, new_output = root.step(ss)
            else:  # Override step_state and output
                new_ss, new_output = step_state, output

            # Update graph state
            new_graph_state = graph_state.replace_step(step=graph_state.step)  # Make sure step is clipped to max_step size
            timing = rjp.tree_take(graph_state.timings_eps[-1][root_slot], i=new_graph_state.step)
            new_graph_state = update_state(graph_state, timing, new_ss, new_output)
            return new_graph_state

        def _skip_root_step() -> CompiledGraphState:
            return graph_state

        # Run root node if step > 0, else skip
        graph_state = jumpy.lax.cond(graph_state.step == 0, _skip_root_step, _run_root_step)
        return graph_state

    # def run(self, graph_state: CompiledGraphState) -> CompiledGraphState:
    #     # todo: can this be standardized (i.e. moved to BaseGraph)?
    #     """Runs graph (incl. root) for one step and returns new graph state.
    #     This means the graph_state *after* the root step is returned.
    #     """
    #     # Runs supergraph (except for root)
    #     graph_state = self.run_until_root(graph_state)
    #
    #     # Runs root node if no step_state or output is provided, otherwise uses provided step_state and output
    #     graph_state = self.run_root(graph_state)
    #     return graph_state
    #
    # def reset(self, graph_state: CompiledGraphState) -> Tuple[CompiledGraphState, StepState]:
    #     # todo: can this be standardized (i.e. moved to BaseGraph)?
    #     """Resets graph and returns before root node would run (follows gym API)."""
    #     # Runs supergraph (except for root)
    #     next_graph_state = self.run_until_root(graph_state)
    #     next_step_state = self.root.get_step_state(next_graph_state)  # Return root node's step state
    #     return next_graph_state, next_step_state
    #
    # def step(self, graph_state: CompiledGraphState, step_state: StepState = None, output: Output = None) -> Tuple[CompiledGraphState, StepState]:
    #     # todo: can this be standardized (i.e. moved to BaseGraph)?
    #     """Runs graph for one step and returns before root node would run (follows gym API).
    #     - If step_state and output are provided, the root node's step is not run and the
    #     provided step_state and output are used instead.
    #     - Calling step() repeatedly is equivalent to calling run() repeatedly, except that
    #     step() returns the root node's step state *before* the root node is run, while run()
    #     returns the root node's step state *after* the root node is run.
    #     - Only calling step() after init() without reset() is possible, but note that step()
    #     starts by running the root node. But, because an episode should start with a run_until_root(),
    #     the first root step call is skipped.
    #     """
    #     # Runs root node (if step_state and output are not provided, otherwise overrides step_state and output with provided values)
    #     new_graph_state = self.run_root(graph_state, step_state, output)
    #
    #     # Runs supergraph (except for root)
    #     next_graph_state = self.run_until_root(new_graph_state)
    #     next_step_state = self.root.get_step_state(next_graph_state)  # Return root node's step state
    #     return next_graph_state, next_step_state

    def max_eps(self, graph_state: CompiledGraphState = None):
        if graph_state is None or graph_state.timings is None:
            assert self._default_timings is not None, "No default timings provided. Cannot determine max episode."
            num_eps = next(iter(self._default_timings[-1].values()))["run"].shape[-2]
        else:
            num_eps = next(iter(graph_state.timings[-1].values()))["run"].shape[-2]
        return num_eps

    def max_steps(self, graph_state: CompiledGraphState = None):
        if graph_state is None or graph_state.timings is None:
            assert self._default_timings is not None, "No default timings provided. Cannot determine max number of steps."
            num_steps = next(iter(self._default_timings[-1].values()))["run"].shape[-1]
        else:
            num_steps = next(iter(graph_state.timings[-1].values()))["run"].shape[-1]
        return num_steps - 1

    def max_starting_step(self, max_steps: int, graph_state: CompiledGraphState = None):
        max_steps_graph = self.max_steps(graph_state=graph_state)
        max_starting_steps = max_steps_graph - max_steps
        assert (
            max_starting_steps >= 0
        ), f"max_steps ({max_steps}) must be smaller than the max number of compiled steps in the graph ({max_steps_graph})"
        return max_starting_steps
