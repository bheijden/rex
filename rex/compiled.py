from typing import Any, Dict, List, Tuple, Callable, Union
import functools
import networkx as nx
from flax.core import FrozenDict
import jax
from jax.random import KeyArray
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as onp
import rex.jax_utils as rjax

from rex.utils import deprecation_warning
from rex.node import Node
from rex.graph import BaseGraph
from rex.base import InputState, StepState, StepStates, CompiledGraphState, GraphState, Output, Timings, GraphBuffer
from rex.supergraph import get_node_data, get_graph_buffer, check_generations_uniformity


int32 = Union[jnp.int32, onp.int32]
float32 = Union[jnp.float32, onp.float32]


def invert_dict_with_list_values(input_dict):
    """
    Inverts a dictionary to create a new dictionary where the keys are the unique values
    of the input dictionary, and the values are lists of keys from the input dictionary
    that corresponded to each unique value.

    :param input_dict: The dictionary to invert.
    :return: An inverted dictionary with lists as values.
    """
    inverted_dict = {}
    for key, value in input_dict.items():
        # Add the key to the list of keys for the particular value
        if value not in inverted_dict:
            inverted_dict[value] = []
        inverted_dict[value].append(key)
    return inverted_dict


def get_buffer_size(buffer: GraphBuffer) -> jnp.int32:
    leaves = jax.tree_util.tree_leaves(buffer)
    size = leaves[0].shape[0] if len(leaves) > 0 else 1
    return size


def update_output(buffer: GraphBuffer, output: Output, seq: int32) -> Output:
    size = get_buffer_size(buffer)
    mod_seq = seq % size
    # new_buffer = jax.tree_map(lambda _b, _o: rjax.index_update(_b, mod_seq, _o, copy=True), buffer, output)
    new_buffer = jax.tree_map(lambda _b, _o: jnp.array(_b).at[mod_seq].set(jnp.array(_o)), buffer, output)
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
            inputs = rjax.tree_take(buffer, mod_seq)
            _new = InputState.from_outputs(t["seq"], t["ts_sent"], t["ts_recv"], inputs, is_data=True)
            new_inputs[input_name] = _new

        return ss.replace(eps=eps, seq=seq, ts=ts_step, inputs=FrozenDict(new_inputs))

    return _update_inputs


def old_make_run_S(nodes: Dict[str, "Node"], S: nx.DiGraph, root_slot: str, generations: List[List[str]], skip: List[str] = None):
    # todo: Buffer size may not be big enough, when updating graph_state during generational loop.
    #       Specifically, get_buffer_sizes_from_timings must be adapted to account for this.
    #       Or, alternatively, we can construct S such that it has a single node per generation (i.e. linear graph).
    #       This is currently not yet enforced.
    INTERMEDIATE_UPDATE = False

    # Define update function
    root = S.nodes[root_slot]["kind"]
    root_gen_idx = [i for i, gen in enumerate(generations) if root_slot in gen][0]
    node_data = get_node_data(S)
    update_input_fns = {name: make_update_inputs(name, data["inputs"]) for name, data in node_data.items()}

    # Determine if all generations contain all slot_kinds
    # NOTE! This assumes that the root is the only node in the last generation.
    is_uniform = check_generations_uniformity(S, generations[:-1])
    slots_to_kinds = {n: data["kind"] for n, data in S.nodes(data=True)}
    kinds_to_slots = invert_dict_with_list_values(slots_to_kinds)
    kinds_to_slots.pop(root)  # remove root from kinds_to_slots
    for key, value in kinds_to_slots.items():
        # sort value based on the generation they belong to.
        kinds_to_slots[key] = sorted(value, key=lambda x: S.nodes[x]["seq"])

    # Determine which slots to skip
    skip_slots = [n for n, data in S.nodes(data=True) if data["kind"] in skip] if skip is not None else []
    skip_slots = skip_slots + skip if skip is not None else skip_slots  # also add kinds to skip slots, because if uniform, then kinds are also slots.

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
            if slot_kind == root_slot or slot_kind in skip_slots:
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
                new_ss, new_output = jax.lax.cond(pred, node_step_fns[kind], no_op, graph_state, timings_node)
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
        return new_graph_state, new_graph_state

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
            lambda _tb, _size: jax.lax.dynamic_slice(_tb, [step] + [0 * s for s in _size], [1] + _size)[0], timings, slice_sizes
        )

        # Run generations
        # NOTE! len(generations)+1 = len(timings_mcs) --> last generation is the root.
        if not is_uniform:
            for gen, timings_gen in zip(generations, timings_mcs):
                assert all([node in gen for node in timings_gen.keys()]), f"Missing nodes in timings: {gen}"
                graph_state, _ = _run_generation(graph_state, timings_gen)
        else:
            flattened_timings = dict()
            # NOTE! This assumes that the root is the only node in the last generation.
            [flattened_timings.update(timings_gen) for timings_gen in timings_mcs[:-1]]  # Remember: this does include root_slot
            slots_timings = {}
            for kind, slots in kinds_to_slots.items():  # Remember: kinds_to_slots does not include root_slot
                timings_to_stack = [flattened_timings[slot] for slot in slots]
                slots_timings[slots[0]] = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *timings_to_stack)
            all_shapes = [v["run"].shape for k, v in slots_timings.items()]
            assert all([s == all_shapes[0] for s in all_shapes]), "Shapes of slots are not equal"
            graph_state, _ = jax.lax.scan(_run_generation, graph_state, slots_timings)

        # Run root input update
        new_ss_root = update_input_fns[root](graph_state, timings_mcs[root_gen_idx][root_slot])
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy({root: new_ss_root}))

        # Increment step (new step may exceed max_step) --> clipping is done at the start of run_S.
        graph_state = graph_state.replace(step=graph_state.step + 1)
        return graph_state

    return _run_S


def new_make_run_S(nodes: Dict[str, "Node"], S: nx.DiGraph, root_slot: str, generations: List[List[str]], skip: List[str] = None):
    # Determine which slots to skip
    skip_slots = [n for n, data in S.nodes(data=True) if data["kind"] in skip] if skip is not None else []

    # Define update function
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
            graph_state = jax.lax.cond(pred, node_step_fns[kind], no_op, graph_state, timings_node)
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
            lambda _tb, _size: jax.lax.dynamic_slice(_tb, [step] + [0 * s for s in _size], [1] + _size)[0], timings, slice_sizes
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


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"], root: Node, S: nx.DiGraph, default_timings: Timings = None, skip: List[str] = None):
        super().__init__(root=root, nodes=nodes, skip=skip)
        self._S = S
        self._default_timings = default_timings

        # if default_timings is None:
        #     deprecation_warning("default_timings is None. This means that the graph will not be able to run without a buffer.", stacklevel=2)

        # Get generations
        self._generations = list(nx.topological_generations(S))
        self._root_slot = [n for n, data in S.nodes(data=True) if data["kind"] == self.root.name][0]
        self._root_gen_idx = [i for i, gen in enumerate(self._generations) if self._root_slot in gen][0]
        self._root_kind = S.nodes[self._root_slot]["kind"]
        self._node_data = get_node_data(S)

        # Make chunk runner
        self.__run_until_root = make_run_S(self.nodes_and_root, S, self._root_slot, self._generations, skip=skip)

    def __getstate__(self):
        args, kwargs = (), dict(nodes=self.nodes, root=self.root, S=self.S, default_timings=self._default_timings)
        return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state
        super().__setstate__((args, kwargs))

    @property
    def S(self):
        return self._S

    def init(self, rng: KeyArray = None, step_states: StepStates = None, starting_step: Union[int, ArrayLike] = 0,
             starting_eps: Union[int, ArrayLike] = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None) -> CompiledGraphState:
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
            timing = rjax.tree_take(graph_state.timings_eps[self._root_gen_idx][root_slot], i=new_graph_state.step)
            new_graph_state = update_state(graph_state, timing, new_ss, new_output)
            return new_graph_state

        def _skip_root_step() -> CompiledGraphState:
            return graph_state

        # Run root node if step > 0, else skip
        graph_state = jax.lax.cond(graph_state.step == 0, _skip_root_step, _run_root_step)
        return graph_state

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

