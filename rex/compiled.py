from typing import Any, Dict, List, Tuple, Callable, Union

import networkx as nx
from flax.core import FrozenDict
import jumpy
import jax
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import rex.jumpy as rjp

from rex.node import Node
from rex.graph import BaseGraph
from rex.base import InputState, StepState, GraphState, Output, Timings, GraphBuffer
from rex.agent import Agent
from rex.tracer import get_node_data, get_graph_buffer


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

    def _update_state(graph_state: GraphState, timing: Dict, step_state: StepState, output: Any) -> GraphState:
        # Define node's step state update
        new_nodes = dict()
        new_outputs = dict()

        # Increment sequence number
        new_ss = step_state.replace(seq=step_state.seq + 1)

        # Add node's step state update
        new_nodes[name] = new_ss
        new_outputs[name] = update_output(graph_state.buffer.outputs[name], output, timing["seq"])

        new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy(new_outputs))
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes), buffer=new_buffer)
        return new_graph_state

    return _update_state


def make_update_inputs(name: str, inputs_data: Dict[str, Dict[str, str]]):

    def _update_inputs(graph_state: GraphState, timings_node: Dict) -> StepState:
        ss = graph_state.nodes[name]
        ts_step = timings_node["ts_step"]
        eps = graph_state.eps
        seq = timings_node["seq"]
        new_inputs = dict()
        for node_name, data in inputs_data.items():
            input_name = data["input_name"]
            t = timings_node["inputs"][input_name]
            buffer = graph_state.buffer.outputs[node_name]
            size = get_buffer_size(buffer)
            mod_seq = t["seq"] % size
            inputs = rjp.tree_take(buffer, mod_seq)
            _new = InputState.from_outputs(t["seq"], t["ts_sent"], t["ts_recv"], inputs, is_data=True)
            new_inputs[input_name] = _new

        return ss.replace(eps=eps, seq=seq, ts=ts_step, inputs=FrozenDict(new_inputs))

    return _update_inputs


def make_run_MCS(nodes: Dict[str, "Node"], MCS: nx.DiGraph, generations: List[List[str]]):
    # Define update function
    node_data = get_node_data(MCS)
    update_input_fns = {name: make_update_inputs(name, data["inputs"]) for name, data in node_data.items()}

    def _run_generation(graph_state: GraphState, timings_gen: Dict):
        new_nodes = dict()
        new_outputs = dict()
        for slot_name, timings_node in timings_gen.items():
            name = MCS.nodes[slot_name]["name"]
            pred = timings_gen[slot_name]["run"]

            # Prepare old states
            _old_ss = graph_state.nodes[name]
            _old_output = graph_state.buffer.outputs[name]

            # Add dummy inputs to old step_state (else jax complains for structure mismatch
            if _old_ss.inputs is None:
                _old_ss = update_input_fns[name](graph_state, timings_node)

            # Define node update function
            def _run_node(graph_state: GraphState) -> Tuple[StepState, Output]:
                # Update inputs
                ss = update_input_fns[name](graph_state, timings_node)
                # ss = _old_ss

                # Run node step
                _new_ss, output = nodes[name].step(ss)

                # Increment sequence number
                _new_seq_ss = _new_ss.replace(seq=_new_ss.seq + 1)

                # Update output buffer
                _new_output = update_output(graph_state.buffer.outputs[name], output, timings_node["seq"])
                # _new_output = graph_state.outputs[name]
                return _new_seq_ss, _new_output

            # Run node step
            new_ss, new_output = jumpy.lax.cond(pred, _run_node, lambda *args: (_old_ss, _old_output), graph_state)

            # Store new state
            new_nodes[name] = new_ss
            new_outputs[name] = new_output

        new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy(new_outputs))
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes),
                                              buffer=new_buffer)
        return new_graph_state

    def _run_MCS(graph_state: GraphState) -> GraphState:
        # Get eps & step  (used to index timings)
        step = graph_state.step

        # Determine slice sizes (depends on window size)
        # [2:] because first two dimensions are episode and step.
        timings = graph_state.buffer.timings
        slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), timings)

        # Slice timings
        timings_mcs = jax.tree_map(lambda _tb, _size: rjp.dynamic_slice(_tb, [step] + [0*s for s in _size], [1] + _size)[0], timings, slice_sizes)

        # Run generations
        # NOTE! len(generations)+1 = len(timings_mcs) --> last generation is the root.
        for gen, timings_gen in zip(generations[:-1], timings_mcs):
            assert all([node in gen for node in timings_gen.keys()]), f"Missing nodes in timings: {gen}"
            graph_state = _run_generation(graph_state, timings_gen)
        return graph_state

    return _run_MCS


def make_graph_reset(MCS: nx.DiGraph, generations: List[List[str]], run_MCS: Callable,
                     default_timings: Timings, default_buffer: GraphBuffer):
    # Determine root node (always in the last generation)
    root_slot = generations[-1][0]
    root = MCS.nodes[root_slot]["name"]

    # Define update function
    node_data = get_node_data(MCS)
    update_input_fns = {name: make_update_inputs(name, data["inputs"]) for name, data in node_data.items()}

    # We pre-compute the modulo sequence for the default timings
    # name_mapping = {n: {v["input_name"]: o for o, v in data["inputs"].items()} for n, data in node_data.items()}
    # for gen in default_timings:
    #     for n, t in gen.items():
    #         node_name = MCS.nodes[n]["name"]
    #         t["seq"] = t["seq"] % jax.tree_util.tree_leaves(default_buffer.outputs[node_name])[0].shape[0]
    #         for i, v in t["inputs"].items():
    #             output_name = name_mapping[node_name][i]
    #             v["seq"] = v["seq"] % jax.tree_util.tree_leaves(default_buffer.outputs[output_name])[0].shape[0]

    def _graph_reset(graph_state: GraphState) -> Tuple[GraphState, StepState]:
        # Get buffer
        buffer = graph_state.buffer if graph_state.buffer is not None else default_buffer
        assert buffer is not None, "The graph_state.buffer is None and no default_buffer was provided."

        # Get timings
        timings = graph_state.timings if graph_state.timings is not None else default_timings
        assert timings is not None, "The graph_state.timings is None and no default_timings were provided."

        # Determine episode timings
        # NOTE! The graph_state.step indexes into the timings and indicates what subgraphs have run.
        #       All subgraphs up until (and including) the graph.state'th subgraph have run.
        #       I.E., graph_state.step=0 means that we have run the first subgraph AFTER this function has run.
        #       We do not increment step+1 here, because we increment before running a chunk in graph_step.
        max_eps, max_step = timings[-1][root_slot]["run"].shape
        eps = jp.clip(graph_state.eps, jp.int32(0), max_eps - 1)
        step = jp.clip(graph_state.step, jp.int32(0), max_step - 1)
        eps_timings = rjp.tree_take(timings, eps)

        # Replace buffer
        new_buffer = buffer.replace(timings=eps_timings)
        graph_state = graph_state.replace(eps=eps, step=step, buffer=new_buffer)

        # Run initial chunk.
        _next_graph_state = run_MCS(graph_state)

        # Update input
        next_timing = rjp.tree_take(graph_state.buffer.timings[-1][root_slot], i=step)
        next_ss = update_input_fns[root](_next_graph_state, next_timing)
        next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({root: next_ss}))
        return next_graph_state, next_ss
    return _graph_reset


def make_graph_step(MCS: nx.DiGraph, generations: List[List[str]], run_MCS: Callable):
    # Determine root node (always in the last generation)
    root_slot = generations[-1][0]
    root = MCS.nodes[root_slot]["name"]

    # Define update function
    node_data = get_node_data(MCS)
    update_state = make_update_state(root)
    update_input = make_update_inputs(root, node_data[root]["inputs"])

    def _graph_step(graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, StepState]:
        # Update graph_state with action
        timing = rjp.tree_take(graph_state.buffer.timings[-1][root_slot], i=graph_state.step)
        new_graph_state = update_state(graph_state, timing, step_state, action)

        # Grab step
        # NOTE! The graph_state.step is used to index into the timings.
        #       Therefore, we increment it before running the subgraph so that we index into the timings of the next step.
        #       Hence, graph_state.step indicates what subgraphs have run.
        max_step = new_graph_state.buffer.timings[-1][root_slot]["run"].shape[0]
        next_step = jp.clip(new_graph_state.step+1, jp.int32(0), max_step - 1)
        graph_state = new_graph_state.replace(step=next_step)

        # Run chunk of next step.
        _next_graph_state = run_MCS(graph_state)

        # Update input
        next_timing = rjp.tree_take(graph_state.buffer.timings[-1][root_slot], i=next_step)
        next_ss = update_input(_next_graph_state, next_timing)
        next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({root: next_ss}))
        return next_graph_state, next_ss

    return _graph_step


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"], root: Agent, MCS: nx.DiGraph, default_timings: Timings = None):
        super().__init__(root=root, nodes=nodes)
        self._MCS = MCS
        self._default_timings = default_timings
        # self._default_outputs = get_outputs_from_timings(MCS, default_timings, self.nodes_and_root) if default_timings is not None else None
        self._default_buffer = get_graph_buffer(MCS, default_timings, self.nodes_and_root) if default_timings is not None else None

        # Get generations
        generations = list(nx.topological_generations(MCS))

        # Make chunk runner
        run_MCS = make_run_MCS(self.nodes_and_root, MCS, generations)

        # Make compiled reset function
        self.__reset = make_graph_reset(MCS, generations, run_MCS, default_timings=self._default_timings, default_buffer=self._default_buffer)

        # Make compiled step function
        self.__step = make_graph_step(MCS, generations, run_MCS)

    def __getstate__(self):
        args, kwargs = (), dict(nodes=self.nodes, root=self.root, MCS=self.MCS, default_timings=self._default_timings)
        return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state
        super().__setstate__((args, kwargs))

    @property
    def MCS(self):
        return self._MCS

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, Any]:
        next_graph_state, next_step_state = self.__reset(graph_state)
        return next_graph_state, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, StepState]:
        next_graph_state, next_step_state = self.__step(graph_state, step_state, output)
        return next_graph_state, next_step_state

    def max_eps(self, graph_state: GraphState = None):
        if graph_state is None or graph_state.timings is None:
            assert self._default_timings is not None, "No default timings provided. Cannot determine max episode."
            num_eps = next(iter(self._default_timings[-1].values()))["run"].shape[-2]
        else:
            num_eps = next(iter(graph_state.timings[-1].values()))["run"].shape[-2]
        return num_eps

    def max_steps(self, graph_state: GraphState = None):
        if graph_state is None or graph_state.timings is None:
            assert self._default_timings is not None, "No default timings provided. Cannot determine max number of steps."
            num_steps = next(iter(self._default_timings[-1].values()))["run"].shape[-1]
        else:
            num_steps = next(iter(graph_state.timings[-1].values()))["run"].shape[-1]
        return num_steps-1

    def max_starting_step(self, max_steps: int, graph_state: GraphState = None):
        max_steps_graph = self.max_steps(graph_state=graph_state)
        max_starting_steps = max_steps_graph - max_steps
        assert max_starting_steps >= 0, f"max_steps ({max_steps}) must be smaller than the max number of compiled steps in the graph ({max_steps_graph})"
        return max_starting_steps