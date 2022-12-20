from typing import Any, Dict, List, Tuple, Generator, Callable, Union
import jumpy as jp
import rex.jumpy as rjp
from rex.graph import BaseGraph
from rex.base import InputState, StepState, GraphState, State
from rex.proto import log_pb2
from rex.agent import Agent
from rex.node import Node


SplitOutput = Dict[str, Union[int, log_pb2.TracedStep, Dict[str, int], List[log_pb2.TracedStep]]]


def make_splitter(nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, name: str) -> Generator[SplitOutput, None, None]:
    # Ticks of every node right before the step is executed.
    tick_state = {n.name: 0 for n in trace.node}
    steps = []
    chunk_index = 0
    for t in trace.used:
        # Update tick state
        tick_state[t.name] = t.tick

        if t.name == name:
            chunk = make_chunk(nodes, t, steps, chunk_index)
            yield dict(index=t.index, steptrace=t, tick_state=tick_state, steps=steps, chunk=chunk, chunk_index=chunk_index)
            chunk_index += 1
            tick_state = dict(**tick_state)  # Creates a deepcopy
            steps = []
        else:
            steps.append(t)


def make_push_output(trace: log_pb2.Dependency) -> Callable[[GraphState, Any], InputState]:
    node_name = trace.target.name
    input_name = trace.target.input_name
    seq = trace.source.tick
    ts_sent = trace.source.ts
    ts_recv = trace.target.ts

    def _push_output(graph_state: GraphState, output: Any) -> InputState:
        return graph_state.nodes[node_name].inputs[input_name].push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=output)

    return _push_output


def make_update_graph_state(name: str, push_outputs: Dict[Tuple[str, str], Callable]) -> Callable[[GraphState, StepState, Any], GraphState]:
    # @jax.jit
    def _update_graph_state(graph_state: GraphState, step_state: StepState, output: Any) -> GraphState:
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy())  # NOTE! This makes a shallow copy
        graph_state.nodes[name] = step_state  # NOTE! This updates a dict in-place
        for (node_name, input_name), push_output in push_outputs.items():
            new_inputs = push_output(graph_state, output)
            graph_state.nodes[node_name] = graph_state.nodes[node_name].replace(inputs=graph_state.nodes[node_name].inputs.copy())  # NOTE! This makes a shallow copy
            graph_state.nodes[node_name].inputs[input_name] = new_inputs  # NOTE: This updates a dict in-place
        return graph_state
    return _update_graph_state


def make_node_step(nodes: Dict[str, "Node"], trace: log_pb2.TracedStep) -> Callable[[GraphState], GraphState]:
    name = trace.name
    node = nodes[trace.name]
    ts_step = jp.float32(trace.ts_step)

    # Pushes the output into the input state of other nodes (excludes state dependency)
    push_outputs = {(d.target.name, d.target.input_name): make_push_output(d) for d in trace.downstream if d.target.name != name}
    update_graph_state = make_update_graph_state(name, push_outputs)

    def _node_step(graph_state: GraphState) -> GraphState:
        # Unpack Step State
        ss = graph_state.nodes[name]

        # Call node step
        new_ss, output = node.step(ts_step, ss)

        # Update graph state with new step_state and push output to downstream dependencies
        new_graph_state = update_graph_state(graph_state, new_ss, output)

        return new_graph_state

    return _node_step


def make_chunk(nodes: Dict[str, "Node"], steptrace: log_pb2.TracedStep, steps: List[log_pb2.TracedStep], chunk_index: int) -> Callable[[GraphState], Tuple[GraphState, jp.float32, StepState]]:
    ts_step = steptrace.ts_step
    name = steptrace.name
    node_steps = [make_node_step(nodes, s) for s in steps]

    def _chunk(graph_state: GraphState) -> Tuple[GraphState, jp.float32, StepState]:
        # Apply chunk of sequential steps
        for step in node_steps:
            graph_state = step(graph_state)

        # Update to next chunk index
        new_graph_state = graph_state.replace(step=jp.int32(chunk_index))  # Increment here?

        # Return new graph state, timestamp and step state of the node that was used to split the trace (usually the agent's)
        return new_graph_state, ts_step, new_graph_state.nodes[name]

    return _chunk


def make_graph_reset(chunks: Dict[int, SplitOutput], start_index: int) -> Callable[[GraphState], Tuple[GraphState, float, StepState]]:
    reset_chunk = chunks[start_index]["chunk"]

    def _graph_reset(graph_state: GraphState) -> Tuple[GraphState, float, StepState]:

        # Run initial chunk.
        next_graph_state, next_ts_step, next_step_state = reset_chunk(graph_state)

        return next_graph_state, next_ts_step, next_step_state
    return _graph_reset


def make_single_step(prev_trace: log_pb2.TracedStep, step_chunk: Callable[[GraphState], Tuple[GraphState, float, StepState]]) \
        -> Callable[[GraphState], Tuple[GraphState, float, StepState]]:
    name = prev_trace.name

    # Pushes the output (action) into the input state of other nodes (excludes state dependency)
    push_outputs = {(d.target.name, d.target.input_name): make_push_output(d) for d in prev_trace.downstream if
                    d.target.name != name}
    update_graph_state = make_update_graph_state(name, push_outputs)

    # @jax.jit
    def _single_step(graph_state: GraphState, step_state: State, action: Any) -> Tuple[GraphState, float, StepState]:
        # print(f"tracing {name} step {prev_trace.index}")

        # Update graph state with step state & action that corresponds to the previous step trace.
        updated_graph_state = update_graph_state(graph_state, step_state, action)

        # Run chunk.
        next_graph_state, next_ts_step, next_step_state = step_chunk(updated_graph_state)

        return next_graph_state, next_ts_step, next_step_state

    return _single_step


def make_graph_step(chunks: Dict[int, SplitOutput], start_index: int, end_index: int)\
        -> Callable[[GraphState, StepState, Any], Tuple[GraphState, jp.float32, StepState]]:
    # Get step chunks and corresponding (previous) step traces
    prev_traces = [chunks[i]["steptrace"] for i in range(start_index, end_index)]
    step_chunks = [chunks[i]["chunk"] for i in range(start_index+1, end_index+1)]

    # Prepare a sequence of step functions (that ca be called based on the graph_state.chunk_index)
    step_chunks = [make_single_step(pt, sc) for pt, sc in zip(prev_traces, step_chunks)]

    def _graph_step(graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, jp.float32, StepState]:
        """Depending on the chunk_index, run the corresponding step chunk."""
        # Determine next chunk index
        next_chunk_index = graph_state.step

        # Run step chunk
        next_graph_state, next_ts_step, next_step_state = rjp.switch(next_chunk_index-jp.int32(start_index), step_chunks, graph_state, step_state, action)

        return next_graph_state, next_ts_step, next_step_state

    return _graph_step


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, agent: Agent, max_steps: int):
        # Split trace into chunks
        splitter = make_splitter(nodes, trace, agent.name)
        self.chunks = {i: d for i, d in enumerate(splitter)}
        self.trace = trace
        self.max_steps = max_steps if max_steps is not None else len(self.chunks)-1
        assert self.max_steps <= len(self.chunks)-1, f"max_steps ({self.max_steps}) must be smaller than the number of chunks ({len(self.chunks)-1})"

        # Compile reset
        self.__reset = make_graph_reset(self.chunks, 0)

        # Compile step
        self.__step = make_graph_step(self.chunks, 0, self.max_steps)

        super().__init__(agent=agent)

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, jp.float32, Any]:
        next_graph_state, next_ts_step, next_step_state = self.__reset(graph_state)
        return next_graph_state, next_ts_step, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, jp.float32, StepState]:
        next_graph_state, next_ts_step, next_step_state = self.__step(graph_state, step_state, output)
        return next_graph_state, next_ts_step, next_step_state
