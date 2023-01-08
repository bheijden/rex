from typing import Any, Dict, List, Tuple, Callable, Union
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import rex.jumpy as rjp
from collections import deque
from copy import deepcopy
import jax
from flax import struct

from rex.constants import SEQUENTIAL, VECTORIZED, BATCHED, WARN, SYNC, FAST_AS_POSSIBLE, PHASE, SIMULATED
from rex.proto import log_pb2
from rex.node import Node
from rex.graph import BaseGraph
from rex.base import InputState, StepState, GraphState, Output
from rex.agent import Agent


int32 = Union[jnp.int32, onp.int32]
float32 = Union[jnp.float32, onp.float32]
SplitOutput = Dict[str, Union[int, log_pb2.TracedStep, Dict[str, int], List[log_pb2.TracedStep]]]
Timings = Dict[str, Union[Dict[str, jp.ndarray], jp.ndarray]]


class TreeLeaf:
    def __init__(self, container):
        self.c = container


def make_depth_grouping(trace: log_pb2.TraceRecord, graph: int) -> List[List[log_pb2.TracedStep]]:
    max_depth = trace.max_depth
    depths = [[] for _ in range(max_depth + 1)]
    for t in trace.used:
        depths[t.depth].append(t)

    if graph == VECTORIZED:
        # We make sure that there are always max_consecutive depths between two consecutive isolated depths.
        # This allows the use of scan, instead of a for loop.
        max_consecutive = trace.max_consecutive
        consecutive = 0
        new_depths = []
        for d in depths:
            has_isolated = any([t.isolate for t in d])
            if has_isolated:
                # Pad with empty lists
                pad = max_consecutive - consecutive
                for _ in range(pad):
                    new_depths.append([])

                # Reset consecutive
                consecutive = 0
            else:
                consecutive += 1

            # Append depth
            new_depths.append(d)
        depths = new_depths
    elif graph == SEQUENTIAL:
        # Place every node in its own depth.
        new_depths = []
        topological_order = []
        for d in depths:
            for t in d:
                topological_order.append(t.index)
                new_depths.append([t])
        assert onp.all(onp.diff(topological_order) > 0), "Topological order is not respected."
        depths = new_depths
    else:
        raise NotImplementedError(f"Graph type {graph} not implemented.")

    return depths


def make_timings(nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord, depths: List[List[log_pb2.TracedStep]]) -> Timings:
    # Number of depths (may be increased if vectorized)
    num_depths = len(depths)

    # Prepare timings pytree
    timings = {n.name: dict(run=onp.repeat(False, num_depths),       # run:= whether the node must run,
                            ts_step=onp.repeat(0., num_depths),      # ts_step:= ts trajectory,
                            tick=onp.repeat(-1, num_depths),         # tick:= tick trajectory,
                            stateful=onp.repeat(False, num_depths),  # stateful:= whether to update the state,
                            inputs={},                              # inputs:= inputs from other nodes to this node
                            ) for i, n in enumerate(trace.node)}
    update, window = dict(), dict()
    for name, t, in timings.items():
        update[name], window[name] = dict(), dict()
        for i in nodes[name].inputs:
            update[name][i.info.name] = deque([False]*i.window, maxlen=i.window)
            window[name][i.info.name] = dict(seq=deque(range(-i.window, 0), maxlen=i.window),
                                             ts_sent=deque([0.]*i.window, maxlen=i.window),
                                             ts_recv=deque([0.]*i.window, maxlen=i.window))
            t["inputs"][i.info.name] = dict(update=[], seq=[], ts_sent=[], ts_recv=[])

    # Populate timings
    for idx, depth in enumerate(depths):
        _update = deepcopy(update)
        for t in depth:
            # Update source node timings
            timings[t.name]["run"][idx] = True
            timings[t.name]["ts_step"][idx] = t.ts_step
            timings[t.name]["tick"][idx] = t.tick
            timings[t.name]["stateful"][idx] = t.stateful or not t.static

            # Sort upstream dependencies per input channel & sequence number
            _sorted_deps = dict()
            for d in t.upstream:
                if not d.used:
                    continue
                if d.source.name == t.name:
                    continue
                assert d.target.name == t.name
                input_name = d.target.input_name
                _sorted_deps[input_name] = _sorted_deps.get(input_name, []) + [d]
            [d_lst.sort(key=lambda d: d.source.tick) for d_lst in _sorted_deps.values()]

            # Update windows
            for input_name, deps in _sorted_deps.items():
                for d in deps:
                    input_name = d.target.input_name
                    _update[t.name][input_name].append(True)
                    window[t.name][input_name]["seq"].append(d.source.tick)
                    window[t.name][input_name]["ts_sent"].append(d.source.ts)
                    window[t.name][input_name]["ts_recv"].append(d.target.ts)

        # Update timings
        for node_name, n in window.items():
            for input_name, w in n.items():
                u = _update[node_name][input_name]
                w = window[node_name][input_name]
                timings[node_name]["inputs"][input_name]["update"].append(onp.array(u))
                timings[node_name]["inputs"][input_name]["seq"].append(onp.array(w["seq"]))
                timings[node_name]["inputs"][input_name]["ts_sent"].append(onp.array(w["ts_sent"]))
                timings[node_name]["inputs"][input_name]["ts_recv"].append(onp.array(w["ts_recv"]))


            # # Update input timings
            # for d in t.downstream:
            #     if d.target.name == t.name:
            #         continue
            #     target_node = d.target.name
            #     input_name = d.target.input_name
            #     timings[target_node]["inputs"][input_name]["update"][idx] = True
            #     timings[target_node]["inputs"][input_name]["seq"][idx] = d.source.tick
            #     timings[target_node]["inputs"][input_name]["ts_sent"][idx] = d.source.ts
            #     timings[target_node]["inputs"][input_name]["ts_recv"][idx] = d.target.ts

    # Stack timings
    for name, t in timings.items():
        for input_name, i in t["inputs"].items():
            for k, v in i.items():
                i[k] = onp.stack(v, axis=0)

    return timings


def make_default_outputs(nodes: Dict[str, "Node"], timings: Timings) -> Dict[str, Output]:
    num_ticks = dict()
    outputs = dict()
    _seed = jp.random_prngkey(0)
    for name, n in nodes.items():
        num_ticks[name] = timings[name]["tick"].max() + 1  # Number of ticks
        outputs[name] = n.default_output(_seed)  # Outputs

    # Stack outputs
    stack_fn = lambda *x: jp.stack(x, axis=0)
    stacked_outputs = dict()
    for name, n in nodes.items():
        stacked_outputs[name] = jp.tree_map(stack_fn, *[outputs[name]]*(num_ticks[name]+1))
    return stacked_outputs


def make_splitter(trace: log_pb2.TraceRecord, timings: Timings, depths: List[List[log_pb2.TracedStep]]) -> Tuple[jp.ndarray, jp.ndarray, Timings]:
    assert trace.isolate
    name = trace.name

    isolate_lst = []
    chunks = []
    substeps = []
    _last_counter = 0
    _last_index = 0
    for i, depth in enumerate(depths):
        _last_counter += 1

        # Check if we have reached the end of a chunk (i.e. an isolated depth)
        if timings[name]["run"][i]:
            assert len(depth) == 1, "Isolated depth must have only a single steptrace."
            assert depth[0].isolate, "Isolated depth must have an isolated steptrace."
            assert depth[0].name == trace.name, "Isolated depth must have a steptrace with the same name as the trace."
            isolate_lst.append(jp.tree_map(lambda _tb: _tb[i], timings))
            chunks.append(_last_index)
            _steps = list(reversed(range(0, _last_counter)))
            substeps += _steps
            _last_counter = 0
            _last_index = i+1
    isolate = jp.tree_map(lambda *args: jp.array(args), *isolate_lst)
    _steps = list(reversed(range(0, _last_counter)))
    substeps += _steps
    assert len(substeps) == len(depths), "Substeps must be the same length as depths."
    assert len(chunks) == len(isolate[name]["run"]), "Chunks must be the same length as the timings of the isolated depths."
    assert jp.all(isolate[name]["run"]), "Isolated depths must have run=True."
    return jp.array(chunks), jp.array(substeps), isolate


def update_output(buffer, output: Output, tick: int32) -> Output:
    new_buffer = jp.tree_map(lambda _b, _o: rjp.index_update(_b, tick, _o, copy=True), buffer, output)
    return new_buffer


def make_update_state(name: str, stateful: bool, static: bool):

    def _update_state(graph_state: GraphState, timing: Dict, step_state: StepState, output: Any) -> GraphState:
        # Define node's step state update
        new_nodes = dict()
        new_outputs = dict()

        # Add node's step state update
        new_nodes[name] = step_state  # todo: Do not update params if static?
        new_outputs[name] = update_output(graph_state.outputs[name], output, timing[name]["tick"])
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes),
                                              outputs=graph_state.outputs.copy(new_outputs))
        return new_graph_state

    return _update_state


def make_update_inputs(name: str, outputs: Dict[str, str], cond: bool = True):

    def __push_input(old: InputState, seq: rjp.int32, ts_sent: rjp.float32, ts_recv: rjp.float32, buffer: Output) -> InputState:
        new_o = rjp.take(buffer, seq)
        return old.push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=new_o)

    def _update_inputs(graph_state: GraphState, timing: Dict) -> StepState:
        ss = graph_state.nodes[name]
        new_inputs = dict()
        for input_name, node_name in outputs.items():
            _new = ss.inputs[input_name]
            t = timing[name]["inputs"][input_name]
            buffer = graph_state.outputs[node_name]
            window = _new.seq.shape[0]
            for j in range(window):
                pred = t["update"][j]
                seq = t["seq"][j]
                ts_sent = t["ts_sent"][j]
                ts_recv = t["ts_recv"][j]
                if cond:
                    _new = rjp.cond(pred, __push_input, lambda _old, *args: _old, _new, seq, ts_sent, ts_recv, buffer)
                else:
                    _update = __push_input(_new, seq, ts_sent, ts_recv, buffer)
                    _new = jp.tree_map(lambda _u, _o: jp.where(pred, _u, _o), _update, _new)
            new_inputs[input_name] = _new

        return ss.replace(inputs=ss.inputs.copy(new_inputs))

    return _update_inputs


def make_run_node(name: str, node: "Node", outputs: Dict[str, str], stateful: bool, static: bool):
    update_inputs = make_update_inputs(name, outputs)
    update_state = make_update_state(name, stateful, static)

    def _run_node(graph_state: GraphState, timing: Dict) -> GraphState:
        # Update inputs
        ss = update_inputs(graph_state, timing)

        # Run node step
        new_ss, output = node.step(timing[name]["ts_step"], ss)

        # Get mask
        new_graph_state = update_state(graph_state, timing, new_ss, output)
        return new_graph_state

    return _run_node


def make_run_batch_chunk(timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray, graph: int,
                         batch_nodes: Dict[str, "Node"], batch_outputs: Dict, stateful: Dict, static: Dict,
                         cond: bool = True):
    if graph == VECTORIZED:
        assert jp.all(substeps[chunks] == substeps[chunks][0]), "All substeps must be equal when vectorized."
        fixed_num_steps = int(substeps[chunks][0])
    elif graph == BATCHED:
        fixed_num_steps = None
    else:
        raise ValueError("Unknown graph type.")

    # Define update function
    update_input_fns = {name: make_update_inputs(name, outputs) for name, outputs in batch_outputs.items()}

    # Determine slice sizes (depends on window size)
    slice_sizes = jp.tree_map(lambda _tb: list(_tb.shape[1:]), timings)

    def _run_batch_step(graph_state: GraphState, timing: Dict):
        new_nodes = dict()
        new_outputs = dict()
        for name, node in batch_nodes.items():
            pred = timing[name]["run"]

            # Prepare old states
            _old_ss = graph_state.nodes[name]
            _old_output = graph_state.outputs[name]

            # Define node update function
            def _run_node(graph_state: GraphState, timing: Dict) -> Tuple[StepState, Output]:
                # Update inputs
                ss = update_input_fns[name](graph_state, timing)

                # Run node step
                new_ss, output = node.step(timing[name]["ts_step"], ss)

                buffer = update_output(graph_state.outputs[name], output, timing[name]["tick"])
                return new_ss, buffer  # todo: Do not update params if static?

            # Run node step
            if cond:
                new_ss, new_output = rjp.cond(pred, _run_node, lambda *args: (_old_ss, _old_output), graph_state, timing)
            else:
                _update_ss, _update_output = _run_node(graph_state, timing)
                new_ss = jp.tree_map(lambda _u, _o: jp.where(pred, _u, _o), _update_ss, _old_ss)
                new_output = jp.tree_map(lambda _u, _o: jp.where(pred, _u, _o), _update_output, _old_output)

            # Store new state
            new_nodes[name] = new_ss
            new_outputs[name] = new_output

        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes),
                                              outputs=graph_state.outputs.copy(new_outputs))
        return new_graph_state, None  # NOTE! carry=graph_state, output=None

    def _run_batch_chunk(graph_state: GraphState) -> GraphState:
        # Get step (used to infer timings)
        step = graph_state.step

        # Run step
        if graph == VECTORIZED:
            # Infer length of chunk
            chunk = rjp.dynamic_slice(chunks, (step,), (1,))[0]  # has len(num_isolated_depths)
            timings_chunk = jp.tree_map(lambda _tb, _size: rjp.dynamic_slice(_tb, [chunk] + [0*s for s in _size], [fixed_num_steps] + _size), timings, slice_sizes)
            # Run chunk
            graph_state, _ = rjp.scan(_run_batch_step, graph_state, timings_chunk, length=fixed_num_steps, unroll=fixed_num_steps)
        else:
            # todo: Can we statically re-compile scan for different depth lengths?
            raise NotImplementedError("batched mode not implemented yet.")

        return graph_state

    return _run_batch_chunk


def make_run_sequential_chunk(timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray, graph: int,
                              batch_nodes: Dict[str, "Node"], batch_outputs: Dict[str, Dict[str, str]], stateful: Dict, static: Dict):

    # Define step functions
    run_node_fns = [make_run_node(name, node, batch_outputs[name], stateful[name], static[name]) for name, node in batch_nodes.items()]

    def _run_step(substep: int32, carry: Tuple[GraphState, int32]):
        # Unpack carry
        graph_state, chunk = carry

        # Get timings of this step
        step_index = chunk + substep
        timings_step = rjp.take(timings, step_index)

        # determine which nodes to run
        must_run_lst = [timings_step[name]["run"] for name in batch_nodes.keys()]
        must_run = jp.argmax(jp.array(must_run_lst))

        # Run node
        # new_graph_state = run_node_fns[0](graph_state, timings_step)
        new_graph_state = rjp.switch(must_run, run_node_fns, graph_state, timings_step)

        return new_graph_state, chunk

    def _run_sequential_chunk(graph_state: GraphState) -> GraphState:
        # Get step (used to infer timings)
        step = graph_state.step

        # Infer length of chunk
        chunk = rjp.dynamic_slice(chunks, (step,), (1,))[0]  # has len(num_isolated_depths)
        num_steps = rjp.dynamic_slice(substeps, (chunk,), (1,))[0]
        # Run chunk
        initial_carry = (graph_state, chunk)
        graph_state, _ = rjp.fori_loop(0, num_steps, _run_step, initial_carry)
        return graph_state

    return _run_sequential_chunk


def make_run_chunk(nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord,
                   timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray,
                   graph: int):
    # Exclude pruned nodes from batch step
    batch_nodes = {node_name: node for node_name, node in nodes.items() if node_name != trace.name and (node_name not in trace.pruned)}
    # Structure is {node_name: {input_name: output_node_name}}
    batch_outputs = {name: {i.input_name: i.output.name for i in n.inputs if i.output.name not in trace.pruned} for name, n in batch_nodes.items()}

    # Infer static and stateful nodes
    node_names = list(batch_nodes.keys())
    stateful, static = {}, {}
    for s in trace.used:
        if s.name in node_names:
            static[s.name] = s.static
            stateful[s.name] = s.stateful
            node_names.remove(s.name)
            if len(node_names) == 0:
                break
    assert len(node_names) == 0, "All nodes must be accounted for."

    if graph in [VECTORIZED, BATCHED]:
        return make_run_batch_chunk(timings, chunks, substeps, graph, batch_nodes, batch_outputs, stateful, static)
    elif graph in [SEQUENTIAL]:
        return make_run_sequential_chunk(timings, chunks, substeps, graph, batch_nodes, batch_outputs, stateful, static)
    else:
        raise ValueError("Unknown graph type.")


def make_graph_reset(trace: log_pb2.TraceRecord, name: str, default_outputs, isolate: Timings, run_chunk: Callable):
    outputs = dict()
    for node_info in trace.node:
        if node_info.name != name:
            continue
        for i in node_info.inputs:
                outputs[i.name] = i.output

    update_input = make_update_inputs(name, outputs)

    def _graph_reset(graph_state: GraphState) -> Tuple[GraphState, jp.float32, StepState]:
        # Update output buffers
        new_outputs = dict()
        for key, value in default_outputs.items():
            if key not in graph_state.outputs:
                new_outputs[key] = value
        graph_state = graph_state.replace(outputs=graph_state.outputs.copy(new_outputs))

        # Grab step
        step = graph_state.step

        # Run initial chunk.
        _next_graph_state = run_chunk(graph_state)

        # Update input
        next_timing = rjp.take(isolate, step)
        next_ss = update_input(_next_graph_state, next_timing)
        next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({name: next_ss}))

        # Determine next ts
        next_ts_step = rjp.dynamic_slice(isolate[name]["ts_step"], (step,), (1,))[0]

        # NOTE! We do not increment step, because graph_state.step is used to index into the timings.
        #       In graph_step we do increment step after running the chunk, because we want to index into the next timings.
        return next_graph_state, next_ts_step, next_ss
    return _graph_reset


def make_graph_step(trace: log_pb2.TraceRecord, name: str, isolate: Timings, run_chunk: Callable):
    # Infer static and stateful nodes
    stateful, static = None, None
    for s in trace.used:
        if s.name == name:
            static = s.static
            stateful = s.stateful
            break
    assert stateful is not None, "Node not found in trace."
    assert static is not None, "Node not found in trace."

    outputs = dict()
    for node_info in trace.node:
        if node_info.name != name:
            continue
        for i in node_info.inputs:
                outputs[i.name] = i.output

    update_state = make_update_state(name, stateful, static)
    update_input = make_update_inputs(name, outputs)

    def _graph_step(graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, jp.float32, StepState]:
        # Update graph_state with action
        timing = rjp.take(isolate, graph_state.step)
        new_graph_state = update_state(graph_state, timing, step_state, action)

        # Grab step
        next_step = new_graph_state.step + 1

        # Run chunk of next step.
        # NOTE! The graph_state.step is used to index into the timings.
        #  Therefore, we increment it before running the chunk so that we index into the timings of the next step.
        _next_graph_state = run_chunk(new_graph_state.replace(step=next_step))

        # Update input
        next_timing = rjp.take(isolate, next_step)
        next_ss = update_input(_next_graph_state, next_timing)
        next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({name: next_ss}))

        # Determine next ts
        next_ts_step = rjp.dynamic_slice(isolate[name]["ts_step"], (next_step,), (1,))[0]

        return next_graph_state, next_ts_step, next_ss

    return _graph_step


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, agent: Agent, graph: int = SEQUENTIAL):
        _assert = len([n for n in nodes.values() if n.name == agent.name]) == 0
        assert _assert, "The agent should be provided separately, so not inside the `nodes` dict"
        nodes = {**nodes, **{agent.name: agent}}

        # Split trace into chunks
        depths = make_depth_grouping(trace, graph=graph)
        timings = make_timings(nodes, trace, depths)
        default_outputs = make_default_outputs(nodes, timings)
        chunks, substeps, isolate = make_splitter(trace, timings, depths)

        # Make chunk runner
        run_chunk = make_run_chunk(nodes, trace, timings, chunks, substeps, graph=graph)

        # Make compiled reset function
        self.__reset = make_graph_reset(trace, trace.name, default_outputs, isolate, run_chunk)

        # make compiled step function
        self.__step = make_graph_step(trace, trace.name, isolate, run_chunk)

        # Store remaining attributes
        self.trace = trace
        self.max_steps = len(chunks)-1
        assert self.max_steps <= len(chunks)-1, f"max_steps ({self.max_steps}) must be smaller than the number of chunks ({len(chunks)-1})"

        super().__init__(agent=agent)

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, jp.float32, Any]:
        # todo: initialize graph_state.outputs with empty arrays
        next_graph_state, next_ts_step, next_step_state = self.__reset(graph_state)
        return next_graph_state, next_ts_step, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, jp.float32, StepState]:
        next_graph_state, next_ts_step, next_step_state = self.__step(graph_state, step_state, output)
        return next_graph_state, next_ts_step, next_step_state
