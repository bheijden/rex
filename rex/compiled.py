from typing import Any, Dict, List, Tuple, Callable, Union
from flax.core import FrozenDict
import jumpy
import jax
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import rex.jumpy as rjp
from collections import deque
from copy import deepcopy

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
                            tick=onp.repeat(0, num_depths),          # tick:= tick trajectory,
                            stateful=onp.repeat(False, num_depths),  # stateful:= whether to update the state,
                            inputs={},                               # inputs:= inputs from other nodes to this node
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
            timings[t.name]["stateful"][idx] = t.stateful

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

    # Sliding max window of ticks
    for name, t in timings.items():
        t["tick"] = onp.maximum.accumulate(t["tick"])
        t["ts_step"] = onp.maximum.accumulate(t["ts_step"])

    # Stack timings
    for name, t in timings.items():
        for input_name, i in t["inputs"].items():
            for k, v in i.items():
                i[k] = onp.stack(v, axis=0)

    return timings


def make_default_outputs(nodes: Dict[str, "Node"], timings: Timings) -> Dict[str, Output]:
    max_window = dict()
    num_ticks = dict()
    outputs = dict()
    _seed = jumpy.random.PRNGKey(0)
    for name, n in nodes.items():
        max_window[name] = n.output.max_window
        num_ticks[name] = timings[name]["tick"].max() + 1 # Number of ticks
        outputs[name] = n.default_output(_seed)  # Outputs

    # Stack outputs
    stack_fn = lambda *x: jp.stack(x, axis=0)
    stacked_outputs = dict()
    for name, n in nodes.items():
        num_buffer = num_ticks[name] + max_window[name]
        if num_buffer > 1:
            stacked_outputs[name] = jax.tree_map(stack_fn, *[outputs[name]] * num_buffer)
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
            isolate_lst.append(jax.tree_map(lambda _tb: _tb[i], timings))
            chunks.append(_last_index)
            _steps = list(reversed(range(0, _last_counter)))
            substeps += _steps
            _last_counter = 0
            _last_index = i+1
    isolate = jax.tree_map(lambda *args: jp.array(args), *isolate_lst)
    _steps = list(reversed(range(0, _last_counter)))
    substeps += _steps
    assert len(substeps) == len(depths), "Substeps must be the same length as depths."
    assert len(chunks) == len(isolate[name]["run"]), "Chunks must be the same length as the timings of the isolated depths."
    assert jp.all(isolate[name]["run"]), "Isolated depths must have run=True."
    return jp.array(chunks), jp.array(substeps), isolate


def update_output(buffer, output: Output, tick: int32) -> Output:
    new_buffer = jax.tree_map(lambda _b, _o: rjp.index_update(_b, tick, _o, copy=True), buffer, output)
    return new_buffer


def make_update_state(name: str, stateful: bool):

    def _update_state(graph_state: GraphState, timing: Dict, step_state: StepState, output: Any) -> GraphState:
        # Define node's step state update
        new_nodes = dict()
        new_outputs = dict()

        # Increment sequence number
        new_ss = step_state.replace(seq=step_state.seq + 1)

        # Add node's step state update
        new_nodes[name] = new_ss
        new_outputs[name] = update_output(graph_state.outputs[name], output, timing[name]["tick"])
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes),
                                              outputs=graph_state.outputs.copy(new_outputs))
        return new_graph_state

    return _update_state


def make_update_inputs(name: str, outputs: Dict[str, str]):

    def _update_inputs(graph_state: GraphState, timing: Dict) -> StepState:
        ss = graph_state.nodes[name]
        ts_step = timing[name]["ts_step"]
        seq = timing[name]["tick"]
        new_inputs = dict()
        for input_name, node_name in outputs.items():
            t = timing[name]["inputs"][input_name]
            buffer = graph_state.outputs[node_name]
            _new = InputState.from_outputs(t["seq"], t["ts_sent"], t["ts_recv"], rjp.tree_take(buffer, t["seq"]), is_data=True)
            new_inputs[input_name] = _new

        return ss.replace(seq=seq, ts=ts_step, inputs=FrozenDict(new_inputs))

    return _update_inputs


def make_run_node(name: str, node: "Node", outputs: Dict[str, str], stateful: bool):
    update_inputs = make_update_inputs(name, outputs)
    update_state = make_update_state(name, stateful)

    def _run_node(graph_state: GraphState, timing: Dict) -> GraphState:
        # Update inputs
        ss = update_inputs(graph_state, timing)

        # Run node step
        new_ss, output = node.step(ss)

        # Get mask
        new_graph_state = update_state(graph_state, timing, new_ss, output)
        return new_graph_state

    return _run_node


def make_run_batch_chunk(timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray, graph: int,
                         batch_nodes: Dict[str, "Node"], batch_outputs: Dict, stateful: Dict,
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
    slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), timings)

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
                new_ss, output = node.step(ss)

                # Increment sequence number
                new_seq_ss = new_ss.replace(seq=new_ss.seq + 1)

                buffer = update_output(graph_state.outputs[name], output, timing[name]["tick"])
                return new_seq_ss, buffer

            # Run node step
            if cond:
                new_ss, new_output = jumpy.lax.cond(pred, _run_node, lambda *args: (_old_ss, _old_output), graph_state, timing)
            else:
                _update_ss, _update_output = _run_node(graph_state, timing)
                new_ss = jax.tree_map(lambda _u, _o: jp.where(pred, _u, _o), _update_ss, _old_ss)
                new_output = jax.tree_map(lambda _u, _o: jp.where(pred, _u, _o), _update_output, _old_output)

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
            timings_chunk = jax.tree_map(lambda _tb, _size: rjp.dynamic_slice(_tb, [chunk] + [0*s for s in _size], [fixed_num_steps] + _size), timings, slice_sizes)
            # Run chunk
            graph_state, _ = rjp.scan(_run_batch_step, graph_state, timings_chunk, length=fixed_num_steps, unroll=fixed_num_steps)
        else:
            # todo: Can we statically re-compile scan for different depth lengths?
            raise NotImplementedError("batched mode not implemented yet.")

        return graph_state

    return _run_batch_chunk


def make_run_sequential_chunk(timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray, graph: int,
                              batch_nodes: Dict[str, "Node"], batch_outputs: Dict[str, Dict[str, str]], stateful: Dict):

    # Define step functions
    run_node_fns = [make_run_node(name, node, batch_outputs[name], stateful[name]) for name, node in batch_nodes.items()]

    def _run_step(substep: int32, carry: Tuple[GraphState, int32]):
        # Unpack carry
        graph_state, chunk = carry

        # Get timings of this step
        step_index = chunk + substep
        timings_step = rjp.tree_take(timings, step_index)

        # determine which nodes to run
        must_run_lst = [timings_step[name]["run"] for name in batch_nodes.keys()]
        must_run = jp.argmax(jp.array(must_run_lst))

        # Run node
        new_graph_state = jumpy.lax.switch(must_run, run_node_fns, graph_state, timings_step)

        return new_graph_state, chunk

    def _run_sequential_chunk(graph_state: GraphState) -> GraphState:
        # Get step (used to infer timings)
        step = graph_state.step

        # Infer length of chunk
        chunk = rjp.dynamic_slice(chunks, (step,), (1,))[0]  # has len(num_isolated_depths)
        num_steps = rjp.dynamic_slice(substeps, (chunk,), (1,))[0]
        # Run chunk
        initial_carry = (graph_state, chunk)
        # todo: fori_loop inhibits the use of reverse-mode differentiation.
        #       Can we use scan instead?
        graph_state, _ = jumpy.lax.fori_loop(0, num_steps, _run_step, initial_carry)
        return graph_state

    return _run_sequential_chunk


def make_run_chunk(nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord,
                   timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray,
                   graph: int):
    # Exclude pruned nodes from batch step
    batch_nodes = {node_name: node for node_name, node in nodes.items() if node_name != trace.name and (node_name not in trace.pruned)}
    # Structure is {node_name: {input_name: output_node_name}}
    batch_outputs = {name: {i.input_name: i.output.name for i in n.inputs if i.output.name not in trace.pruned} for name, n in batch_nodes.items()}

    # Infer stateful nodes
    node_names = list(batch_nodes.keys())
    stateful = {}
    for s in trace.used:
        if s.name in node_names:
            stateful[s.name] = s.stateful
            node_names.remove(s.name)
            if len(node_names) == 0:
                break
    assert len(node_names) == 0, "All nodes must be accounted for."

    if graph in [VECTORIZED, BATCHED]:
        return make_run_batch_chunk(timings, chunks, substeps, graph, batch_nodes, batch_outputs, stateful)
    elif graph in [SEQUENTIAL]:
        return make_run_sequential_chunk(timings, chunks, substeps, graph, batch_nodes, batch_outputs, stateful)
    else:
        raise ValueError("Unknown graph type.")


def make_graph_reset(nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord, name: str, default_outputs, isolate: Timings, run_chunk: Callable):
    # Exclude pruned nodes from batch step
    batch_nodes = {node_name: node for node_name, node in nodes.items() if (node_name not in trace.pruned)}
    batch_outputs = {name: {i.input_name: i.output.name for i in n.inputs if i.output.name not in trace.pruned} for name, n in batch_nodes.items()}

    # Prepare inputs update function isolated node (e.g. root).
    max_step = isolate[name]["tick"].shape[0]
    isolate_outputs = dict()
    for node_info in trace.node:
        if node_info.name != name:
            continue
        for i in node_info.inputs:
                isolate_outputs[i.name] = i.output

    # Define update function
    dummy_timing = rjp.tree_take(isolate, 0)
    update_input_fns = {name: make_update_inputs(name, outputs) for name, outputs in batch_outputs.items()}

    def _update_outputs(graph_state: GraphState) -> GraphState:
        new_outputs = dict()
        new_nodes = dict()
        for node_name, outs in default_outputs.items():
            if node_name not in graph_state.outputs:
                max_window = nodes[node_name].output.max_window

                # Sample rngs for outputs, and replace node rng.
                new_rng, *rngs_out = jumpy.random.split(graph_state.nodes[node_name].rng, num=1 + max_window)
                new_nodes[node_name] = graph_state.nodes[node_name].replace(rng=new_rng)

                # Overwrite outputs from [-max_win, ..., -1] with default. (This is usually the case for training)
                win_outs = [nodes[node_name].default_output(_rng, graph_state) for _rng in rngs_out]
                if max_window > 0:
                    win_outs = jax.tree_util.tree_map(lambda *x: jp.stack(x, axis=0), *win_outs)
                    new_outs = jax.tree_util.tree_map(lambda x, y: rjp.index_update(x, jp.arange(-max_window, 0), y, copy=False), outs, win_outs)
                new_outputs[node_name] = new_outs if max_window > 0 else outs

        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes), outputs=graph_state.outputs.copy(new_outputs))
        return new_graph_state

    def _graph_reset(graph_state: GraphState) -> Tuple[GraphState, StepState]:
        # Update output buffers
        graph_state = _update_outputs(graph_state)

        # Update inputs (with dummy inputs, so that pytree structure matches)
        new_nodes = dict()
        for node_name, input_fn in update_input_fns.items():
            new_nodes[node_name] = input_fn(graph_state, dummy_timing)
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))

        # Grab step
        # NOTE! The graph_state.step indexes into the timings and indicates what chunks have run.
        #       All chunks up until (and including) the graph.state'th chunks have been run.
        #       I.E., graph_state.step=0 means that we have run the first chunk AFTER this function has run.
        #       We do not increment step+1 here, because we increment before running a chunk in graph_step.
        step = jp.clip(graph_state.step, jp.int32(0), max_step - 1)

        # Run initial chunk.
        _next_graph_state = run_chunk(graph_state)

        # Update input
        next_timing = rjp.tree_take(isolate, step)
        next_ss = update_input_fns[name](_next_graph_state, next_timing)
        next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({name: next_ss}))
        return next_graph_state, next_ss
    return _graph_reset


def make_graph_step(trace: log_pb2.TraceRecord, name: str, isolate: Timings, run_chunk: Callable):
    max_step = isolate[name]["tick"].shape[0]

    # Infer stateful nodes
    stateful = None
    for s in trace.used:
        if s.name == name:
            stateful = s.stateful
            break
    assert stateful is not None, "Node not found in trace."

    outputs = dict()
    for node_info in trace.node:
        if node_info.name != name:
            continue
        for i in node_info.inputs:
                outputs[i.name] = i.output

    update_state = make_update_state(name, stateful)
    update_input = make_update_inputs(name, outputs)

    def _graph_step(graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, StepState]:
        # Update graph_state with action
        timing = rjp.tree_take(isolate, graph_state.step)
        new_graph_state = update_state(graph_state, timing, step_state, action)

        # Grab step
        # NOTE! The graph_state.step is used to index into the timings.
        #       Therefore, we increment it before running the chunk so that we index into the timings of the next step.
        #       Hence, graph_state.step indicates what chunks have run.
        next_step = jp.clip(new_graph_state.step + 1, jp.int32(0), max_step - 1)

        # Run chunk of next step.
        _next_graph_state = run_chunk(new_graph_state.replace(step=next_step))

        # Update input
        next_timing = rjp.tree_take(isolate, next_step)
        next_ss = update_input(_next_graph_state, next_timing)
        next_graph_state = _next_graph_state.replace(nodes=_next_graph_state.nodes.copy({name: next_ss}))
        return next_graph_state, next_ss

    return _graph_step


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord, root: Agent, graph_type: int = SEQUENTIAL):
        super().__init__(root=root, nodes=nodes)

        # Split trace into chunks
        depths = make_depth_grouping(trace, graph=graph_type)
        timings = make_timings(self.nodes_and_root, trace, depths)
        default_outputs = make_default_outputs(self.nodes_and_root, timings)
        chunks, substeps, isolate = make_splitter(trace, timings, depths)

        # Make chunk runner
        run_chunk = make_run_chunk(self.nodes_and_root, trace, timings, chunks, substeps, graph=graph_type)

        # Make compiled reset function
        self.__reset = make_graph_reset(self.nodes_and_root, trace, trace.name, default_outputs, isolate, run_chunk)

        # make compiled step function
        self.__step = make_graph_step(trace, trace.name, isolate, run_chunk)

        # Store remaining attributes
        self._default_outputs = default_outputs
        self._depths = depths
        self._timings = timings
        self._chunks = chunks
        self._substeps = substeps
        self._isolate = isolate
        self.trace: log_pb2.TraceRecord = trace
        self.max_steps = len(chunks)-1
        assert self.max_steps <= len(chunks)-1, f"max_steps ({self.max_steps}) must be smaller than the number of chunks ({len(chunks)-1})"

    def __getstate__(self):
        trace_str = self.trace.SerializeToString()
        args, kwargs = (), dict(nodes=self.nodes, trace_str=trace_str, agent=self.root)
        return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state
        kwargs["trace"] = log_pb2.TraceRecord.FromString(kwargs.pop("trace_str"))
        super().__setstate__((args, kwargs))

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, Any]:
        next_graph_state, next_step_state = self.__reset(graph_state)
        return next_graph_state, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, StepState]:
        next_graph_state, next_step_state = self.__step(graph_state, step_state, output)
        return next_graph_state, next_step_state
