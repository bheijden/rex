from typing import Any, Dict, List, Tuple, Callable, Union
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import rex.jumpy as rjp
import jax
from flax import struct

from rex.constants import SEQUENTIAL, VECTORIZED, BATCHED, WARN, SYNC, FAST_AS_POSSIBLE, PHASE, SIMULATED
from rex.proto import log_pb2
from rex.node import Node
from rex.graph import BaseGraph
from rex.base import InputState, StepState, GraphState, State
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
    timings = {n.name: dict(run=jp.repeat(False, num_depths),       # run:= whether the node must run,
                            ts_step=jp.repeat(0., num_depths),      # ts_step:= ts trajectory,
                            stateful=jp.repeat(False, num_depths),  # stateful:= whether to update the state,
                            inputs={},                               # inputs:= inputs from other nodes to this node
                            ) for i, n in enumerate(trace.node)}
    for name, t, in timings.items():
        for i in nodes[name].inputs:
            t["inputs"][i.info.name] = dict(update=jp.repeat(False, num_depths),   # update:= whether to update the input with output,
                                            seq=jp.repeat(-1, num_depths),         # seq:= seq trajectory,
                                            ts_sent=jp.repeat(-1., num_depths),    # ts_sent:= ts trajectory
                                            ts_recv=jp.repeat(-1., num_depths))    # ts_recv:= ts trajectory

    # Populate timings
    for idx, depth in enumerate(depths):
        for t in depth:
            # Update source node timings
            timings[t.name]["run"][idx] = True
            timings[t.name]["ts_step"][idx] = t.ts_step
            timings[t.name]["stateful"][idx] = t.stateful or not t.static
            # Update input timings
            for d in t.downstream:
                if d.target.name == t.name:
                    continue
                target_node = d.target.name
                input_name = d.target.input_name
                timings[target_node]["inputs"][input_name]["update"][idx] = True
                timings[target_node]["inputs"][input_name]["seq"][idx] = d.source.tick
                timings[target_node]["inputs"][input_name]["ts_sent"][idx] = d.source.ts
                timings[target_node]["inputs"][input_name]["ts_recv"][idx] = d.target.ts

    return timings


def make_splitter(trace: log_pb2.TraceRecord, timings, depths) -> Tuple[jp.ndarray, jp.ndarray, Timings]:
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


def make_update_state(name: str, outputs: List[Tuple[str, str]], stateful: bool, static: bool, with_mask: bool):

    def _update_ss(old: StepState, new: StepState) -> StepState:
        # If we use a mask, we do not replace step_state because the mask will take care of filtering.
        if with_mask:
            return new
        elif stateful and not static:
            return old.replace(rng=new.rng, state=new.state, params=new.params)
        elif stateful and static:
            return old.replace(rng=new.rng, state=new.state)
        elif (not stateful) and static:
            return old.replace(rng=new.rng)
        elif (not stateful) and not static:
            return old.replace(rng=new.rng, params=new.params)
        else:
            raise ValueError(f"Invalid combination of stateful={stateful} and static={static}.")

    def _update_input(old: InputState, seq: int32, ts_sent: float32, ts_recv: float32, output: Any) -> InputState:
        return old.push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=output)

    def _update_state(graph_state: GraphState, timing: Dict, step_state: StepState, output: Any) -> GraphState:
        # Define node's step state update
        new_nodes = dict()

        # Add node's step state update
        new_nodes[name] = _update_ss(graph_state.nodes[name], step_state)

        # Define node's output push to other nodes
        for node_name, input_name in outputs:
            # if not jp._in_jit() and update:
            #     assert timing[name]["run"], "Node must run if it is to update another node."
            old = graph_state.nodes[node_name].inputs[input_name]
            update = timing[node_name]["inputs"][input_name]["update"]
            seq = timing[node_name]["inputs"][input_name]["seq"]
            ts_sent = timing[node_name]["inputs"][input_name]["ts_sent"]
            ts_recv = timing[node_name]["inputs"][input_name]["ts_recv"]

            # Push output
            if with_mask:
                new_input = _update_input(old, seq, ts_sent, ts_recv, output)
            else:
                new_input = rjp.cond(update, _update_input, lambda *args: old, old, seq, ts_sent, ts_recv, output)

            # Update node's input state
            new_inputs = graph_state.nodes[node_name].inputs.copy({input_name: new_input})
            # NOTE! Currently, nodes cannot self-connect or have multiple inputs from the same node.
            assert node_name not in new_nodes, "Overwriting node. Should implement merging instead."
            new_nodes[node_name] = graph_state.nodes[node_name].replace(inputs=new_inputs)

        # Define new graph_state and mask
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
        return new_graph_state

    return _update_state


def make_update_mask(name: str, outputs: List[Tuple[str, str]], stateful: bool, static: bool):
    # Define update function
    set_mask = lambda tree, val: jp.tree_map(lambda x: val, tree)

    def _update_mask(mask: GraphState, timing: Dict) -> Tuple[GraphState, GraphState]:
        must_run = timing[name]["run"]

        # Define node's step state update
        new_nodes_mask = dict()

        # Create update mask for node stepstate
        _replace = dict(rng=set_mask(mask.nodes[name].rng, must_run))
        if stateful:
            _replace["state"] = set_mask(mask.nodes[name].state, must_run)
        if not static:
            _replace["params"] = set_mask(mask.nodes[name].params, must_run)
        new_nodes_mask[name] = mask.nodes[name].replace(**_replace)

        # Define node's output push to other nodes
        for node_name, input_name in outputs:
            update = timing[node_name]["inputs"][input_name]["update"]
            if not jp._in_jit() and update:
                assert must_run, "Node must run if it is to update another node."

            # Push output
            new_input_mask = set_mask(mask.nodes[node_name].inputs[input_name], update)
            new_inputs_mask = mask.nodes[node_name].inputs.copy({input_name: new_input_mask})
            # NOTE! Currently, nodes cannot self-connect or have multiple inputs from the same node.
            new_nodes_mask[node_name] = mask.nodes[node_name].replace(inputs=new_inputs_mask)

        # Define new graph_state and mask
        new_mask = mask.replace(nodes=mask.nodes.copy(new_nodes_mask))
        return new_mask

    return _update_mask


def make_run_node(name: str, node: "Node", outputs: List[Tuple[str, str]], stateful: bool, static: bool):
    update = make_update_state(name, outputs, stateful, static, with_mask=False)

    def _run_node(graph_state: GraphState, timing: Dict) -> GraphState:
        # Run node step
        ss = graph_state.nodes[name]
        ts_step = timing[name]["ts_step"]
        new_ss, output = node.step(ts_step, ss)

        # Get mask
        new_graph_state = update(graph_state, timing, new_ss, output)
        return new_graph_state

    return _run_node


def make_run_batch_chunk(timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray, graph: int,
                         batch_nodes: Dict[str, "Node"], batch_outputs: Dict, stateful: Dict, static: Dict):
    if graph == VECTORIZED:
        assert jp.all(substeps[chunks] == substeps[chunks][0]), "All substeps must be equal when vectorized."
        fixed_num_steps = int(substeps[chunks][0])
    elif graph == BATCHED:
        fixed_num_steps = None
    else:
        raise ValueError("Unknown graph type.")

    # Define update function
    update_state_fns = {name: make_update_state(name, batch_outputs[name], stateful[name], static[name], with_mask=True) for name in batch_nodes.keys()}
    update_mask_fns = {name: make_update_mask(name, batch_outputs[name], stateful[name], static[name]) for name in batch_nodes.keys()}

    def _run_batch_step(graph_state: GraphState, timing: Dict):
        mask = jp.tree_map(lambda x: False, graph_state)

        gs_lst = []
        mask_lst = []
        for name, node in batch_nodes.items():
            # todo: Skip nodes that are not used in this chunk
            must_run = timing[name]["run"]

            # Run node step
            ss = graph_state.nodes[name]
            ts_step = timing[name]["ts_step"]
            new_ss, output = node.step(ts_step, ss)

            # Get mask
            new_gs = update_state_fns[name](graph_state, timing, new_ss, output)
            new_mask = update_mask_fns[name](mask, timing)

            gs_lst.append(new_gs)
            mask_lst.append(new_mask)

        gs_choice = jp.tree_map(lambda *args: TreeLeaf(args), *gs_lst)
        mask_choice = jp.tree_map(lambda *args: TreeLeaf(args), *mask_lst)
        new_graph_state = jp.tree_map(lambda mask, next_gs, prev_gs: rjp.select(mask.c, next_gs.c, prev_gs), mask_choice, gs_choice, graph_state)
        return new_graph_state, None  # NOTE! carry=graph_state, output=None

    def _run_batch_chunk(graph_state: GraphState) -> GraphState:
        # Get step (used to infer timings)
        step = graph_state.step

        # Run step
        if graph == VECTORIZED:
            # Infer length of chunk
            chunk = rjp.dynamic_slice(chunks, (step,), (1,))[0]  # has len(num_isolated_depths)
            timings_chunk = jp.tree_map(lambda _tb: rjp.dynamic_slice(_tb, (chunk,), (fixed_num_steps,)), timings)
            # Run chunk
            graph_state, _ = rjp.scan(_run_batch_step, graph_state, timings_chunk, length=fixed_num_steps, unroll=fixed_num_steps)
        else:
            # todo: Can we statically re-compile scan for different depth lengths?
            raise NotImplementedError("batched mode not implemented yet.")

        return graph_state

    return _run_batch_chunk


def make_run_sequential_chunk(timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray, graph: int,
                              batch_nodes: Dict[str, "Node"], batch_outputs: Dict, stateful: Dict, static: Dict):

    # Define step functions
    run_node_fns = [make_run_node(name, node, batch_outputs[name], stateful[name], static[name]) for name, node in batch_nodes.items()]

    def _run_step(substep: int32, carry: Tuple[GraphState, int32]):
        # Unpack carry
        graph_state, chunk = carry

        # Get timings of this step
        step_index = chunk + substep
        timings_step = jp.tree_map(lambda _tb: rjp.dynamic_slice(_tb, (step_index,), (1,))[0], timings)

        # determine which nodes to run
        must_run_lst = [timings_step[name]["run"] for name in batch_nodes.keys()]
        must_run = jp.argmax(jp.array(must_run_lst))

        # Run node
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
    # NOTE! Some weird output name grabbing happens here. Could become a source of bugs.
    batch_nodes = {node_name: node for node_name, node in nodes.items() if node_name != trace.name and (node_name not in trace.pruned)}
    batch_outputs = {name: [(nn.node.name, nn.input_name) for nn in n.output.inputs if nn.node.name not in trace.pruned] for name, n in batch_nodes.items()}

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


def make_graph_reset(name: str, isolate: Timings, run_chunk: Callable):
    def graph_reset(graph_state: GraphState) -> Tuple[GraphState, jp.float32, StepState]:
        # Grab step
        step = graph_state.step

        # Run initial chunk.
        next_graph_state = run_chunk(graph_state)

        # Determine next ts
        next_ts_step = rjp.dynamic_slice(isolate[name]["ts_step"], (step,), (1,))[0]

        # NOTE! We do not increment step, because graph_state.step is used to index into the timings.
        #       In graph_step we do increment step after running the chunk, because we want to index into the next timings.
        return next_graph_state, next_ts_step, next_graph_state.nodes[name]
    return graph_reset


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

    outputs = []
    for node_info in trace.node:
        for i in node_info.inputs:
            if i.output == name:
                outputs.append((node_info.name, i.name))

    update = make_update_state(name, outputs, stateful, static, with_mask=False)

    def graph_step(graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, jp.float32, StepState]:
        # Update graph_state with action
        timing = jp.tree_map(lambda _tb: rjp.dynamic_slice(_tb, (graph_state.step,), (1,))[0], isolate)
        new_graph_state = update(graph_state, timing, step_state, action)

        # Grab step
        next_step = new_graph_state.step + 1

        # Run chunk of next step.
        # NOTE! The graph_state.step is used to index into the timings.
        #  Therefore, we increment it before running the chunk so that we index into the timings of the next step.
        next_graph_state = run_chunk(new_graph_state.replace(step=next_step))

        # Determine next ts
        next_ts_step = rjp.dynamic_slice(isolate[name]["ts_step"], (next_step,), (1,))[0]

        return next_graph_state, next_ts_step, next_graph_state.nodes[name]

    return graph_step


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, agent: Agent, graph: int = SEQUENTIAL):
        assert len([n for n in nodes.values() if
                    n.name == agent.name]) == 0, "The agent should be provided separately, so not inside the `nodes` dict"
        nodes = {**nodes, **{agent.name: agent}}

        # Split trace into chunks
        depths = make_depth_grouping(trace, graph=graph)
        timings = make_timings(nodes, trace, depths)
        chunks, substeps, isolate = make_splitter(trace, timings, depths)

        # Make batch chunk runner (runs a single nodes every step)
        run_chunk = make_run_chunk(nodes, trace, timings, chunks, substeps, graph=graph)

        # Compile reset
        self.__reset = make_graph_reset(trace.name, isolate, run_chunk)

        # Compile step
        self.__step = make_graph_step(trace, trace.name, isolate, run_chunk)

        self.trace = trace
        self.max_steps = len(chunks)-1
        assert self.max_steps <= len(chunks)-1, f"max_steps ({self.max_steps}) must be smaller than the number of chunks ({len(chunks)-1})"

        super().__init__(agent=agent)

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, jp.float32, Any]:
        next_graph_state, next_ts_step, next_step_state = self.__reset(graph_state)
        return next_graph_state, next_ts_step, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, jp.float32, StepState]:
        next_graph_state, next_ts_step, next_step_state = self.__step(graph_state, step_state, output)
        return next_graph_state, next_ts_step, next_step_state
