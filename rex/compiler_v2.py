from functools import partial
from typing import Any, Dict, List, Tuple, Generator, Callable, Union
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import rex.jumpy as rjp
import jax
from flax import struct

from rex.constants import WARN, SYNC, FAST_AS_POSSIBLE, PHASE, SIMULATED
from rex.proto import log_pb2
from rex.node import Node
from rex.graph import BaseGraph
from rex.base import InputState, StepState, GraphState, State
from rex.agent import Agent


SplitOutput = Dict[str, Union[int, log_pb2.TracedStep, Dict[str, int], List[log_pb2.TracedStep]]]

Timings = Dict[str, Union[Dict[str, jp.ndarray], jp.ndarray]]


class TreeLeaf:
    def __init__(self, container):
        self.c = container


def make_depth_grouping(trace: log_pb2.TraceRecord, vectorized: bool) -> List[List[log_pb2.TracedStep]]:
    max_depth = trace.max_depth
    depths = [[] for _ in range(max_depth + 1)]
    for t in trace.used:
        depths[t.depth].append(t)

    # If vectorized, we that there are always max_consecutive depths between two consecutive isolated depths.
    # This allows the use of scan, instead of a for loop.
    if vectorized:
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

    return depths


def make_batch_timings(nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord, vectorized: bool) -> Tuple[Timings, List[List[log_pb2.TracedStep]]]:
    # Make depth groupings (if vectorized, this will be padded)
    depths = make_depth_grouping(trace, vectorized=vectorized)

    # Number of depths (may be increased if vectorized)
    num_depths = len(depths)

    # Prepare timings pytree
    timings = {n.name: dict(run=jp.repeat(False, num_depths),       # run:= whether the node must run,
                            ts_step=jp.repeat(0., num_depths),      # ts_step:= ts trajectory,
                            stateful=jp.repeat(False, num_depths),  # stateful:= whether to update the state,
                            inputs={},                               # inputs:= inputs from other nodes to this node
                            # outputs=[],                               # outputs:= outputs to other node from this node
                            ) for i, n in enumerate(trace.node)}
    for name, t, in timings.items():
        for i in nodes[name].inputs:
            t["inputs"][i.info.name] = dict(update=jp.repeat(False, num_depths),   # update:= whether to update the input with output,
                                            seq=jp.repeat(-1, num_depths),         # seq:= seq trajectory,
                                            ts_sent=jp.repeat(-1., num_depths),    # ts_sent:= ts trajectory
                                            ts_recv=jp.repeat(-1., num_depths))    # ts_recv:= ts trajectory
            # timings[i.info.output]["outputs"].append((name, i.info.name))

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

    return timings, depths


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


def push_output(input_state: InputState, seq: jp.int32, ts_sent: jp.float32, ts_recv: jp.float32, output: Any) -> InputState:
    return input_state.push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=output)


def make_run_node_step(node: "Node", outputs: Dict[str, str]):

    def _run_node_step():
        pass

    return _run_node_step


def make_update(name: str, outputs: List[Tuple[str, str]], stateful: bool, static: bool):
    # Define update function
    set_mask = lambda tree, val: jp.tree_map(lambda x: val, tree)

    def _update(graph_state: GraphState, timing: Dict, step_state: StepState, output: Any) -> Tuple[GraphState, GraphState]:
        mask = jp.tree_map(lambda x: False, graph_state)
        must_run = timing[name]["run"]

        # Define node's step state update
        new_nodes = dict()
        new_nodes_mask = dict()

        # Add node's step state update
        new_nodes[name] = step_state

        # Create update mask for node stepstate
        new_state_mask = set_mask(mask.nodes[name].state, stateful and must_run)
        new_params_mask = set_mask(mask.nodes[name].params, must_run and not static)
        new_rng_mask = set_mask(mask.nodes[name].rng, must_run)
        new_nodes_mask[name] = mask.nodes[name].replace(state=new_state_mask, params=new_params_mask, rng=new_rng_mask)

        # Define node's output push to other nodes
        for node_name, input_name in outputs:
            update = timing[node_name]["inputs"][input_name]["update"]
            if update:
                assert must_run, "Node must run if it is to update another node."
            seq = timing[node_name]["inputs"][input_name]["seq"]
            ts_sent = timing[node_name]["inputs"][input_name]["ts_sent"]
            ts_recv = timing[node_name]["inputs"][input_name]["ts_recv"]

            # Push output
            new_input = graph_state.nodes[node_name].inputs[input_name].replace(seq=seq,
                                                                                ts_sent=ts_sent,
                                                                                ts_recv=ts_recv,
                                                                                data=output)
            new_inputs = graph_state.nodes[node_name].inputs.copy({input_name: new_input})
            new_input_mask = set_mask(mask.nodes[node_name].inputs[input_name], update)
            new_inputs_mask = mask.nodes[node_name].inputs.copy({input_name: new_input_mask})
            # NOTE! Currently, nodes cannot self-connect or have multiple inputs from the same node.
            assert node_name not in new_nodes, "Overwriting node. Should implement merging instead."
            new_nodes[node_name] = graph_state.nodes[node_name].replace(inputs=new_inputs)
            new_nodes_mask[node_name] = mask.nodes[node_name].replace(inputs=new_inputs_mask)

        # Define new graph_state and mask
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
        new_mask = mask.replace(nodes=mask.nodes.copy(new_nodes_mask))
        return new_graph_state, new_mask
    return _update


def make_run_chunk(nodes: Dict[str, "Node"], trace: log_pb2.TraceRecord,
                   timings: Timings, chunks: jp.ndarray, substeps: jp.ndarray,
                   vectorized: bool):
    if vectorized:
        # todo: All substeps must be equal and vectorized
        # assert jp.all(substeps[chunks] == substeps[chunks][0]), "All substeps must be equal when vectorized."
        fixed_num_steps = int(substeps.max())
    else:
        fixed_num_steps = None

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

    # Define update function
    update_fns = {name: make_update(name, batch_outputs[name], stateful[name], static[name]) for name in batch_nodes.keys()}

    def _run_batch_step(graph_state: GraphState, timing: Dict):
        # NOTE! carry=graph_state, x=timing

        gs_lst = []
        mask_lst = []
        for name, node in batch_nodes.items():
            # todo: Skip nodes that are not used in this chunk
            must_run = timing[name]["run"]

            # Run node step
            ss = graph_state.nodes[name]
            ts_step = timing[name]["ts_step"]
            new_ss, output = node.step(ts_step, ss)

            new_gs, new_mask = update_fns[name](graph_state, timing, new_ss, output)

            # # Define node's step state update
            # new_nodes = dict()
            # new_nodes_mask = dict()
            #
            # new_state_mask = set_mask(mask.nodes[name].state, stateful[name] and must_run)
            # new_params_mask = set_mask(mask.nodes[name].params, must_run and not static[name])
            # new_rng_mask = set_mask(mask.nodes[name].rng, must_run)
            # new_nodes_mask[name] = mask.nodes[name].replace(state=new_state_mask, params=new_params_mask, rng=new_rng_mask)
            #
            # # Define node's output push to other nodes
            # for node_name, input_name in batch_outputs[name]:
            #     update = timing[node_name]["inputs"][input_name]["update"]
            #     if update:
            #         assert must_run, "Node must run if it is to update another node."
            #     seq = timing[node_name]["inputs"][input_name]["seq"]
            #     ts_sent = timing[node_name]["inputs"][input_name]["ts_sent"]
            #     ts_recv = timing[node_name]["inputs"][input_name]["ts_recv"]
            #
            #     # Push output
            #     new_input = graph_state.nodes[node_name].inputs[input_name].replace(seq=seq,
            #                                                                         ts_sent=ts_sent,
            #                                                                         ts_recv=ts_recv,
            #                                                                         data=output)
            #     new_inputs = graph_state.nodes[node_name].inputs.copy({input_name: new_input})
            #     new_input_mask = set_mask(mask.nodes[node_name].inputs[input_name], update)
            #     new_inputs_mask = mask.nodes[node_name].inputs.copy({input_name: new_input_mask})
            #     # NOTE! Currently, nodes cannot self-connect or have multiple inputs from the same node.
            #     assert node_name not in new_nodes, "Overwriting node. Should implement merging instead."
            #     new_nodes[node_name] = graph_state.nodes[node_name].replace(inputs=new_inputs)
            #     new_nodes_mask[node_name] = mask.nodes[node_name].replace(inputs=new_inputs_mask)
            #
            # # Define new graph_state and mask
            # new_gs = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
            # new_mask = mask.replace(nodes=mask.nodes.copy(new_nodes_mask))

            # Append to list
            gs_lst.append(new_gs)
            mask_lst.append(new_mask)

        gs_choice = jp.tree_map(lambda *args: TreeLeaf(args), *gs_lst)
        mask_choice = jp.tree_map(lambda *args: TreeLeaf(args), *mask_lst)
        graph_state = jp.tree_map(lambda mask, next_gs, prev_gs: rjp.select(mask.c, next_gs.c, prev_gs), mask_choice, gs_choice,
                             graph_state)

        # todo: define make_node_step
        return graph_state, None  # NOTE! carry=graph_state, output=None

    def _run_chunk(graph_state: GraphState) -> GraphState:
        # Get step (used to infer timings)
        step = graph_state.step

        # Run step
        if vectorized:
            # Infer length of chunk
            # todo: indexing not working with JIT
            chunk = chunks[step]  # has len(num_isolated_depths)
            timings_chunk = jp.tree_map(lambda _tb: _tb[chunk:chunk+fixed_num_steps], timings)
            # Run chunk # todo: Does this also with for varying lengths (i.e. statically re-compile for every possible length).
            graph_state, _ = rjp.scan(_run_batch_step, graph_state, timings_chunk, length=fixed_num_steps, unroll=fixed_num_steps)
        else:
            chunk = chunks[step]  # has len(num_isolated_depths)
            num_steps = substeps[chunk]  # Has len(depths)--> includes isolated depths
            timings_chunk = jp.tree_map(lambda _tb: _tb[chunk:chunk + num_steps], timings)
            # Run chunk. todo: use for loop, or recompile for different depth lengths
            # graph_state, _ = rjp.scan(_run_batch_step, graph_state, timings_chunk, unroll=unroll)
            raise NotImplementedError("Non-vectorized mode not implemented yet.")
        return graph_state

    return _run_chunk


def make_graph_reset(name: str, isolate: Timings, run_chunk: Callable):
    def graph_reset(graph_state: GraphState) -> Tuple[GraphState, jp.float32, StepState]:
        # Grab step
        step = graph_state.step

        # Run initial chunk.
        next_graph_state = run_chunk(graph_state)

        # Determine next ts
        next_ts_step = isolate[name]["ts_step"][step]

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

    update = make_update(name, outputs, stateful, static)

    def graph_step(graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, jp.float32, StepState]:
        # todo: push action into the step_state of nodes that have the agent as input.
        # todo: update the agent's step_state.
        # Update graph_state with action
        timing = jp.tree_map(lambda _tb: _tb[graph_state.step], isolate)
        gs_choice, mask = update(graph_state, timing, step_state, action)
        graph_state = jp.tree_map(lambda mask, next_gs, prev_gs: rjp.select([mask], [next_gs], prev_gs), mask, gs_choice, graph_state)

        # Grab step
        next_step = graph_state.step + 1

        # Run chunk of next step.
        # NOTE! The graph_state.step is used to index into the timings.
        #  Therefore, we increment it before running the chunk so that we index into the timings of the next step.
        next_graph_state = run_chunk(graph_state.replace(step=next_step))

        # Determine next ts
        next_ts_step = isolate[name]["ts_step"][next_step]

        return next_graph_state, next_ts_step, next_graph_state.nodes[name]

    return graph_step


class CompiledGraph(BaseGraph):
    def __init__(self, nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, agent: Agent, max_steps: int, vectorized: bool = True):
        # Split trace into chunks
        name = trace.name
        timings, depths = make_batch_timings(nodes, trace, vectorized=vectorized)
        chunks, substeps, isolate = make_splitter(trace, timings, depths)

        run_chunk = make_run_chunk(nodes, trace, timings, chunks, substeps, vectorized=vectorized)

        # Compile reset
        self.__reset = make_graph_reset(trace.name, isolate, run_chunk)

        # Compile step
        self.__step = make_graph_step(trace, trace.name, isolate, run_chunk)

        self.trace = trace
        self.max_steps = max_steps if max_steps is not None else len(chunks)-1
        assert self.max_steps <= len(chunks)-1, f"max_steps ({self.max_steps}) must be smaller than the number of chunks ({len(chunks)-1})"

        super().__init__(agent=agent)

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, jp.float32, Any]:
        next_graph_state, next_ts_step, next_step_state = self.__reset(graph_state)
        return next_graph_state, next_ts_step, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, jp.float32, StepState]:
        next_graph_state, next_ts_step, next_step_state = self.__step(graph_state, step_state, output)
        return next_graph_state, next_ts_step, next_step_state

