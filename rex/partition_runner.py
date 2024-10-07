from typing import Any, Dict, List, Tuple, Callable, Union
import functools
import networkx as nx
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import numpy as onp
import rex.jax_utils as rjax
import equinox as eqx
from rex.utils import check_generations_uniformity
from rex.node import BaseNode
from rex.base import InputState, StepState, GraphState, Output, GraphBuffer, Timings, SlotVertex, EpisodeRecord, StepRecord


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
    def _update_state(graph_state: GraphState, timing: SlotVertex, step_state: StepState, output: Any, output_record: Any) -> GraphState:
        # Define node's step state update
        new_step_states = dict()
        new_outputs = dict()

        # Increment sequence number
        new_ss = step_state.replace(seq=step_state.seq + 1)

        # Add node's step state update
        new_step_states[name] = new_ss
        new_outputs[name] = update_output(graph_state.buffer[name], output, timing.seq)

        graph_state = graph_state.replace_buffer(new_outputs)
        new_graph_state = graph_state.replace_step_states(new_step_states)

        # Update record
        record = graph_state.aux.get("record", None)
        if record is not None and output_record is not None:
            new_outputs = jax.tree_map(lambda _b, _o: jnp.array(_b).at[timing.seq].set(jnp.array(_o)),
                                       record.nodes[name].steps.output, output_record)
            new_graph_state = eqx.tree_at(lambda _gs: _gs.aux["record"].nodes[name].steps.output, new_graph_state, new_outputs)
        return new_graph_state

    return _update_state


def make_update_inputs(node: "BaseNode"):
    def _update_inputs(graph_state: GraphState, timings_node: SlotVertex) -> StepState:
        ss = graph_state.step_state[node.name]
        ts_start = timings_node.ts_start
        eps = graph_state.eps
        seq = timings_node.seq
        new_inputs = dict()
        for input_name, c in node.inputs.items():
            t = timings_node.windows[c.output_node.name]
            buffer = graph_state.buffer[c.output_node.name]
            size = get_buffer_size(buffer)
            mod_seq = t.seq % size
            inputs = rjax.tree_take(buffer, mod_seq)
            prev_delay_dist = ss.inputs[input_name].delay_dist  # This is important, as it substitutes the delay_dist with the previous one.
            _inputs_undelayed = InputState.from_outputs(t.seq, t.ts_sent, t.ts_recv, inputs, delay_dist=prev_delay_dist, is_data=True)
            if not c.delay_dist.equivalent(_inputs_undelayed.delay_dist):
                raise ValueError(f"Delay distributions are not equivalent for input `{input_name}` of node `{node.name}`: "
                                 f"{c.delay_dist} != {_inputs_undelayed.delay_dist}. \n"
                                 f"Compare the delay distributions provided to .connect(dela_dist=...) with graph_state.inputs[{node.name}][{c.output_node.name}].delay_dist.")
            _inputs = _inputs_undelayed.delay_dist.apply_delay(c.output_node.rate, _inputs_undelayed, ts_start)
            new_inputs[input_name] = _inputs
        return ss.replace(eps=eps, seq=seq, ts=ts_start, inputs=FrozenDict(new_inputs))

    return _update_inputs


def make_run_partition_excl_supervisor(
    nodes: Dict[str, "BaseNode"], timings: Timings, S: nx.DiGraph, supervisor_slot: str, skip: List[str] = None
):
    INTERMEDIATE_UPDATE = False
    RETURN_OUTPUT = True

    # Define input function
    update_input_fns = {name: make_update_inputs(n) for name, n in nodes.items()}

    # Define update function
    supervisor = timings.slots[supervisor_slot].kind
    supervisor_gen_idx = timings.slots[supervisor_slot].generation

    # Determine if all generations contain all slot_kinds
    # NOTE! This assumes that the supervisor is the only node in the last generation.
    generations = timings.to_generation()
    is_uniform = check_generations_uniformity(generations[:-1])
    slots_to_kinds = {n: v.kind for n, v in timings.slots.items()}
    kinds_to_slots = invert_dict_with_list_values(slots_to_kinds)
    kinds_to_slots.pop(supervisor)  # remove supervisor from kinds_to_slots
    for key, value in kinds_to_slots.items():
        # sort value based on the generation they belong to.
        kinds_to_slots[key] = sorted(value, key=lambda x: timings.slots[x].generation)
    if len(kinds_to_slots) == 0:
        raise ValueError("There are no nodes in the partition (excl. supervisor).")

    # Determine which slots to skip
    skip_slots = [n for n, v in timings.slots.items() if v.kind in skip] if skip is not None else []
    skip_slots = (
        skip_slots + skip if skip is not None else skip_slots
    )  # also add kinds to skip slots, because if uniform, then kinds are also slots.

    def _run_node(kind: str, graph_state: GraphState, timings_node: SlotVertex):
        # Update inputs
        ss = update_input_fns[kind](graph_state, timings_node)
        # ss = _old_ss

        # Run node step
        _new_ss, output = nodes[kind].step(ss)

        # Increment sequence number
        _new_seq_ss = _new_ss.replace(seq=_new_ss.seq + 1)

        # Update record # todo: update record
        if "record" in graph_state.aux:
            record: EpisodeRecord = graph_state.aux["record"]
            new_step_record = StepRecord(
                eps=ss.eps,
                seq=ss.seq,
                ts_start=ss.ts,
                ts_end=timings_node.ts_end,
                delay=timings_node.ts_end - ss.ts,
                rng=ss.rng if record.nodes[kind].steps.rng is not None else None,
                inputs=ss.inputs if record.nodes[kind].steps.inputs is not None else None,
                state=ss.state if record.nodes[kind].steps.state is not None else None,
                output=output if record.nodes[kind].steps.output is not None else None,
            )
        else:
            new_step_record = None

        # Update output buffer
        if not RETURN_OUTPUT:
            output = update_output(graph_state.buffer[kind], output, timings_node.seq)
        # _new_output = graph_state.outputs[kind]
        return _new_seq_ss, output, new_step_record

    node_step_fns = {kind: functools.partial(_run_node, kind) for kind in nodes.keys()}

    def _run_generation(graph_state: GraphState, timings_gen: Dict[str, SlotVertex]):
        record: EpisodeRecord = graph_state.aux.get("record", None)
        new_records = dict()
        new_step_states = dict()
        new_outputs = dict()
        for slot_kind, timings_node in timings_gen.items():
            # Skip slots
            if slot_kind == supervisor_slot or slot_kind in skip_slots:
                continue

            if INTERMEDIATE_UPDATE:
                new_step_states = dict()
                new_outputs = dict()
            kind = timings.slots[slot_kind].kind  # Node kind to run
            pred = timings_gen[slot_kind].run  # Predicate for running node step

            # Prepare old states
            noop_step_record = rjax.tree_take(record.nodes[kind].steps, timings_node.seq) if record is not None else None
            noop_ss = graph_state.step_state[kind]
            if RETURN_OUTPUT:
                size = get_buffer_size(graph_state.buffer[kind])
                noop_output = rjax.tree_take(graph_state.buffer[kind], timings_node.seq % size)
            else:
                noop_output = graph_state.buffer[kind]

            if noop_ss.inputs is None:
                raise DeprecationWarning("Inputs should not be None, but pre-filled via graph.init")

            # Run node step
            no_op = lambda *args: (noop_ss, noop_output, noop_step_record)
            try:
                new_ss, output, new_step_record = jax.lax.cond(pred, node_step_fns[kind], no_op, graph_state, timings_node)
            except TypeError as e:
                print(f"TypeError: kind={kind}:",  e)
                new_ss, output, new_step_record = node_step_fns[kind](graph_state, timings_node)
                raise e

            # Update record
            if record is not None:
                steps = record.nodes[kind].steps
                new_steps = jax.tree_map(lambda _b, _o: jnp.array(_b).at[timings_node.seq].set(jnp.array(_o)), steps, new_step_record)
                new_records[kind] = record.nodes[kind].replace(steps=new_steps)

            # Store new state
            new_step_states[kind] = new_ss
            if RETURN_OUTPUT:
                new_outputs[kind] = update_output(graph_state.buffer[kind], output, timings_node.seq)
            else:
                new_outputs[kind] = output

            # Update buffer
            if INTERMEDIATE_UPDATE:
                # todo: Incorrect? new_outputs/new_step_states are not emptied to {} after the lines below.
                graph_state = graph_state.replace_buffer(new_outputs)
                graph_state = graph_state.replace_step_states(new_step_states)
                raise NotImplementedError("Intermediate record update not yet implemented")

        if INTERMEDIATE_UPDATE:
            new_graph_state = graph_state
        else:
            graph_state = graph_state.replace_buffer(new_outputs)
            new_graph_state = graph_state.replace_step_states(new_step_states)
            if record is not None:
                new_nodes = record.nodes.copy()
                new_nodes.update(new_records)
                new_graph_state = new_graph_state.replace_aux({"record": record.replace(nodes=new_nodes)})
        return new_graph_state, new_graph_state

    def _run_S(graph_state: GraphState) -> GraphState:
        # Get eps & step  (used to index timings)
        graph_state = graph_state.replace_step(timings, step=graph_state.step)  # Make sure step is clipped to max_step size
        step = graph_state.step

        # Determine slice sizes (depends on window size)
        # [1:] because first dimension is step.
        timings_eps = graph_state.timings_eps
        slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), timings_eps)

        # Slice timings
        timings_mcs = jax.tree_map(
            lambda _tb, _size: jax.lax.dynamic_slice(_tb, [step] + [0 * s for s in _size], [1] + _size)[0],
            timings_eps,
            slice_sizes,
        )
        timings_mcs = timings_mcs.to_generation()

        # Run generations
        # NOTE! len(generations) = len(timings_mcs) --> last generation is the supervisor.
        if not is_uniform:
            for gen, timings_gen in zip(generations[:-1], timings_mcs):
                assert all([node in gen for node in timings_gen.keys()]), f"Missing nodes in timings: {gen}"
                graph_state, _ = _run_generation(graph_state, timings_gen)
        else:
            # raise NotImplementedError("Uniform generations not yet validated")
            flattened_timings = dict()
            # NOTE! This assumes that the supervisor is the only node in the last generation.
            [
                flattened_timings.update(timings_gen) for timings_gen in timings_mcs[:-1]
            ]  # Remember: this does include supervisor_slot
            slots_timings = {}
            for kind, slots in kinds_to_slots.items():  # Remember: kinds_to_slots does not include supervisor_slot
                timings_to_stack = [flattened_timings[slot].replace(generation=None) for slot in slots]
                slots_timings[slots[0]] = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *timings_to_stack)
            all_shapes = [v.run.shape for k, v in slots_timings.items()]
            assert all([s == all_shapes[0] for s in all_shapes]), "Shapes of slots are not equal"
            graph_state, _ = jax.lax.scan(_run_generation, graph_state, slots_timings)

        # Run supervisor input update
        new_ss_supervisor = update_input_fns[supervisor](graph_state, timings_mcs[supervisor_gen_idx][supervisor_slot])
        graph_state = graph_state.replace_step_states({supervisor: new_ss_supervisor})

        record = graph_state.aux.get("record", None)
        if record is not None:
            timing_sup = rjax.tree_take(graph_state.timings_eps.slots[supervisor_slot], i=graph_state.step)
            new_step_record = StepRecord(
                eps=new_ss_supervisor.eps,
                seq=new_ss_supervisor.seq,
                ts_start=new_ss_supervisor.ts,
                ts_end=timing_sup.ts_end,
                delay=timing_sup.ts_end - new_ss_supervisor.ts,
                rng=new_ss_supervisor.rng if record.nodes[supervisor].steps.rng is not None else None,
                inputs=new_ss_supervisor.inputs if record.nodes[supervisor].steps.inputs is not None else None,
                state=new_ss_supervisor.state if record.nodes[supervisor].steps.state is not None else None,
                # Output is recorded AFTER the supervisor's step,
                # while the other statistics are recorded BEFORE the supervisor.
                output=None,
            )
            # Therefore, we perform some output abracadabra here.
            # We set output to none to match the static shape of new_step_record.
            # Then, we add the original output back to the record.
            new_steps = jax.tree_map(lambda _b, _o: jnp.array(_b).at[graph_state.step].set(jnp.array(_o)),
                                     record.nodes[supervisor].steps.replace(output=None), new_step_record)
            new_steps = new_steps.replace(output=record.nodes[supervisor].steps.output)
            graph_state = eqx.tree_at(lambda _gs: _gs.aux["record"].nodes[supervisor].steps, graph_state, new_steps)

        # Increment step (new step may exceed max_step) --> clipping is done at the start of run_S.
        graph_state = graph_state.replace(step=graph_state.step + 1)
        return graph_state

    return _run_S
