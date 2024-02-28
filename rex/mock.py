from itertools import chain
import tqdm
from typing import Tuple, Deque, Dict, Union, List


import jax
import jax.numpy as jnp
import numpy as onp

from rex.asynchronous import AsyncGraph
import rex.constants as const
from rex.distributions import GMM
from rex.node import Node
from rex.base import StepState, GraphState, Empty
from rex.proto import log_pb2


class MockNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> Empty:
        return Empty()

    def step(self, step_state: StepState) -> Tuple[StepState, Empty]:
        return step_state, Empty()


class Mock:
    def __init__(self, nodes: Dict[str, "Node"]):
        self._nodes = nodes
        self._node_infos = {}
        for name, node in self._nodes.items():
            info = node.info
            assert name == node.name, f"Node name {name} does not match node.name {node.name}"
            assert info.name == node.name, f"Node info name {info.name} does not match node.name {node.name}"
            self._node_infos[info.name] = node.info

    def augment(self, record: log_pb2.EpisodeRecord, ts_end, rng: jax.Array = None, progress_bar: bool = False, copy_data: bool = False) -> log_pb2.EpisodeRecord:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        eps = None

        # Get node infos
        node_infos = {}
        node_records = {}
        # max_ts_output = {}
        # max_ts_step = {}
        for node_record in record.node:
            node_infos[node_record.info.name] = node_record.info
            node_records[node_record.info.name] = node_record
            # max_ts_output[node_record.info.name] = max([step.ts_output for step in node_record.steps])
            # max_ts_step[node_record.info.name] = max([step.ts_step for step in node_record.steps])
            # To determine the eps, we need to find an InputRecord.grouped[0].messages[0].sent.eps from one o real node
            if eps is None:
                for input_record in node_record.inputs:
                    for grouped_record in input_record.grouped:
                        for message_record in grouped_record.messages:
                            eps = message_record.sent.eps
                            break
                        if eps is not None:
                            break
                    if eps is not None:
                        break
        assert eps is not None, "Could not determine eps from the record."

        # Determine subset of nodes that are not in the recording
        sim_nodes = {}
        real_nodes = {}
        for name, info in self._node_infos.items():
            if name in node_infos:
                real_nodes[name] = self._nodes[name]
            else:
                sim_nodes[name] = self._nodes[name]

        # Determine connections between sim_nodes and real_nodes
        sim_connections = {}
        real_connections = {}
        sim2real_connections = {}
        real2sim_connections = {}
        SIM, REAL = 0, 1
        for name_in, info in self._node_infos.items():
            type_in = SIM if name_in in sim_nodes else REAL
            for i in info.inputs:
                name_out = i.output
                type_out = SIM if name_out in sim_nodes else REAL
                if type_out == REAL and type_in == SIM:
                    real2sim_connections[name_in] = real2sim_connections.get(name_in, [])
                    real2sim_connections[name_in].append(i)
                    # todo: support jitter modes and non-blocking connections in real2sim
                    assert i.jitter == const.LATEST, f"real2sim connection {name_in} -> {name_out}: Expected jitter to be {const.JITTER_MODES[const.LATEST]}, but got {const.JITTER_MODES[i.jitter]}"
                    assert i.blocking == False, f"real2sim connection {name_in} -> {name_out}: Expected blocking to be {False}, but got {i.blocking}"
                elif type_out == SIM and type_in == REAL:
                    sim2real_connections[name_in] = sim2real_connections.get(name_in, [])
                    sim2real_connections[name_in].append(i)
                    # todo: support jitter modes and non-blocking connections in sim2real
                    assert i.jitter == const.LATEST, f"sim2real connection {name_in} -> {name_out}: Expected jitter to be {const.JITTER_MODES[const.LATEST]}, but got {const.JITTER_MODES[i.jitter]}"
                    assert i.blocking == False, f"sim2real connection {name_in} -> {name_out}: Expected blocking to be {False}, but got {i.blocking}"
                elif type_out == REAL and type_in == REAL:
                    real_connections[name_in] = real_connections.get(name_in, [])
                    real_connections[name_in].append(i)
                elif type_out == SIM and type_in == SIM:
                    sim_connections[name_in] = sim_connections.get(name_in, [])
                    sim_connections[name_in].append(i)
                else:
                    raise ValueError(f"Unexpected combination of type_in {type_in} and type_out {type_out}")

        # Create mock nodes
        mock_nodes = {name: MockNode.subclass_from_info(node.info) for name, node in sim_nodes.items()}

        # Connect mock nodes
        [node.connect_from_info(sim_connections[name], mock_nodes) for name, node in mock_nodes.items() if name in sim_connections]

        # Create mock graph
        root = max(mock_nodes.values(), key=lambda n: n.rate)  # Highest rate, so that we can exist as soon as possible
        graph = AsyncGraph(mock_nodes, root, clock=const.SIMULATED, real_time_factor=const.FAST_AS_POSSIBLE)

        # Initialize pbar
        pbar = tqdm.tqdm(total=ts_end, desc=f"Augmenting", disable=not progress_bar)

        import rex.utils as rutils
        rutils.set_log_level(const.WARN)

        # todo: deadlock #1
        # graph.start(gs)  # this results in stopping (because it is scheduled after?)
        # gs, _ = graph.reset(gs)
        # gs, _ = graph.step(gs)

        # todo: deadlock #2
        # graph.start(gs)
        # gs = graph.run(gs)

        # Initialize mock graph
        gs = graph.init(starting_eps=eps)
        # graph.start(gs)
        gs, _ = graph.reset(gs)
        while True:
            gs, _ = graph.step(gs)
            # gs = graph.run(gs)

            # Break if all nodes have passed the ts_end
            ts_steps = {name: ss.ts for name, ss in gs.nodes.items()}
            seqs = {name: ss.seq for name, ss in gs.nodes.items()}
            ts_seqs = {name: f"{ts_steps[name]:.2f} sec, {seqs[name]+1} nodes" for name in ts_steps}
            msg = "|".join([f"{name}: {ts_seqs[name]}" for name in ts_seqs])

            # Determine how many nodes have passed the ts_end since the last update
            num_passed = sum([ts_steps[name] >= ts_end for name in ts_steps])
            mean_ts_steps = sum([ts_steps[name] for name in ts_steps]) / len(ts_steps)
            pbar.update(mean_ts_steps - pbar.n)

            pbar.set_postfix_str(msg)
            pbar.refresh()
            if num_passed == len(mock_nodes):
                graph.stop()
                pbar.close()
                break

        # Prepare augmented record (sim + real)
        aug_node_records = {name: node.record() for name, node in mock_nodes.items()}

        # Add records of the original record
        for name, node_record in node_records.items():
            record_cp = log_pb2.NodeRecord()
            record_cp.CopyFrom(node_record)
            if not copy_data:
                [record_cp.ClearField(f) for f in ["outputs", "rngs", "states", "params", "step_states"]]
            aug_node_records[name] = record_cp

        # Add input records
        for name_in, lst_input_infos in chain(sim2real_connections.items(), real2sim_connections.items()):
            for input_info in lst_input_infos:
                name_out = input_info.output
                record_out = aug_node_records[name_out]
                record_in = aug_node_records[name_in]

                # Add input_info to record_in
                record_in.info.inputs.append(input_info)

                # Initialize empty InputRecord
                rng, rng_dist = jax.random.split(rng)
                input_record = log_pb2.InputRecord(info=input_info,
                                                   rng=rng_dist.tolist())

                # Sample communication delays
                ts_output_arr = onp.array([step.ts_output for step in record_out.steps])
                seq_arr = onp.array([step.tick for step in record_out.steps])
                dist = GMM.from_info(input_info.delay_sim)
                dist_state = dist.reset(rng_dist)
                _, delays = dist.sample(dist_state, shape=(len(ts_output_arr),))
                communication_delays_arr = onp.array(delays)

                # Get ts_output
                ts_sent_arr = ts_output_arr
                ts_recv_arr = ts_sent_arr + communication_delays_arr
                ts_step_arr = onp.array([step.ts_step for step in record_in.steps])

                # Group messages
                skip = input_info.skip
                grouped = [log_pb2.GroupedRecord(num_msgs=0) for _ in ts_step_arr]
                input_record.grouped.extend(grouped)
                messages = []
                iter_ts_step = iter(enumerate(ts_step_arr))
                iter_out = iter(zip(seq_arr, ts_sent_arr, ts_recv_arr, communication_delays_arr))
                idx, ts_step = next(iter_ts_step)
                try:
                    while True:
                        seq, ts_sent, ts_recv, delay = next(iter_out)
                        while True:
                            if ts_recv > ts_step or (skip and ts_recv == ts_step):
                                # Group messages
                                input_record.grouped[idx].messages.extend(messages)
                                input_record.grouped[idx].num_msgs = len(messages)
                                msgs_recvs = [msg.received.ts.sc for msg in messages]
                                msgs_seqs = [msg.received.seq for msg in messages]
                                assert all([msg_recv <= ts_step for msg_recv in msgs_recvs]), f"Expected all messages to be received before ts_step, but got {msgs_recvs} > {ts_step:.2f}"
                                assert all([(msgs_seqs[i+1] - msgs_seqs[i]) == 1 for i in range(len(msgs_seqs)-1)]), f"Expected messages to be in order, but got {msgs_seqs}"
                                # if name_in == "sensor":
                                #     str_msgs_recvs = " ".join([f"{msg:.2f}" for msg in msgs_recvs])
                                #     print(f"{name_out} -> {name_in}({idx}): ({str_msgs_recvs}) < {ts_step:.2f}")
                                # Clear message queue
                                messages = []
                                # Get next ts_step
                                idx, ts_step = next(iter_ts_step)
                            else:
                                break
                        sent = log_pb2.Header(eps=eps, seq=seq, ts=log_pb2.Time(sc=float(ts_sent), wc=float(ts_sent)))
                        recv = log_pb2.Header(eps=eps, seq=seq, ts=log_pb2.Time(sc=float(ts_recv), wc=float(ts_recv)))
                        comm_delay = log_pb2.Time(sc=float(delay), wc=float(delay))
                        message = log_pb2.MessageRecord(sent=sent, received=recv, comm_delay=comm_delay, delay=delay)
                        messages.append(message)
                except StopIteration:
                    # either no more incoming messages (i.e. no more iter_out)
                    # or no more steps of the "inbound" node on the receiving end (i.e. no more iter_ts_step)
                    pass
                record_in.inputs.append(input_record)
                assert len(input_record.grouped) == len(ts_step_arr), f"Expected {len(ts_step_arr)} groups, but got {len(input_record.grouped)}"
                assert len(record_in.inputs) == len(record_in.info.inputs), f"Expected {len(record_in.info.inputs)} inputs, but got {len(record_in.inputs)}"

        augmented_record = log_pb2.EpisodeRecord()
        [augmented_record.node.append(node_record) for node_record in aug_node_records.values()]
        return augmented_record



















