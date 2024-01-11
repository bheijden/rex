from functools import partial
import jax
import jax.numpy as jnp
import numpy as onp
import numpy.ma as ma
import dill as pickle
from flax.core import FrozenDict
from typing import Tuple, List, Dict, Union
from copy import deepcopy
from collections import deque
from google.protobuf.pyext._message import RepeatedCompositeContainer
import networkx as nx

import rex.open_colors as oc
from rex.utils import deprecation_warning
from rex.node import Node
from rex.proto import log_pb2
from rex.base import SeqsMapping, BufferSizes, NodeTimings, Timings, Output, GraphBuffer, GraphState
import supergraph


def get_node_y_position(G: nx.DiGraph) -> Dict[str, int]:
    """Get the order of the nodes in the graph."""
    node_data = get_node_data(G)
    y = {k: v["order"] for k, v in node_data.items()}
    return y


def set_node_order(G: nx.DiGraph, order: List[str]):
    """Set the order of the nodes in the graph."""
    assert isinstance(order, list), "Order must be a list."
    node_data = get_node_data(G)
    order = order + [k for k in node_data.keys() if k not in order]
    y = {name: i for i, name in enumerate(order)}
    for node in G.nodes:
        d = G.nodes[node]
        G.nodes[node].update({"position": (d["position"][0], y[d["kind"]]), "order": y[d["kind"]]})


def set_node_colors(G: nx.DiGraph, cscheme: Dict[str, str]):
    """Set the colors of the nodes in the graph."""
    assert isinstance(cscheme, dict), "Color scheme must be a dict."
    node_data = get_node_data(G)
    cscheme = {**{k: "gray" for k in node_data.keys() if k not in cscheme}, **cscheme}
    ecolor, fcolor = oc.cscheme_fn(cscheme)
    for node in G.nodes:
        d = G.nodes[node]
        G.nodes[node].update({"color": cscheme[d["kind"]], "edgecolor": ecolor[d["kind"]], "facecolor": fcolor[d["kind"]]})


def get_node_colors(G: nx.DiGraph) -> Dict[str, str]:  # Tuple[Dict[str, str], Dict[str, str]]:
    """Get the colors of the nodes in the graph."""
    node_data = get_node_data(G)
    cscheme = {k: v["color"] for k, v in node_data.items()}
    return cscheme


def get_node_data(G: nx.DiGraph):
    """Get structural node data from graph."""
    node_data = {}
    for node in G.nodes:
        data = G.nodes[node]
        if ("pruned" not in data or not data["pruned"]) and data["kind"] not in node_data:
            node_data[data["kind"]] = {
                k: val
                for k, val in data.items()
                if k in ["kind", "inputs", "stateful", "color", "edgecolor", "facecolor", "alpha", "order"]
            }
    return deepcopy(node_data)


def create_graph(record: log_pb2.EpisodeRecord, excludes_inputs: List[str] = None) -> nx.DiGraph:
    # Create empty list if excludes_inputs is None
    excludes_inputs = excludes_inputs or []

    # Create graph
    G_full = nx.DiGraph()

    # Set default color scheme
    cscheme = {record_node.info.name: "gray" for record_node in record.node}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Determine order
    order = [record_node.info.name for record_node in record.node]

    # Layout properties
    y = {name: i for i, name in enumerate(order)}

    # Prepare data
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.0}
    node_data = {}
    for record_node in record.node:
        inputs = {i.output: {"input_name": i.name, "window": i.window} for i in record_node.info.inputs}
        inputs = inputs if record_node.info.name not in excludes_inputs else {}
        node_data[record_node.info.name] = {
            "kind": record_node.info.name,
            "inputs": inputs,
            "stateful": record_node.info.stateful,
            "order": y[record_node.info.name],
            "color": cscheme[record_node.info.name],
            "edgecolor": ecolor[record_node.info.name],
            "facecolor": fcolor[record_node.info.name],
            "alpha": 1.0,
        }

    # Get all nodes
    for record_node in record.node:
        # Add nodes
        for i_step, record_step in enumerate(record_node.steps):
            data = {
                "seq": record_step.tick,
                "ts_step": record_step.ts_step,
                "ts_sent": record_step.ts_output,
                "pruned": False,
                "super": False,
                "position": (record_step.ts_step, y[record_node.info.name]),
            }
            data.update(node_data[record_node.info.name])
            id = f'{data["kind"]}_{data["seq"]}'
            G_full.add_node(id, **data)

            # Add edge for stateful nodes
            if record_step.tick > 0:
                pruned = not record_node.info.stateful
                data = {
                    "kind": record_node.info.name,
                    "output": record_node.info.name,
                    "window": 1,
                    "seq": record_step.tick - 1,
                    "ts_sent": record_step.ts_output_prev,
                    "ts_recv": record_step.ts_output_prev,
                    "stateful": True,
                    "pruned": pruned,
                }
                data.update(**edge_data)
                if pruned:
                    data.update(**{"color": oc.ecolor.pruned, "linestyle": "--", "alpha": 0.5})
                id_source = f"{record_node.info.name}_{record_step.tick - 1}"
                id_target = f"{record_node.info.name}_{record_step.tick}"
                G_full.add_edge(id_source, id_target, **data)

        # Add edges
        for record_input in record_node.inputs:
            window = record_input.info.window
            edge_window = deque(maxlen=window)
            for i_step, (record_grouped, record_step) in enumerate(zip(record_input.grouped, record_node.steps)):
                for i_msg, record_msg in enumerate(reversed(record_grouped.messages)):
                    pruned = True if i_msg >= window or record_node.info.name in excludes_inputs else False

                    data = {
                        "kind": record_input.info.name,
                        "output": record_input.info.output,
                        "window": record_input.info.window,
                        "seq": record_msg.sent.seq,
                        "ts_sent": record_msg.sent.ts.sc,
                        "ts_recv": record_msg.received.ts.sc,
                        "stateful": False,
                        "pruned": pruned,
                    }
                    data.update(**edge_data)
                    if pruned:
                        data.update(**{"color": oc.ecolor.pruned, "linestyle": "--", "alpha": 0.5})
                    id_source = f"{record_input.info.output}_{record_msg.sent.seq}"
                    id_target = f"{record_node.info.name}_{record_step.tick}"
                    if pruned:
                        G_full.add_edge(id_source, id_target, **data)
                    else:
                        edge_window.append((id_source, data))

                # Add all messages in window as edge
                id_target = f"{record_node.info.name}_{record_step.tick}"
                for id_source, data in edge_window:
                    G_full.add_edge(id_source, id_target, **deepcopy(data))
    return G_full


def prune_graph(G: nx.DiGraph, copy: bool = True) -> nx.DiGraph:
    G = prune_nodes(G, copy=copy)
    G = prune_edges(G, copy=copy)
    return G


def prune_edges(G: nx.DiGraph, copy: bool = True) -> nx.DiGraph:
    G_pruned = G.copy(as_view=False) if copy else G
    remove_edges = [(u, v) for u, v, data in G_pruned.edges(data=True) if data["pruned"]]
    G_pruned.remove_edges_from(remove_edges)
    return G_pruned


def prune_nodes(G: nx.DiGraph, copy: bool = True) -> nx.DiGraph:
    G_pruned = G.copy(as_view=False) if copy else G
    remove_nodes = [n for n, data in G_pruned.nodes(data=True) if data["pruned"]]
    G_pruned.remove_nodes_from(remove_nodes)
    return G_pruned


def get_root_nodes(G: nx.DiGraph, root: str) -> Dict[str, Dict]:
    root_nodes = {n: data for n, data in G.nodes(data=True) if n.startswith(root)}
    return root_nodes


def trace_root(G: nx.DiGraph, root: str, seq: int) -> nx.DiGraph:
    node_data = get_node_data(G)
    assert node_data[root]["stateful"], f"Root node {root} must be stateful."

    # Get root nodes
    G_traced = G.copy(as_view=False)
    root_nodes = get_root_nodes(G_traced, root)  # {n: data for n, data in G_traced.nodes(data=True) if n.startswith(root)}

    # Trace
    seq = seq if seq >= 0 else len(root_nodes) + seq
    root_id = f"{root}_{seq}"
    [G_traced.nodes[n].update({"root": True}) for n in root_nodes.keys()]

    # Trace
    G_pruned_edge = prune_edges(G)  # Prune unused edges (due to e.g. windowing)
    ancestors = nx.ancestors(G_pruned_edge, root_id)
    pruned_nodes = [n for n in G_traced.nodes() if n not in ancestors and n != root_id]
    data_pruned = {"pruned": True, "alpha": 0.5, "edgecolor": oc.ecolor.pruned, "facecolor": oc.fcolor.pruned}
    [G_traced.nodes[n].update(data_pruned) for n in pruned_nodes]
    pruned_edges = [(u, v) for u, v in G_traced.edges() if u in pruned_nodes or v in pruned_nodes]
    data_pruned = {"pruned": True, "alpha": 0.5, "color": oc.ecolor.pruned, "linestyle": "--"}
    [G_traced.edges[u, v].update(data_pruned) for u, v in pruned_edges]
    return G_traced


def get_network_record(
    records: Union[log_pb2.EpisodeRecord, List[log_pb2.EpisodeRecord]],
    root: str,
    seq: int = None,
    S_init: nx.DiGraph = None,
    supergraph_mode: str = "MCS",
    cscheme: Dict[str, str] = None,
    order: List[str] = None,
    excludes_inputs: List[str] = None,
    progress_bar: bool = False,
    validate: bool = True,
    backtrack: int = 20,
) -> Tuple[log_pb2.NetworkRecord, nx.DiGraph, Dict[str, str], List[nx.DiGraph], List[Dict[str, Tuple[int, str]]]]:
    excludes_inputs = excludes_inputs or []

    # Convert to list of records
    records = records if isinstance(records, (list, RepeatedCompositeContainer)) else [records]

    # Set default cscheme and node ordering
    cscheme = cscheme or {}
    order = order or []

    # Assert that all episode records have the root
    root_records = [r for eps_record in records for r in eps_record.node if r.info.name == root]
    assert len(root_records) == len(records), "Not all episode records have the root node."

    # Assert that all records have at least seq number of root steps.
    num_seqs = [len(r.steps) for r in root_records]
    min_seqs = min(num_seqs)
    seq = seq if seq is not None and seq > 0 else min_seqs - 1
    assert min_seqs > seq, f"Not all episode records ('{num_seqs}') have at least seq={seq} number of root steps."

    # Get all graphs
    Gs = []
    Gs_raw = []
    for i, record in enumerate(records):
        G_raw = create_graph(record, excludes_inputs=excludes_inputs)
        Gs_raw.append(G_raw)

        # Set edge and node properties
        set_node_order(G_raw, order)
        set_node_colors(G_raw, cscheme)

        # Trace root node (not pruned yet)
        G_traced = trace_root(G_raw, root=root, seq=seq)

        # Prune unused nodes (not in computation graph of traced root)
        G_traced_pruned = prune_graph(G_traced)
        Gs.append(G_traced_pruned)

    if supergraph_mode == "MCS":
        # Run evaluation
        S, S_init_to_S, Gs_monomorphism = supergraph.grow_supergraph(
            Gs,
            root,
            S_init=S_init,
            combination_mode="linear",
            backtrack=backtrack,
            progress_fn=None,
            progress_bar=progress_bar,
            validate=validate,
        )
    elif supergraph_mode == "topological":
        from supergraph.evaluate import baselines_S

        S, _ = baselines_S(Gs, root)
        S_init_to_S = {n: n for n in S.nodes()}
        Gs_monomorphism = supergraph.evaluate_supergraph(Gs, S, progress_bar=progress_bar)
    elif supergraph_mode == "generational":
        from supergraph.evaluate import baselines_S

        _, S = baselines_S(Gs, root)
        S_init_to_S = {n: n for n in S.nodes()}
        Gs_monomorphism = supergraph.evaluate_supergraph(Gs, S, progress_bar=progress_bar)
    else:
        raise ValueError(f"Unknown supergraph mode '{supergraph_mode}'.")

    # Save traced network record
    record_network = log_pb2.NetworkRecord()
    record_network.episode.extend(records)
    record_network.root = root
    record_network.seq = seq
    record_network.supergraph_mode = supergraph_mode
    record_network.excludes_inputs.extend(excludes_inputs)
    record_network.S = pickle.dumps(S)
    return record_network, S, S_init_to_S, Gs_raw, Gs_monomorphism


def _get_timings_template(S: nx.DiGraph, num_root_steps: int) -> Timings:
    # Get supergraph timings template
    timings = []
    generations = list(nx.topological_generations(S))
    for gen in generations:
        t_gen = dict()
        timings.append(t_gen)
        for n in gen:
            data = S.nodes[n]
            inputs = {}
            for v in data["inputs"].values():
                inputs[v["input_name"]] = {
                    "seq": onp.vstack([onp.array([-1] * v["window"], dtype=onp.int32)] * num_root_steps),
                    "ts_sent": onp.vstack([onp.array([0.0] * v["window"], dtype=onp.float32)] * num_root_steps),
                    "ts_recv": onp.vstack([onp.array([0.0] * v["window"], dtype=onp.float32)] * num_root_steps),
                }
            t_slot = {
                "run": onp.repeat(False, num_root_steps),
                "ts_step": onp.repeat(0.0, num_root_steps),
                "seq": onp.repeat(0, num_root_steps),
                "inputs": inputs,
            }
            t_gen[n] = t_slot
    return timings


def get_timings(S: nx.DiGraph, G: nx.DiGraph, G_monomorphism: Dict[str, Tuple[int, str]], num_root_steps: int, root: str):
    # Get supergraph timings
    timings = _get_timings_template(S, num_root_steps)
    # Fill in timings for each mapped node
    for n_step, (i_step, n_mcs) in G_monomorphism.items():
        gen = S.nodes[n_mcs]["generation"]
        t_slot = timings[gen][n_mcs]
        ndata = G.nodes[n_step]
        t_slot["run"][i_step] = True
        t_slot["seq"][i_step] = ndata["seq"]
        t_slot["ts_step"][i_step] = ndata["ts_step"]

        # Sort input timings
        outputs = {k: [] for k, v in ndata["inputs"].items()}
        for u, v, edata in G.in_edges(n_step, data=True):
            u_kind = G.nodes[u]["kind"]
            _v_kind = G.nodes[v]["kind"]
            # if u_name == v_name or edata["pruned"]:
            if edata["stateful"] or edata["pruned"]:
                continue
            outputs[u_kind].append(edata)
        inputs = {v["input_name"]: outputs[k] for k, v in ndata["inputs"].items()}

        # Update input timings
        for input_name, input_edata in inputs.items():
            # Sort inputs by sequence number
            input_edata.sort(reverse=False, key=lambda x: x["seq"])
            seqs = [data["seq"] for data in input_edata]
            ts_sent = [data["ts_sent"] for data in input_edata]
            ts_recv = [data["ts_recv"] for data in input_edata]
            # TODO: VERIFY FOR WINDOW > 1 THAT IDX IS CORRECT
            idx = t_slot["inputs"][input_name]["seq"][i_step].shape[0] - len(seqs)
            t_slot["inputs"][input_name]["seq"][i_step][idx:] = seqs
            t_slot["inputs"][input_name]["ts_sent"][i_step][idx:] = ts_sent
            t_slot["inputs"][input_name]["ts_recv"][i_step][idx:] = ts_recv

    # # todo: {TEST} convert to old format
    # monomorphisms = {i: {} for i in range(num_root_steps)}
    # for n_step, (i_step, n_mcs) in G_monomorphism.items():
    #     monomorphisms[i_step][n_mcs] = n_step
    # # Get supergraph timings
    # timings_test = timings
    # timings = _get_timings_template(S, num_root_steps)
    # for i_step, (root_test, mcs) in enumerate(monomorphisms.items()):
    #     # Add root node to mapping (root is always the only node in the last generation)
    #     assert f"s{root}_0" in S, "Root node not found in S."
    #     # Update timings of step nodes
    #     for n_MCS, n_step in mcs.items():
    #         gen = S.nodes[n_MCS]["generation"]
    #         t_slot = timings[gen][n_MCS]
    #         ndata = G.nodes[n_step]
    #         t_slot["run"][i_step] = True
    #         t_slot["seq"][i_step] = ndata["seq"]
    #         t_slot["ts_step"][i_step] = ndata["ts_step"]
    #
    #         # Sort input timings
    #         outputs = {k: [] for k, v in ndata["inputs"].items()}
    #         inputs = {v["input_name"]: outputs[k] for k, v in ndata["inputs"].items()}
    #         for u, v, edata in G.in_edges(n_step, data=True):
    #             u_kind = G.nodes[u]["kind"]
    #             v_kind = G.nodes[v]["kind"]
    #             # if u_kind == v_kind or edata["pruned"]:
    #             if edata["stateful"] or edata["pruned"]:
    #                 continue
    #             outputs[u_kind].append(edata)
    #
    #         # Update input timings
    #         for input_name, input_edata in inputs.items():
    #             # Sort inputs by sequence number
    #             input_edata.sort(reverse=False, key=lambda x: x["seq"])
    #             seqs = [data["seq"] for data in input_edata]
    #             ts_sent = [data["ts_sent"] for data in input_edata]
    #             ts_recv = [data["ts_recv"] for data in input_edata]
    #             # TODO: VERIFY FOR WINDOW > 1 THAT IDX IS CORRECT
    #             idx = t_slot["inputs"][input_name]["seq"][i_step].shape[0] - len(seqs)
    #             t_slot["inputs"][input_name]["seq"][i_step][idx:] = seqs
    #             t_slot["inputs"][input_name]["ts_sent"][i_step][idx:] = ts_sent
    #             t_slot["inputs"][input_name]["ts_recv"][i_step][idx:] = ts_recv
    # for t1, t2 in zip(timings, timings_test):
    #     for (s1, ts1), (s2, ts2) in zip(t1.items(), t2.items()):
    #         assert all(ts1["run"] == ts2["run"]), f"Run mismatch in step {s1}."
    #         assert all(ts1["seq"] == ts2["seq"]), f"Seq mismatch in step {s1}."
    #         assert all(ts1["ts_step"] == ts2["ts_step"]), f"ts_step mismatch in step {s1}."
    #         for (i1, ti1), (i2, ti2) in zip(ts1["inputs"].items(), ts2["inputs"].items()):
    #             assert (ti1["seq"] == ti2["seq"]).all(), f"Seq mismatch in step {s1}, input {i1}."
    #             assert (ti1["ts_sent"] == ti2["ts_sent"]).all(), f"ts_sent mismatch in step {s1}, input {i1}."
    #             assert (ti1["ts_recv"] == ti2["ts_recv"]).all(), f"ts_recv mismatch in step {s1}, input {i1}."

    return timings


def get_timings_from_network_record(
    network_record: log_pb2.NetworkRecord,
    Gs: List[nx.DiGraph] = None,
    Gs_monomorphism: List[Dict[str, Tuple[int, str]]] = None,
    progress_bar: bool = False,
) -> Timings:
    assert Gs is None or len(Gs) == len(network_record.episode), "Number of graphs does not match number of steps."
    assert Gs_monomorphism is None or len(Gs_monomorphism) == len(
        network_record.episode
    ), "Number of monomorphisms does not match number of episodes."

    # Prepare graphs
    Gs = Gs or [None] * len(network_record.episode)
    Gs_monomorphism = Gs_monomorphism or [None] * len(network_record.episode)

    # Prepare supergraph
    S = pickle.loads(network_record.S)

    # Get all monomorphisms
    for i, (record, G, G_monomorphism) in enumerate(zip(network_record.episode, Gs, Gs_monomorphism)):
        # Create graph if not provided
        if G is None:
            G = create_graph(record)
            Gs[i] = G

        # Get subgraphs if not provided
        if G_monomorphism is None:
            # Trace root node
            G_traced = trace_root(G, root=network_record.root, seq=network_record.seq)

            # Prune unused nodes (not in computation graph of traced root)
            G_traced_pruned = prune_graph(G_traced)

            # Get monomorphisms
            Gs_monomorphism[i] = supergraph.evaluate_supergraph([G_traced_pruned], S, progress_bar=progress_bar)[0]

    # Get monomorphisms
    timings = []
    for i, (G, G_monomorphism) in enumerate(zip(Gs, Gs_monomorphism)):
        t = get_timings(S, G, G_monomorphism, num_root_steps=network_record.seq + 1, root=network_record.root)
        timings.append(t)

    # Stack timings
    timings = jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *timings)
    return timings


def get_outputs_from_timings(
    S: nx.DiGraph, timings: Timings, nodes: Dict[str, "Node"], extra_padding: int = 0
) -> Dict[str, Output]:
    """Get output buffer from timings."""
    # get seq state
    timings = get_timings_after_root_split(S, timings)

    # Get output buffer sizes (+1, to add default output)
    num_outputs = {k: v["seq"].max() for k, v in timings.items()}
    buffer_size = {k: v + 1 + extra_padding for k, v in num_outputs.items()}

    # Fill output buffer
    output_buffer = {}
    stack_fn = lambda *x: onp.stack(x, axis=0)
    rng = jax.random.PRNGKey(0)
    for node, size in buffer_size.items():
        assert node in nodes, f"Node `{node}` not found in nodes."
        step_buffer = jax.tree_util.tree_map(stack_fn, *[nodes[node].default_output(rng)] * size)
        eps_buffer = jax.tree_util.tree_map(stack_fn, *[step_buffer] * timings[node]["seq"].shape[0])
        output_buffer[node] = eps_buffer
    return output_buffer


def get_timings_after_root_split(S: nx.DiGraph, timings: Timings):
    """Get every node's latest sequence number at every root step."""
    # Flatten timings
    timings_flat = {slot: t for gen in timings for slot, t in gen.items()}

    # Get node names
    node_kinds = set([data["kind"] for n, data in S.nodes(data=True)])

    # Sort slots
    slots = {k: [] for k in node_kinds}
    [slots[data["kind"]].append(timings_flat[n]) for n, data in S.nodes(data=True)]
    slots = {k: jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *v) for k, v in slots.items()}

    # Get seq state
    timings = {}
    for name, t in slots.items():
        max_seq = onp.maximum.accumulate(onp.amax(t["seq"], axis=0), axis=1)
        max_ts_step = onp.maximum.accumulate(onp.amax(t["ts_step"], axis=0), axis=1)
        timings[name] = dict(seq=max_seq, ts_step=max_ts_step)
    return timings


def get_chronological_timings(S: nx.DiGraph, timings: Timings, eps: int) -> NodeTimings:
    # Take only one episode
    timings = jax.tree_util.tree_map(lambda x: x[eps], timings)

    # Flatten timings
    timings_flat = {slot: t for gen in timings for slot, t in gen.items()}

    # Get node names
    node_kinds = set([data["kind"] for n, data in S.nodes(data=True)])

    # Sort slots
    slots = {k: [] for k in node_kinds}
    [slots[data["kind"]].append(timings_flat[n]) for n, data in S.nodes(data=True)]
    slots = {k: jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *v) for k, v in slots.items()}

    # Only keep timings with run=True, sort by seq
    slots_run = {k: jax.tree_util.tree_map(lambda _arr: _arr[v["run"]], v) for k, v in slots.items()}
    sort = {k: onp.argsort(v["seq"]) for k, v in slots_run.items()}
    slots_chron = {k: jax.tree_util.tree_map(lambda _arr: _arr[sort[k]], v) for k, v in slots_run.items()}
    return slots_chron


def get_masked_timings(S: nx.DiGraph, timings: Timings) -> NodeTimings:
    # generations = list(nx.topological_generations(S))

    # Get node names
    node_kinds = set([data["kind"] for n, data in S.nodes(data=True)])

    # Get node data
    slot_node_data = {n: data for n, data in S.nodes(data=True)}
    node_data = {}
    [node_data.update({d["kind"]: d}) for slot, d in slot_node_data.items() if d["kind"] not in node_data]

    # Get output buffer sizes
    masked_timings_slot = []
    for i_gen, gen in enumerate(timings):
        t_flat = {slot: t for slot, t in gen.items()}
        slots = {k: [] for k in node_kinds}
        [slots[S.nodes[n]["kind"]].append(t_flat[n]) for n in gen]
        [slots.pop(k) for k in list(slots.keys()) if len(slots[k]) == 0]
        # slots:= [eps, step, slot_idx, window=optional]
        slots = {k: jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=2), *v) for k, v in slots.items()}

        def _mask(mask, arr):
            # Repeat mask in extra dimensions of arr (for inputs)
            if arr.ndim > mask.ndim:
                extra_dim = tuple([mask.ndim + a for a in range(arr.ndim - mask.ndim)])
                new_mask = onp.expand_dims(mask, axis=extra_dim)
                for i in extra_dim:
                    new_mask = onp.repeat(new_mask, arr.shape[i], axis=-1)
            else:
                new_mask = mask
            # print(mask.shape, arr.shape, new_mask.shape)
            masked_arr = ma.masked_array(arr, mask=new_mask)
            return masked_arr

        masked_slots = {k: jax.tree_util.tree_map(partial(_mask, ~v["run"]), v) for k, v in slots.items()}
        masked_timings_slot.append(masked_slots)

    def _update_mask(j, arr):
        arr.mask[:, :, :, j] = True
        return arr

    def _concat_arr(a, b):
        return ma.concatenate((a, b), axis=2)

    # Combine timings for each slot. masked_timings := [eps, step, slot_idx, gen_idx, window=optional]
    masked_timings = {}
    for i_gen, gen in enumerate(masked_timings_slot):
        for key, t in gen.items():
            # Repeat mask in extra dimensions of arr (for number of gens, and mask all but the current i_gen)
            t = {
                k: jax.tree_util.tree_map(lambda x: onp.repeat(x[:, :, :, None], len(timings), axis=3), v)
                for k, v in t.items()
            }

            # Update mask to be True for all other gens
            for j in range(len(timings)):
                if j == i_gen:
                    continue
                jax.tree_util.tree_map(partial(_update_mask, j), t)

            # Add to masked_timings
            if key not in masked_timings:
                # Add as new entry
                masked_timings[key] = t
            else:
                # Concatenate with existing entry
                masked_timings[key] = jax.tree_util.tree_map(_concat_arr, masked_timings[key], t)
    return masked_timings


def get_buffer_sizes_from_timings(S: nx.DiGraph, timings: Timings) -> BufferSizes:
    # Get masked timings:= [eps, step, slot_idx, gen_idx, window=optional]
    masked_timings = get_masked_timings(S, timings)

    # Get node data
    slot_node_data = {n: data for n, data in S.nodes(data=True)}
    node_data = {}
    [node_data.update({d["kind"]: d}) for slot, d in slot_node_data.items() if d["kind"] not in node_data]

    # Get min buffer size for each node
    name_mapping = {n: {v["input_name"]: o for o, v in data["inputs"].items()} for n, data in node_data.items()}
    min_buffer_sizes = {
        k: {input_name: output_name for input_name, output_name in inputs.items()} for k, inputs in name_mapping.items()
    }
    node_buffer_sizes = {n: [] for n in node_data.keys()}
    for n, inputs in name_mapping.items():
        t = masked_timings[n]
        for input_name, output_name in inputs.items():
            # Determine min input sequence per generation (i.e. we reduce over all slots within a generation & window)
            seq_in = onp.amin(t["inputs"][input_name]["seq"], axis=(2, 4))
            seq_in = seq_in.reshape(*seq_in.shape[:-2], -1)   # flatten over generation & step dimension (i.e. [s1g1, s1g2, ..], [s2g1, s2g2, ..], ..)
            # NOTE: fill masked steps with max value (to not influence buffer size)
            ma.set_fill_value(seq_in, onp.iinfo(onp.int32).max)  # Fill with max value, because it will not influence the min
            filled_seq_in = seq_in.filled()
            max_seq_in = onp.minimum.accumulate(filled_seq_in[:, ::-1], axis=-1)[:, ::-1]

            # Determine max output sequence per generation
            seq_out = onp.amax(masked_timings[output_name]["seq"], axis=(2,))  #  (i.e. we reduce over all slots within a generation)
            seq_out = seq_out.reshape(*seq_out.shape[:-2], -1)  # flatten over generation & step dimension (i.e. [s1g1, s1g2, ..], [s2g1, s2g2, ..], ..)
            ma.set_fill_value(seq_out, onp.iinfo(onp.int32).min)  # todo: CHECK! changed from -1 to onp.iinfo(onp.int32).min to deal with negative seq numbers
            filled_seq_out = seq_out.filled()
            max_seq_out = onp.maximum.accumulate(filled_seq_out, axis=-1)

            # Calculate difference to determine buffer size
            # NOTE: Offset output sequence by +1, because the output is written to the buffer AFTER the buffer is read
            offset_max_seq_out = onp.roll(max_seq_out, shift=1, axis=1)
            offset_max_seq_out[:, 0] = onp.iinfo(onp.int32).min  # todo: CHANGED to min value compared to --> NOTE: First step is always -1, because no node has run at this point.
            s = offset_max_seq_out - max_seq_in

            # NOTE! +1, because, for example, when offset_max_seq_out = 0, and max_seq_in = 0, we need to buffer 1 step.
            max_s = s.max() + 1

            # Store min buffer size
            min_buffer_sizes[n][input_name] = max_s
            node_buffer_sizes[output_name].append(max_s)

    return node_buffer_sizes


def get_graph_buffer(
    S: nx.DiGraph, timings: Timings, nodes: Dict[str, "Node"], sizes: BufferSizes = None, extra_padding: int = 0, graph_state: GraphState = None,
) -> GraphBuffer:
    if graph_state is None:
        deprecation_warning("graph_state should be provided per default.", stacklevel=2)

    # Get buffer sizes if not provided
    if sizes is None:
        sizes = get_buffer_sizes_from_timings(S, timings)

    # Create output buffers
    buffers = {}
    stack_fn = lambda *x: jnp.stack(x, axis=0)
    rng = jax.random.PRNGKey(0)
    for n, s in sizes.items():
        assert n in nodes, f"Node `{n}` not found in nodes."
        buffer_size = max(s) + extra_padding if len(s) > 0 else max(1, extra_padding)
        assert buffer_size > 0, f"Buffer size for node `{n}` is 0."
        b = jax.tree_util.tree_map(stack_fn, *[nodes[n].default_output(rng, graph_state=graph_state)] * buffer_size)
        buffers[n] = b
    return FrozenDict(buffers)


def get_seqs_mapping(S: nx.DiGraph, timings: Timings, buffer: GraphBuffer) -> Tuple[SeqsMapping, SeqsMapping]:
    # generations = list(nx.topological_generations(S))

    def _get_buffer_size(b):
        leaves = jax.tree_util.tree_leaves(b)
        size = leaves[0].shape[0] if len(leaves) > 0 else 1
        return size

    # Get buffer sizes
    buffer_sizes = {n: _get_buffer_size(b) for n, b in buffer.items()}

    # Get masked timings:= [eps, step, slot_idx, gen_idx, window=optional]
    masked_timings = get_masked_timings(S, timings)

    # Determine absolute sequence numbers in buffer
    # seqs:=[eps, step, slot_idx, gen_idx, seq]
    # updated:=[eps, step, slot_idx, gen_idx, updated]
    seqs = {}
    updated = {}
    for n, t in masked_timings.items():
        # Take max over slots in same generation
        seq_out = onp.amax(t["seq"], axis=(2,))
        # Record shape of seq_out [num_eps, num_steps*num_MCS_gens]
        shape_seq_out = seq_out.shape
        # Reshape to [num_eps, num_steps, num_MCS_gens]
        seq_out = seq_out.reshape(*shape_seq_out[:-2], -1)
        ma.set_fill_value(seq_out, -1)  # todo: effect of fill value?
        filled_seq_out = seq_out.filled()
        # Get max executed seq per generation
        max_seq_out = onp.maximum.accumulate(filled_seq_out, axis=-1)
        # Create buffers for seqs, and sort based on modulo with buffer size
        # NOTE: min buffer_seq = -1 here
        buffer_seqs = onp.stack([onp.maximum(-1, max_seq_out - s) for s in range(buffer_sizes[n])], axis=-1)
        idx_seqs = onp.argsort(buffer_seqs % buffer_sizes[n], axis=-1)
        sorted_seqs = onp.take_along_axis(buffer_seqs, idx_seqs, axis=-1)

        # NOTE! updated seqs = True if seq is updated AFTER this step is executed
        updated_seqs = sorted_seqs != onp.roll(sorted_seqs, shift=1, axis=1)  # Roll in gen axis
        updated_seqs[:, 0, :] = False  # First step is never updated

        # Reshape to step shape
        sorted_seqs = sorted_seqs.reshape(*shape_seq_out, buffer_sizes[n])
        updated_seqs = updated_seqs.reshape(*shape_seq_out, buffer_sizes[n])

        # Store
        seqs[n] = sorted_seqs
        updated[n] = updated_seqs

    return seqs, updated


def get_step_seqs_mapping(S: nx.DiGraph, timings: Timings, buffer: GraphBuffer) -> Tuple[SeqsMapping, SeqsMapping]:
    # Get update mask
    seqs, updated = get_seqs_mapping(S, timings, buffer)

    # Get updated seqs in buffer AFTER step is executed
    updated_step = {n: onp.any(v[:, :, :, :], axis=2) for n, v in updated.items()}

    # Get absolute seqs in buffer BEFORE step is executed
    after_seqs_step = {n: v[:, :, -1, :] for n, v in seqs.items()}
    init_seqs_step = {n: onp.full(arr[:, [0], :].shape, fill_value=-1) for n, arr in after_seqs_step.items()}
    before_seqs_step = {n: onp.concatenate([init_seqs_step[n], arr], axis=1) for n, arr in after_seqs_step.items()}

    return before_seqs_step, updated_step


def check_generations_uniformity(S: nx.DiGraph, generations: List[List]):
    """
    Checks if all generations have the same kinds of nodes and the same number of instances of each kind.

    :param generations: A list of generations, where each generation is a set of node IDs.
    :return: True if all generations are uniform in terms of node kinds and their counts, False otherwise.
    """

    # Dictionary to store the kind count of the first generation
    first_gen_kind_count = None

    for gen in generations:
        gen_kind_count = dict()
        for node_id in gen:
            kind = S.nodes[node_id]["kind"]
            gen_kind_count[kind] = gen_kind_count.get(kind, 0) + 1

        if first_gen_kind_count is None:
            first_gen_kind_count = gen_kind_count
        else:
            if gen_kind_count != first_gen_kind_count:
                return False

    return True
