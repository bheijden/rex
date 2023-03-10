import jax
import jumpy
import numpy as onp
import jumpy.numpy as jp
import dill as pickle
from typing import Tuple, List, Dict, Union
from copy import deepcopy
from collections import deque

import networkx as nx
from networkx.algorithms import isomorphism

from rex.constants import INFO
import rex.open_colors as oc
from rex.node import Node
from rex.proto import log_pb2
from rex.mcs import Deque, QueuePolicy, find_largest_motifs
from rex.utils import timer, log
from rex.base import Timings, Output


def node_match(n1, n2):
    """A function that returns True iff node n1 in G1 and n2 in G2
       should be considered equal during the isomorphism test. The
       function will be called like::

          node_match(G1.nodes[n1], G2.nodes[n2])

          G1: Supergraph
          G2: Motif

       That is, the function will receive the node attribute dictionaries
       of the nodes under consideration. If None, then no attributes are
       considered when testing for an isomorphism."""
    # todo: Test whether "sub_longest_path_length" is needed (maybe, vf2 already does this).
    return n1["name"] == n2["name"] and n1["sub_longest_path_length"] >= n2["sub_longest_path_length"]


def edge_match(e1, e2):
        """If G2 (motif) has an edge, G1 (supergraph) must have an edge.

           A function that returns True iff the edge attribute dictionary for
           the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be
           considered equal during the isomorphism test. The function will be
           called like::

              edge_match(G1[u1][v1], G2[u2][v2])

              G1: Supergraph
              G2: Motif

           That is, the function will receive the edge attribute dictionaries
           of the edges under consideration. If None, then no attributes are
           considered when testing for an isomorphism."""
        return True


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
        G.nodes[node].update({"position": (d["position"][0], y[d["name"]]), "order": y[d["name"]]})


def set_node_colors(G: nx.DiGraph, cscheme: Dict[str, str]):
    """Set the colors of the nodes in the graph."""
    assert isinstance(cscheme, dict), "Color scheme must be a dict."
    node_data = get_node_data(G)
    cscheme = {**{k: "gray" for k in node_data.keys() if k not in cscheme}, **cscheme}
    ecolor, fcolor = oc.cscheme_fn(cscheme)
    for node in G.nodes:
        d = G.nodes[node]
        G.nodes[node].update({"color": cscheme[d["name"]], "edgecolor": ecolor[d["name"]], "facecolor": fcolor[d["name"]]})


def get_node_colors(G: nx.DiGraph) -> Dict[str, str]: #Tuple[Dict[str, str], Dict[str, str]]:
    """Get the colors of the nodes in the graph."""
    node_data = get_node_data(G)
    cscheme = {k: v["color"] for k, v in node_data.items()}
    return cscheme
    # ecolor = {k: v["edgecolor"] for k, v in node_data.items()}
    # fcolor = {k: v["facecolor"] for k, v in node_data.items()}
    # return ecolor, fcolor


def get_node_data(G: nx.DiGraph):
    """Get structural node data from graph."""
    node_data = {}
    for node in G.nodes:
        data = G.nodes[node]
        if ("pruned" not in data or not data["pruned"]) and data["name"] not in node_data:
            node_data[data["name"]] = {k: val for k, val in data.items() if k in ["name", "inputs", "stateful", "color", "edgecolor", "facecolor", "alpha", "order"]}
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
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
    node_data = {}
    for record_node in record.node:
        inputs = {i.output: {"input_name": i.name, "window": i.window} for i in record_node.info.inputs}
        inputs = inputs if record_node.info.name not in excludes_inputs else {}
        node_data[record_node.info.name] = {"name": record_node.info.name,
                                            "inputs": inputs,
                                            "stateful": record_node.info.stateful,
                                            "order": y[record_node.info.name],
                                            "color": cscheme[record_node.info.name],
                                            "edgecolor": ecolor[record_node.info.name],
                                            "facecolor": fcolor[record_node.info.name],
                                            "alpha": 1.}

    # Get all nodes
    for record_node in record.node:
        # Add nodes
        for i_step, record_step in enumerate(record_node.steps):
            data = {
                "seq": record_step.tick,
                "ts_step": record_step.ts_step,
                "ts_sent": record_step.ts_output,
                "pruned": False,
                "root": False,
                "root_index": None,
                "generation": None,
                "sub_generation": None,
                "sub_longest_path_length": None,
                "super": False,
                "position": (record_step.ts_step, y[record_node.info.name])}
            data.update(node_data[record_node.info.name])
            id = f'{data["name"]}_{data["seq"]}'
            G_full.add_node(id, **data)

            # Add edge for stateful nodes
            if record_step.tick > 0:
                pruned = not record_node.info.stateful
                data = {"name": record_node.info.name,
                        "output": record_node.info.name,
                        "window": 1,
                        "seq": record_step.tick - 1,
                        "ts_sent": record_step.ts_output_prev,
                        "ts_recv": record_step.ts_output_prev,
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

                    data = {"name": record_input.info.name,
                            "output": record_input.info.output,
                            "window": record_input.info.window,
                            "seq": record_msg.sent.seq,
                            "ts_sent": record_msg.sent.ts.sc,
                            "ts_recv": record_msg.received.ts.sc,
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


def prune_graph(G: nx.DiGraph) -> nx.DiGraph:
    G = prune_nodes(G)
    G = prune_edges(G)
    return G


def prune_edges(G: nx.DiGraph) -> nx.DiGraph:
    G_pruned = G.copy(as_view=False)
    remove_edges = [(u, v) for u, v, data in G_pruned.edges(data=True) if data['pruned']]
    G_pruned.remove_edges_from(remove_edges)
    return G_pruned


def prune_nodes(G: nx.DiGraph) -> nx.DiGraph:
    G_pruned = G.copy(as_view=False)
    remove_nodes = [n for n, data in G_pruned.nodes(data=True) if data['pruned']]
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


def get_subgraphs(G: nx.DiGraph, split_mode: str = "generational") -> Dict[str, nx.DiGraph]:
    G = G.copy(as_view=False)
    if split_mode not in ["generational"]:
        raise NotImplementedError(f"Split mode {split_mode} not supported.")

    # Define generations
    generations = list(nx.topological_generations(G))
    for i_gen, gen in enumerate(generations):
        for n in gen:
            G.nodes[n].update({"generation": i_gen})

    # Define subgraphs:= bwd_subgraphs = {root_seq: nx.ancestors(G_traced_pruned, root_seq)}
    bwd_subgraphs = {}  # Places the root in an isolated generation *before* the generation it was originally in.
    new_subgraph = []
    for gen in generations:
        is_root = False
        gen_has_root = False
        for n in gen:
            assert not (is_root and G.nodes[n][
                "root"]), "Multiple roots in a generation. Make sure the root node is stateful."
            is_root = G.nodes[n]["root"]
            gen_has_root = is_root or gen_has_root

        if gen_has_root:
            root = [n for n in gen if G.nodes[n]["root"]][0]
            data_root = {"root_index": len(bwd_subgraphs), "root": root}
            G.nodes[root].update(data_root)
            [G.nodes[n].update(data_root) for gen in new_subgraph for n in gen]
            bwd_subgraphs[root] = new_subgraph
            next_gen = [n for n in gen if not G.nodes[n]["root"]]
            new_subgraph = [next_gen]
        else:
            new_subgraph.append(gen)

    # Create subgraphs
    G_subgraphs = dict()
    for i_root, (root, subgraph) in enumerate(bwd_subgraphs.items()):
        nodes = [n for gen in subgraph for n in gen]
        G_sub = G.subgraph(nodes).copy(as_view=False)
        sub_generations = list(nx.topological_generations(G_sub))
        # Calculate longest path
        for n in G_sub.nodes():
            desc = nx.descendants(G_sub, n)
            desc.add(n)
            G_desc = G_sub.subgraph(desc).copy(as_view=True)
            longest_path_length = nx.dag_longest_path_length(G_desc)
            # longest_path = nx.dag_longest_path(G_desc)
            G_sub.nodes[n].update({"sub_longest_path_length": longest_path_length})
        # Calculate sub generations
        for i_sub_gen, sub_gen in enumerate(sub_generations):
            for n in sub_gen:
                G_sub.nodes[n].update({"sub_generation": i_sub_gen})
        G_subgraphs[root] = G_sub
    return G_subgraphs


def get_unique_motifs(G_subgraphs: Dict[str, nx.DiGraph], validate: bool = True) -> Dict[str, nx.DiGraph]:
    G_motifs = {}
    for root_test, G_test in G_subgraphs.items():
        G_test: nx.DiGraph
        is_unique = True
        for root_unique in list(G_motifs.keys()):
            G_unique: nx.DiGraph = G_motifs.pop(root_unique)

            # Determine super vs motif graph
            n_nodes_test = G_test.number_of_nodes()
            n_nodes_unique = G_unique.number_of_nodes()
            G_super = G_test if n_nodes_test > n_nodes_unique else G_unique
            root_super = root_test if n_nodes_test > n_nodes_unique else root_unique
            G_motif = G_test if n_nodes_test <= n_nodes_unique else G_unique

            # Define matcher
            matcher = isomorphism.DiGraphMatcher(G_super, G_motif, node_match=node_match, edge_match=edge_match)

            if matcher.subgraph_is_monomorphic():
                G_motifs[root_super] = G_super
                is_unique = False
            # break # Do not break, but pop
            else:
                G_motifs[root_unique] = G_unique

        if is_unique:
            G_motifs[root_test] = G_test

    if validate:
        # Verify that all unique subgraphs are not monomorphic with any other unique subgraph
        for root_motif, G_motif in G_motifs.items():
            for root_motif_other, G_motif_other in G_motifs.items():
                if root_motif == root_motif_other:
                    continue
                n_nodes_motif = G_motif.number_of_nodes()
                n_nodes_motif_other = G_motif_other.number_of_nodes()
                G_large = G_motif if n_nodes_motif > n_nodes_motif_other else G_motif_other
                G_small = G_motif if n_nodes_motif <= n_nodes_motif_other else G_motif_other
                matcher = isomorphism.DiGraphMatcher(G_large, G_small, node_match=node_match, edge_match=edge_match)
                assert not matcher.subgraph_is_monomorphic(), "Identified unique subgraphs are monomorphic."

        # Verify that all subgraphs are monomorphic with one of the identified unique subgraphs
        for root_test, G_test in G_subgraphs.items():
            is_unique = True
            for root_motif, G_motif in G_motifs.items():
                matcher = isomorphism.DiGraphMatcher(G_motif, G_test, node_match=node_match, edge_match=edge_match)
                if matcher.subgraph_is_monomorphic():
                    is_unique = False
                    break
            assert not is_unique, "Subgraph is unique, but not included into G_unique."

    return G_motifs


def as_MCS(G: nx.DiGraph) -> nx.DiGraph:
    # todo: as_MCS should only have edges between nodes in the partite groups that exist in the unique patterns.
    node_data = get_node_data(G)
    order = sorted(node_data.keys(), key=lambda k: node_data[k]["order"])
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
    generations = list(nx.topological_generations(G))

    # Sort all nodes in generation
    [g.sort(key=lambda k: order.index(G.nodes[k]["name"])) for g in generations]

    # Enforce sequential order
    nodes_super = [{} for _ in generations]
    G_super = nx.DiGraph()
    slots = {n: 0 for n in order}
    for i_gen, gen in enumerate(generations):
        sub_longest_path_length = len(generations) - i_gen
        for i_node, n in enumerate(gen):
            # Determine slot & name
            name = G.nodes[n]["name"]
            n_name = f"{name}_s{slots[name]}"
            # Add node
            position = (i_gen, i_node)
            data = {"seq": slots[name], "position": position,
                    "sub_longest_path_length": sub_longest_path_length,
                    "sub_generation": i_gen,
                    "generation": i_gen,
                    "super": True}
            data.update(node_data[name])
            nodes_super[i_gen][n_name] = data
            G_super.add_node(n_name, **data)
            # Increase slot number
            slots[name] += 1

    # Add multi partite directed edges between all nodes of G_super directed in descending generation order
    for i_gen, u in enumerate(nodes_super):
        for j_gen, v in enumerate(nodes_super[i_gen + 1:]):
            for u_name in u:
                for v_name in v:
                    # todo: must ensure that root is last in generation before filtering edges here.
                    # is_same = G_super.nodes[u_name]["name"] == G_super.nodes[v_name]["name"]
                    # is_input = G_super.nodes[u_name]["name"] in G_super.nodes[v_name]["inputs"]
                    # if not(is_same or is_input):
                    # 	continue
                    G_super.add_edge(u_name, v_name, **edge_data)

    return G_super


def emb(G1, G2):
    # G1 is a subgraph of G2
    E = [e for e in G1.edges
         if e[0] in G1 and e[1] in G2
         or e[0] in G2 and e[1] in G1]
    return E


def unify(G1, G2, E):
    # E is the edge embedding of G1 in G2
    # G1 is unified with G2 by adding the edges in E
    G = G1.copy(as_view=False)
    G.add_nodes_from(G2.nodes(data=True))
    G.add_edges_from(G2.edges(data=True))
    G.add_edges_from(E)
    return G


def difference(G1, G2):
    # G1 is a subgraph of G2
    G = G2.copy(as_view=False)
    G.remove_nodes_from(G1.nodes)
    return G


def _get_MCS(G1, G2, max_evals: int = None):
    # todo: check if we cannot use isomorphism.vf2. Seems alot faster.
    queue = Deque(policy=QueuePolicy.BREADTHFIRST)
    num_evals, is_monomorphism, largest_motif = find_largest_motifs(G2, G1,
                                                                    queue_=queue,
                                                                    max_evals=max_evals,
                                                                    # interestingness={"observer_6": 1.0, "actuator_0": 0.9},
                                                                    # is_node_structural_match=_is_node_structural_match,
                                                                    # is_node_attr_match=_is_node_attr_match,
                                                                    # is_edge_attr_match=_is_edge_attr_match
                                                                    )
    # Do not modify G1 if it is a monomorphism
    if is_monomorphism:
        return num_evals, G1
    # Else, prepare to unify G1 and G2
    mcs = {v: k for k, v in largest_motif[0].items()}
    mcs_G1 = G1.subgraph(mcs.keys())
    mcs_G2 = nx.relabel_nodes(G1.subgraph(mcs.keys()), mcs, copy=True)
    E1 = emb(mcs_G1, G1)
    E2 = emb(mcs_G2, G2)
    _MCS = nx.relabel_nodes(unify(mcs_G2, G2, E2), {v: k for k, v in mcs.items()}, copy=True)
    MCS = unify(_MCS, G1, E1)
    return num_evals, as_MCS(MCS)


def get_minimum_common_supergraph(G_motifs: Dict[str, nx.DiGraph], max_evals_per_motif: int = None, max_total_evals: int = 100_000):
    """
    Find the minimum common (monomorphic) supergraph (MCS) of a list of graphs.

    Arguments:
        G_motifs (List[nx.DiGraph]): The list of graphs
        max_eval_per_motif (int): The maximum number of evaluations per motif

    Returns:
        nx.DiGraph: The minimum common supergraph (MCS)
    """

    # Select the largest motif as the initial supergraph
    G_motifs_max = G_motifs.get(max(G_motifs, key=lambda x: G_motifs[x].number_of_nodes()))

    # Convert to supergarph (partite DAG)
    G_MCS = as_MCS(G_motifs_max.copy(as_view=False))

    # Iterate over all motifs
    num_evals = 0
    for i_motif, (root_name, G_motif) in enumerate(G_motifs.items()):
        if G_motif is G_motifs_max:
            continue
        max_evals_per_motif = min(max_evals_per_motif,
                                  max_total_evals - num_evals) if max_evals_per_motif is not None else max_total_evals - num_evals
        prev_num_nodes = G_MCS.number_of_nodes()
        # Find the MCS
        num_evals_motif, G_MCS = _get_MCS(G_MCS, G_motif, max_evals=max_evals_per_motif)
        # Update the total number of evaluations
        num_evals += num_evals_motif
        next_num_nodes = G_MCS.number_of_nodes()
        if num_evals >= max_total_evals:
            print(
                f"max_total_evals={max_total_evals} exceeded. Stopping. MCS covers {i_motif + 1}/{len(G_motifs)} motifs. May be excessively large.")
            break
        print(
            f"motif=`{root_name}` | nodes+={next_num_nodes - prev_num_nodes} | num_nodes={next_num_nodes} | motif_evals={num_evals_motif} | total_evals={num_evals}")
    return G_MCS


def _add_root_to_supergraph(G_MCS: nx.DiGraph, root_data: Dict) -> nx.DiGraph:
    G_with_root = G_MCS.copy(as_view=False)

    # Add root node slot as the last generation
    has_root = any([True if data["name"] == root_data["name"] else False for n, data in G_with_root.nodes(data=True)])
    assert not has_root, f"Root node already exists in graph: {root_data['name']}."
    *_, last_gen = nx.topological_generations(G_with_root)
    G_with_root.add_node(root_data["name"], **root_data)

    # Add edges between root node and last generation
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
    for node in last_gen:
        G_with_root.add_edge(node, root_data["name"], **edge_data)

    return as_MCS(G_with_root)


def validate_subgraphs(G_MCS: nx.DiGraph, G_subgraphs: Dict[str, nx.DiGraph]) -> Dict[str, bool]:
    # Test all subgraphs for monomorphism with the supergraph
    is_monomorphic = {}
    for root_test, G_test in G_subgraphs.items():
        matcher = isomorphism.DiGraphMatcher(G_MCS, G_test, node_match=node_match, edge_match=edge_match)
        is_monomorphic[root_test] = matcher.subgraph_is_monomorphic()
    return is_monomorphic


def get_subgraph_monomorphisms(G_MCS: nx.DiGraph, G_subgraphs: Dict[str, nx.DiGraph]) -> Dict[str, bool]:
    # Get all subgraph monomorphisms with the supergraph
    monomorphisms = {}
    for root_test, G_test in G_subgraphs.items():
        matcher = isomorphism.DiGraphMatcher(G_MCS, G_test, node_match=node_match, edge_match=edge_match)
        mcs = next(matcher.subgraph_monomorphisms_iter())
        monomorphisms[root_test] = mcs
    return monomorphisms


def get_network_record(records: Union[log_pb2.EpisodeRecord, List[log_pb2.EpisodeRecord]], root: str, seq: int = None, split_mode: str = "generational",
                       cscheme: Dict[str, str] = None, order: List[str] = None, max_evals_per_motif: int = None, excludes_inputs: List[str] = None,
                       max_total_evals: int = 100_000, log_level: int = INFO) -> Tuple[log_pb2.NetworkRecord, nx.DiGraph, List[nx.DiGraph], List[Dict[str, nx.DiGraph]]]:
    # Convert to list of records
    records = records if isinstance(records, list) else [records]

    # Set default cscheme and node ordering
    cscheme = cscheme or {}
    order = order or []

    # Assert that all episode records have the root
    root_records = [r for eps_record in records for r in eps_record.node if r.info.name == root]
    assert len(root_records) == len(records), "Not all episode records have the root node."

    # Assert that all records have at least seq number of root steps.
    num_seqs = [len(r.steps) for r in root_records]
    min_seqs = min(num_seqs)
    seq = seq if seq is not None and seq > 0 else min_seqs-1
    assert min_seqs > seq, f"Not all episode records ('{num_seqs}') have at least seq={seq} number of root steps."

    # Get all subgraphs
    with timer("Preparing subgraphs", log_level=log_level):
        node_data = {}
        lst_G_full = []
        lst_G_subgraphs = []
        G_subgraphs = {}
        for i, record in enumerate(records):
            G_full = create_graph(record, excludes_inputs=excludes_inputs)
            lst_G_full.append(G_full)

            # Get node_data
            node_data.update(get_node_data(G_full))

            # Set edge and node properties
            set_node_order(G_full, order)
            set_node_colors(G_full, cscheme)

            # Trace root node (not pruned yet)
            G_traced = trace_root(G_full, root=root, seq=seq)

            # Prune unused nodes (not in computation graph of traced root)
            G_traced_pruned = prune_graph(G_traced)

            # Define subgraphs
            G_subgraphs_eps = get_subgraphs(G_traced_pruned, split_mode=split_mode)
            lst_G_subgraphs.append(G_subgraphs_eps)

            # Prepend episode to subgraphs keys
            G_subgraphs_eps = {f"eps_{i}_{k}": v for k, v in G_subgraphs_eps.items()}

            # Add to subgraphs
            G_subgraphs.update(G_subgraphs_eps)

    # Determine unique motifs
    with timer("Determining motifs", log_level=log_level):
        G_motifs = get_unique_motifs(G_subgraphs)

    # Determine minimum common supergraph
    with timer("Determining MCS", log_level=log_level):
        G_MCS = get_minimum_common_supergraph(G_motifs, max_total_evals=max_total_evals, max_evals_per_motif=max_evals_per_motif)
        G_MCS_root = _add_root_to_supergraph(G_MCS, node_data[root])
    log(name="tracer", color="white", log_level=log_level, id="MCS", msg=f"num_nodes={G_MCS.number_of_nodes()} | num_edges={G_MCS.number_of_edges()}")

    # Verify that all subgraphs are monomorphic with the supergraph
    with timer("Check subgraph monomorphism", log_level=log_level):
        assert all(validate_subgraphs(G_MCS, G_subgraphs).values()), "Not all subgraphs are monomorphic with the supergraph."

    # Save traced network record
    record_network = log_pb2.NetworkRecord()
    record_network.episode.extend(records)
    record_network.root = root
    record_network.seq = seq
    record_network.split_mode = split_mode
    record_network.motifs = pickle.dumps(G_motifs)
    record_network.MCS = pickle.dumps(G_MCS_root)
    return record_network, G_MCS_root, lst_G_full, lst_G_subgraphs


def _get_timings_template(G_MCS: nx.DiGraph, num_root_steps: int) -> Timings:
    # Get supergraph timings template
    timings = []
    generations = list(nx.topological_generations(G_MCS))
    for gen in generations:
        t_gen = dict()
        timings.append(t_gen)
        for n in gen:
            data = G_MCS.nodes[n]
            inputs = {}
            for v in data["inputs"].values():
                inputs[v["input_name"]] = {"seq": onp.vstack([onp.array([-1] * v["window"], dtype=onp.int32)] * num_root_steps),
                                           "ts_sent": onp.vstack(
                                               [onp.array([0.] * v["window"], dtype=onp.float32)] * num_root_steps),
                                           "ts_recv": onp.vstack(
                                               [onp.array([0.] * v["window"], dtype=onp.float32)] * num_root_steps)}
            t_slot = {"run": onp.repeat(False, num_root_steps), "ts_step": onp.repeat(0., num_root_steps),
                      "seq": onp.repeat(0, num_root_steps), "inputs": inputs}
            t_gen[n] = t_slot
    return timings


def get_timings(G_MCS: nx.DiGraph, G: nx.DiGraph, G_subgraphs: Dict[str, nx.DiGraph], num_root_steps: int, root: str):
    # Get supergraph timings
    timings = _get_timings_template(G_MCS, num_root_steps)
    # Fill in timings for each subgraph
    for i_step, (root_test, G_step) in enumerate(G_subgraphs.items()):
        matcher = isomorphism.DiGraphMatcher(G_MCS, G_step, node_match=node_match, edge_match=edge_match)
        mcs = next(matcher.subgraph_monomorphisms_iter())
        # Add root node to mapping (root is always the only node in the last generation)
        root_slot = f"{root}_s0"
        assert root_slot in G_MCS, "Root node not found in MCS."
        mcs.update({root_slot: root_test})
        # Update timings of step nodes
        for n_MCS, n_step in mcs.items():
            gen = G_MCS.nodes[n_MCS]["generation"]
            t_slot = timings[gen][n_MCS]
            ndata = G.nodes[n_step]
            t_slot["run"][i_step] = True
            t_slot["seq"][i_step] = ndata["seq"]
            t_slot["ts_step"][i_step] = ndata["ts_step"]

            # Sort input timings
            outputs = {k: [] for k, v in ndata["inputs"].items()}
            inputs = {v["input_name"]: outputs[k] for k, v in ndata["inputs"].items()}
            for u, v, edata in G.in_edges(n_step, data=True):
                u_name = G.nodes[u]["name"]
                v_name = G.nodes[v]["name"]
                if u_name == v_name or edata["pruned"]:
                    continue
                outputs[u_name].append(edata)

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
    return timings


def get_timings_from_network_record(network_record: log_pb2.NetworkRecord, G: List[nx.DiGraph] = None, G_subgraphs: List[Dict[str, nx.DiGraph]] = None,
                                    log_level: int = INFO) -> Timings:
    assert G is None or len(G) == len(network_record.episode), "Number of graphs does not match number of steps."
    assert G_subgraphs is None or len(G_subgraphs) == len(network_record.episode), "Number of subgraphs does not match number of steps."

    # Prepare graphs
    G = G or [None] * len(network_record.episode)
    G_subgraphs = G_subgraphs or [None] * len(network_record.episode)

    # Prepare supergraph
    G_MCS = pickle.loads(network_record.MCS)

    # Get all subgraphs
    timings = []
    with timer("Get timings", log_level=log_level):
        for i, (record, _G, _G_subgraphs) in enumerate(zip(network_record.episode, G, G_subgraphs)):
            # Create graph if not provided
            if _G is None:
                _G = create_graph(record)

            # Get subgraphs if not provided
            if _G_subgraphs is None:
                # Trace root node
                G_traced = trace_root(_G, root=network_record.root, seq=network_record.seq)

                # Prune unused nodes (not in computation graph of traced root)
                G_traced_pruned = prune_graph(G_traced)

                # Define subgraphs
                _G_subgraphs = get_subgraphs(G_traced_pruned, split_mode=network_record.split_mode)

            t = get_timings(G_MCS, _G, _G_subgraphs, num_root_steps=network_record.seq+1, root=network_record.root)
            timings.append(t)

        # Stack timings
        timings = jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *timings)
    return timings


def get_output_buffers_from_timings(G_MCS: nx.DiGraph, timings: Timings, nodes: Dict[str, "Node"], extra_padding: int = 0) -> Dict[str, Output]:
    """Get output buffer from timings."""
    # get seq state
    timings = get_timings_after_root_split(G_MCS, timings)

    # Get output buffer sizes (+1, to add default output)
    num_outputs = {k: v["seq"].max() for k, v in timings.items()}
    buffer_size = {k: v+1+extra_padding for k, v in num_outputs.items()}

    # Fill output buffer
    output_buffer = {}
    stack_fn = lambda *x: jp.stack(x, axis=0)
    rng = jumpy.random.PRNGKey(0)
    for node, size in buffer_size.items():
        assert node in nodes, f"Node `{node}` not found in nodes."
        step_buffer = jax.tree_util.tree_map(stack_fn, *[nodes[node].default_output(rng)] * size)
        eps_buffer = jax.tree_util.tree_map(stack_fn, *[step_buffer] * timings[node]["seq"].shape[0])
        output_buffer[node] = eps_buffer
    return output_buffer


def get_timings_after_root_split(G_MCS: nx.DiGraph, timings: Timings):
    """Get every node's latest sequence number at every root step."""
    # Flatten timings
    timings_flat = {slot: t for gen in timings for slot, t in gen.items()}

    # Get node names
    node_names = set([data["name"] for n, data in G_MCS.nodes(data=True)])

    # Sort slots
    slots = {k: [] for k in node_names}
    [slots[data["name"]].append(timings_flat[n]) for n, data in G_MCS.nodes(data=True)]
    slots = {k: jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *v) for k, v in slots.items()}

    # Get seq state
    timings = {}
    for name, t in slots.items():
        max_seq = onp.maximum.accumulate(jp.amax(t["seq"], axis=0), axis=1)
        max_ts_step = onp.maximum.accumulate(jp.amax(t["ts_step"], axis=0), axis=1)
        timings[name] = dict(seq=max_seq, ts_step=max_ts_step)
    return timings

