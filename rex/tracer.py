from functools import partial
import multiprocessing
import jax
import jumpy
import numpy as onp
import jumpy.numpy as jp
import numpy.ma as ma
import dill as pickle
from flax.core import FrozenDict
from typing import Tuple, List, Dict, Union
from copy import deepcopy
from collections import deque
from google.protobuf.pyext._message import RepeatedCompositeContainer
import networkx as nx
from networkx.algorithms import isomorphism

from rex.constants import INFO
import rex.open_colors as oc
from rex.node import Node
from rex.proto import log_pb2
from rex.mcs import Deque, QueuePolicy, find_largest_motifs
from rex.utils import timer, log
from rex.base import SeqsMapping, BufferSizes, NodeTimings, Timings, Output, GraphBuffer
from rex.multiprocessing import new_process


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

                    data = {"name": record_input.info.name,
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


def _update_subgraph_data(G_sub: nx.DiGraph) -> nx.DiGraph:
    G_sub = G_sub.copy(as_view=False)
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
    return G_sub


def split_generational(G: nx.DiGraph) -> Dict[str, nx.DiGraph]:
    # Copy Graph
    G = G.copy(as_view=False)

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
        G_sub = G.subgraph(nodes).copy(as_view=True)
        G_subgraphs[root] = _update_subgraph_data(G_sub)
    return G_subgraphs


def split_topological(G: nx.DiGraph) -> Dict[str, nx.DiGraph]:
    # Copy Graph
    G = G.copy(as_view=False)

    # Get & sort root nodes
    roots = {n: data for n, data in G.nodes(data=True) if data["root"]}
    roots = {k: roots[k] for k in sorted(roots.keys(), key=lambda k: roots[k]["seq"])}

    # Define subgraphs
    G_subgraphs = {}
    for i_root, (root, data) in enumerate(roots.items()):
        data_root = {"root_index": i_root, "root": root}
        G.nodes[root].update(data_root)
        ancestors = nx.ancestors(G, root)
        [G.nodes[n].update(data_root) for n in ancestors]
        G_sub = G.subgraph(ancestors).copy(as_view=True)
        G_subgraphs[root] = _update_subgraph_data(G_sub)
        G.remove_nodes_from(list(ancestors) + [root])
    return G_subgraphs


def get_subgraphs(G: nx.DiGraph, split_mode: str = "generational") -> Dict[str, nx.DiGraph]:
    if split_mode == "generational":
        return split_generational(G)
    elif split_mode == "topological":
        return split_topological(G)
    else:
        raise NotImplementedError(f"Split mode {split_mode} not supported.")


def as_topological_subgraphs(G_subgraphs: Dict[str, nx.DiGraph]) -> Dict[str, nx.DiGraph]:
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
    G_subgraphs_topo = {}
    for root, G_sub in G_subgraphs.items():
        # Create new subgraph without edges
        G_sub_topo = G_sub.copy(as_view=False)
        G_sub_topo.remove_edges_from(list(G_sub_topo.edges()))

        # Connect nodes with edges according to topological order
        sort = list(nx.topological_sort(G_sub))
        for i, n in enumerate(sort):
            if i > 0:
                G_sub_topo.add_edge(sort[i - 1], n, **edge_data)

        # Update subgraph data
        G_subgraphs_topo[root] = _update_subgraph_data(G_sub_topo)
    return G_subgraphs_topo


def determine_is_super(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    """Determine if G1 (=True) or G2 (=False) should be the larger graph to test for monomorphism."""
    n_nodes_1 = G1.number_of_nodes()
    n_nodes_2 = G2.number_of_nodes()
    if n_nodes_1 == n_nodes_2:
        # If the number of nodes is the same, the graph with the most edges is the supergraph.
        n_edges_1 = G1.number_of_edges()
        n_edges_2 = G2.number_of_edges()
        if n_edges_1 == n_edges_2:
            is_super = True
        else:
            is_super = n_edges_1 > n_edges_2
    else:
        # If the number of nodes is different, the graph with the most nodes is the supergraph.
        is_super = n_nodes_1 > n_nodes_2
    return is_super


def get_unique_motifs(G_subgraphs: Dict[str, nx.DiGraph], validate: bool = True, workers: int = None) -> Dict[str, nx.DiGraph]:
    G_motifs = {}
    G_as_MCS = {}
    for root_test, G_test in G_subgraphs.items():
        G_test: nx.DiGraph
        is_unique = True
        for root_unique in list(G_motifs.keys()):
            G_unique: nx.DiGraph = G_motifs.pop(root_unique)

            # Determine super vs motif graph
            is_super = determine_is_super(G_unique, G_test)
            root_super = root_unique if is_super else root_test
            if root_super not in G_as_MCS:
                G_as_MCS[root_super] = as_MCS(G_test.copy(as_view=False))
            G_super = G_unique if is_super else G_test
            G_super_as_MCS = G_as_MCS[root_super]
            G_motif = G_unique if not is_super else G_test

            # Define matcher
            matcher = isomorphism.DiGraphMatcher(G_super_as_MCS, G_motif, node_match=node_match, edge_match=edge_match)

            if matcher.subgraph_is_monomorphic():
                # If the subgraph is monomorphic, add the supergraph as the unique motif (can either be root_test or root_unique)
                G_motifs[root_super] = G_super
                is_unique = False
            else:
                # If the subgraph is not monomorphic, add the (prev. popped) unique motif back to the motifs dict
                G_motifs[root_unique] = G_unique

        if is_unique:
            # If the subgraph is not monomorphic with any motifs, add it to the motifs dict.
            # At this point, it may have already been added to the motifs dict if it was the super graph, but that's fine.
            G_motifs[root_test] = G_test
            G_as_MCS[root_test] = as_MCS(G_test.copy(as_view=False))

    if validate:
        # Verify that all unique subgraphs are not monomorphic with any other unique subgraph
        for root_motif, G_motif in G_motifs.items():
            for root_motif_other, G_motif_other in G_motifs.items():
                if root_motif == root_motif_other:
                    continue
                is_super = determine_is_super(G_motif, G_motif_other)
                root_super = root_motif if is_super else root_motif_other
                assert root_super in G_as_MCS, "Root motif not in G_as_MCS."
                G_super_as_MCS = G_as_MCS[root_super]
                # G_super = G_motif if is_super else G_motif_other
                G_motif = G_motif if not is_super else G_motif_other
                matcher = isomorphism.DiGraphMatcher(G_super_as_MCS, G_motif, node_match=node_match, edge_match=edge_match)
                assert not matcher.subgraph_is_monomorphic(), "Identified unique subgraphs are monomorphic."

        # Verify that all subgraphs are monomorphic with one of the identified unique subgraphs
        # todo: parallelize
        for root_test, G_test in G_subgraphs.items():
            is_unique = True
            for root_motif, G_motif in G_motifs.items():
                assert root_super in G_as_MCS, "Root_motif not in G_as_MCS."
                G_super_as_MCS = G_as_MCS[root_motif]
                matcher = isomorphism.DiGraphMatcher(G_super_as_MCS, G_test, node_match=node_match, edge_match=edge_match)
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


def get_minimum_common_supergraph(G_motifs: Dict[str, nx.DiGraph], max_evals_per_motif: int = None, max_total_evals: int = 100_000) -> nx.DiGraph:
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


def get_topological_supergraph(G_motifs: Dict[str, nx.DiGraph], **kwargs) -> nx.DiGraph:
    """
    Find the topological supergraph of a list of graphs.

    Arguments:
        G_motifs (List[nx.DiGraph]): The list of graphs

    Returns:
        nx.DiGraph: The topological supergraph
    """
    # Get node data
    node_data = {}
    [node_data.update(get_node_data(G)) for G in G_motifs.values()]

    # Find the motif with the most nodes
    G_motifs_max = G_motifs.get(max(G_motifs, key=lambda x: G_motifs[x].number_of_nodes()))

    # Add len(G_motifs_max.number_of_nodes()) generations, where each generation has a slot for every node type.
    num_nodes = G_motifs_max.number_of_nodes()
    G_super = nx.DiGraph()
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
    for i in range(num_nodes):
        for name, data in node_data.items():
            G_super.add_node(f"{name}_{i}", **data)
            if i > 0:
                G_super.add_edge(f"{name}_{i - 1}", f"{name}_{i}", **edge_data)
    G_topo = as_MCS(G_super)
    return G_topo


def get_supergraph(G_motifs: Dict[str, nx.DiGraph], supergraph_mode: str = "MCS", **kwargs) -> nx.DiGraph:
    if supergraph_mode == "MCS":
        return get_minimum_common_supergraph(G_motifs, **kwargs)
    elif supergraph_mode == "topological":
        return get_topological_supergraph(G_motifs, **kwargs)
    else:
        raise ValueError(f"Unknown supergraph mode: {supergraph_mode}")


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


def validate_subgraphs(G_MCS: nx.DiGraph, G_subgraphs: Dict[str, nx.DiGraph], workers: int = None) -> Dict[str, bool]:

    def _validate_is_monomorphic(G_MCS, G_test) -> Dict[str, str]:
        matcher = isomorphism.DiGraphMatcher(G_MCS, G_test, node_match=node_match, edge_match=edge_match)
        return matcher.subgraph_is_monomorphic()

    max_workers = workers or multiprocessing.cpu_count()
    validate_is_monomorphic = new_process(_validate_is_monomorphic, max_workers=max_workers)

    # Test all subgraphs for monomorphism with the supergraph
    is_monomorphic = {}
    for root_test, G_test in G_subgraphs.items():
        # matcher = isomorphism.DiGraphMatcher(G_MCS, G_test, node_match=node_match, edge_match=edge_match)
        is_monomorphic[root_test] = validate_is_monomorphic.submit(G_MCS, G_test)

    # Wait for all processes to finish
    for root_test, future in is_monomorphic.items():
        is_monomorphic[root_test] = future.result()

    # Shutdown executor
    validate_is_monomorphic.shutdown()
    return is_monomorphic


def get_subgraph_monomorphisms(G_MCS: nx.DiGraph, G_subgraphs: Dict[str, nx.DiGraph], workers: int = None) -> Dict[str, Dict[str, str]]:

    def _get_monomorphism(G_MCS, G_test) -> Dict[str, str]:
        matcher = isomorphism.DiGraphMatcher(G_MCS, G_test, node_match=node_match, edge_match=edge_match)
        return next(matcher.subgraph_monomorphisms_iter())

    max_workers = workers or multiprocessing.cpu_count()
    get_monomorphism = new_process(_get_monomorphism, max_workers=max_workers)

    # Get all subgraph monomorphisms with the supergraph
    monomorphisms = {}
    for root_test, G_test in G_subgraphs.items():
        monomorphisms[root_test] = get_monomorphism.submit(G_MCS, G_test)

    # Wait for all processes to finish
    for root_test, future in monomorphisms.items():
        monomorphisms[root_test] = future.result()

    # Shutdown executor
    get_monomorphism.shutdown()
    return monomorphisms


def get_network_record(records: Union[log_pb2.EpisodeRecord, List[log_pb2.EpisodeRecord]], root: str, seq: int = None, split_mode: str = "generational",
                       supergraph_mode: str = "MCS",
                       cscheme: Dict[str, str] = None, order: List[str] = None, max_evals_per_motif: int = None, excludes_inputs: List[str] = None, workers: int = None,
                       max_total_evals: int = 100_000, log_level: int = INFO, validate: bool = True) -> Tuple[log_pb2.NetworkRecord, nx.DiGraph, List[nx.DiGraph], List[Dict[str, nx.DiGraph]]]:
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

            # Set edge and node properties
            set_node_order(G_full, order)
            set_node_colors(G_full, cscheme)

            # Get node_data
            node_data.update(get_node_data(G_full))

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
        G_motifs = get_unique_motifs(G_subgraphs, validate=validate, workers=workers)

    # Determine minimum common supergraph
    with timer(f"Determining {supergraph_mode}", log_level=log_level):
        G_MCS = get_supergraph(G_motifs, supergraph_mode=supergraph_mode, max_total_evals=max_total_evals, max_evals_per_motif=max_evals_per_motif)
        G_MCS_root = _add_root_to_supergraph(G_MCS, node_data[root])
    log(name="tracer", color="white", log_level=log_level, id="supergraph", msg=f"num_nodes={G_MCS.number_of_nodes()} | num_edges={G_MCS.number_of_edges()}")

    # Verify that all subgraphs are monomorphic with the supergraph
    if validate:
        with timer("Check subgraph monomorphism", log_level=log_level):
            assert all(validate_subgraphs(G_MCS, G_subgraphs, workers=workers).values()), "Not all subgraphs are monomorphic with the supergraph."

    # Save traced network record
    record_network = log_pb2.NetworkRecord()
    record_network.episode.extend(records)
    record_network.root = root
    record_network.seq = seq
    record_network.split_mode = split_mode
    record_network.supergraph_mode = supergraph_mode
    record_network.excludes_inputs.extend(excludes_inputs)
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


def get_timings(G_MCS: nx.DiGraph, G: nx.DiGraph, monomorphisms: Dict[str, Dict[str, str]], num_root_steps: int, root: str, workers: int = None):
    # Get supergraph timings
    timings = _get_timings_template(G_MCS, num_root_steps)
    # Fill in timings for each subgraph
    for i_step, (root_test, mcs) in enumerate(monomorphisms.items()):
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
                # if u_name == v_name or edata["pruned"]:
                if edata["stateful"] or edata["pruned"]:
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
                                    log_level: int = INFO, workers: int = None) -> Timings:
    assert G is None or len(G) == len(network_record.episode), "Number of graphs does not match number of steps."
    assert G_subgraphs is None or len(G_subgraphs) == len(network_record.episode), "Number of subgraphs does not match number of steps."

    # Prepare graphs
    G = G or [None] * len(network_record.episode)
    G_subgraphs = G_subgraphs or [None] * len(network_record.episode)

    # Prepare supergraph
    G_MCS = pickle.loads(network_record.MCS)

    # Get all subgraphs
    with timer("Get timings", log_level=log_level):
        # Get all subgraphs
        keys_subgraphs = {}
        all_subgraphs = {}
        for i, (record, _G, _G_subgraphs) in enumerate(zip(network_record.episode, G, G_subgraphs)):
            # Create graph if not provided
            if _G is None:
                _G = create_graph(record)
                G[i] = _G

            # Get subgraphs if not provided
            if _G_subgraphs is None:
                # Trace root node
                G_traced = trace_root(_G, root=network_record.root, seq=network_record.seq)

                # Prune unused nodes (not in computation graph of traced root)
                G_traced_pruned = prune_graph(G_traced)

                # Define subgraphs
                _G_subgraphs = get_subgraphs(G_traced_pruned, split_mode=network_record.split_mode)
                G_subgraphs[i] = _G_subgraphs

            # Convert to topological subgraphs
            if network_record.supergraph_mode == "topological":
                _G_subgraphs = as_topological_subgraphs(_G_subgraphs)

            keys_subgraphs[i] = {f"{i}_{k}": k for k in _G_subgraphs.keys()}
            eps_subgraphs = {f"{i}_{k}": v for k, v in _G_subgraphs.items()}
            all_subgraphs.update(eps_subgraphs)

        # Get monomorphisms
        monomorphisms = get_subgraph_monomorphisms(G_MCS, all_subgraphs, workers=workers)
        monomorphisms_eps = {i: {v: monomorphisms[k] for k, v in key_map.items()} for i, key_map in keys_subgraphs.items()}
        timings = []
        for i, mono in monomorphisms_eps.items():
            t = get_timings(G_MCS, G[i], mono, num_root_steps=network_record.seq+1, root=network_record.root, workers=workers)
            timings.append(t)

        # Stack timings
        timings = jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *timings)
    return timings


def get_outputs_from_timings(G_MCS: nx.DiGraph, timings: Timings, nodes: Dict[str, "Node"], extra_padding: int = 0) -> Dict[str, Output]:
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


def get_chronological_timings(G_MCS: nx.DiGraph, timings: Timings, eps: int) -> NodeTimings:
    # Take only one episode
    timings = jax.tree_util.tree_map(lambda x: x[eps], timings)

    # Flatten timings
    timings_flat = {slot: t for gen in timings for slot, t in gen.items()}

    # Get node names
    node_names = set([data["name"] for n, data in G_MCS.nodes(data=True)])

    # Sort slots
    slots = {k: [] for k in node_names}
    [slots[data["name"]].append(timings_flat[n]) for n, data in G_MCS.nodes(data=True)]
    slots = {k: jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *v) for k, v in slots.items()}

    # Only keep timings with run=True, sort by seq
    slots_run = {k: jax.tree_util.tree_map(lambda _arr: _arr[v["run"]], v) for k, v in slots.items()}
    sort = {k: onp.argsort(v["seq"]) for k, v in slots_run.items()}
    slots_chron = {k: jax.tree_util.tree_map(lambda _arr: _arr[sort[k]], v) for k, v in slots_run.items()}
    return slots_chron


def get_masked_timings(G_MCS: nx.DiGraph, timings: Timings) -> NodeTimings:
    # generations = list(nx.topological_generations(G_MCS))

    # Get node names
    node_names = set([data["name"] for n, data in G_MCS.nodes(data=True)])

    # Get node data
    slot_node_data = {n: data for n, data in G_MCS.nodes(data=True)}
    node_data = {}
    [node_data.update({d["name"]: d}) for slot, d in slot_node_data.items() if d["name"] not in node_data]

    # Get output buffer sizes
    masked_timings_slot = []
    for i_gen, gen in enumerate(timings):
        t_flat = {slot: t for slot, t in gen.items()}
        slots = {k: [] for k in node_names}
        [slots[G_MCS.nodes[n]["name"]].append(t_flat[n]) for n in gen]
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
            t = {k: jax.tree_util.tree_map(lambda x: onp.repeat(x[:, :, :, None], len(timings), axis=3), v) for k, v in
                 t.items()}

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


def get_buffer_sizes_from_timings(G_MCS: nx.DiGraph, timings: Timings) -> BufferSizes:
    # Get masked timings:= [eps, step, slot_idx, gen_idx, window=optional]
    masked_timings = get_masked_timings(G_MCS, timings)

    # Get node data
    slot_node_data = {n: data for n, data in G_MCS.nodes(data=True)}
    node_data = {}
    [node_data.update({d["name"]: d}) for slot, d in slot_node_data.items() if d["name"] not in node_data]

    # Get min buffer size for each node
    name_mapping = {n: {v["input_name"]: o for o, v in data["inputs"].items()} for n, data in node_data.items()}
    min_buffer_sizes = {k: {input_name: output_name for input_name, output_name in inputs.items()} for k, inputs in name_mapping.items()}
    node_buffer_sizes = {n: [] for n in node_data.keys()}
    for n, inputs in name_mapping.items():
        t = masked_timings[n]
        for input_name, output_name in inputs.items():
            # Determine min input sequence per generation
            seq_in = onp.amin(t["inputs"][input_name]["seq"], axis=(2, 4))
            seq_in = seq_in.reshape(*seq_in.shape[:-2], -1)
            # NOTE: fill masked steps with max value (to not influence buffer size)
            ma.set_fill_value(seq_in, onp.iinfo(onp.int32).max)
            filled_seq_in = seq_in.filled()
            max_seq_in = onp.minimum.accumulate(filled_seq_in[:, ::-1], axis=-1)[:, ::-1]

            # Determine max output sequence per generation
            seq_out = onp.amax(masked_timings[output_name]["seq"], axis=(2,))
            seq_out = seq_out.reshape(*seq_out.shape[:-2], -1)
            ma.set_fill_value(seq_out, -1)
            filled_seq_out = seq_out.filled()
            max_seq_out = onp.maximum.accumulate(filled_seq_out, axis=-1)

            # Calculate difference to determine buffer size
            # NOTE: Offset output sequence by +1, because the output is written to the buffer AFTER the buffer is read
            offset_max_seq_out = onp.roll(max_seq_out, shift=1, axis=1)
            offset_max_seq_out[:, 0] = -1  # NOTE: First step is always -1, because no node has run at this point.
            s = offset_max_seq_out - max_seq_in

            # NOTE! +1, because, for example, when offset_max_seq_out = 0, and max_seq_in = 0, we need to buffer 1 step.
            max_s = s.max() + 1

            # Store min buffer size
            min_buffer_sizes[n][input_name] = max_s
            node_buffer_sizes[output_name].append(max_s)

    return node_buffer_sizes


def get_graph_buffer(G_MCS: nx.DiGraph, timings: Timings, nodes: Dict[str, "Node"], sizes: BufferSizes = None, extra_padding: int = 0) -> GraphBuffer:
    # Get buffer sizes if not provided
    if sizes is None:
        sizes = get_buffer_sizes_from_timings(G_MCS, timings)

    # Create output buffers
    buffers = {}
    stack_fn = lambda *x: jp.stack(x, axis=0)
    rng = jumpy.random.PRNGKey(0)
    for n, s in sizes.items():
        assert n in nodes, f"Node `{n}` not found in nodes."
        buffer_size = max(s) + extra_padding if len(s) > 0 else max(1, extra_padding)
        b = jax.tree_util.tree_map(stack_fn, *[nodes[n].default_output(rng)] * buffer_size)
        buffers[n] = b

    # Get dummy timings (to infer static shapes)
    eps_timing = jax.tree_util.tree_map(lambda x: x[0], timings)

    return GraphBuffer(outputs=FrozenDict(buffers), timings=eps_timing)


def get_seqs_mapping(G_MCS: nx.DiGraph, timings: Timings, buffer: GraphBuffer) -> Tuple[SeqsMapping, SeqsMapping]:
    # generations = list(nx.topological_generations(G_MCS))

    def _get_buffer_size(b):
        leaves = jax.tree_util.tree_leaves(b)
        size = leaves[0].shape[0] if len(leaves) > 0 else 1
        return size

    # Get buffer sizes
    buffer_sizes = {n: _get_buffer_size(b) for n, b in buffer.outputs.items()}

    # Get masked timings:= [eps, step, slot_idx, gen_idx, window=optional]
    masked_timings = get_masked_timings(G_MCS, timings)

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
        updated_seqs = (sorted_seqs != onp.roll(sorted_seqs, shift=1, axis=1))  # Roll in gen axis
        updated_seqs[:, 0, :] = False  # First step is never updated

        # Reshape to step shape
        sorted_seqs = sorted_seqs.reshape(*shape_seq_out, buffer_sizes[n])
        updated_seqs = updated_seqs.reshape(*shape_seq_out, buffer_sizes[n])

        # Store
        seqs[n] = sorted_seqs
        updated[n] = updated_seqs

    return seqs, updated


def get_step_seqs_mapping(G_MCS: nx.DiGraph, timings: Timings, buffer: GraphBuffer) -> Tuple[SeqsMapping, SeqsMapping]:
    # Get update mask
    seqs, updated = get_seqs_mapping(G_MCS, timings, buffer)

    # Get updated seqs in buffer AFTER step is executed
    updated_step = {n: onp.any(v[:, :, :, :], axis=2) for n, v in updated.items()}

    # Get absolute seqs in buffer BEFORE step is executed
    after_seqs_step = {n: v[:, :, -1, :] for n, v in seqs.items()}
    init_seqs_step = {n: onp.full(arr[:, [0], :].shape, fill_value=-1) for n, arr in after_seqs_step.items()}
    before_seqs_step = {n: onp.concatenate([init_seqs_step[n], arr], axis=1) for n, arr in after_seqs_step.items()}

    return before_seqs_step, updated_step
