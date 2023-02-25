import numpy as np
import numpy as onp
from functools import lru_cache
from typing import Tuple, List
from copy import deepcopy
from collections import deque
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import seaborn as sns

sns.set()

import networkx as nx
from networkx.algorithms import isomorphism

import rex.open_colors as oc
from rex.proto import log_pb2


if __name__ == "__main__":
    # todo: [DONE] Define graph in networkx
    # todo: [DONE] Store information to build timings as attributes in nodes and edges
    #        - Nodes: ts_step, ts_sent, seq, stateful | edgecolor (str), facecolor (str), alpha (float), position (ts_step, y), gen_index (int), pruned (bool), chunk_index (int), root (bool)
    #        - Edges: seq (int), ts_sent (float), ts_recv (float) | pruned (bool), color (str), alpha (float), linestyle (str),
    # todo: [DONE] Get all ancestors of a root node --> Prune nodes that are not ancestors of the root node.
    # todo: [LATER] Sort pruned graph in iterative topological order.
    # todo: [DONE] Sort pruned graph in topological generations order.
    # todo: [DONE] Isolate root nodes into separate generations
    # todo: [DONE] Place isolate before his generation in graph
    # todo: [LATER] See if nodes in end generation of a motif be merged into the next generation
    # todo: [DONE] Define N motifs where every motif is defined as a subgraph consisting of the generations in-between subsequent root nodes.
    # todo: [DONE] Find U unique motifs based on the N motifs, by filtering out motifs that are (subgraph) isomorphic to each other. Always keep the one with the highest number of nodes.
    # todo: [LATER] Goal: Provided a root node, what is the minimal supergraph and root motif split, so that all root motifs are semantically monomorphic to a subgraph of this supergraph (MCS).
    # todo: [DONE] Goal: Find a minimal supergraph that contains a subgraph that is semantically monomorphic to every motif (MCS).
    # todo: [DONE] Extract timings from graph
    # todo: Extract tick state per chunk
    # todo: Convert scratch_networkx.py to a module
    # todo: How to hierarchically save graphs?
    #       1. Save the full graph (unpruned) (G_full).
    #       2. Given a root node and split, save unique motifs and supergraph.
    #       3. Save pruned full graph with supergraph mapping? Or dynamically create that by loading the full graph and the supergraph?

    # Load record
    record = log_pb2.EpisodeRecord()
    with open("eps_record_1.pb", "rb") as f:
        record.ParseFromString(f.read())

    node_size = 200
    node_fontsize = 10
    edge_linewidth = 2.0
    node_linewidth = 1.5
    arrowsize = 10
    arrowstyle = "->"
    connectionstyle = "arc3"
    edge_bbox = None
    order = ["world", "sensor", "observer", "agent", "actuator"]
    cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}

    # Create graph
    G_full = nx.DiGraph()

    # Determine edge bbox
    if edge_bbox is None:
        edge_bbox = dict(boxstyle="round", fc=oc.ccolor("gray"), ec=oc.ccolor("gray"), alpha=1.0)

    # Determine color scheme
    cscheme = cscheme if isinstance(cscheme, dict) else {}
    cscheme = {**{record_node.info.name: "gray" for record_node in record.node if record_node.info.name not in cscheme},
               **cscheme}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Determine order
    order = order if isinstance(order, list) else []
    order = order + [record_node.info.name for record_node in record.node if record_node.info.name not in order]

    # Layout properties
    y = {name: i for i, name in enumerate(order)}

    # Prepare data
    edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
    node_data = {}
    for record_node in record.node:
        inputs = {i.output: {"input_name": i.name, "window": i.window} for i in record_node.info.inputs}
        node_data[record_node.info.name] = {"name": record_node.info.name,
                                            "inputs": inputs,
                                            "stateful": record_node.info.stateful,
                                            "edgecolor": ecolor[record_node.info.name],
                                            "facecolor": fcolor[record_node.info.name],
                                            "alpha": 1.}

    # Get all nodes
    for record_node in record.node:
        # record_node.info.stateful = True
        # Add nodes
        for i_step, record_step in enumerate(record_node.steps):
            # print(f"name={record_node.info.name} | step={i_step} | ts_step={record_step.ts_step} | ts_sent={record_step.ts_output} | seq={record_step.tick}")
            data = {  # "name": record_node.info.name,
                # "stateful": record_node.info.stateful,
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
                # "edgecolor": ecolor[record_node.info.name],
                # "facecolor": fcolor[record_node.info.name],
                # "alpha": 1.,
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
                        "color": oc.ecolor.pruned if pruned else oc.ecolor.used,
                        "linestyle": "--" if pruned else "-",
                        "alpha": 0.5 if pruned else 1.,
                        }
                id_source = f"{record_node.info.name}_{record_step.tick - 1}"
                id_target = f"{record_node.info.name}_{record_step.tick}"
                G_full.add_edge(id_source, id_target, **data)

        # Add edges
        for record_input in record_node.inputs:
            name = record_input.info.name
            output = record_input.info.output
            window = record_input.info.window
            edge_window = deque(maxlen=window)
            for i_step, (record_grouped, record_step) in enumerate(zip(record_input.grouped, record_node.steps)):
                for i_msg, record_msg in enumerate(reversed(record_grouped.messages)):
                    pruned = True if i_msg >= window else False

                    data = {"name": record_input.info.name,
                            "output": record_input.info.output,
                            "window": record_input.info.window,
                            "seq": record_msg.sent.seq,
                            "ts_sent": record_msg.sent.ts.sc,
                            "ts_recv": record_msg.received.ts.sc,
                            "pruned": pruned,
                            "color": oc.ecolor.pruned if pruned else oc.ecolor.used,
                            "linestyle": "--" if pruned else "-",
                            "alpha": 0.5 if pruned else 1.,
                            }
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

    # Get edge and node properties
    G_pruned_edge = G_full.copy(as_view=False)
    remove_edges = [(u, v) for u, v, data in G_pruned_edge.edges(data=True) if data['pruned']]
    G_pruned_edge.remove_edges_from(remove_edges)


    def plot_graph(G, ax=None):
        # Get edge and node properties
        edges = G.edges(data=True)
        nodes = G.nodes(data=True)
        edge_color = [data['color'] for u, v, data in edges]
        edge_alpha = [data['alpha'] for u, v, data in edges]
        edge_style = [data['linestyle'] for u, v, data in edges]
        node_alpha = [data['alpha'] for n, data in nodes]
        node_ecolor = [data['edgecolor'] for n, data in nodes]
        node_fcolor = [data['facecolor'] for n, data in nodes]

        # Prepare node labels
        node_labels = {n: data["seq"] for n, data in nodes}
        node_position = {n: data["position"] for n, data in nodes}

        # Create new plot
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 5)
            ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])

        nx.draw_networkx_nodes(G, ax=ax, pos=node_position, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                               node_size=node_size, linewidths=node_linewidth)
        nx.draw_networkx_edges(G, ax=ax, pos=node_position, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                               arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                               width=edge_linewidth, node_size=node_size)
        nx.draw_networkx_labels(G, node_position, node_labels, font_size=node_fontsize)

        # Set yticks
        yticks = list(y.values())
        ylabels = order
        ax.set_yticks(yticks, labels=ylabels)
        ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)


    # Show graph
    # plot_graph(G_full)
    # plot_graph(G_pruned_edge)
    # plt.show()

    # Trace root node
    # Traced node must be stateful. Otherwise, chronological sorting is not guaranteed.
    G_traced = G_pruned_edge.copy(as_view=False)
    root_name = "agent"
    root_seq_trace = 200
    root_id = f"{root_name}_{root_seq_trace}"
    root_nodes = {n: data for n, data in G_traced.nodes(data=True) if n.startswith(root_name)}
    [G_traced.nodes[n].update({"root": True}) for n in root_nodes.keys()]

    # Trace
    ancestors = nx.ancestors(G_traced, root_id)
    pruned_nodes = [n for n in G_traced.nodes() if n not in ancestors and n != root_id]
    data_pruned = {"pruned": True, "alpha": 0.5, "edgecolor": oc.ecolor.pruned, "facecolor": oc.fcolor.pruned}
    [G_traced.nodes[n].update(data_pruned) for n in pruned_nodes]
    pruned_edges = [(u, v) for u, v in G_traced.edges() if u in pruned_nodes or v in pruned_nodes]
    data_pruned = {"pruned": True, "alpha": 0.5, "color": oc.ecolor.pruned, "linestyle": "--"}
    [G_traced.edges[u, v].update(data_pruned) for u, v in pruned_edges]

    # Define new graph
    G_traced_pruned = G_traced.copy(as_view=False)
    remove_nodes = [n for n, data in G_traced_pruned.nodes(data=True) if data['pruned']]
    G_traced_pruned.remove_nodes_from(remove_nodes)

    # Show traced graph
    # plot_graph(G_traced)
    # plot_graph(G_traced_pruned)
    # plt.show()

    # Define generations
    generations = list(nx.topological_generations(G_traced_pruned))
    for i_gen, gen in enumerate(generations):
        for n in gen:
            G_traced_pruned.nodes[n].update({"generation": i_gen})

    # Define subgraphs
    bwd_subgraphs = {}  # Places the root in an isolated generation *before* the generation it was originally in.
    new_subgraph = []
    for gen in generations:
        is_root = False
        gen_has_root = False
        for n in gen:
            assert not (is_root and G_traced_pruned.nodes[n][
                "root"]), "Multiple roots in a generation. Make sure the root node is stateful."
            is_root = G_traced_pruned.nodes[n]["root"]
            gen_has_root = is_root or gen_has_root

        if gen_has_root:
            root = [n for n in gen if G_traced_pruned.nodes[n]["root"]][0]
            data_root = {"root_index": len(bwd_subgraphs), "root": root}
            G_traced_pruned.nodes[root].update(data_root)
            [G_traced_pruned.nodes[n].update(data_root) for gen in new_subgraph for n in gen]
            bwd_subgraphs[root] = new_subgraph
            next_gen = [n for n in gen if not G_traced_pruned.nodes[n]["root"]]
            new_subgraph = [next_gen]
        else:
            new_subgraph.append(gen)

    # Create subgraphs
    G_full_subgraphs = G_traced_pruned.copy(as_view=False)
    G_subgraphs = dict()
    for i_root, (root, subgraph) in enumerate(bwd_subgraphs.items()):
        nodes = [n for gen in subgraph for n in gen]
        G_root = G_full_subgraphs.subgraph(nodes).copy(as_view=False)
        sub_generations = list(nx.topological_generations(G_root))
        # Calculate longest path
        for n in G_root.nodes():
            desc = nx.descendants(G_root, n)
            desc.add(n)
            G_desc = G_root.subgraph(desc).copy(as_view=True)
            longest_path_length = nx.dag_longest_path_length(G_desc)
            # longest_path = nx.dag_longest_path(G_desc)
            G_root.nodes[n].update({"sub_longest_path_length": longest_path_length})
        # Calculate sub generations
        for i_sub_gen, sub_gen in enumerate(sub_generations):
            for n in sub_gen:
                G_root.nodes[n].update({"sub_generation": i_sub_gen})
        G_subgraphs[root] = G_root

    # TODO: REMOVE
    keys = ["agent_0", "agent_2"]
    G_subgraphs["agent_0"] = G_full_subgraphs.subgraph(["actuator_0", "sensor_0", "sensor_1", "observer_4", "observer_5", "observer_6", "observer_7"]).copy(
        as_view=False)
    G_subgraphs["agent_2"] = G_full_subgraphs.subgraph(["actuator_0", "actuator_1", "actuator_2"]).copy(as_view=False)
    for k in keys:
        for n in G_subgraphs[k].nodes():
            desc = nx.descendants(G_subgraphs[k], n)
            desc.add(n)
            G_desc = G_subgraphs[k].subgraph(desc).copy(as_view=True)
            longest_path_length = nx.dag_longest_path_length(G_desc)
            # longest_path = nx.dag_longest_path(G_desc)
            G_subgraphs[k].nodes[n].update({"sub_longest_path_length": longest_path_length})

    # # Plot subgraphs
    # if False:
    # 	fig, ax = plt.subplots(figsize=(12, 5))
    # 	ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
    # 	for root, G_root in G_subgraphs.items():
    # 		plot_graph(G_root, ax=ax)
    # 	plt.show()


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
        # todo: Test whether "sbu_longest_path_length" is needed (maybe, vf2 already does this).
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


    # Determine unique motifs
    G_full_subgraphs = G_traced_pruned.copy(as_view=False)
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
            root_motif = root_test if n_nodes_test <= n_nodes_unique else root_unique

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

    # The lower-bound on the optimal solution adds N nodes, where N is the max difference over every mcs and corresponding target graph.
    # The worst-case optimal solution M nodes, where M is the sum of node differences between every mcs and corresponding target graph.
    # Use the relation between Maximal Common Subgraph (mcs) and Minimal Common Supergraph (MCS) to prove the calculation of the minimal common supergraph (MCS).
    # Smartly define search space for the optimal solution (i.e. minimal number of each node type).

    # Plot unique motifs
    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
        for root, G_unique in G_motifs.items():
            plot_graph(G_unique, ax=ax)
        plt.show()


    def as_MCS(G: nx.DiGraph) -> nx.DiGraph:
        # todo: as_MCS should only have edges between nodes in the partite groups that exist in the unique patterns.
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
                        # todo: must ensure that root is first in generation before filtering edges here.
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

    @lru_cache()
    def _is_node_attr_match(
            motif_node_id: str, host_node_id: str, motif: nx.Graph, host: nx.Graph
    ) -> bool:
        """
        Check if a node in the host graph matches the attributes in the motif.

        Arguments:
            motif_node_id (str): The motif node ID
            host_node_id (str): The host graph ID
            motif (nx.Graph): The motif graph
            host (nx.Graph): The host graph

        Returns:
            bool: True if the host node matches the attributes in the motif

        """
        motif_node = motif.nodes[motif_node_id]
        host_node = host.nodes[host_node_id]

        return node_match(host_node, motif_node)

    @lru_cache()
    def _is_edge_attr_match(
            motif_edge_id: Tuple[str, str],
            host_edge_id: Tuple[str, str],
            motif: nx.Graph,
            host: nx.Graph,
    ) -> bool:
        """
        Check if an edge in the host graph matches the attributes in the motif.

        Arguments:
            motif_edge_id (str): The motif edge ID
            host_edge_id (str): The host edge ID
            motif (nx.Graph): The motif graph
            host (nx.Graph): The host graph

        Returns:
            bool: True (Always)

        """

        return True


    @lru_cache()
    def _is_node_structural_match(
            motif_node_id: str, host_node_id: str, motif: nx.Graph, host: nx.Graph
    ) -> bool:
        """
        Check if the motif node here is a valid structural match.

        Specifically, this requires that a host node has at least the degree as the
        motif node.

        Arguments:
            motif_node_id (str): The motif node ID
            host_node_id (str): The host graph ID
            motif (nx.Graph): The motif graph
            host (nx.Graph): The host graph

        Returns:
            bool: True if the motif node maps to this host node

        """
        return host.degree(host_node_id) >= motif.degree(motif_node_id)

    from grandiso import find_largest_motifs, find_motifs
    from grandiso.queues import Deque, QueuePolicy

    def MCS(G1, G2, max_evals: int = None):
        # todo: check if we cannot use isomorphism.vf2. Seems alot faster.
        queue = Deque(policy=QueuePolicy.BREADTHFIRST)
        num_evals, is_monomorphism, largest_motif = find_largest_motifs(G2, G1,
                                                                        queue_=queue,
                                                                        max_evals=max_evals,
                                                                        # interestingness={"observer_6": 1.0, "actuator_0": 0.9},
                                                                        is_node_structural_match=_is_node_structural_match,
                                                                        is_node_attr_match=_is_node_attr_match,
                                                                        is_edge_attr_match=_is_edge_attr_match)
        # Do not modify G1 if it is a monomorphism
        if is_monomorphism:
            return num_evals, G1
        # Else, prepare to unify G1 and G2
        mcs = {v: k for k, v in largest_motif[0].items()}
        mcs_G1 = G1.subgraph(mcs.keys())
        mcs_G2 = nx.relabel_nodes(G1.subgraph(mcs.keys()), mcs, copy=True)
        E1 = emb(mcs_G1, G1)
        E2 = emb(mcs_G2, G2)
        _MCS = nx.relabel_nodes(unify(mcs_G2, G2, E2), {v: k for k,v in mcs.items()}, copy=True)
        MCS = unify(_MCS, G1, E1)
        return num_evals, as_MCS(MCS)

    def get_minimum_common_supergraph(G_motifs: List[nx.DiGraph], max_evals_per_motif: int = None, max_total_evals: int = 100_000):
        """
        Find the minimum common monomorphic supergraph of a list of graphs.

        Arguments:
            G_motifs (List[nx.DiGraph]): The list of graphs
            max_eval_per_motif (int): The maximum number of evaluations per motif

        Returns:
            nx.DiGraph: The minimum common supergraph

        """

        # Construct initial supergraph from largest motif
        G_motifs_max = G_motifs.get(max(G_motifs, key=lambda x: G_motifs[x].number_of_nodes()))
        G_MCS = as_MCS(G_motifs_max.copy(as_view=False))

        # Iterate over all motifs
        num_evals = 0
        for i_motif, (root_name, G_motif) in enumerate(G_motifs.items()):
            if G_motif is G_motifs_max:
                continue
            max_evals_per_motif = min(max_evals_per_motif, max_total_evals-num_evals) if max_evals_per_motif is not None else max_total_evals-num_evals
            prev_num_nodes = G_MCS.number_of_nodes()
            # Find the MCS
            num_evals_motif, G_MCS = MCS(G_MCS, G_motif, max_evals=max_evals_per_motif)
            # Update the total number of evaluations
            num_evals += num_evals_motif
            next_num_nodes = G_MCS.number_of_nodes()
            if num_evals >= max_total_evals:
                print(f"max_total_evals={max_total_evals} exceeded. Stopping. MCS covers {i_motif+1}/{len(G_motifs)} motifs. May be excessively large.")
                break
            print(f"motif=`{root_name}` | nodes+={next_num_nodes - prev_num_nodes} | num_nodes={next_num_nodes} | motif_evals={num_evals_motif} | total_evals={num_evals}")
        return G_MCS

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_graph(G_motif, ax=ax)
        generations = list(nx.topological_generations(G_motif))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="generation", yticks=[], xlim=[-1, 0.3])
        plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_graph(G_test, ax=ax)
        generations = list(nx.topological_generations(G_test))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="generation", yticks=[], xlim=[-1, 0.3])
        plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_graph(G_MCS, ax=ax)
        generations = list(nx.topological_generations(G_MCS))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="generation", yticks=[], xlim=[-1, len(generations)])
        plt.show()

    G_MCS = get_minimum_common_supergraph(G_motifs, max_total_evals=100_000)
    print(f"num_nodes={G_MCS.number_of_nodes()} | num_edges={G_MCS.number_of_edges()}")

    # Add root node slot as the first generation
    generations = next(nx.topological_generations(G_MCS))
    some_root = next(iter(root_nodes.keys()))
    G_MCS.add_node(some_root, **root_nodes[some_root])

    # Add edges between root node and first generation
    for node in generations:
        G_MCS.add_edge(some_root, node, **edge_data)

    # Convert to super graph
    G_MCS = as_MCS(G_MCS)

    # Verify that all subgraphs are monomorphic with the supergraph
    for root_test, G_test in G_subgraphs.items():
        matcher = isomorphism.DiGraphMatcher(G_MCS, G_test, node_match=node_match, edge_match=edge_match)
        assert matcher.subgraph_is_monomorphic(), "Subgraph is not monomorphic with the supergraph."

    def get_MCS_template(G_MCS, num_root_steps: int):
        # Get supergraph timings template
        MCS_template = []
        generations = list(nx.topological_generations(G_MCS))
        for gen in generations:
            t_gen = dict()
            MCS_template.append(t_gen)
            for n in gen:
                data = G_MCS.nodes[n]
                inputs = {}
                for v in data["inputs"].values():
                    inputs[v["input_name"]] = {"seq": onp.vstack([onp.arange(-v["window"], 0, dtype=onp.int32)]*num_root_steps),
                                               "ts_sent": onp.vstack([onp.array([0.]*v["window"], dtype=onp.float32)]*num_root_steps),
                                               "ts_recv": onp.vstack([onp.array([0.]*v["window"], dtype=onp.float32)]*num_root_steps)}
                t_slot = {"run": onp.repeat(False, num_root_steps), "ts_step": onp.repeat(0., num_root_steps), "seq": onp.repeat(0., num_root_steps), "inputs": inputs}
                t_gen[n] = t_slot
        return MCS_template

    # Get timings template
    MCS_template = get_MCS_template(G_MCS, num_root_steps=len(root_nodes))

    # Fill template via mapping from supergraph to subgraphs
    from rex.utils import timer
    with timer("get_mappings", log_level=50):
        for i_step, (root_test, G_step) in enumerate(G_subgraphs.items()):
            matcher = isomorphism.DiGraphMatcher(G_MCS, G_step, node_match=node_match, edge_match=edge_match)
            mcs = next(matcher.subgraph_monomorphisms_iter())
            # Add root node to mapping (root is always the only node in the first generation)
            root_slot = f"{G_full.nodes[root_test]['name']}_s0"
            assert root_slot in G_MCS, "Root node not found in MCS."
            mcs.update({root_slot: root_test})
            # Update timings of step nodes
            for n_MCS, n_step in mcs.items():
                gen = G_MCS.nodes[n_MCS]["generation"]
                t_slot = MCS_template[gen][n_MCS]
                ndata = G_full.nodes[n_step]
                t_slot["run"][i_step] = True
                t_slot["seq"][i_step] = ndata["seq"]
                t_slot["ts_step"][i_step] = ndata["ts_step"]

                # Sort input timings
                outputs = {k: [] for k, v in ndata["inputs"].items()}
                inputs = {v["input_name"]: outputs[k] for k, v in ndata["inputs"].items()}
                for u, v, edata in G_full.in_edges(n_step, data=True):
                    u_name = G_full.nodes[u]["name"]
                    v_name = G_full.nodes[v]["name"]
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

    input("Press enter to continue...")