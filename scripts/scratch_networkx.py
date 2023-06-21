import jumpy
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import seaborn as sns

sns.set()

import networkx as nx

from rex.utils import timer
import rex.tracer as tracer
import rex.open_colors as oc
from rex.proto import log_pb2
from rex.compiled import CompiledGraph
from rex.constants import ERROR, INFO, DEBUG, WARN, SILENT

from scripts.dummy import build_dummy_env, DummyEnv


# Plot
def plot_graph(G,
               ax=None,
               node_size=200,
               node_fontsize=10,
               edge_linewidth=2.0,
               node_linewidth=1.5,
               arrowsize=10,
               arrowstyle="->",
               connectionstyle="arc3",
               ):
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
    y = tracer.get_node_y_position(G)
    yticks = list(y.values())
    ylabels = list(y.keys())
    ax.set_yticks(yticks, labels=ylabels)
    ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)

if __name__ == "__main__":
    # todo: [DONE] Define graph in networkx
    # todo: [DONE] Store information to build timings as attributes in nodes and edges
    #        - Nodes: ts_step, ts_sent, seq, stateful | edgecolor (str), facecolor (str), alpha (float), position (ts_step, y), gen_index (int), pruned (bool), chunk_index (int), root (bool)
    #        - Edges: seq (int), ts_sent (float), ts_recv (float) | pruned (bool), color (str), alpha (float), linestyle (str),
    # todo: [DONE] Get all ancestors of a root node --> Prune nodes that are not ancestors of the root node.
    # todo: [DONE] Use a topological sort for the split the graph into subgraphs.
    # todo: [DONE] Sort pruned graph in topological generations order.
    # todo: [DONE] Isolate root nodes into separate generations
    # todo: [DONE] Place isolate before his generation in graph
    # todo: [LATER] See if nodes in end generation of a motif be merged into the next generation
    # todo: [DONE] Define N motifs where every motif is defined as a subgraph consisting of the generations in-between subsequent root nodes.
    # todo: [DONE] Find U unique motifs based on the N motifs, by filtering out motifs that are (subgraph) isomorphic to each other. Always keep the one with the highest number of nodes.
    # todo: [LATER] Goal: Provided a root node, what is the minimal supergraph and root motif split, so that all root motifs are semantically monomorphic to a subgraph of this supergraph (MCS).
    # todo: [DONE] Goal: Find a minimal supergraph that contains a subgraph that is semantically monomorphic to every motif (MCS).
    # todo: [DONE] Extract timings from graph
    # todo: [DONE] Convert scratch_networkx.py to a module
    # todo: [DONE] How to hierarchically save graphs?
    #       1. Save the full graph (unpruned) (G_full).
    #       2. Given a root node and split, save- unique motifs and MCS.
    #       3. Save pruned full graph with supergraph mapping? Or dynamically create that by loading the full graph and the supergraph?
    # todo: [DONE] Combine timings of multiple episodes.
    # todo: [DONE] Provide list of log_pb2.EpisodeRecords to function that identifies MCS
    # todo: [LATER] Create separate function to merge NetworkRecords.
    # todo: [DONE] Create networkx tracer tests.
    # todo: [DONE] Extract tick state per subgraph
    # todo: [DONE] Extract min buffer sizes per node
    # todo: [DONE] min buffer size of 1? Only root may have no "subscribers" --> the case for estimators.
    # todo: [DONE] BUGG? loss or new_loss?
    # todo: [DONE] remove outputs from base.GraphState
    # todo: [DONE] convert eps, step, buffer to update mask and absolute seq's
    # todo: [DONE] set default GraphState.step to 0
    # todo: [DONE] make estimator work again --> Test with perfect initialization (should result in 0 loss).
    # todo: [DONE] verify buffer implementation (vary window, connections, etc.. in dummy env)
    # todo: [DONE] modify tests to get code coverage sufficiently high
    # todo: starting step & episode in double pendulum & pendulum envs
    # todo: Add outputs to state of wrapped nodes
    # todo: default rng = None? --> Problematic for switching between backends (numpy, jax)
    # todo: resolve future error "OSError: handle is closed" --> had to do with new_process and tracer. Shutdown should be handled better.
    # todo: [DONE] speed improvements full rollout jit, vs jit per step
    # todo: maybe, no speed up because policy does not use the observations --> therefore nodes are optimized away.

    env, nodes = build_dummy_env()

    # Simulate
    tstart = time.time()
    graph_state, obs = env.reset(jumpy.random.PRNGKey(0))
    steps = 0
    while True:
        steps += 1
        graph_state, obs, reward, done, info = env.step(graph_state, None)
        if done:
            tend = time.time()
            env.stop()
            print(f"agent_steps={steps} | t={(tend - tstart): 2.4f} sec | fps={steps / (tend - tstart): 2.4f}")
            break

    # Gather the records
    record = env.graph.get_episode_record()

    order = ["world", "sensor", "observer", "agent", "actuator"]
    cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}
    root_name = "agent"
    root_seq = 100
    split_mode = "generational"
    supergraph_mode = "MCS"

    # Load records
    records = [record, record]
    # for i in range(1, 2):
    #     record = log_pb2.EpisodeRecord()
    #     with open(f"eps_record_{i}.pb", "rb") as f:
    #         record.ParseFromString(f.read())
    #     records.append(record)

    # Get network record
    record_network, G_MCS, G, G_subgraphs = tracer.get_network_record(records, root=root_name, seq=root_seq, split_mode=split_mode, supergraph_mode=supergraph_mode, cscheme=cscheme, order=order, log_level=50, workers=None)

    # Convert to topologically sorted subgraphs.
    # G_subgraphs_seq = tracer.as_topological_subgraphs(G_subgraphs[0])

    # Get timings
    # timings = tracer.get_timings_from_network_record(record_network, log_level=50)
    timings = tracer.get_timings_from_network_record(record_network, G, G_subgraphs, log_level=50, workers=None)

    # Get graph buffers
    buffer = tracer.get_graph_buffer(G_MCS, timings, nodes)

    # Determine step_update_mask
    seqs_step, updated_step = tracer.get_step_seqs_mapping(G_MCS, timings, buffer)

    # Get update mask
    # seqs, updated = tracer.get_seqs_mapping(G_MCS, timings, buffer)

    # Get masked_timings
    masked_timings = tracer.get_masked_timings(G_MCS, timings)

    # Initialize dummy env
    graph = CompiledGraph(nodes=nodes, root=nodes["agent"], MCS=G_MCS, default_timings=timings)
    env_mcs = DummyEnv(graph=graph, max_steps=env.max_steps, name="dummy_env_mcs")

    # Simulate
    tstart = time.time()
    graph_state, obs = env_mcs.reset(jumpy.random.PRNGKey(0))
    steps = 0
    while True:
        steps += 1
        graph_state, obs, reward, done, info = env_mcs.step(graph_state, None)
        if done:
            tend = time.time()
            env.stop()
            print(f"agent_steps={steps} | t={(tend - tstart): 2.4f} sec | fps={steps / (tend - tstart): 2.4f}")
            break

    # Get default outputs
    outputs = tracer.get_outputs_from_timings(G_MCS, timings, nodes)

    # Prepare graphs for plotting
    G_subgraphs = G_subgraphs[0]
    G_full = G[0]
    G_MCS_root = G_MCS
    G_motifs = tracer.get_unique_motifs(G_subgraphs)

    if False:
        plot_graph(G_full)
        # plot_graph(G_pruned_edge)
        # plot_graph(G_traced)
        # plot_graph(G_traced_pruned)
        plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
        for root, G_root in G_subgraphs.items():
            # if root in ["root_0", "root_1", "root_2", "root_3", "root_4"]:
            plot_graph(G_root, ax=ax)
        plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
        for root, G_unique in G_motifs.items():
            plot_graph(G_unique, ax=ax)
        plt.show()

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
        plot_graph(G_MCS_root, ax=ax)
        generations = list(nx.topological_generations(G_MCS_root))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="generation", yticks=[], xlim=[-1, len(generations)])
        plt.show()

    exit()

    # Define graph
    with timer("G_full", log_level=50):
        G_full = tracer.create_graph(record)

    # Set edge and node properties
    tracer.set_node_order(G_full, order)
    tracer.set_node_colors(G_full, cscheme)

    # Prune unused edges (due to windowing)
    G_pruned_edge = tracer.prune_edges(G_full)

    # Trace root node
    G_traced = tracer.trace_root(G_pruned_edge, root=root_name, seq=root_seq)

    # Prune unused nodes (not in computation graph of traced root)
    G_traced_pruned = tracer.prune_nodes(G_traced)

    # Define subgraphs
    G_subgraphs = tracer.get_subgraphs(G_traced_pruned, split_mode=split_mode)

    # TODO: REMOVE
    keys = ["agent_0", "agent_2"]
    G_subgraphs["agent_0"] = G_traced_pruned.subgraph(["actuator_0", "sensor_0", "sensor_1", "observer_4", "observer_5", "observer_6", "observer_7"]).copy(
        as_view=False)
    G_subgraphs["agent_2"] = G_traced_pruned.subgraph(["actuator_0", "actuator_1", "actuator_2"]).copy(as_view=False)
    for k in keys:
        for n in G_subgraphs[k].nodes():
            desc = nx.descendants(G_subgraphs[k], n)
            desc.add(n)
            G_desc = G_subgraphs[k].subgraph(desc).copy(as_view=True)
            longest_path_length = nx.dag_longest_path_length(G_desc)
            # longest_path = nx.dag_longest_path(G_desc)
            G_subgraphs[k].nodes[n].update({"sub_longest_path_length": longest_path_length})

    # Determine unique motifs
    G_motifs = tracer.get_unique_motifs(G_subgraphs)

    # Determine minimum common supergraph
    G_MCS = tracer.get_minimum_common_supergraph(G_motifs, max_total_evals=100_000)
    print(f"num_nodes={G_MCS.number_of_nodes()} | num_edges={G_MCS.number_of_edges()}")

    # Verify that all subgraphs are monomorphic with the supergraph
    assert all(tracer.validate_subgraphs(G_MCS, G_subgraphs).values()), "Not all subgraphs are monomorphic with the supergraph."

    # The lower-bound on the optimal solution adds N nodes, where N is the max difference over every mcs and corresponding target graph.
    # The worst-case optimal solution M nodes, where M is the sum of node differences between every mcs and corresponding target graph.
    # Use the relation between Maximal Common Subgraph (mcs) and Minimal Common Supergraph (MCS) to prove the calculation of the minimal common supergraph (MCS).
    # Smartly define search space for the optimal solution (i.e. minimal number of each node type).

    # Save traced network record
    import dill as pickle
    record_network = log_pb2.NetworkRecord()
    record_network.episode.CopyFrom(record)
    record_network.root = root_name
    record_network.seq = root_seq
    record_network.split_mode = split_mode
    with timer("pickle G_full", log_level=50):
        record_network.graph = pickle.dumps(G_full)
    with timer("pickle G_motifs", log_level=50):
        record_network.motifs = pickle.dumps(G_motifs)
    with timer("pickle G_MCS", log_level=50):
        record_network.MCS = pickle.dumps(G_MCS)

    # Save network protobuf record
    with open("network_record.pb", "wb") as f:
        f.write(record_network.SerializeToString())

    # Load network protobuf record
    with open("network_record.pb", "rb") as f:
        record_network = log_pb2.NetworkRecord()
        record_network.ParseFromString(f.read())
        G_full = pickle.loads(record_network.graph)
        G_motifs = pickle.loads(record_network.motifs)
        G_MCS = pickle.loads(record_network.MCS)

    # Get timings
    with timer("get_mappings", log_level=50):
        timings = tracer.get_timings(G_MCS, G_full, G_subgraphs, num_root_steps=root_seq+1, root=root_name)

    if False:
        plot_graph(G_full)
        plot_graph(G_pruned_edge)
        plot_graph(G_traced)
        plot_graph(G_traced_pruned)
        plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
        for root, G_root in G_subgraphs.items():
            # if root in ["root_0", "root_1", "root_2", "root_3", "root_4"]:
            plot_graph(G_root, ax=ax)
        plt.show()

    if False:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
        for root, G_unique in G_motifs.items():
            plot_graph(G_unique, ax=ax)
        plt.show()

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
        plot_graph(G_MCS_root, ax=ax)
        generations = list(nx.topological_generations(G_MCS_root))
        ax.set(facecolor=oc.ccolor("gray"), xlabel="generation", yticks=[], xlim=[-1, len(generations)])
        plt.show()

    input("Press enter to continue...")