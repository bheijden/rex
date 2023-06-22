import time
from functools import partial, lru_cache
from typing import List, Dict, Tuple, Union, Callable, Set, Any
from math import ceil, floor
import numpy as np
import rex.open_colors as oc
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import rex.mcs as rex_mcs
from rex.multiprocessing import new_process
import grandiso

edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
pruned_edge_data = {"color": oc.ecolor.pruned, "linestyle": "--", "alpha": 0.5}
delayed_edge_data = {"color": oc.ecolor.pruned, "linestyle": "-", "alpha": 1.0}


def plot_graph(ax, _G,
               node_size: int = 300,
               node_fontsize=10,
               edge_linewidth=2.0,
               node_linewidth=1.5,
               arrowsize=10,
               arrowstyle="->",
               connectionstyle="arc3,rad=0.1"):
    edges = _G.edges(data=True)
    nodes = _G.nodes(data=True)
    # edge_color = [data['color'] for u, v, data in edges]
    # edge_style = [data['linestyle'] for u, v, data in edges]
    edge_color = [data.get('color', edge_data['color']) for u, v, data in edges]
    edge_alpha = [data.get('alpha', edge_data['alpha']) for u, v, data in edges]
    edge_style = [data.get('linestyle', edge_data['linestyle']) for u, v, data in edges]
    node_alpha = [data['alpha'] for n, data in nodes]
    node_ecolor = [data['edgecolor'] for n, data in nodes]
    node_fcolor = [data['facecolor'] for n, data in nodes]
    node_labels = {n: data["seq"] for n, data in nodes}

    # Get positions
    pos = {n: data["position"] for n, data in nodes}

    # Draw graph
    nx.draw_networkx_nodes(_G, ax=ax, pos=pos, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                           node_size=node_size, linewidths=node_linewidth)
    nx.draw_networkx_edges(_G, ax=ax, pos=pos, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                           arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                           width=edge_linewidth, node_size=node_size)
    nx.draw_networkx_labels(_G, pos, node_labels, ax=ax, font_size=node_fontsize)

    # Set ticks
    # node_order = {data["kind"]: data["position"][1] for n, data in nodes}
    # yticks = list(node_order.values())
    # ylabels = list(node_order.keys())
    # ax.set_yticks(yticks, labels=ylabels)
    # ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def ancestral_partition(G, root_kind):
    # Copy Graph
    G = G.copy(as_view=False)

    # Get & sort root nodes
    roots = {n: data for n, data in G.nodes(data=True) if data["kind"] == root_kind}
    roots = {k: roots[k] for k in sorted(roots.keys(), key=lambda k: roots[k]["seq"])}

    P = {}
    G_partition = {}
    for k, (root, data) in enumerate(roots.items()):
        ancestors = list(nx.ancestors(G, root))
        ancestors_root = ancestors + [root]

        # Get subgraph
        g = nx.DiGraph()
        g.add_node(root, **G.nodes[root])
        for n in ancestors_root:
            g.add_node(n, **G.nodes[n])
        for (o, i, data) in G.edges(ancestors_root, data=True):
            if o in ancestors_root and i in ancestors_root:
                g.add_edge(o, i, **data)
        # Add generation information to nodes
        generations = nx.topological_generations(g)
        for i_gen, gen in enumerate(generations):
            for n in gen:
                g.nodes[n]["generation"] = i_gen
        P[k] = g

        # Get subgraph
        G_partition[k] = G.subgraph(ancestors_root).copy(as_view=False)

        # Remove nodes
        G.remove_nodes_from(ancestors_root)

    # for k in P.keys():
    #     assert P[k].edges() == G_partition[k].edges()
    #     for n, data in P[k].nodes(data=True):
    #         for key, val in data.items():
    #             if key == "generation":
    #                 continue
    #             print(key, val==G_partition[k].nodes[n][key])

    return P


def find_motif(motif: nx.DiGraph, host: nx.DiGraph, interestingness: Dict = None, hints: List[Dict] = None):
    _is_node_attr_match = lambda motif_node_id, host_node_id, motif, host: host.nodes[host_node_id]["kind"] == \
                                                                           motif.nodes[motif_node_id]["kind"]
    _is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: True
    _is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: host.in_degree(
        host_node_id) >= motif.in_degree(motif_node_id) and host.out_degree(host_node_id) >= motif.out_degree(motif_node_id)

    # Cache functions
    _is_node_attr_match = lru_cache()(_is_node_attr_match)
    _is_edge_attr_match = lru_cache()(_is_edge_attr_match)
    _is_node_structural_match = lru_cache()(_is_node_structural_match)

    # Sort nodes by interestingness
    # interestingness = interestingness or uniform_node_interestingness(motif)

    # Make copies
    host = host.copy(as_view=False)
    motif = motif.copy(as_view=False)
    nodes = motif.nodes(data=True)

    # Count number of nodes of each type
    node_kinds: Dict[str, Dict[str, Dict]] = {}
    for n, data in host.nodes(data=True):
        if data["kind"] not in node_kinds:
            node_kinds[data["kind"]] = dict()
        node_kinds[data["kind"]][n] = data

    # Get generations
    host_generations = list(nx.topological_generations(host))
    motif_generations = list(nx.topological_generations(motif))

    #
    host_generations_dict = {n: [[] for _ in range(len(host_generations))] for n in node_kinds}
    for i_gen, gen in enumerate(host_generations):
        for n in gen:
            host_generations_dict[n] = i_gen

    host_generations_dict = {n: i_gen for i_gen, gen in enumerate(host_generations) for n in gen}

    # Prepare
    candidates = []
    for i_gen, gen in enumerate(motif_generations):
        for n in gen:
            data = {"in_degree": motif.in_degree(n),
                    # "out_degree": motif.out_degree(n),
                    # "p_max": None,
                    # "p": None,
                    "g_edge": {}
                    }
            motif.nodes[n].update(data)

            # Start traversing graph from nodes with in_degree=0 (i.e. add them to the candidate list)
            if data["in_degree"] == 0:
                candidates.append(n)
        # if i_gen == 0:
        #     assert all([c==g for c, g in zip(candidates, gen)])

    monomorphism = {}
    while len(candidates) > 0:
        # Get next candidate
        c = candidates.pop(0)
        kind = nodes[c]["kind"]

        # Identify the largest host generation of incoming edges
        assert len(nodes[c]["p_edge"]) == nodes[c]["in_degree"], "The number of edge partitions must be equal to the in_degree of the node."
        start_gen = max(list(nodes[c]["g_edge"].values()) + [0])  # Add 0 to g_edge to account for nodes with in_degree=0

        # Look for a match starting from the identified starting host generation of incoming edges
        for delta_gen, gen in enumerate(host_generations[start_gen:]):
            g = start_gen + delta_gen


def balanced_partition(G, root_kind=None):
    # Copy Graph
    G = G.copy(as_view=False)
    nodes = G.nodes(data=True)

    # Check that graph is DAG
    assert nx.is_directed_acyclic_graph(G), "The graph must be a connected DAG"

    # Check that graph is connected
    assert nx.is_weakly_connected(G), "The graph must be a connected DAG (no separate components)"

    # Count number of nodes of each type
    node_types: Dict[str, Dict[str, Dict]] = {}
    for n, data in G.nodes(data=True):
        if data["kind"] not in node_types:
            node_types[data["kind"]] = dict()
        node_types[data["kind"]][n] = data

    # Sort nodes of each type by sequence number
    for k, v in node_types.items():
        node_types[k] = {k: v[k] for k in sorted(v.keys(), key=lambda k: v[k]["seq"])}

    # Define ideal partition size (either number of root nodes or number of nodes of least common)
    num_partitions = min([len(v) for v in node_types.values()]) if root_kind is None else len(node_types[root_kind])
    size_ideal = {k: -(-len(v) // num_partitions) for k, v in node_types.items()}
    assert min(
        [len(v) for v in node_types.values()]) > 0, "The minimum number of nodes of every type per partition must be > 0."

    # Prepare partitioning
    candidates = []
    for (n, data) in nodes:
        # Initialize every node with its in_degree, p_max=None, and p=None, p_edge=[]
        data_partition = {"in_degree": G.in_degree(n),
                          # "out_degree": G.out_degree(n),
                          "p_max": None,
                          # "p": None,
                          "p_edge": {}}
        data.update(**data_partition)

        # Start traversing graph from nodes with in_degree=0 (i.e. add them to the candidate list)
        if data["in_degree"] == 0:
            candidates.append(n)

    # [OPTIONAL] set max_partition to the partition of e.g. an ancestral partition.
    if root_kind is not None:
        G_ancestral = ancestral_partition(G, root_kind)
        for i, _G in G_ancestral.items():
            for n in _G.nodes:
                G.nodes[n]["p_max"] = i

    # Start traversing graph from nodes with in_degree=0.
    G_partition = {}
    sizes = [size_ideal.copy()]
    while len(candidates) > 0:
        # Get next candidate
        c = candidates.pop(0)
        kind = nodes[c]["kind"]

        # Determine partition of incoming edges
        assert len(nodes[c]["p_edge"]) == nodes[c]["in_degree"], "The number of edge partitions must be equal to the in_degree of the node."
        p_edge = max(list(nodes[c]["p_edge"].values()) + [0])  # Add 0 to p_edge to account for nodes with in_degree=0
        if p_edge >= len(sizes):  # If p is larger than the number of partitions, add new partition
            sizes.append(size_ideal.copy())
        p_balanced = p_edge if sizes[p_edge][kind] > 0 else p_edge + 1

        # [OPTIONAL] Overwrite partition of node if max_partition is set
        p = min(p_balanced, nodes[c]["p_max"]) if nodes[c]["p_max"] is not None else p_balanced

        # Update partition of node
        nodes[c].update(p=p)
        G_partition[p] = G_partition.get(p, []) + [c]

        # Update sizes
        if p >= len(sizes):  # If p is larger than the number of partitions, add new partition
            sizes.append(size_ideal.copy())
        sizes[p][kind] -= 1

        # Iterate over outgoing edges
        for _c, t, data in G.out_edges(c, data=True):
            # Add edge partition
            # [OPTIONAL] To ensure that root is always at the end of the partition, add 1 to p if kind == root_kind
            nodes[t]["p_edge"][c] = p if kind != root_kind else p + 1

            # Add node to candidates if in_degree == len(p_edge)
            if nodes[t]["in_degree"] == len(nodes[t]["p_edge"]):
                candidates.append(t)

    # Sort partitions nodes
    G_partition = {k: sorted(v, key=lambda n: n) for k, v in G_partition.items()}

    # Convert partition to subgraph
    P = {}
    for k, v in G_partition.items():
        g = nx.DiGraph()
        for n in v:
            g.add_node(n, **G.nodes[n])
        for (o, i, data) in G.edges(v, data=True):
            if o in v and i in v:
                g.add_edge(o, i, **data)
        # Add generation information to nodes
        generations = nx.topological_generations(g)
        for i_gen, gen in enumerate(generations):
            for n in gen:
                g.nodes[n]["generation"] = i_gen
        P[k] = g

    # NOTE! Lines below change order of nodes in subgraphs
    Psub = {k: G.subgraph(v).copy(as_view=False) for k, v in G_partition.items()}
    # Check that all nodes in P and Psub are the same
    assert all([sorted(P[k].nodes) == sorted(Psub[k].nodes) for k in P.keys()]), "Not all nodes in P and Psub are the same."
    # Check that all edges in P and Psub are the same
    assert all([sorted(P[k].edges) == sorted(Psub[k].edges) for k in P.keys()]), "Not all edges in P and Psub are the same."
    return P


def get_excalidraw_graph():
    def _get_ts(_f, _T):
        # Get the number of time steps
        _n = floor(_T * _f) + 1

        # Get the time steps
        _ts = np.linspace(0, (_n - 1) / _f, _n)

        return _ts

    order = ["world", "sensor", "actuator", "agent"]
    order = {k: i for i, k in enumerate(order)}
    cscheme = {"world": "grape", "sensor": "pink", "agent": "teal", "actuator": "orange"}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Graph definition
    T = 0.3
    f = dict(world=20, sensor=20, agent=10, actuator=10)
    phase = dict(world=0., sensor=0.2 / 20, agent=0.4 / 20, actuator=1.2 / 20)
    ts = {k: _get_ts(fn, T) + phase[k] for k, fn in f.items()}

    # Initialize a Directed Graph
    G0 = nx.DiGraph()

    # Add nodes
    node_kinds = dict()
    for n in ts.keys():
        node_kinds[n] = []
        for i, _ts in enumerate(ts[n]):
            data = dict(kind=n, seq=i, ts=_ts, order=order[n], edgecolor=ecolor[n], facecolor=fcolor[n],
                        position=(_ts, order[n]), alpha=1.0)
            id = f"{n}_{i}"
            G0.add_node(id, **data)
            node_kinds[n].append(id)

    # Add edges
    edges = [('world', 'sensor'), ('sensor', 'agent'), ('agent', 'actuator'), ('actuator', 'world')]
    for (i, o) in edges:
        for idx, id_seq in enumerate(node_kinds[i]):
            data_source = G0.nodes[id_seq]
            # Add stateful edge
            if idx > 0:
                data = {"delay": 0., "pruned": False}
                data.update(**edge_data)
                G0.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            for id_tar in node_kinds[o]:
                data_target = G0.nodes[id_tar]
                if data_target['ts'] >= data_source['ts']:
                    data = {"delay": 0., "pruned": False}
                    data.update(**edge_data)
                    G0.add_edge(id_seq, id_tar, **data)
                    break


    def get_delayed_graph(_G, _delay, _node_kinds):
        _Gd = _G.copy(as_view=False)

        for (source, target, d) in _delay:
            ts_recv = _Gd.nodes[source]["ts"] + d
            undelayed_data = _Gd.edges[(source, target)]
            target_data = _Gd.nodes[target]
            target_kind = target_data["kind"]
            _Gd.remove_edge(source, target)
            for id_tar in _node_kinds[target_kind]:
                data_target = _Gd.nodes[id_tar]
                if data_target['ts'] >= ts_recv:
                    delayed_data = undelayed_data.copy()
                    delayed_data.update(**{"delay": d, "pruned": False})
                    delayed_data.update(**delayed_edge_data)
                    _Gd.add_edge(source, id_tar, **delayed_data)
                    break
        return _Gd


    # Apply delays
    delays_1 = [("actuator_0", "world_2", 0.9 / 20),
                ("world_1", "sensor_1", 0.9 / 20),
                ("world_4", "sensor_4", 0.9 / 20)]
    G1 = get_delayed_graph(G0, delays_1, node_kinds)

    # Apply delays
    delays_2 = [("sensor_2", "agent_1", 0.9 / 20),
                ("actuator_1", "world_4", 0.9 / 20)]
    G2 = get_delayed_graph(G0, delays_2, node_kinds)

    return G0, G1, G2


def run_excalidraw_example():
    # Get excalidraw graph
    G0, G1, G2 = get_excalidraw_graph()

    # Split
    root_kind = "agent"
    G0_partition = balanced_partition(G0, root_kind)
    G1_partition = balanced_partition(G1, root_kind)
    G2_partition = balanced_partition(G2, root_kind)

    # Get all topological sorts
    G0_partition_topo = {k: list(nx.all_topological_sorts(G0_partition[k])) for k in G0_partition.keys()}
    G1_partition_topo = {k: list(nx.all_topological_sorts(G1_partition[k])) for k in G1_partition.keys()}
    G2_partition_topo = {k: list(nx.all_topological_sorts(G2_partition[k])) for k in G2_partition.keys()}

    # kind_to_index, unique_kinds = get_unique_kinds(G0)

    # G0_partition_onehot = {k: [topo_to_onehot([G0.nodes[n] for n in g], kind_to_index) for g in G0_partition_topo[k]] for k in G0_partition_topo.keys()}

    # Create new plot
    fig, axes = plt.subplots(nrows=3)
    fig.set_size_inches(12, 15)
    plot_graph(axes[0], G0)
    plot_graph(axes[1], G1)
    plot_graph(axes[2], G2)

    fig, axes = plt.subplots(nrows=3)
    fig.set_size_inches(12, 15)
    for idx, G_partition in enumerate([G0_partition, G1_partition, G2_partition]):
        for k, p in G_partition.items():
            # Create new plot
            print(f"Partition {k}")
            plot_graph(axes[idx], p)
    plt.show()

    # Given
    for P in [G0_partition_topo, G1_partition_topo, G2_partition_topo]:
        s = 0
        for k, v in P.items():
            s += len(v)
            print(k, len(v))
        print(s)


def ornstein_uhlenbeck_generator(rng, theta, mu, sigma, dt, x0):
    """Ornstein-Uhlenbeck generator.
    X(t) is the value of the process at time t,
    θ is the speed of reversion to the mean,
    μ is the long-term mean,
    σ is the volatility (i.e. the standard deviation of the changes in the process),
    and dW(t) is a Wiener Process (or Brownian motion).
    """
    X = x0
    while True:
        dW = rng.normal(0, np.sqrt(dt))
        X = X + theta * (mu - X) * dt + sigma * dW
        yield X


def ornstein_uhlenbeck_samples(rng, theta, mu, sigma, dt, x0, n):
    X = np.zeros(n)
    X[0] = x0
    for i in range(1, n):
        dW = rng.normal(0, np.sqrt(dt))
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
    return X


def create_graph(fs: List[float], edges: Set[Tuple[int, int]], T: float, seed: int = 0, theta: float = 0.07, sigma: float = 0.1):
    assert all([T > 1/_f for _f in fs]), "The largest sampling time must be smaller than the simulated time T."

    _all_colors = ['red', 'pink', 'grape', 'violet', 'indigo', 'blue', 'cyan', 'teal', 'green', 'lime', 'yellow', 'orange',  'gray']
    cscheme = {i: _all_colors[i % len(_all_colors)] for i in range(len(fs))}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Function body
    rng = np.random.default_rng(seed=seed)
    fn_dt = [partial(ornstein_uhlenbeck_samples, rng=rng, theta=theta, mu=1 / f, sigma=sigma / f, dt=1 / f, x0=1 / f) for f in fs]
    dt = [np.stack((np.linspace(1/f, T, ceil(T * f)), np.clip(fn(n=ceil(T * f)), 1/f, np.inf))) for f, fn in zip(fs, fn_dt)]
    ts = [np.stack((_dt[0], np.cumsum(_dt[1]))) for _dt in dt]
    ts = [np.concatenate((np.array([[0.], [0.]]), _ts), axis=1) for _ts in ts]

    # Find where entries ts are ts < min_ts
    min_ts = np.min([_ts[1][-1] for _ts in ts])
    idx = [np.where(_ts[1] < min_ts)[0] for _ts in ts]
    ts = [_ts[:, _idx] for _idx, _ts in zip(idx, ts)]

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # [ax.plot(*_dt) for _dt in dt]
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # [ax.plot(_ts[0], _ts[1]) for _ts in ts]
    #
    # plt.show()

    # Initialize a Directed Graph
    G = nx.DiGraph()

    # Add nodes
    node_kinds = dict()
    for n, (_f, _ts) in enumerate(zip(fs, ts)):
        node_kinds[n] = []
        for i, __ts in enumerate(_ts[1]):
            data = dict(kind=n, seq=i, ts=__ts, order=n, edgecolor=ecolor[n], facecolor=fcolor[n],
                        position=(__ts, n), alpha=1.0)
            id = f"{n}_{i}"
            G.add_node(id, **data)
            node_kinds[n].append(id)

    # Add edges
    for (i, o) in edges:
        for idx, id_seq in enumerate(node_kinds[i]):
            data_source = G.nodes[id_seq]
            if idx > 0 and i == o:  # Add stateful edge
                data = {"delay": 0., "pruned": False}
                data.update(**edge_data)
                G.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            else:
                for id_tar in node_kinds[o]:
                    data_target = G.nodes[id_tar]
                    if i < o:
                        must_connect = data_target['ts'] >= data_source['ts']
                    else:  # To break cycles
                        must_connect = data_target['ts'] > data_source['ts']
                    if must_connect:
                        data = {"delay": 0., "pruned": False}
                        data.update(**edge_data)
                        G.add_edge(id_seq, id_tar, **data)
                        break

    # Check that G is a DAG
    assert nx.is_directed_acyclic_graph(G), "The graph is not a DAG."
    return G


def get_set_of_feasible_edges(P: Dict[int, nx.DiGraph]) -> Set[Tuple[int, int]]:
    # Determine all feasible edges between node types (E_val) = MCS
    # [OPTIONAL] provide E_val as input instead of looking at all partitions --> Don't forget state-full edges.
    E_val = set()
    for k, p in P.items():
        for o, i in p.edges:
            i_type = p.nodes[i]["kind"]
            o_type = p.nodes[o]["kind"]
            E_val.add((o_type, i_type))
    return E_val


def as_MCS(P: nx.DiGraph, E_val: Set[Tuple[int, int]], num_topo: int = 1, as_tc: bool = False) -> Tuple[nx.DiGraph, Dict[Any, Any]]:
    # Grab topological sort of P and add all feasible edges (TC(topo(P)) | e in E_val) = MCS
    assert num_topo > 0, "num_topo must be greater than 0."
    gen_all_sorts = nx.all_topological_sorts(P)
    MCS, monomorphism = sort_to_MCS(P, next(gen_all_sorts), E_val, as_tc=as_tc)

    # [OPTIONAL] Repeat for all topological sorts of P and pick the one with the most feasible edges.
    for i in range(1, num_topo):
        try:
            new_MCS, new_monomorphism = sort_to_MCS(P, next(gen_all_sorts), E_val, as_tc=as_tc)
        except StopIteration:
            break
        # print(MCS.number_of_edges(), new_MCS.number_of_edges())
        if new_MCS.number_of_edges() > MCS.number_of_edges():
            MCS = new_MCS
            monomorphism = new_monomorphism
    return MCS, monomorphism


def sort_to_MCS(P, sort, E_val, as_tc: bool = False) -> Tuple[nx.DiGraph, Dict[Any, Any]]:
    attribute_set = {"kind", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {P.nodes[n]["kind"]: data for n, data in P.nodes(data=True)}
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}
    MCS = nx.DiGraph()
    slots = {k: 0 for k in kinds}
    monomorphism = dict()
    for n in sort:
        k = P.nodes[n]["kind"]
        s = slots[k]
        # Add monomorphism map
        name = f"{k}_s{s}"
        monomorphism[n] = name
        # Add node and data
        data = kinds[k].copy()
        data.update({"seq": s})
        MCS.add_node(name, **data)
        # Increase slot count
        slots[k] += 1

    # Add feasible edges
    for i, n_out in enumerate(sort):
        name_out = monomorphism[n_out]
        for j, n_in in enumerate(sort[i+1:]):
            name_in = monomorphism[n_in]
            e_P = (n_out, n_in)  # Edge in P
            e_MCS = (name_out, name_in)  # Corresponding edge in MCS
            e_kind = (MCS.nodes[name_out]["kind"], MCS.nodes[name_in]["kind"])  # Kind of edge

            # Add edge if it is in P or if we are adding all feasible edges
            if e_P in P.edges:
                MCS.add_edge(*e_MCS, **edge_data)
            elif as_tc and e_kind in E_val:
                MCS.add_edge(*e_MCS, **edge_data)

    # Set positions of nodes
    generations = list(nx.topological_generations(MCS))
    for i_gen, gen in enumerate(generations):
        for i_node, n in enumerate(gen):
            MCS.nodes[n]["position"] = (i_gen, i_node)
            MCS.nodes[n]["generation"] = i_gen
    return MCS, monomorphism


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


def uniform_node_interestingness(motif: nx.Graph) -> dict:
    """
    Sort the nodes in a motif by their interestingness.

    Most interesting nodes are defined to be those that most rapidly filter the
    list of nodes down to a smaller set.

    """
    return {n: 1 for n in motif.nodes}




if __name__ == "__main__":
    # todo: Perform transitive reduction on P before matching? (i.e. remove redundant edges)
    # todo: Check both in and out degree in _is_node_structural_match for finding exact monomorphism.
    # todo: Remove degree constraint in _is_node_structural_match for finding largest monomorphism.
    # todo: Determine starting pattern
    #  - Determine all valid edges between node types (E_val) = MCS
    #  - Select partition P that has the most edges (i.e. edges are interpreted as constraints).
    #  - Grab topological sort of P and add all feasible edges (TC(topo(P)) | e in E_val) = MCS
    #  - [OPTIONAL] Repeat for all topological sorts of P and pick the one with the most feasible edges.
    #  - Sort partitions by the number of edges.
    #  - Iterate over partitions and find largest (subgraph) monomorphism with MCS.
    #  - If perfect match, store mapping (P -> MCS).
    #  - If no subgraph monomorphism, then merge MCS with with non-matched nodes in partition.
    #       - Grab topological sort of merged(MCS, P) and add all feasible edges (TC(topo(P)) | e in E_feas) = new_MCS
    #       - Remap mappings from MCS to new_MCS.
    #       - Add mapping (P -> new_MCS).
    #  - Repeat until all partitions are matched.

    # run_excalidraw_example()

    # Record depth of each node in partition
    # Check tentative_candidates

    # Function inputs
    MUST_PLOT = False
    SEED = 29  # 50
    THETA = 0.07
    SIGMA = 0.3
    NUM_NODES = 3  # 7
    T = 100

    ROOT_KIND = None
    AS_TC = True
    NUM_TOPO = 1
    MAX_EVALS = 10000

    # Define graph
    fs = [float(i) for i in range(1, NUM_NODES + 1)]
    edges = {(i, (i + 1) % NUM_NODES) for i in range(NUM_NODES-1)}  # Add forward edges
    edges.update({(j, i) for i, j in edges})  # Add reverse edges
    edges.update({(i, i) for i in range(NUM_NODES)})  # Stateful edges

    # Create graph
    G = create_graph(fs, edges, T, seed=SEED, theta=THETA, sigma=SIGMA)

    # Partition
    P = balanced_partition(G, root_kind=ROOT_KIND)

    # Record original mapping
    partition_order = list(P.keys())

    # Determine all feasible edges between node types (E_val) = MCS
    E_val = get_set_of_feasible_edges(P)

    # Select partition P that has the most edges (i.e. edges are interpreted as constraints).
    key_MCS, P_MCS = sorted(P.items(), key=lambda item: item[1].number_of_edges(), reverse=True)[0]

    # Determine starting pattern. monomorphism = {P_node: MCS_0_node}
    MCS, monomorphism = as_MCS(P_MCS, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)

    # Iterate over partitions sorted by the number of edges in descending order.
    P_sorted = {k: P for k, P in sorted(P.items(), key=lambda item: item[1].number_of_edges(), reverse=True)}

    # Find monomorphism
    # find_motif(P_sorted[102], MCS)

    # Define graph isomorphism function
    _is_node_attr_match = lambda motif_node_id, host_node_id, motif, host: host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]
    _is_node_attr_match = lru_cache()(_is_node_attr_match)
    _is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: True
    _is_edge_attr_match = lru_cache()(_is_edge_attr_match)
    # _L_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: (host.nodes[host_edge_id[0]]["generation"] - host.nodes[host_edge_id[1]]["generation"]) >= (motif.nodes[motif_edge_id[0]]["generation"] - motif.nodes[motif_edge_id[1]]["generation"])
    _L_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: host.nodes[host_edge_id[0]]["generation"] < host.nodes[host_edge_id[1]]["generation"]
    # _L_is_edge_attr_match = lru_cache()(_L_is_edge_attr_match)
    _L_is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: True
    # _L_is_node_structural_match = lru_cache()(_L_is_node_structural_match)
    _is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: host.in_degree(host_node_id) >= motif.in_degree(motif_node_id) and host.out_degree(host_node_id) >= motif.out_degree(motif_node_id)
    _is_node_structural_match = lru_cache()(_is_node_structural_match)

    if MUST_PLOT:
        fig, axes = plt.subplots(nrows=2)
        fig.set_size_inches(12, 15)
        plot_graph(axes[0], MCS)
        plt.show()

    # Find MCS
    metric = {"match": 0, "no_match": 0}
    MCS_history: Dict[nx.DiGraph, int] = {MCS: 0}
    MCS_to_MCS: Dict[nx.DiGraph, Dict[nx.DiGraph, Dict[str, str]]] = {}
    P_to_MCS: Dict[int, Dict[nx.DiGraph, Dict[str, str]]] = {}
    for k, P in P_sorted.items():

        # Find subgraph monomorphism
        gr_start = time.time()
        mono = grandiso.find_motifs(P, MCS, limit=1, directed=True,
                                    is_node_attr_match=_is_node_attr_match,
                                    is_edge_attr_match=_is_edge_attr_match,
                                    is_node_structural_match=_is_node_structural_match)
        t_find_motifs = time.time() - gr_start

        if len(mono) > 0:  # Perfect match found! Save mapping: P -> subgraph(MCS)
            P_monomorphism = mono[0]
            P_to_MCS[k] = {MCS: P_monomorphism}
            metric["match"] += 1
            print(f"k={k} | Match    | find={t_find_motifs:.3f} sec | nodes=({MCS.number_of_nodes()}/{P.number_of_nodes()}) | edges=({MCS.number_of_edges()}/{P.number_of_edges()})")
        else:  # No match. Merge MCS with non-matched nodes in partition.
            rex_start = time.time()
            num_evals, is_match, largest_mono_lst = rex_mcs.find_largest_motifs(P, MCS,
                                                                                max_evals=MAX_EVALS,
                                                                                queue_=rex_mcs.Deque(policy=rex_mcs.QueuePolicy.BREADTHFIRST),
                                                                                is_node_attr_match=_is_node_attr_match,
                                                                                is_edge_attr_match=_L_is_edge_attr_match,
                                                                                is_node_structural_match=_L_is_node_structural_match)
            largest_mono = largest_mono_lst[0]
            t_find_large_motifs = time.time() - rex_start

            assert not is_match, "Should not have found a match."
            unmatched_nodes = len(P.nodes) - max([len(largest_mono), 0])

            # Prepare to unify MCS and P
            mcs = {v: k for k, v in largest_mono.items()}
            mcs_MCS = MCS.subgraph(mcs.keys())
            mcs_P = nx.relabel_nodes(MCS.subgraph(mcs.keys()), mcs, copy=True)
            E1 = emb(mcs_MCS, MCS)
            E2 = emb(mcs_P, P)
            tmp = nx.relabel_nodes(unify(mcs_P, P, E2), largest_mono, copy=True)
            new_P_MCS = unify(tmp, MCS, E1)

            if not nx.is_directed_acyclic_graph(new_P_MCS):
                print(largest_mono)
                # Color nodes
                for k, v in largest_mono.items():
                    new_P_MCS.nodes[v].update({"edgecolor": "green"})
                    MCS.nodes[v].update({"edgecolor": "green"})
                    P.nodes[k].update({"edgecolor": "green"})
                # Color edges
                for u, v in P.edges:
                    if u in largest_mono and v in largest_mono:
                        MCS.edges[(largest_mono[u], largest_mono[v])].update({"color": "green"})
                        new_P_MCS.edges[(largest_mono[u], largest_mono[v])].update({"color": "green"})
                        P.edges[(u, v)].update({"color": "green"})
                # Find cycles
                cycle = list(nx.find_cycle(new_P_MCS))
                print(cycle)
                for u, v in cycle:
                    if new_P_MCS[u][v]["color"] == "green":
                        continue
                    new_P_MCS[u][v]["color"] = "red"
                # Set edge alpha
                for u, v in new_P_MCS.edges:
                    if new_P_MCS[u][v]["color"] in ["green", "red"]:
                        new_P_MCS[u][v]["alpha"] = 1.0
                    else:
                        new_P_MCS[u][v]["alpha"] = 0.
                for u, v in MCS.edges:
                    if MCS[u][v]["color"] in ["green", "red"]:
                        MCS[u][v]["alpha"] = 1.0
                    else:
                        MCS[u][v]["alpha"] = 0.3
                for u, v in P.edges:
                    if P[u][v]["color"] in ["green", "red"]:
                        P[u][v]["alpha"] = 1.0
                    else:
                        P[u][v]["alpha"] = 0.5

                # Color cycles
                fig, axes = plt.subplots(nrows=3)
                fig.set_size_inches(12, 15)
                plot_graph(axes[0], P)
                plot_graph(axes[1], MCS)
                plot_graph(axes[2], new_P_MCS)
                plt.show()

            assert nx.is_directed_acyclic_graph(new_P_MCS), "Should be a DAG."

            new_MCS, monomorphism = as_MCS(new_P_MCS, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)

            # Determine P_monomorphism
            P_monomorphism = {n: None for n in P.nodes}
            for k, v in P_monomorphism.items():
                if k in largest_mono:
                    P_monomorphism[k] = monomorphism[largest_mono[k]]
                elif k in monomorphism:
                    P_monomorphism[k] = monomorphism[k]

            # Verify that P_monomorphism is a valid mapping
            # They hint cannot already be a perfect match --> limitation of find_motifs.
            hints = [{k: v for idx, (k, v) in enumerate(P_monomorphism.items()) if idx > 0}]
            mono = grandiso.find_motifs(P, new_MCS, limit=1, directed=True, is_node_attr_match=_is_node_attr_match,
                                        is_edge_attr_match=_is_edge_attr_match, hints=hints)
            assert len(mono) > 0, "P_monomorphism is not a valid mapping."
            P_to_MCS[k] = {new_MCS: P_monomorphism}

            # Filter out added nodes from MCS_monomorphism
            MCS_monomorphism = {k: v for k, v in monomorphism.items() if k in MCS.nodes}

            # Remap mappings from MCS to new_MCS.
            idx_MCS = len(MCS_history)  # NOTE! not thread safe
            MCS_history[new_MCS] = idx_MCS
            MCS_to_MCS[MCS] = {new_MCS: MCS_monomorphism}

            # Update MCS
            # MCS = new_MCS
            metric["no_match"] += 1
            print(f"k={k} | No Match | find={t_find_motifs:.3f} sec | nodes=({MCS.number_of_nodes()}/{P.number_of_nodes()}) | edges=({MCS.number_of_edges()}/{P.number_of_edges()}) | large={t_find_large_motifs:.3f} sec | num_evals={num_evals} | unmatched_nodes={unmatched_nodes}")
    print(f"Matched {metric['match']} | No Match {metric['no_match']}")

    exit()

    metric = {"match": 0, "no_match": 0}
    for k, P in P_sorted.items():

        # Time function
        gr_start = time.time()
        queue = rex_mcs.Deque(policy=rex_mcs.QueuePolicy.BREADTHFIRST)
        try:
            x = next(grandiso.find_motifs_iter(P, MCS, directed=True, is_node_attr_match=_is_node_attr_match, is_edge_attr_match=_is_edge_attr_match))
            gr_match = True
        except StopIteration:
            gr_match = False
        gr_end = time.time()

        # Time function
        rex_start = time.time()
        queue = rex_mcs.Deque(policy=rex_mcs.QueuePolicy.BREADTHFIRST)
        num_evals, rex_match, largest_motif = rex_mcs.find_largest_motifs(P, MCS, queue_=queue, is_node_attr_match=_is_node_attr_match, is_edge_attr_match=_is_edge_attr_match)
        unmatched_nodes = len(P.nodes) - max([len(m) for m in largest_motif]+[0])
        rex_end = time.time()

        # Time function
        nx_start = time.time()
        try:
            matcher = isomorphism.GraphMatcher(MCS, P, node_match=isomorphism.categorical_node_match(attr="kind", default=None))
            x = next(matcher.subgraph_monomorphisms_iter())
            nx_match = True
            # nx_match = rex_match
        except StopIteration:
            nx_match = False
        nx_end = time.time()

        assert gr_match == rex_match, f"gr_match={gr_match} | rex_match={rex_match}"
        # assert gr_match == nx_match, f"gr_match={gr_match} | nx_match={nx_match}"
        if gr_match != nx_match:
            print(f"k={k} | gr_match={gr_match} | nx_match={nx_match}")

        if nx_match:
            metric["match"] += 1
            print(f"Match    | nx={nx_end-nx_start:.3f} sec | gr={gr_end-gr_start:.3f} sec | rex={rex_end-rex_start:.3f} sec | key={k} | nodes={P.number_of_nodes()} | edges={P.number_of_edges()}")
        else:
            metric["no_match"] += 1
            print(f"No Match | nx={nx_end-nx_start:.3f} sec | gr={gr_end-gr_start:.3f} sec | rex={rex_end-rex_start:.3f} sec | key={k} | nodes={P.number_of_nodes()} | edges={P.number_of_edges()} | unmatched_nodes={unmatched_nodes}")
    print(metric)
    input("Press Enter to continue...")
    exit()

    # # Define transitive closure (not perfect, because it does not add possible backward edges)
    # # Instead, should make transitive closure of each topo sort
    # G_tc = {}
    # for p_idx, g in G_partition.items():
    #     tc = g.copy()
    #     for o in tc:
    #         descendants = nx.descendants(g, o)
    #         tc_edges = [(o, i) for i in descendants if o not in tc[o] and (g.nodes[o]["kind"], g.nodes[i]["kind"]) in edge_set]
    #         tc.add_edges_from(tc_edges, **edge_data)
    #     G_tc[p_idx] = tc
    #
    # # Create new plot
    # fig, axes = plt.subplots(nrows=3)
    # fig.set_size_inches(12, 15)
    # plot_graph(axes[0], G)
    #
    # for idx, G_partition in enumerate([G_partition]):
    #     for k, p in G_partition.items():
    #         # Create new plot
    #         print(f"Partition {k}")
    #         plot_graph(axes[idx+1], p)
    #         plot_graph(axes[idx+2], G_tc[k])
    #         # tc = nx.transitive_closure(p)
    #         # plot_graph(axes[idx+3], tc)
    #
    # plt.show()


