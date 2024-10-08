from typing import Any, Tuple, List, TypeVar, Dict, Union, Callable, TYPE_CHECKING
import functools

import distrax
import networkx as nx
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
import time

import supergraph
from supergraph import EDGE_DATA
from supergraph import open_colors as oc
from rex.base import Graph, WindowedGraph, Window, Vertex, Edge, WindowedVertex, Timings, SlotVertex, TrainableDist, NodeInfo
from rex.constants import LogLevel, Jitter

from threading import current_thread
from os import getpid
from termcolor import colored
import matplotlib
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from rex.node import BaseNode


# Global log levels
LOG_LEVEL = LogLevel.WARN
NODE_LOG_LEVEL = {}
NODE_COLOR = {}
NODE_LOGGING_ENABLED = True


def log(
    name: str,
    color: str,
    log_level: int,
    id: str,
    msg=None,
):
    if log_level >= LOG_LEVEL:
        # Add process ID, thread ID, name (somewhat expensive)
        log_msg = f"[{str(getpid())[:5].ljust(5)}][{current_thread().name.ljust(25)}][{str(name).ljust(20)}][{str(id).ljust(20)}]"
        if msg is not None:
            log_msg += f" {msg}"
        print(colored(log_msg, color))


def set_log_level(log_level: int, node: "BaseNode" = None, color: str = None):
    if node is not None:
        NODE_LOG_LEVEL[node] = log_level
        if color is not None:
            NODE_COLOR[node] = color
    else:
        global LOG_LEVEL
        LOG_LEVEL = log_level
        assert color is None, "Cannot set color without node"


def apply_window(nodes: Dict[str, "BaseNode"], graphs: Graph) -> WindowedGraph:
    """Apply the window to the edges."""

    @struct.dataclass
    class IndexedWindow(Window):
        seq_in: Union[int, jax.Array]

        def to_window(self) -> Window:
            return Window(seq=self.seq, ts_sent=self.ts_sent, ts_recv=self.ts_recv)

    def _scan_body(vertex: Vertex, window: Window, edge: Edge):
        ts_sent = jnp.take(vertex.ts_end, edge.seq_out)  # vertex.ts_end[edge.seq_out]
        ts_recv = edge.ts_recv
        seq = edge.seq_out
        seq_in = jnp.where(edge.seq_out == -1, -1, edge.seq_in)
        new_window = window.push(seq, ts_sent, ts_recv)
        return new_window, IndexedWindow(new_window.seq, new_window.ts_sent, new_window.ts_recv, seq_in)

    def _apply_window(graph):
        windows = dict()

        for (n1, n2), e in graph.edges.items():
            c = nodes[n1].outputs[n2]  # Connection
            if isinstance(c.delay_dist, TrainableDist):
                if c.blocking is True:
                    raise NotImplementedError("Cannot have trainable distribution for blocking connection.")
                if c.jitter is Jitter.BUFFER:
                    raise NotImplementedError("Cannot have trainable distribution for jitter buffer.")

            # Initialize window (window + window caused by trainable delay)
            win = c.window + c.delay_dist.window(nodes[n1].rate)
            seq = jnp.array([-1] * win, dtype=onp.int32)
            ts_sent = jnp.array([0.0] * win, dtype=onp.float32)
            ts_recv = jnp.array([0.0] * win, dtype=onp.float32)
            init_window = Window(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv)
            indexed_init_window = IndexedWindow(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, seq_in=-1)

            # Get all windows
            scan_body_seq = functools.partial(_scan_body, graph.vertices[n1])
            last_window, indexed_windows = jax.lax.scan(scan_body_seq, init_window, e)

            # Append init_window
            extended_indexed_windows = jax.tree_map(
                lambda w_lst, w: jnp.concatenate([w_lst, jnp.array(w)[None]]), indexed_windows, indexed_init_window
            )

            # Replace -1 with largest seq_in so that it can never be selected
            win_seq_in = jnp.where(indexed_windows.seq_in == -1, jnp.array(2**31 - 1, dtype=int), indexed_windows.seq_in)
            indexed_windows = indexed_windows.replace(seq_in=win_seq_in)

            def _get_window_index(_seq):
                reversed_seq_in = jnp.flip(indexed_windows.seq_in)
                idx = jnp.argwhere(reversed_seq_in <= _seq, size=1, fill_value=-1)[0, 0]
                idx = jnp.where(
                    idx != -1, indexed_windows.seq_in.shape[0] - idx - 1, idx
                )  # -1 to account for 0-based indexing
                return idx

            # Take windows based on indices
            win_indices = jax.vmap(_get_window_index)(graph.vertices[n2].seq)

            # Append
            window = jax.tree_map(lambda w: w[win_indices], extended_indexed_windows)
            windows[(n1, n2)] = window

        vertices = dict()
        for n, v in graph.vertices.items():
            vertex_windows = {
                c.output_node.name: windows[(c.output_node.name, c.input_node.name)].to_window()
                for c in nodes[n].inputs.values()
            }
            vertices[n] = WindowedVertex(seq=v.seq, ts_start=v.ts_start, ts_end=v.ts_end, windows=vertex_windows)
        return WindowedGraph(vertices=vertices)

    if next(iter(graphs.vertices.values())).seq.ndim == 1:
        windowed_graphs = _apply_window(graphs)
        return windowed_graphs
    else:
        windowed_graphs = jax.vmap(_apply_window, in_axes=0)(graphs)
        return windowed_graphs


def to_networkx_graph(graph: Graph, nodes: Dict[str, "BaseNode"] = None, validate: bool = False) -> nx.DiGraph:
    graph = jax.tree_util.tree_map(lambda x: onp.array(x), graph)
    order = {n: nodes[n].order for n in nodes} if nodes is not None else {n: None for n in enumerate(graph.vertices.keys())}
    order_filter = list(filter(None, order.values()))
    max_val = max(order_filter) if len(order_filter) > 0 else 0
    increment = max_val + 1
    for key in order:
        if order[key] is None:
            order[key] = increment
            increment += 1
    colors = {n: nodes[n].color for n in nodes} if nodes is not None else {n: "gray" for n in enumerate(graph.vertices.keys())}
    colors = {n: c if isinstance(c, str) else "gray" for n, c in colors.items()}
    ecolors, fcolors = oc.cscheme_fn(colors)

    # Create networkx graph
    G = nx.DiGraph()

    # Add vertices
    for n, v in graph.vertices.items():
        static_data = dict(kind=n, facecolor=fcolors[n], edgecolor=ecolors[n], order=order[n])
        for seq, ts_start, ts_end in zip(v.seq, v.ts_start, v.ts_end):
            if seq == -1:
                continue
            vname = f"{n}_{seq}"
            position = (ts_start, order[n])
            G.add_node(vname, seq=seq, ts=ts_start, ts_start=ts_start, ts_end=ts_end, position=position, **static_data)

            if seq > 0:  # Adds stateful edges between consecutive vertices of the same kind
                uname = f"{n}_{seq-1}"
                G.add_edge(uname, vname)

    # Add edges
    for (n1, n2), e in graph.edges.items():
        for seq_out, seq_in, ts_recv in zip(e.seq_out, e.seq_in, e.ts_recv):
            if seq_out == -1 or seq_in == -1:
                continue
            u = f"{n1}_{seq_out}"
            v = f"{n2}_{seq_in}"
            if validate:
                # if v == "world_60":
                #     print(f"Adding edge {u} -> {v}")
                assert u in G.nodes, f"Node {u} not found in graph"
                assert v in G.nodes, f"Node {v} not found in graph"
            G.add_edge(u, v, ts_recv=ts_recv)

    return G


def to_connected_graph(G: nx.DiGraph, supervisor: "BaseNode", nodes: Dict[str, "BaseNode"] = None, validate: bool = False) -> nx.DiGraph:
    # todo: Get all nodes that are not ancestors of a supervisor vertex
    # todo: Determine ts_start for all supervisor vertices
    # todo: Connect every non-ancestor to a supervisor vertex such that ts_end of the non-ancestor <= ts_start of the supervisor vertex
    G = G.copy()
    nodes_sup = {n: data for n, data in G.nodes(data=True) if data["kind"] == supervisor.name}
    nodes_sup = sorted(nodes_sup, key=lambda x: G.nodes[x]["ts_start"])
    ancestors = nx.ancestors(G, nodes_sup[-1])
    ancestors.add(nodes_sup[-1])
    non_ancestors = set(G.nodes) - ancestors
    non_ancestors = sorted(non_ancestors, key=lambda x: G.nodes[x]["ts_end"])
    for n_sup in nodes_sup:
        if len(non_ancestors) == 0:
            break
        while len(non_ancestors) > 0:
            n_non = G.nodes[non_ancestors[0]]
            if n_non["ts_end"] <= G.nodes[n_sup]["ts_start"]:
                G.add_edge(non_ancestors[0], n_sup)
                non_ancestors.pop(0)
            else:
                break
    return G


def to_timings(
    graphs: WindowedGraph, S: nx.DiGraph, Gs: List[nx.DiGraph], Gs_monomorphism: List[Dict[str, str]], supervisor: str
) -> Timings:
    # Convert graphs to numpy
    graphs = jax.tree_util.tree_map(lambda val: onp.array(val), graphs)

    # Determine number of episodes
    num_episodes = graphs.vertices[supervisor].seq.shape[0]
    # The smallest number of partitions across all episodes is the number of partitions
    num_partitions = onp.where(graphs.vertices[supervisor].seq.min(axis=-2) >= 0, 1, 0).sum()
    # num_partitions = graphs.vertices[supervisor].seq.shape[-1]

    # Prepare template for timings (that we fill in later with data according to Gs_monomorphism and graphs)
    slots = dict()
    timings = Timings(slots=slots)
    generations = list(supergraph.topological_generations(S))
    for idx_gen, gen in enumerate(generations):
        for s2 in gen:
            data = S.nodes[s2]
            kind = data["kind"]
            v = graphs.vertices[kind]
            # Prepare masked slot data (later we will fill in the data)
            run = onp.zeros((num_episodes, num_partitions)).astype(bool)
            seq_in = onp.zeros((num_episodes, num_partitions)).astype(int)
            ts_start = onp.zeros((num_episodes, num_partitions)).astype(float)
            ts_end = onp.zeros((num_episodes, num_partitions)).astype(float)
            # Replace windows with zeros
            windows = dict()
            for n1, w in v.windows.items():
                num_win = w.seq.shape[-1]
                seq_out = -onp.ones((num_episodes, num_partitions, num_win)).astype(int)
                ts_sent = onp.zeros((num_episodes, num_partitions, num_win)).astype(float)
                ts_recv = onp.zeros((num_episodes, num_partitions, num_win)).astype(float)
                windows[n1] = Window(seq=seq_out, ts_sent=ts_sent, ts_recv=ts_recv)
            slot = SlotVertex(
                seq=seq_in, ts_start=ts_start, ts_end=ts_end, windows=windows, run=run, kind=kind, generation=idx_gen
            )
            slots[s2] = slot

    # Gather indices to fill in the timings
    fill_idx = {s2: dict(slots=[], fill=[]) for s2 in timings.slots.keys()}
    for eps_idx, G_monomorphism in enumerate(Gs_monomorphism):
        G = Gs[eps_idx]
        for n2, (partition_idx, s2) in G_monomorphism.items():
            # print(f"eps_idx={eps_idx} | partition_idx={partition_idx} | s2={s2}")
            if not partition_idx < num_partitions:  # Skip if partition index is out of bounds
                continue
            data = G.nodes[n2]
            seq = int(data["seq"])
            fill_idx[s2]["slots"].append((eps_idx, partition_idx))
            fill_idx[s2]["fill"].append((eps_idx, seq))

    # Fill in the timings
    for key, indices in fill_idx.items():
        slot = timings.slots[key]
        v = graphs.vertices[slot.kind]
        fill_idx = onp.array(indices["fill"])
        slot_idx = onp.array(indices["slots"])
        if len(fill_idx) == 0:  # Some slots may never be used, so we skip them to avoid IndexErrors
            continue
        slot.seq[slot_idx[:, 0], slot_idx[:, 1]] = v.seq[fill_idx[:, 0], fill_idx[:, 1]]
        slot.ts_start[slot_idx[:, 0], slot_idx[:, 1]] = v.ts_start[fill_idx[:, 0], fill_idx[:, 1]]
        slot.ts_end[slot_idx[:, 0], slot_idx[:, 1]] = v.ts_end[fill_idx[:, 0], fill_idx[:, 1]]
        slot.run[slot_idx[:, 0], slot_idx[:, 1]] = True
        for n1, w in v.windows.items():
            window = slot.windows[n1]
            window.seq[slot_idx[:, 0], slot_idx[:, 1]] = w.seq[fill_idx[:, 0], fill_idx[:, 1]]
            window.ts_sent[slot_idx[:, 0], slot_idx[:, 1]] = w.ts_sent[fill_idx[:, 0], fill_idx[:, 1]]
            window.ts_recv[slot_idx[:, 0], slot_idx[:, 1]] = w.ts_recv[fill_idx[:, 0], fill_idx[:, 1]]
    return timings


def check_generations_uniformity(generations: List[Dict[str, SlotVertex]]):
    """
    Checks if all generations have the same kinds of nodes and the same number of instances of each kind.

    :param generations: A list of generations, where each generation is a set of node IDs.
    :return: True if all generations are uniform in terms of node kinds and their counts, False otherwise.
    """

    # Dictionary to store the kind count of the first generation
    first_gen_kind_count = None

    for gen in generations:
        gen_kind_count = dict()
        for node_id, v in gen.items():
            kind = v.kind
            gen_kind_count[kind] = gen_kind_count.get(kind, 0) + 1

        if first_gen_kind_count is None:
            first_gen_kind_count = gen_kind_count
        else:
            if gen_kind_count != first_gen_kind_count:
                return False

    return True


def mixture_distribution_quantiles(dist, probs, N_grid_points: int = int(1e3), grid_min: float = None, grid_max: float = None):
    """More info: https://github.com/tensorflow/probability/issues/659"""
    base_grid = onp.linspace(grid_min, grid_max, num=int(N_grid_points))
    shape = (dist.batch_shape, 1) if len(dist.batch_shape) else [1]
    full_grid = onp.transpose(onp.tile(base_grid, shape))
    try:
        cdf_grid = dist.cdf(full_grid)  # this is fully parallelized and even uses GPU
    except NotImplementedError as e:
        if isinstance(dist, distrax.MixtureSameFamily):
            cdist = dist.components_distribution
            cweights = dist.mixture_distribution.probs
            cdist_cdfs = cdist.cdf(full_grid[..., None])
            cdf_grid = onp.sum(cdist_cdfs * cweights[None], axis=-1)
        else:
            raise e

    grid_check = (cdf_grid.min(axis=0).max() <= min(probs)) & (max(probs) <= cdf_grid.max(axis=0).min())
    if not grid_check:
        print(f"Grid min: {grid_min}, max: {grid_max} | CDF min: {cdf_grid.min(axis=0).max()}, max: {cdf_grid.max(axis=0).min()} | Probs min: {min(probs)}, max: {max(probs)}")
        raise RuntimeError("Grid does not span full CDF range needed for interpolation!")

    probs_row_grid = onp.transpose(onp.tile(onp.array(probs), (cdf_grid.shape[0], 1)))

    def get_quantiles_for_one_observation(cdf_grid_one_obs):
        return base_grid[onp.argmax(onp.greater(cdf_grid_one_obs, probs_row_grid), axis=1)]

    # TODO: this is the main performance bottleneck. uses only one CPU core
    quantiles_grid = onp.apply_along_axis(
        func1d=get_quantiles_for_one_observation,
        axis=0,
        arr=cdf_grid,
    )
    return quantiles_grid


class timer:
    def __init__(self, name: str = None, log_level: int = LogLevel.WARN, repeat: int = 1):
        self.name = name or "timer"
        self.repeat = repeat
        self.log_level = log_level
        self.duration = None
        self.msg = "No message."

    def __enter__(self):
        self.tstart = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.duration = time.perf_counter() - self.tstart
        if self.log_level >= LOG_LEVEL:
            if self.repeat == 1:
                self.msg = f"Elapsed: {self.duration:.4f} sec"
            else:
                self.msg = f"Elapsed: {self.duration / self.repeat:.4f} sec (x{self.repeat} repeats = {self.duration:.4f} sec)"
            log(name="tracer", color="white", log_level=self.log_level, id=f"{self.name}", msg=self.msg)


def get_subplots(tree, figsize=(10, 10), sharex=False, sharey=False, major="row", is_leaf: Callable[[Any], bool] = None):
    _, treedef = jax.tree_util.tree_flatten(tree, is_leaf=is_leaf)
    num = treedef.num_leaves
    nrows, ncols = onp.ceil(onp.sqrt(num)).astype(int), onp.ceil(onp.sqrt(num)).astype(int)
    if nrows * (ncols - 1) >= num:
        if major == "row":
            ncols -= 1
        else:
            nrows -= 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    tree_axes = jax.tree_util.tree_unflatten(treedef, axes.flatten()[0:num].tolist())
    if len(axes.flatten()) > num:
        for ax in axes.flatten()[num:]:
            ax.remove()
    return fig, tree_axes


def plot_graph(
    G,
    font_size=10,
    edge_linewidth=1.0,
    arrowsize=8,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=-0.1",
    show_labels=True,
    max_x=None,
    ax=None,
    label_map: Dict = None,
    label_loc="center",  # "center", "bottom", "top"
    height: float = 0.6,
    show_stateful_edges=False,
    message_arrow_timing_mode="arrival",
):
    """

    :param G:
    :param font_size:
    :param edge_linewidth: Edge line width for the message arrows
    :param arrowsize: Arrow size for the message arrows
    :param arrowstyle: Arrow style for the message arrows
    :param connectionstyle: Connection style for the message arrows
    :param show_labels: Whether to draw labels on the nodes
    :param max_x: Maximum x value to plot
    :param ax: Matplotlib axis
    :param label_map: A dictionary mapping node names to labels
    :param label_loc: Location of the label: "center", "bottom", "top"
    :param height: Height of the computation blocks
    :param show_stateful_edges: Whether to show stateful edges between consecutive vertices of the same kind.
    :param message_arrow_timing_mode: "arrival" represents the message arrival timem, while
                                      "usage" represents the time when the message is actually used in processing.
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1)
        fig.set_size_inches(12, 5)

    # Remove nodes until max_x
    x = {n: data.get("ts", data.get("seq", "")) for idx, (n, data) in enumerate(G.nodes(data=True))}
    if max_x is not None:
        x = {k: v for k, v in x.items() if v <= max_x}
        G = G.subgraph(x.keys()).copy()

    # Get all nodes and edges
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)

    # Remove stateful edges
    if not show_stateful_edges:
        # Remove stateful edges
        rm_edges = [(u, v) for u, v, data in edges if G.nodes[u]["kind"] == G.nodes[v]["kind"]]
        G.remove_edges_from(rm_edges)
        edges = G.edges(data=True)

    # Get color, order per node kind
    node_ecolor, node_fcolor, node_order, node_alpha = {}, {}, {}, {}
    for n, data in nodes:
        node_ecolor[data["kind"]] = data.get("edgecolor", "#212529")
        node_fcolor[data["kind"]] = data.get("facecolor", "#868e96")
        node_order[data["kind"]] = data.get("order", None)
        node_alpha[data["kind"]] = data.get("alpha", 1.0)

    # Add unassigned nodes an order
    orders = []
    for idx, (k, o) in enumerate(node_order.items()):
        if o is None:
            node_order[k] = k if isinstance(k, int) else idx
        assert node_order[k] not in orders, "Order must be unique"
        orders.append(node_order[k])

    # Get edge attributes
    edges_msg = []
    pos_msg = {}
    edge_style, edge_alpha, edge_color = [], [], []
    for u, v, data in edges:
        edge_color.append(data.get("color", EDGE_DATA["color"]))
        edge_alpha.append(data.get("alpha", EDGE_DATA["alpha"]))
        edge_style.append(data.get("linestyle", EDGE_DATA["linestyle"]))
        # Determine position of begin and end of message
        name = f"{u}_{v}"
        name_out = name + "_out"
        name_int = name + "_int"
        edges_msg.append((name_out, name_int))
        # Output messages always start from the end of the sender
        offset = height / 2 if node_order[nodes[v]["kind"]] > node_order[nodes[u]["kind"]] else -height / 2  # Offset if sender is higher
        offset = 0 if nodes[u]["kind"] == nodes[v]["kind"] else offset  # No offset if same kind
        pos_msg[name_out] = (nodes[u]["ts_end"], node_order[nodes[u]["kind"]]+offset)
        # Input messages always end at the start of the receiver, with a height offset
        offset = height / 2 if node_order[nodes[u]["kind"]] > node_order[nodes[v]["kind"]] else -height / 2  # Offset if sender is higher
        offset = 0 if nodes[u]["kind"] == nodes[v]["kind"] else offset  # No offset if same kind
        if message_arrow_timing_mode == "arrival":
            pos_msg[name_int] = (data["ts_recv"], node_order[nodes[v]["kind"]]+offset)
        elif message_arrow_timing_mode == "usage":
            pos_msg[name_int] = (nodes[v]["ts_start"], node_order[nodes[v]["kind"]]+offset)
        else:
            raise ValueError(f"Invalid message_arrow_timing_mode: {message_arrow_timing_mode}. Valid options: ['arrival', 'usage']")
    G_msg = nx.DiGraph()
    G_msg.add_edges_from(edges_msg)

    # Get vertex attributes
    node_ts_start = {k: [] for k in node_order.keys()}
    node_ts_end = {k: [] for k in node_order.keys()}
    node_labels, pos, pos_labels = {}, {}, {}
    for idx, (n, data) in enumerate(nodes):
        node_labels[n] = data.get("seq", "")
        ts_start = data.get("ts_start")
        ts_end = data.get("ts_end")
        node_ts_start[data["kind"]].append(ts_start)
        node_ts_end[data["kind"]].append(ts_end)
        pos[n] = (ts_start + (ts_end - ts_start) / 2, node_order[data["kind"]])
        offset = (height)/2*1.25
        offset = offset if label_loc == "top" else -offset if label_loc == "bottom" else 0
        pos_labels[n] = (ts_start + (ts_end - ts_start) / 2, node_order[data["kind"]]+offset)

    # Sort node_ts_start and node_ts_end
    for k in node_ts_start.keys():
        node_ts_start[k] = onp.array(sorted(node_ts_start[k]))
        node_ts_end[k] = onp.array(sorted(node_ts_end[k]))

    node_phase, node_computation, node_idle = {}, {}, {}
    for k in node_ts_start.keys():
        node_phase[k] = node_ts_start[k][0]
        node_computation[k] = node_ts_end[k] - node_ts_start[k]
        node_idle[k] = node_ts_start[k][1:] - node_ts_end[k][:-1]

    # Draw computation delay
    c = 1.0
    for k in node_ts_start.keys():
        ax.barh(y=node_order[k], height=height*c, width=node_phase[k], left=0, color=oc.fcolor.phase, alpha=0.5, label="phase shift")
        ax.barh(y=node_order[k], height=height*c, width=node_idle[k], left=node_ts_end[k][:-1], color=oc.fcolor.sleep, alpha=0.5, label="idle")
        ax.barh(y=node_order[k], height=height, width=node_computation[k], left=node_ts_start[k], color=node_fcolor[k], edgecolor=node_ecolor[k], alpha=node_alpha[k], label=k)
        # ax.scatter(node_ts_end[k], [node_order[k]] * len(node_ts_end[k]), s=(height*10) ** 2, edgecolors=node_ecolor[k], facecolors=node_fcolor[k], marker="o", alpha=node_alpha[k],
        #            label=f"output {k}")

    # Draw graph
    nx.draw_networkx_edges(
        G_msg,
        ax=ax,
        pos=pos_msg,
        edge_color=edge_color,
        alpha=edge_alpha,
        style=edge_style,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        width=edge_linewidth,
        node_size=0,
    )
    if show_labels:
        nx.draw_networkx_labels(G, pos_labels, node_labels, ax=ax, font_size=font_size, font_weight="bold")

    # Overwrite label_map with dict
    label_map = label_map or {}

    # Set ticks
    yticks = list(node_order.values())
    ylabels = [label_map.get(k, k) for k in node_order.keys()]
    ax.set_yticks(yticks, labels=ylabels)
    ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)

    # Create legend with the following handles
    new_handles, new_labels = [], []
    handles, labels = ax.get_legend_handles_labels()
    new_handles.append(handles[labels.index("phase shift")]), new_labels.append("phase shift")
    new_handles.append(handles[labels.index("idle")]), new_labels.append("idle")
    arrow_handle = matplotlib.lines.Line2D([0], [1], color='black', marker=None, linewidth=2, linestyle='-',
                                           markerfacecolor='black', markersize=10)  # For the arrow
    new_handles.append(arrow_handle), new_labels.append("message")
    # dot_handle = matplotlib.lines.Line2D([0], [0], color='black', marker='o', markerfacecolor='grey', markersize=8, linestyle='None')
    # new_handles.append(dot_handle), new_labels.append("output channel")

    new_handles.append(matplotlib.lines.Line2D([], [], color='none'))  # Add empty entry as a header for the 'Nodes' category
    new_labels.append(r'$\bf{Comp.}$ $\bf{delays}$')
    for k in node_order.keys():
        new_handles.append(handles[labels.index(k)]), new_labels.append(k)
    ax.legend(handles=new_handles, labels=new_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def plot_system(
    node_infos: Dict[str, NodeInfo],
    pos: Dict[str, Tuple[float, float]] = None,
    k: float = 1.0,
    node_size: int = 2000,
    font_size=9,
    edge_linewidth=3.0,
    node_linewidth=3.0,
    arrowsize=15,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=-0.2",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(nrows=1)
        fig.set_size_inches(6, 6)

    # Add color of nodes that are not in the cscheme
    cscheme = {n.name: n.color if n.color is not None else "gray" for k, n in node_infos.items()}

    # Generate node color scheme
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Determine node position
    if pos is not None:
        fixed_pos: Dict[str, bool] = {key: True for key in pos.keys()}
    else:
        fixed_pos = None

    # Generate graph
    G = nx.MultiDiGraph()
    for name, info in node_infos.items():
        # name = f"{info.name}\n{info.rate} Hz"
        G.add_node(
            info.name,
            kind=info.name,
            rate=info.rate,
            advance=info.advance,
            phase=info.phase,
            delay=info.delay,
            edgecolor=ecolor[info.name],
            facecolor=fcolor[info.name],
            alpha=1.0,
        )
        for input_name, i in info.inputs.items():
            G.add_edge(
                i.output,
                info.name,
                name=i.name,
                blocking=i.blocking,
                skip=i.skip,
                delay=i.delay,
                window=i.window,
                jitter=i.jitter,
                phase=i.phase,
                color=oc.ecolor.skip if i.skip else oc.ecolor.normal,
                linestyle="-" if i.blocking else "--",
                alpha=1.0,
            )

    # Get label map and relabel nodes
    label_map = {v.name: k for k, v in node_infos.items()}
    G = nx.relabel_nodes(G, label_map)  # Relabel nodes

    # Get labels
    node_labels = {n: f"{n}\n{data['rate']} Hz" for n, data in G.nodes(data=True)}

    # Get edge and node properties
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data["color"] for u, v, data in edges]
    edge_alpha = [data["alpha"] for u, v, data in edges]
    edge_style = [data["linestyle"] for u, v, data in edges]
    node_alpha = [data["alpha"] for n, data in nodes]
    node_ecolor = [data["edgecolor"] for n, data in nodes]
    node_fcolor = [data["facecolor"] for n, data in nodes]

    # Get position
    pos = nx.spring_layout(G, pos=pos, fixed=fixed_pos, k=k)

    # Draw graph
    nx.draw_networkx_nodes(
        G,
        ax=ax,
        pos=pos,
        node_color=node_fcolor,
        alpha=node_alpha,
        edgecolors=node_ecolor,
        node_size=node_size,
        linewidths=node_linewidth,
    )
    nx.draw_networkx_edges(
        G,
        ax=ax,
        pos=pos,
        edge_color=edge_color,
        alpha=edge_alpha,
        style=edge_style,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        width=edge_linewidth,
        node_size=node_size,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=font_size, font_weight="bold")

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.normal, label="blocking")
    ax.plot([], [], color=oc.ecolor.skip, label="skip")
    ax.plot([], [], color=oc.ecolor.normal, label="non-blocking", linestyle="--")
    return ax