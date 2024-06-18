from typing import Any, Tuple, List, TypeVar, Dict, Union, Callable, TYPE_CHECKING
import functools
import networkx as nx
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import supergraph
from supergraph import open_colors as oc
from rexv2.base import Graph, WindowedGraph, Window, Vertex, Edge, WindowedVertex, Timings, SlotVertex, TrainableDist
from rexv2.constants import LogLevel, Jitter

from threading import current_thread
from os import getpid
from termcolor import colored

if TYPE_CHECKING:
    from rexv2.node import BaseNode


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
        ts_sent = vertex.ts_end[edge.seq_out]
        ts_recv = edge.ts_recv
        seq = edge.seq_out
        new_window = window.push(seq, ts_sent, ts_recv)
        return new_window, IndexedWindow(new_window.seq, new_window.ts_sent, new_window.ts_recv, edge.seq_in)

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
    max_val = max(filter(None, order.values()))
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


def to_timings(
    graphs: WindowedGraph, S: nx.DiGraph, Gs: List[nx.DiGraph], Gs_monomorphism: List[Dict[str, str]], supervisor: str
) -> Timings:
    # Convert graphs to numpy
    graphs = jax.tree_util.tree_map(lambda val: onp.array(val), graphs)

    # Determine number of
    num_episodes = graphs.vertices[supervisor].seq.shape[0]
    num_partitions = graphs.vertices[supervisor].seq.shape[-1]

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


def promote_to_no_weak_type(_x):
    # Applies jnp.promote_types to itself to promote to no weak type
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.promote_types.html#jax.numpy.promote_types
    _y = jnp.array(_x)
    _z = _y.astype(jnp.promote_types(_y.dtype, _y.dtype))
    return _z


def no_weaktype(identifier: str = None):
    def _no_weaktype(fn):
        def no_weaktype_wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            return jax.tree_util.tree_map(lambda x: promote_to_no_weak_type(x), res)

        no_weaktype_wrapper = functools.wraps(fn)(no_weaktype_wrapper)
        if identifier is not None:
            # functools.update_wrapper(no_weaktype_wrapper, fn)
            no_weaktype_wrapper.__name__ = identifier
        return no_weaktype_wrapper
    return _no_weaktype


def mixture_distribution_quantiles(dist, probs, N_grid_points: int = int(1e3), grid_min: float = None, grid_max: float = None):
    """More info: https://github.com/tensorflow/probability/issues/659"""
    base_grid = onp.linspace(grid_min, grid_max, num=int(N_grid_points))
    shape = (dist.batch_shape, 1) if len(dist.batch_shape) else [1]
    full_grid = onp.transpose(onp.tile(base_grid, shape))
    cdf_grid = dist.cdf(full_grid)  # this is fully parallelized and even uses GPU
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
