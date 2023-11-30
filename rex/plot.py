from typing import TYPE_CHECKING, Dict, List, Tuple, Union
import numpy as onp
import networkx as nx
import jax

import rex.supergraph
from rex.utils import AttrDict
from rex.constants import SIMULATED
from rex.distributions import GMM, Gaussian
from rex.proto import log_pb2
import rex.open_colors as oc

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib import collections as mc
import matplotlib.patches as mpatches

if TYPE_CHECKING:
    from rex.distributions import Distribution


def get_subplots(tree, figsize=(10, 10), sharex=False, sharey=False, major="row"):
    _, treedef = jax.tree_util.tree_flatten(tree)
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


def plot_input_thread(
    ax: "matplotlib.Axes",
    record: log_pb2.InputRecord,
    ystart: float,
    dy: float,
    name: str = None,
    xstart: float = None,
    dx: float = None,
    ecolor: AttrDict = None,
    fcolor: AttrDict = None,
) -> float:
    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Calculate xy-coordinates
    xstart = 0.0 if xstart is None else xstart
    if dx is None:
        for g in reversed(record.grouped):
            if len(g.messages) > 0:
                dx = g.messages[-1].received.ts.sc - xstart
                break
    assert dx > 0, "dx must be > 0."

    # Prepare timings
    phase_in = [(0, record.info.phase)]
    recv = []
    for g in record.grouped:
        for m in g.messages:
            recv.append((m.sent.ts.sc, m.delay))

    # Plot
    ax.broken_barh(
        phase_in, (ystart, dy), facecolors=fcolor.phase_input, edgecolor=ecolor.phase_input, hatch="", label="phase (expected)"
    )
    ax.broken_barh(recv, (ystart, dy), facecolors=fcolor.communication, edgecolor=ecolor.communication, label="communication")

    # Set ticks
    name = name if isinstance(name, str) else record.info.name
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    yticks = ax.get_yticks().tolist()
    ylabels.append(name)
    yticks.append(ystart + dy / 2)
    ax.set_yticks(yticks, labels=ylabels)

    return ystart + dy


def plot_event_thread(
    ax: "matplotlib.Axes",
    record: log_pb2.NodeRecord,
    ystart: float,
    dy: float,
    name: str = None,
    xstart: float = None,
    dx: float = None,
    ecolor: AttrDict = None,
    fcolor: AttrDict = None,
) -> float:
    name = name if isinstance(name, str) else record.info.name

    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Calculate xy-coordinates
    ystart_delay = ystart + 3 * dy / 4
    dy_delay = -dy / 2
    xstart = 0.0 if xstart is None else xstart
    dx = record.steps[-1].ts_output - xstart if dx is None else dx
    assert dx > 0, "dx must be > 0."

    # Prepare timings
    phase = [(0, record.info.phase)]
    step_comp = []
    step_sleep = []
    step_scheduled = []
    step_delay = []
    step_advanced = []
    phase_scheduled = []
    last_phase_scheduled = 0.0
    for step in record.steps:
        if step.ts_step > xstart + dx:
            break
        step_comp.append((step.ts_step, step.delay))
        step_scheduled.append(step.ts_scheduled + step.phase_scheduled)
        step_sleep.append((step.ts_output_prev, step.ts_step - step.ts_output_prev))
        if round(step.phase_scheduled, 6) > 0:
            if not last_phase_scheduled > round(step.phase_scheduled, 6):
                phase_scheduled.append((step.ts_scheduled + last_phase_scheduled, step.phase_scheduled - last_phase_scheduled))
            last_phase_scheduled = round(step.phase_scheduled, 6)
        if round(step.phase - step.phase_scheduled, 6) > 0:
            step_delay.append((step.ts_step, -max(0, step.phase - step.phase_scheduled)))
        if round(step.phase - step.phase_scheduled, 6) < 0:
            step_advanced.append((step.ts_step, -min(0, step.phase - step.phase_scheduled)))

    # Plot
    ax.broken_barh(step_sleep, (ystart, dy), facecolors=fcolor.sleep, edgecolor=ecolor.sleep, label="sleep")
    ax.broken_barh(phase, (ystart, dy), facecolors=fcolor.phase, edgecolor=ecolor.phase, hatch="", label="phase (expected)")
    ax.broken_barh(step_comp, (ystart, dy), facecolors=fcolor.computation, edgecolor=ecolor.computation, label="computation")
    ax.broken_barh(
        phase_scheduled,
        (ystart_delay, dy_delay),
        facecolors=fcolor.phase,
        edgecolor=ecolor.phase,
        hatch="////",
        label="phase (scheduled)",
    )
    ax.broken_barh(
        step_advanced,
        (ystart_delay, dy_delay),
        facecolors=fcolor.advanced,
        edgecolor=ecolor.advanced,
        label="phase (advanced)",
    )
    ax.broken_barh(
        step_delay, (ystart_delay, dy_delay), facecolors=fcolor.delay, edgecolor=ecolor.delay, label="phase (delayed)"
    )

    # Plot scheduled ts
    ymin = ystart + dy if dy < 0 else ystart
    ymax = ystart if dy < 0 else ystart + dy
    ax.vlines(step_scheduled, ymin=ymin, ymax=ymax, facecolors=fcolor.scheduled, edgecolor=ecolor.scheduled, label="scheduled")

    # Set ticks
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    yticks = ax.get_yticks().tolist()
    ylabels.append(name)
    yticks.append(ystart + dy / 2)
    ax.set_yticks(yticks, labels=ylabels)

    return ystart + dy


class HandlerPatchCollection:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        p = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            hatch=orig_handle.get_hatch(),
            lw=orig_handle.get_linewidth(),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(p)
        return p


# Register patch artist for legends
Legend.update_default_handler_map({mc.PatchCollection: HandlerPatchCollection()})


def broken_bar(ax: "matplotlib.Axes", ranges: List[Tuple[float, float, float, float]], **kwargs):
    patches = []
    for x, dx, y, dy in ranges:
        patches.append(matplotlib.patches.Rectangle((x, y), dx, dy))
    pc = mc.PatchCollection(patches, **kwargs)
    ax.add_collection(pc)
    return pc


def plot_grouped(
    ax: "matplotlib.Axes",
    record: log_pb2.NodeRecord,
    name: str = None,
    xstart: float = None,
    dx: float = None,
    ecolor: AttrDict = None,
    fcolor: AttrDict = None,
    max_num: int = None,
):
    # Get input record
    name = name if isinstance(name, str) else record.inputs[0].info.name
    record_in = [i for i in record.inputs if name == i.info.name]
    assert len(record_in) > 0, f"No input with the name `{name}` for node `{record.info.name}`."
    record_in = record_in[0]

    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Calculate xy-coordinates
    xstart = 0.0 if xstart is None else xstart
    dx = record.steps[-1].ts_output - xstart if dx is None else dx
    assert dx > 0, "dx must be > 0."

    # Determine max number of messages to assume
    max_num = max([g.num_msgs for g in record_in.grouped]) if max_num is None else max_num
    assert max_num > 0, "max_num must be > 0."

    # Prepare timings
    phase = [(0, record.info.phase, 0, record.steps[0].ts_step)]
    phase_in = [(0, record_in.info.phase)]
    rate = record_in.info.rate
    input_advanced = []
    input_delayed = []
    input_received = []
    step_scheduled = []
    step_comp = []
    step_sleep = []
    last_ts_step = 0.0
    last_delay = None
    for i, (step, g) in enumerate(zip(record.steps, record_in.grouped)):
        # Process step info
        if step.ts_step > xstart + dx:
            break
        y, dy = last_ts_step, step.ts_step - last_ts_step
        xy = [(step.ts_scheduled + step.phase_scheduled, y), (step.ts_scheduled + step.phase_scheduled, y + dy)]
        step_scheduled.append(xy)
        if last_delay is not None:
            step_comp.append((last_ts_step, last_delay, y, dy))
        step_sleep.append((step.ts_output_prev, step.ts_step - step.ts_output_prev, y, dy))
        last_ts_step = step.ts_step
        last_delay = step.delay

        # Process messages
        if not i + 1 < len(record.steps):
            continue
        next_step = record.steps[i + 1].ts_step
        y, dy = step.ts_step, next_step - step.ts_step

        offset = max_num + 1
        yy, dyy = y + 0.5 * dy / offset, dy / offset
        yy += 0.5 * dyy * (max_num - g.num_msgs)
        for i, m in enumerate(g.messages):
            seq, ts = m.received.seq, round(m.received.ts.sc, 6)
            ts_expected = round(seq / rate + record_in.info.phase, 6)
            yyy, dyyy = yy + i * dyy, dyy

            input_received.append([(m.received.ts.sc, yyy), (m.received.ts.sc, yyy + dyyy)])
            dt = ts_expected - ts
            if ts_expected < ts:
                input_delayed.append((ts, dt, yyy, dyyy))
            elif ts_expected > ts:
                input_advanced.append((ts, dt, yyy, dyyy))

    # Plot event thread
    broken_bar(ax, step_sleep, facecolor=fcolor.sleep, edgecolor=ecolor.sleep, label="sleep")
    broken_bar(ax, phase, facecolor=fcolor.phase, edgecolor=ecolor.phase, hatch="", label="phase (expected)")
    broken_bar(ax, step_comp, facecolor=fcolor.computation, edgecolor=ecolor.computation, label="computation")

    # Plot input thread
    broken_bar(ax, input_delayed, facecolor=fcolor.delay, edgecolor=ecolor.delay, label="phase (delayed)")
    broken_bar(ax, input_advanced, facecolor=fcolor.advanced, edgecolor=ecolor.advanced, label="phase (advanced)")

    # Plot input messages
    lc = mc.LineCollection(input_received, color=fcolor.scheduled, label="received")
    ax.add_collection(lc)


record_types = Union[log_pb2.NodeRecord, log_pb2.InputRecord]


def plot_delay(
    ax: "matplotlib.Axes",
    records: Union[List[record_types], record_types],
    dist: "Distribution" = None,
    name: str = None,
    low: float = None,
    high: float = None,
    clock: int = SIMULATED,
    num: int = 1000,
    ecolor: AttrDict = None,
    fcolor: AttrDict = None,
    plot_dist: bool = True,
    **kde_kwargs,
):
    name = name if isinstance(name, str) else "distribution"
    records = records if isinstance(records, list) else [records]
    assert len(records) > 0, "The provided record is empty."

    # Determine communication or computation delay
    if isinstance(records[0], log_pb2.NodeRecord):
        delay_type = "computation"
    else:
        assert isinstance(records[0], log_pb2.InputRecord)
        delay_type = "communication"

    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Get distributions
    dist = GMM.from_info(records[0].info.delay_sim) if dist is None else dist

    # Convert to GMM
    if isinstance(dist, Gaussian):
        dist = GMM([dist], [1.0])

    # Get sampled delays
    delay_sc = []
    delay_wc = []
    for record in records:
        if delay_type == "computation":
            for step in record.steps:
                delay_sc.append(step.delay)
                delay_wc.append(step.comp_delay.wc)
        else:
            for group in record.grouped:
                for m in group.messages:
                    delay_sc.append(m.delay)
                    delay_wc.append(m.comm_delay.wc)

    # Determine delay based on selected clock
    if clock in [SIMULATED]:
        delay = delay_sc
    else:
        delay = delay_wc

    # Determine low and high
    low = dist.low - 1e-6 if low is None else low
    high = dist.high + 1e-6 if high is None else high
    low = min(low, min(delay), high - 1e-6)
    high = max(high, max(delay), high + 1e-6)

    # Insert all mean values
    t = onp.linspace(low, high, num=num)
    ii = onp.searchsorted(t, dist.means)
    t = onp.insert(t, ii, dist.means)

    # Determine colors
    edgecolor = ecolor.computation if delay_type == "computation" else ecolor.communication
    facecolor = fcolor.computation if delay_type == "computation" else fcolor.communication

    # Plot distribution
    if plot_dist:
        ax.plot(t, dist.pdf(t), color=edgecolor, linestyle="--", label=name)

    # Plot kde/histogram
    import seaborn as sns

    sns.histplot(delay, ax=ax, stat="density", label="data", color=edgecolor, fill=facecolor)
    # sns.kdeplot(delay, ax=ax, warn_singular=False, clip=[low, high], color=edgecolor, fill=facecolor, label="kde estimate",
    #             **kde_kwargs) # todo: TURN ON AGAIN!


def plot_step_timing(
    ax: "matplotlib.Axes",
    record: log_pb2.NodeRecord,
    kind: Union[List[str], str],
    name: str = None,
    low: float = None,
    high: float = None,
    ecolor: AttrDict = None,
    fcolor: AttrDict = None,
    plot_hist: bool = True,
    plot_kde: bool = True,
    **kde_kwargs,
):
    name = name if isinstance(name, str) else "data"
    kind = kind if isinstance(kind, list) else [kind]
    assert all([k in ["advanced", "ontime", "delayed"] for k in kind])

    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Prepare timings
    delay = []
    advanced = []
    ontime = []
    for step in record.steps:
        dt = step.phase - step.phase_scheduled
        if round(dt, 6) > 0:
            delay.append(max(0, dt))
        elif round(dt, 6) < 0:
            advanced.append(min(0, dt))
        else:
            ontime.append(0.0)

    # Determine colors
    if "ontime" in kind or len(kind) > 1:
        edgecolor = ecolor.sleep
        facecolor = fcolor.sleep
    elif "delayed" in kind:
        edgecolor = ecolor.delay
        facecolor = fcolor.delay
    else:
        edgecolor = ecolor.advanced
        facecolor = fcolor.advanced

        # Determine data
    data = []
    if "ontime" in kind:
        data += ontime
    if "delayed" in kind:
        data += delay
    if "advanced" in kind:
        data += advanced

    # Determine low and high
    low = -1e-5 if low is None else low
    high = 1e-5 if high is None else high
    low = min([low] + advanced)
    high = max([high] + delay)

    # Plot kde/histogram
    import seaborn as sns

    if plot_hist:
        sns.histplot(data, ax=ax, stat="density", label=name, color=edgecolor, fill=facecolor)
    if plot_kde:
        sns.kdeplot(
            data,
            ax=ax,
            warn_singular=False,
            clip=[low, high],
            color=edgecolor,
            fill=facecolor,
            label="kde estimate",
            **kde_kwargs,
        )


def plot_computation_graph(
    ax: "matplotlib.Axes",
    G: nx.DiGraph,
    root: str = None,
    seq: int = -1,
    xmax: float = None,
    order: List[str] = None,
    cscheme: Dict[str, str] = None,
    node_labeltype: str = "seq",
    node_size: int = 300,
    node_fontsize=10,
    edge_linewidth=2.0,
    node_linewidth=1.5,
    arrowsize=10,
    arrowstyle="->",
    connectionstyle="arc3",
    draw_nodelabels=True,
    draw_pruned=True,
):
    """

    :param ax:
    :param G: Computation graph.
    :param root: Root node to trace in the computation graph.
    :param seq: Sequence number of the traced root step.
    :param xmax: Maximum time to plot the computation graph for.
    :param order: Order in which the nodes are placed in y-direction.
    :param cscheme: Color scheme for the nodes.
    :param node_labeltype:
    :param node_size:
    :param node_fontsize:
    :param edge_linewidth:
    :param node_linewidth:
    :param arrowsize:
    :param arrowstyle:
    :param connectionstyle:
    :param draw_nodelabels: Draw node labels with ts/tick (=True) or not (=False).
    :param draw_pruned: Draw pruned nodes (=True) or not (=False).
    :return:
    """
    # Make a copy of the graph
    G = G.copy(as_view=False)

    # Set cscheme & order
    order = order or []
    cscheme = cscheme or {}
    assert all([v != "red" for v in cscheme.values()]), "Color red is reserved for excluded nodes."

    # Set edge and node properties
    # Get all non-pruned nodes
    unpruned_nodes = [n for n in G.nodes if not G.nodes[n]["pruned"]]
    rex.supergraph.set_node_order(G, order)
    rex.supergraph.set_node_colors(G.subgraph(unpruned_nodes), cscheme)
    y = rex.supergraph.get_node_y_position(G)
    cscheme = rex.supergraph.get_node_colors(G)

    # Generate node color scheme
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Trace root
    if root is not None:
        G = rex.supergraph.trace_root(G, root, seq)

    # Prune
    if not draw_pruned:
        G = rex.supergraph.prune_graph(G)

    # Only plot nodes up to xmax
    if xmax is not None:
        draw_nodes = [n for n in G.nodes if G.nodes[n]["ts_step"] <= xmax]
        G = G.subgraph(draw_nodes)

    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data["color"] for u, v, data in edges]
    edge_alpha = [data["alpha"] for u, v, data in edges]
    edge_style = [data["linestyle"] for u, v, data in edges]
    node_alpha = [data["alpha"] for n, data in nodes]
    node_ecolor = [data["edgecolor"] for n, data in nodes]
    node_fcolor = [data["facecolor"] for n, data in nodes]

    # Get labels
    if node_labeltype == "seq":
        node_labels = {n: data["seq"] for n, data in nodes}
    elif node_labeltype == "ts":
        node_labels = {n: f"{data['ts_step']:.3f}" for n, data in nodes}
    else:
        raise NotImplementedError("label_type must be 'seq' or 'ts'")

    # Get positions
    pos = {n: data["position"] for n, data in nodes}

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
    if draw_nodelabels:
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=node_fontsize)

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.used, label="dependency")
    for (name, e), (_, f) in zip(ecolor.items(), fcolor.items()):
        ax.scatter([], [], edgecolor=e, facecolor=f, label=name)

    if draw_pruned:
        ax.plot([], [], color=oc.ecolor.pruned, label="pruned dependency", alpha=0.5)
        ax.scatter([], [], edgecolor=oc.ecolor.pruned, facecolor=oc.fcolor.pruned, alpha=0.5, label="pruned step")

    # Set ticks
    yticks = list(y.values())
    ylabels = list(y.keys())
    ax.set_yticks(yticks, labels=ylabels)
    ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)


# def plot_topological_order(
#     ax: "matplotlib.Axes",
#     G: nx.DiGraph,
#     root: str,
#     seq: int = -1,
#     xmax: float = None,
#     order: List[str] = None,
#     cscheme: Dict[str, str] = None,
#     node_labeltype: str = "seq",
#     node_size: int = 250,
#     node_fontsize=10,
#     edge_linewidth=2.0,
#     node_linewidth: float = 1.5,
#     arrowsize: int = 10,
#     arrowstyle: str = "->",
#     connectionstyle: str = "arc3,rad=0.1",
#     draw_nodelabels: bool = True,
#     draw_excess: bool = True,
#     draw_root_excess: bool = True,
# ):
#     """Plot topological order of a trace record.
#
#     Args:
#     :param ax: Matplotlib axes.
#     :param G: Computation graph.
#     :param root: Root node to trace in the computation graph.
#     :param seq: Sequence number of the traced root step.
#     :param xmax: Maximum x position of the graph.
#     :param order: Node order.
#     :param cscheme: Color scheme.
#     :param node_labeltype: Node label type. Can be "seq" or "ts".
#     :param node_size: Node size.
#     :param node_fontsize: Node font size.
#     :param edge_linewidth: Edge line width.
#     :param node_linewidth: Node line width.
#     :param arrowsize: Arrow size.
#     :param arrowstyle: Arrow style.
#     :param connectionstyle: Connection style.
#     :param draw_nodelabels: Draw node labels with ts/tick (=True) or not (=False).
#     :param draw_excess: Draw excess step calls (=True) or not (=False).
#     :param draw_root_excess: Draw excess step calls for root (=True) or not (=False).
#     """
#     # Make a copy of the graph
#     G = G.copy(as_view=False)
#
#     # Set cscheme
#     order = order or []
#     cscheme = cscheme or {}
#     assert all([v != "red" for v in cscheme.values()]), "Color red is reserved for excluded nodes."
#
#     # Set edge and node properties
#     rex.supergraph.set_node_colors(G, cscheme)
#     rex.supergraph.set_node_order(G, order)
#     y = rex.supergraph.get_node_y_position(G)
#     cscheme = rex.supergraph.get_node_colors(G)
#
#     # Generate node color scheme
#     ecolor, fcolor = oc.cscheme_fn(cscheme)
#
#     # Trace root
#     G = rex.supergraph.trace_root(G, root, seq)
#     G = rex.supergraph.prune_graph(G)
#
#     # Get node data
#     node_data = rex.supergraph.get_node_data(G)
#
#     # Topological sort
#     topo = list(nx.topological_sort(G))
#
#     # Set x positions according to topological order
#     data_excess = {"alpha": 0.5, "edgecolor": oc.ecolor.pruned, "facecolor": oc.fcolor.pruned}
#     for idx, n in enumerate(topo):
#         d = G.nodes[n]
#         G.nodes[n].update({"position": (idx, y[d["kind"]])})
#         if draw_excess:
#             if d["kind"] == root and not draw_root_excess:
#                 continue
#             for key, val in node_data.items():
#                 if key == root and not draw_root_excess:
#                     continue
#                 if d["kind"] == key:
#                     continue
#                 position = (idx, y[key])
#                 excess = {"pruned": False, "position": position, "ts_step": d["ts_step"]}
#                 excess.update(val)
#                 excess.update(data_excess)
#                 G.add_node(f"{key}_excess_{idx}", **excess)
#
#     # Only plot nodes up to xmax
#     if xmax is not None:
#         draw_nodes = [n for n in G.nodes if G.nodes[n]["ts_step"] <= xmax]
#         G = G.subgraph(draw_nodes)
#
#     # Get edge and node properties
#     edges = G.edges(data=True)
#     nodes = G.nodes(data=True)
#     edge_color = [data["color"] for u, v, data in edges]
#     edge_alpha = [data["alpha"] for u, v, data in edges]
#     edge_style = [data["linestyle"] for u, v, data in edges]
#     node_alpha = [data["alpha"] for n, data in nodes]
#     node_ecolor = [data["edgecolor"] for n, data in nodes]
#     node_fcolor = [data["facecolor"] for n, data in nodes]
#
#     # Get labels
#     if node_labeltype == "seq":
#         node_labels = {n: data["seq"] if "seq" in data else "" for n, data in nodes}
#     elif node_labeltype == "ts":
#         node_labels = {n: f"{data['ts_step']:.3f}" if "ts_step" in data else "" for n, data in nodes}
#     else:
#         raise NotImplementedError("label_type must be 'seq' or 'ts'")
#
#     # Get positions
#     pos = {n: data["position"] for n, data in nodes}
#
#     # Draw graph
#     nx.draw_networkx_nodes(
#         G,
#         ax=ax,
#         pos=pos,
#         node_color=node_fcolor,
#         alpha=node_alpha,
#         edgecolors=node_ecolor,
#         node_size=node_size,
#         linewidths=node_linewidth,
#     )
#     nx.draw_networkx_edges(
#         G,
#         ax=ax,
#         pos=pos,
#         edge_color=edge_color,
#         alpha=edge_alpha,
#         style=edge_style,
#         arrowsize=arrowsize,
#         arrowstyle=arrowstyle,
#         connectionstyle=connectionstyle,
#         width=edge_linewidth,
#         node_size=node_size,
#     )
#
#     # Draw labels
#     if draw_nodelabels:
#         nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=node_fontsize)
#
#     # Add empty plot with correct color and label for each node
#     ax.plot([], [], color=oc.ecolor.used, label="dependency")
#     for (name, e), (_, f) in zip(ecolor.items(), fcolor.items()):
#         ax.scatter([], [], edgecolor=e, facecolor=f, label=name)
#
#     # Add excess with correct color and label to legend
#     if draw_excess:
#         ax.scatter([], [], edgecolor=oc.ecolor.excluded, facecolor=oc.fcolor.excluded, alpha=0.5, label="excess step")
#
#     # Set ticks
#     yticks = list(y.values())
#     ylabels = list(y.keys())
#     ax.set_yticks(yticks, labels=ylabels)
#     ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)
#
#
# def plot_depth_order(
#     ax: "matplotlib.Axes",
#     G: nx.DiGraph,
#     root: str,
#     MCS: nx.DiGraph,
#     seq: int = -1,
#     split_mode: str = "generational",
#     supergraph_mode: str = "MCS",
#     xmax: float = None,
#     order: List[str] = None,
#     cscheme: Dict[str, str] = None,
#     node_labeltype: str = "seq",
#     node_size: int = 300,
#     node_fontsize=10,
#     edge_linewidth=2.0,
#     node_linewidth=1.5,
#     arrowsize=10,
#     arrowstyle="->",
#     connectionstyle="arc3,rad=0.1",
#     draw_nodelabels=True,
#     draw_excess=True,
#     workers: int = None,
# ):
#     """
#     :param ax:
#     :param G: Computation graph.
#     :param root: Root node to trace in the computation graph.
#     :param MCS: minimum common supergraph.
#     :param seq: Sequence number of the traced root step.
#     :param split_mode: Split mode for the subgraphs graph.
#     :param supergraph_mode: The type of supergraph for the subgraphs graph.
#     :param xmax: Maximum time to plot the computation graph for.
#     :param order: Order in which the nodes are placed in y-direction.
#     :param cscheme: Color scheme for the nodes.
#     :param node_labeltype:
#     :param node_size:
#     :param node_fontsize:
#     :param edge_linewidth:
#     :param node_linewidth:
#     :param arrowsize:
#     :param arrowstyle:
#     :param connectionstyle:
#     :param draw_nodelabels: Draw node labels with ts/tick (=True) or not (=False).
#     :param draw_excess: Draw excess step calls (=True) or not (=False).
#     :param workers: Number of workers to use for parallelization.
#     :return:
#     """
#     # Create copy of graphs
#     G = G.copy(as_view=False)
#     MCS = MCS.copy(as_view=False)
#
#     # Set cscheme & order
#     order = order or []
#     cscheme = cscheme or {}
#     assert all([v != "red" for v in cscheme.values()]), "Color red is reserved for excluded nodes."
#
#     # Determine root node (always in the last generation)
#     generations = list(nx.topological_generations(MCS))
#     num_gens = len(generations)
#     root_slot = generations[-1][0]
#     assert MCS.nodes[root_slot]["kind"] == root, f"Root node {root} not found in last generation of MCS."
#
#     # Set edge and node properties
#     rex.supergraph.set_node_order(G, order)
#     rex.supergraph.set_node_colors(G, cscheme)
#     cscheme = rex.supergraph.get_node_colors(G)
#
#     # Generate node color scheme
#     ecolor, fcolor = oc.cscheme_fn(cscheme)
#
#     # Trace root node (not pruned yet)
#     G = rex.supergraph.trace_root(G, root=root, seq=seq)
#
#     # Prune unused nodes (not in computation graph of traced root)
#     G = rex.supergraph.prune_graph(G)
#
#     # Get subgraphs
#     G_subgraphs = rex.supergraph.get_subgraphs(G, split_mode=split_mode)
#
#     # If supergraph_mode is "topological"
#     if supergraph_mode == "topological":
#         G_subgraphs = rex.supergraph.as_topological_subgraphs(G_subgraphs)
#
#     # Set MCS properties
#     rex.supergraph.set_node_order(MCS, order)
#     rex.supergraph.set_node_colors(MCS, cscheme)
#     MCS = rex.supergraph.as_MCS(MCS)
#
#     # Get monomorphisms
#     monomorphisms = rex.supergraph.get_subgraph_monomorphisms(MCS, G_subgraphs, workers=workers)
#
#     # Add root node to mapping (root is always the only node in the last generation)
#     root_slot = f"{root}_s0"
#     assert root_slot in MCS, "Root node not found in MCS."
#     monomorphisms = {k: {**v, root_slot: k} for k, v in monomorphisms.items()}
#
#     # Sort monomorphisms by root sequence number
#     monomorphisms = {k: v for k, v in sorted(monomorphisms.items(), key=lambda key_val: G.nodes[key_val[0]]["seq"])}
#
#     # Set node positions
#     node_data = rex.supergraph.get_node_data(MCS)
#     data_excess = {"alpha": 0.5, "edgecolor": oc.ecolor.pruned, "facecolor": oc.fcolor.pruned}
#     for i_root, (root_seq, mapping) in enumerate(monomorphisms.items()):
#         # swapped = {v: k for k, v in mapping.items()}
#         for i_gen, gen in enumerate(generations):
#             x = i_root * num_gens + i_gen
#             for node in gen:
#                 y = MCS.nodes[node]["position"][1]
#                 position = (x, y)
#                 if node in mapping:
#                     G.nodes[mapping[node]].update({"position": position})
#                 elif draw_excess:
#                     excess = {"pruned": False, "position": position, "ts_step": G.nodes[root_seq]["ts_step"]}
#                     excess.update(node_data[MCS.nodes[node]["kind"]])
#                     excess.update(data_excess)
#                     G.add_node(f"{node}_excess_{x}", **excess)
#
#     # Only plot nodes up to xmax
#     if xmax is not None:
#         draw_nodes = [n for n in G.nodes if G.nodes[n]["ts_step"] <= xmax]
#         G = G.subgraph(draw_nodes)
#
#     # Get edge and node properties
#     edges = G.edges(data=True)
#     nodes = G.nodes(data=True)
#     edge_color = [data["color"] for u, v, data in edges]
#     edge_alpha = [data["alpha"] for u, v, data in edges]
#     edge_style = [data["linestyle"] for u, v, data in edges]
#     node_alpha = [data["alpha"] for n, data in nodes]
#     node_ecolor = [data["edgecolor"] for n, data in nodes]
#     node_fcolor = [data["facecolor"] for n, data in nodes]
#
#     # Get labels
#     if node_labeltype == "seq":
#         node_labels = {n: data["seq"] if "seq" in data else "" for n, data in nodes}
#     elif node_labeltype == "ts":
#         node_labels = {n: f"{data['ts_step']:.3f}" if "ts_step" in data else "" for n, data in nodes}
#     else:
#         raise NotImplementedError("label_type must be 'seq' or 'ts'")
#
#     # Get positions
#     pos = {n: data["position"] for n, data in G.nodes(data=True)}
#
#     # Draw graph
#     nx.draw_networkx_nodes(
#         G,
#         ax=ax,
#         pos=pos,
#         node_color=node_fcolor,
#         alpha=node_alpha,
#         edgecolors=node_ecolor,
#         node_size=node_size,
#         linewidths=node_linewidth,
#     )
#     nx.draw_networkx_edges(
#         G,
#         ax=ax,
#         pos=pos,
#         edge_color=edge_color,
#         alpha=edge_alpha,
#         style=edge_style,
#         arrowsize=arrowsize,
#         arrowstyle=arrowstyle,
#         connectionstyle=connectionstyle,
#         width=edge_linewidth,
#         node_size=node_size,
#     )
#
#     # Draw labels
#     if draw_nodelabels:
#         nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=node_fontsize)
#
#     # Add empty plot with correct color and label for each node
#     ax.plot([], [], color=oc.ecolor.used, label="dependency")
#     for (name, e), (_, f) in zip(ecolor.items(), fcolor.items()):
#         ax.scatter([], [], edgecolor=e, facecolor=f, label=name)
#
#     if draw_excess:
#         ax.scatter([], [], edgecolor=oc.ecolor.excluded, facecolor=oc.fcolor.excluded, alpha=0.5, label="excess step")
#
#     # Set ticks
#     yticks = list(range(max([len(gen) for gen in generations])))
#     ylabels = ["" for _ in yticks]
#     ax.set_yticks(yticks, labels=ylabels)
#     # ax.set_yticks(yticks)
#     ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)


def plot_graph(
    ax: "matplotlib.Axes",
    record: log_pb2.EpisodeRecord,
    cscheme: Dict[str, str] = None,
    pos: Dict[str, Tuple[float, float]] = None,
    node_size: int = 2000,
    node_fontsize=10,
    edge_linewidth=3.0,
    node_linewidth=2.0,
    arrowsize=10,
    arrowstyle="->",
    connectionstyle="arc3,rad=0.2",
):
    # Add color of nodes that are not in the cscheme
    cscheme = cscheme if isinstance(cscheme, dict) else {}
    for n in record.node:
        if n.info.name not in cscheme:
            cscheme[n.info.name] = "gray"
        else:
            assert cscheme[n.info.name] != "red", "Color red is a reserved color."

    # Generate node color scheme
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Determine node position
    if pos is not None:
        fixed_pos: Dict[str, bool] = {key: True for key in pos.keys()}
    else:
        fixed_pos = None

    # Generate graph
    G = nx.MultiDiGraph()
    for n in record.node:
        edgecolor = ecolor[n.info.name]
        facecolor = fcolor[n.info.name]
        name = f"{n.info.name}\n{n.info.rate} Hz"  # \n{n.info.delay:.3f} s\n{n.info.phase: .3f} s"
        G.add_node(
            n.info.name,
            kind=name,
            rate=n.info.rate,
            advance=n.info.advance,
            phase=n.info.phase,
            delay=n.info.delay,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=1.0,
        )
        for i in n.inputs:
            linestyle = "-" if i.info.blocking else "--"
            color = oc.ecolor.skip if i.info.skip else oc.ecolor.normal
            G.add_edge(
                i.info.output,
                n.info.name,
                name=i.info.name,
                blocking=i.info.blocking,
                skip=i.info.skip,
                delay=i.info.delay,
                window=i.info.window,
                jitter=i.info.jitter,
                phase=i.info.phase,
                color=color,
                linestyle=linestyle,
                alpha=1.0,
            )

    # Get edge and node properties
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data["color"] for u, v, data in edges]
    edge_alpha = [data["alpha"] for u, v, data in edges]
    edge_style = [data["linestyle"] for u, v, data in edges]
    node_alpha = [data["alpha"] for n, data in nodes]
    node_ecolor = [data["edgecolor"] for n, data in nodes]
    node_fcolor = [data["facecolor"] for n, data in nodes]

    # Get labels
    # edge_labels = {(u, v): f"{data['delay']:.3f}" for u, v, data in edges}
    node_labels = {n: data["kind"] for n, data in nodes}

    # Get position
    pos = nx.spring_layout(G, pos=pos, fixed=fixed_pos)

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
        node_shape="s",
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
    nx.draw_networkx_labels(G, pos, node_labels, font_size=node_fontsize)
    # if draw_edgelabels:
    # 	nx.draw_networkx_edge_labels(G, pos, edge_labels, rotate=True, bbox=edge_bbox, font_size=edge_fontsize)

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.normal, label="blocking")
    ax.plot([], [], color=oc.ecolor.skip, label="skip")
    ax.plot([], [], color=oc.ecolor.normal, label="non-blocking", linestyle="--")
