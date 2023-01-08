from typing import TYPE_CHECKING, Dict, List, Tuple, Union
import numpy as onp
import networkx as nx

from rex.utils import AttrDict
from rex.constants import SIMULATED
from rex.distributions import GMM, Gaussian
from rex.proto import log_pb2

import matplotlib
from matplotlib.legend import Legend
from matplotlib import collections as mc
import matplotlib.patches as mpatches

if TYPE_CHECKING:
    from rex.distributions import Distribution


def plot_input_thread(ax: "matplotlib.Axes",
                      record: log_pb2.InputRecord,
                      ystart: float,
                      dy: float,
                      name: str = None,
                      xstart: float = None,
                      dx: float = None,
                      ecolor: AttrDict = None,
                      fcolor: AttrDict = None) -> float:
    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Calculate xy-coordinates
    xstart = 0. if xstart is None else xstart
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
    ax.broken_barh(phase_in, (ystart, dy), facecolors=fcolor.phase_input, edgecolor=ecolor.phase_input, hatch="",
                   label="phase (expected)")
    ax.broken_barh(recv, (ystart, dy), facecolors=fcolor.communication, edgecolor=ecolor.communication, label="communication")

    # Set ticks
    name = name if isinstance(name, str) else record.info.name
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    yticks = ax.get_yticks().tolist()
    ylabels.append(name)
    yticks.append(ystart + dy / 2)
    ax.set_yticks(yticks, labels=ylabels)

    return ystart + dy


def plot_event_thread(ax: "matplotlib.Axes",
                      record: log_pb2.NodeRecord,
                      ystart: float,
                      dy: float,
                      name: str = None,
                      xstart: float = None,
                      dx: float = None,
                      ecolor: AttrDict = None,
                      fcolor: AttrDict = None) -> float:
    name = name if isinstance(name, str) else record.info.name

    # Get color scheme
    if ecolor is None:
        from rex.open_colors import ecolor
    if fcolor is None:
        from rex.open_colors import fcolor

    # Calculate xy-coordinates
    ystart_delay = ystart + 3 * dy / 4
    dy_delay = -dy / 2
    xstart = 0. if xstart is None else xstart
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
    last_phase_scheduled = 0.
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
    ax.broken_barh(phase_scheduled, (ystart_delay, dy_delay), facecolors=fcolor.phase, edgecolor=ecolor.phase, hatch="////",
                   label="phase (scheduled)")
    ax.broken_barh(step_advanced, (ystart_delay, dy_delay), facecolors=fcolor.advanced, edgecolor=ecolor.advanced,
                   label="phase (advanced)")
    ax.broken_barh(step_delay, (ystart_delay, dy_delay), facecolors=fcolor.delay, edgecolor=ecolor.delay,
                   label="phase (delayed)")

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
        p = mpatches.Rectangle([x0, y0], width, height,
                               facecolor=orig_handle.get_facecolor(),
                               edgecolor=orig_handle.get_edgecolor(),
                               hatch=orig_handle.get_hatch(),
                               lw=orig_handle.get_linewidth(),
                               transform=handlebox.get_transform())
        handlebox.add_artist(p)
        return p


# Register patch artist for legends
Legend.update_default_handler_map({mc.PatchCollection: HandlerPatchCollection()})


def broken_bar(ax: "matplotlib.Axes", ranges: List[Tuple[float, float, float, float]], **kwargs):
    patches = []
    for (x, dx, y, dy) in ranges:
        patches.append(matplotlib.patches.Rectangle((x, y),dx, dy))
    pc = mc.PatchCollection(patches, **kwargs)
    ax.add_collection(pc)
    return pc


def plot_grouped(ax: "matplotlib.Axes",
                 record: log_pb2.NodeRecord,
                 name: str = None,
                 xstart: float = None,
                 dx: float = None,
                 ecolor: AttrDict = None,
                 fcolor: AttrDict = None,
                 max_num: int = None):
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
    xstart = 0. if xstart is None else xstart
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
    last_ts_step = 0.
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
        next_step = record.steps[i+1].ts_step
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


def plot_delay(ax: "matplotlib.Axes",
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
               **kde_kwargs):
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


def plot_step_timing(ax: "matplotlib.Axes",
                     record: log_pb2.NodeRecord,
                     kind: Union[List[str], str],
                     name: str = None,
                     low: float = None,
                     high: float = None,
                     ecolor: AttrDict = None,
                     fcolor: AttrDict = None,
                     plot_hist: bool = True,
                     plot_kde: bool = True,
                     **kde_kwargs):
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
            ontime.append(0.)

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
        sns.kdeplot(data, ax=ax, warn_singular=False, clip=[low, high], color=edgecolor, fill=facecolor,
                    label="kde estimate",
                    **kde_kwargs)


def plot_computation_graph(ax: "matplotlib.Axes",
                           record: log_pb2.TraceRecord,
                           xmax: float = None,
                           order: List[str] = None,
                           cscheme: Dict[str, str] = None,
                           node_labeltype: str = "tick",
                           node_size: int = 300,
                           edge_fontsize=10,
                           node_fontsize=10,
                           edge_linewidth=2.0,
                           node_linewidth=1.5,
                           arrowsize=10,
                           arrowstyle="->",
                           connectionstyle="arc3",
                           edge_bbox=None,
                           draw_edgelabels=False,
                           draw_nodelabels=True,
                           draw_excluded=True,
                           draw_stateless=True):
    """

    :param ax:
    :param record:
    :param xmax: Maximum time to plot the computation graph for.
    :param order: Order in which the nodes are placed in y-direction.
    :param cscheme: Color scheme for the nodes.
    :param node_labeltype:
    :param node_size:
    :param edge_fontsize:
    :param node_fontsize:
    :param edge_linewidth:
    :param node_linewidth:
    :param arrowsize:
    :param arrowstyle:
    :param connectionstyle:
    :param edge_bbox:
    :param draw_edgelabels: Draw edge labels with ts (=True) or not (=False).
    :param draw_nodelabels: Draw node labels with ts/tick (=True) or not (=False).
    :param draw_excluded: Draw excluded nodes (=True) or not (=False).
    :param draw_stateless: Draw monotonic time constraint (=True) or not (=False). Only relevant for stateless nodes when tracing with static=True.
    :return:
    """
    import rex.open_colors as oc

    # Determine edge bbox
    if edge_bbox is None:
        edge_bbox = dict(boxstyle="round", fc=oc.ccolor("gray"), ec=oc.ccolor("gray"), alpha=1.0)

    # Determine fixed node
    max_dt = max([1 / n.rate for n in record.node])
    order = order if isinstance(order, list) else []
    order = order + [info.name for info in record.node if info.name not in order]
    y = {name: i * max_dt for i, name in enumerate(order)}
    fixed_pos: Dict[str, bool] = {}
    pos: Dict[str, Tuple[float, float]] = {}

    # Add color of nodes that are not in the cscheme
    cscheme = cscheme if isinstance(cscheme, dict) else {}
    for n in record.node:
        if n.name not in cscheme:
            cscheme[n.name] = "gray"
        else:
            assert cscheme[n.name] != "red", "Color red is reserved for excluded nodes."

    # Generate node color scheme
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Generate graph
    G = nx.MultiDiGraph()
    steptraces = [record.used, record.excluded] if draw_excluded else [record.used]
    for _steptraces in steptraces:
        for t in _steptraces:
            # Skip if trace is past tmax
            if xmax is not None and t.ts_step > xmax:
                continue

            # Add node to graph
            name = f"{t.name}({t.tick})"
            edgecolor = ecolor[t.name] if t.used else oc.ecolor.excluded
            facecolor = fcolor[t.name] if t.used else oc.fcolor.excluded
            alpha = 1.0 if t.used else 0.5
            G.add_node(name, trace=t, name=t.name, used=t.used, tick=t.tick, ts_step=t.ts_step, edgecolor=edgecolor,
                       facecolor=facecolor, alpha=alpha)

            # Add (initial) position
            pos[name] = (t.ts_step, y[t.name])

            # Add fixed position (not used)
            fixed_pos[name] = True

            # Add upstream dependencies as edges
            for d in t.upstream:
                if not d.used and not draw_excluded:
                    continue
                # Do not draw links between stateless nodes
                if not draw_stateless and not t.stateful and t.static and d.source.name == d.target.name:
                    continue
                if t.used:
                    is_rerouted = True if d.target.rerouted.name != '' else False
                    alpha = 1.0 if d.used else 0.5
                    color = oc.ecolor.used if d.used else oc.ecolor.excluded
                else:
                    # is_rerouted = True if d.target.rerouted.name == 'REROUTED' else False
                    is_rerouted = False
                    alpha = 0.5
                    color = oc.ecolor.excluded
                linestyle = "--" if is_rerouted else "-"
                source_name = f"{d.source.name}({d.source.tick})"
                target_name = f"{d.target.name}({d.target.tick})"

                G.add_edge(source_name, target_name, dependency=d, used=d.used, ts=d.target.ts, rerouted=d.target.rerouted,
                           is_rerouted=is_rerouted, color=color, linestyle=linestyle, alpha=alpha)

    # Get edge and node properties
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data['color'] for u, v, data in edges]
    edge_alpha = [data['alpha'] for u, v, data in edges]
    edge_style = [data['linestyle'] for u, v, data in edges]
    node_alpha = [data['alpha'] for n, data in nodes]
    node_ecolor = [data['edgecolor'] for n, data in nodes]
    node_fcolor = [data['facecolor'] for n, data in nodes]

    # Get labels
    edge_labels = {(u, v): f"{data['ts']:.3f}" for u, v, data in edges}
    if node_labeltype == "tick":
        node_labels = {n: data["tick"] for n, data in nodes}
    elif node_labeltype == "ts":
        node_labels = {n: f"{data['ts_step']:.3f}" for n, data in nodes}
    else:
        raise NotImplementedError("label_type must be 'tick' or 'ts'")

    # Get position
    # pos = nx.spring_layout(G, pos=pos, fixed=fixed_pos)

    # Draw graph
    nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                           node_size=node_size, linewidths=node_linewidth)
    nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                           arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                           width=edge_linewidth, node_size=node_size)

    # Draw labels
    if draw_nodelabels:
        nx.draw_networkx_labels(G, pos, node_labels, font_size=node_fontsize)
    if draw_edgelabels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, rotate=True, bbox=edge_bbox, font_size=edge_fontsize)

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.used, label="dep")
    ax.plot([], [], color=oc.ecolor.used, label="dep (rerouted)", linestyle="--")
    for (name, e), (_, f) in zip(ecolor.items(), fcolor.items()):
        ax.scatter([], [], edgecolor=e, facecolor=f, label=name)

    if draw_excluded:
        ax.plot([], [], color=oc.ecolor.excluded, label="excl. dep", alpha=0.5)
        ax.plot([], [], color=oc.ecolor.excluded, label="excl. dep (rerouted)", linestyle="--", alpha=0.5)
        ax.scatter([], [], edgecolor=oc.ecolor.excluded, facecolor=oc.fcolor.excluded, alpha=0.5, label="excl. step")

    # Set ticks
    yticks = ax.get_yticks().tolist()
    [yticks.append(i) for _, i in y.items()]
    ax.set_yticks(yticks)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def plot_topological_order(
        ax: "matplotlib.Axes",
        record: log_pb2.TraceRecord,
        y: float = 0.0,
        xmax: float = None,
        cscheme: Dict[str, str] = None,
        node_labeltype: str = "tick",
        node_size: int = 250,
        edge_fontsize=8,
        node_fontsize=10,
        edge_linewidth=2.0,
        node_linewidth=1.5,
        arrowsize=10,
        arrowstyle="->",
        connectionstyle="arc3",
        edge_bbox=None,
        draw_edgelabels=True,
        draw_nodelabels=True
):
    """Plot topological order of a trace record.

    Args:
    :param ax: Matplotlib axes.
    :param record: Trace record.
    :param y: Y position of the graph.
    :param xmax: Maximum x position of the graph.
    :param cscheme: Color scheme.
    :param node_labeltype: Node label type. Can be "tick" or "ts".
    :param node_size: Node size.
    :param edge_fontsize: Edge font size.
    :param node_fontsize: Node font size.
    :param edge_linewidth: Edge line width.
    :param node_linewidth: Node line width.
    :param arrowsize: Arrow size.
    :param arrowstyle: Arrow style.
    :param connectionstyle: Connection style.
    :param edge_bbox: Edge bbox.
    :param draw_edgelabels: Draw edge labels with ts (=True) or not (=False).
    :param draw_nodelabels: Draw node labels with ts/tick (=True) or not (=False).
    """
    import rex.open_colors as oc

    # Determine edge bbox
    if edge_bbox is None:
        edge_bbox = dict(boxstyle="round", fc=oc.ccolor("gray"), ec=oc.ccolor("gray"), alpha=0.0)

    # Add color of nodes that are not in the cscheme
    cscheme = cscheme if isinstance(cscheme, dict) else {}
    for n in record.node:
        if n.name not in cscheme:
            cscheme[n.name] = "gray"
        else:
            assert cscheme[n.name] != "red", "Color red is reserved for excluded nodes."

    # Generate node color scheme
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Determine position
    fixed_pos: Dict[str, bool] = {}
    pos: Dict[str, Tuple[float, float]] = {}

    # Generate graph
    G = nx.MultiDiGraph()
    for idx, t in enumerate(record.used):
        # Skip if trace is past tmax
        if xmax is not None and t.ts_step > xmax:
            break

        # Add node to graph
        name = f"{t.name}({t.tick})"
        edgecolor = ecolor[t.name]
        facecolor = fcolor[t.name]
        alpha = 1.0
        G.add_node(name, trace=t, name=t.name, used=t.used, tick=t.tick, ts_step=t.ts_step, edgecolor=edgecolor,
                   facecolor=facecolor, alpha=alpha)

        # Add (initial) position
        pos[name] = (t.index, y)

        # Add fixed position (not used)
        fixed_pos[name] = True

        # Add edge with previous node
        if t.index > 0:
            alpha = 1.0
            linestyle = "-"
            color = oc.ecolor.used
            source = record.used[t.index - 1]
            source_name = f"{source.name}({source.tick})"
            target_name = f"{t.name}({t.tick})"
            num_downstream = len(source.downstream)
            G.add_edge(source_name, target_name, num_downstream=num_downstream, source=source, target=t, color=color,
                       linestyle=linestyle, alpha=alpha)

    # Get edge and node properties
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data['color'] for u, v, data in edges]
    edge_alpha = [data['alpha'] for u, v, data in edges]
    edge_style = [data['linestyle'] for u, v, data in edges]
    node_alpha = [data['alpha'] for n, data in nodes]
    node_ecolor = [data['edgecolor'] for n, data in nodes]
    node_fcolor = [data['facecolor'] for n, data in nodes]

    # Get labels
    edge_labels = {(u, v): f"{data['num_downstream']}" for u, v, data in edges}
    if node_labeltype == "tick":
        node_labels = {n: data["tick"] for n, data in nodes}
    elif node_labeltype == "ts":
        node_labels = {n: f"{data['ts_step']:.3f}" for n, data in nodes}
    else:
        raise NotImplementedError("label_type must be 'tick' or 'ts'")

    # Get position
    # pos = nx.spring_layout(G, pos=pos, fixed=fixed_pos)

    # Draw graph
    nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                           node_size=node_size, linewidths=node_linewidth)
    nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                           arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                           width=edge_linewidth, node_size=node_size)

    # Draw labels
    if draw_nodelabels:
        nx.draw_networkx_labels(G, pos, node_labels, font_size=node_fontsize)
    if draw_edgelabels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, rotate=True, bbox=edge_bbox, font_size=edge_fontsize,
                                     verticalalignment="bottom", label_pos=0.6)

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.used, label="update deps")
    for (name, e), (_, f) in zip(ecolor.items(), fcolor.items()):
        ax.scatter([], [], edgecolor=e, facecolor=f, label=name)

    # Set ticks
    yticks = ax.get_yticks().tolist()
    yticks.append(y)
    ax.set_yticks(yticks)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def plot_depth_order(ax: "matplotlib.Axes",
                     record: log_pb2.TraceRecord,
                     xmax: float = None,
                     order: List[str] = None,
                     cscheme: Dict[str, str] = None,
                     node_labeltype: str = "tick",
                     node_size: int = 300,
                     edge_fontsize=10,
                     node_fontsize=10,
                     edge_linewidth=2.0,
                     node_linewidth=1.5,
                     arrowsize=10,
                     arrowstyle="->",
                     connectionstyle="arc3",
                     edge_bbox=None,
                     draw_edgelabels=False,
                     draw_nodelabels=True,
                     draw_excess=True,
                     draw_stateless=True):
    """
    :param ax:
    :param record:
    :param xmax: Maximum time to plot the computation graph for.
    :param order: Order in which the nodes are placed in y-direction.
    :param cscheme: Color scheme for the nodes.
    :param node_labeltype:
    :param node_size:
    :param edge_fontsize:
    :param node_fontsize:
    :param edge_linewidth:
    :param node_linewidth:
    :param arrowsize:
    :param arrowstyle:
    :param connectionstyle:
    :param edge_bbox:
    :param draw_edgelabels: Draw edge labels with ts (=True) or not (=False).
    :param draw_nodelabels: Draw node labels with ts/tick (=True) or not (=False).
    :param draw_excess: Draw excess step calls (=True) or not (=False).
    :param draw_stateless: Draw monotonic time constraint (=True) or not (=False). Only relevant for stateless nodes when tracing with static=True.
    :return:
    """
    import rex.open_colors as oc

    # Determine edge bbox
    if edge_bbox is None:
        edge_bbox = dict(boxstyle="round", fc=oc.ccolor("gray"), ec=oc.ccolor("gray"), alpha=1.0)

    # Determine fixed node
    max_dt = 1
    order = order if isinstance(order, list) else []
    order = order + [info.name for info in record.node if info.name not in order]
    y = {name: i * max_dt for i, name in enumerate(order)}
    fixed_pos: Dict[str, bool] = {}
    pos: Dict[str, Tuple[float, float]] = {}

    # Add color of nodes that are not in the cscheme
    cscheme = cscheme if isinstance(cscheme, dict) else {}
    for n in record.node:
        if n.name not in cscheme:
            cscheme[n.name] = "gray"
        else:
            assert cscheme[n.name] != "red", "Color red is reserved for excluded nodes."

    # Generate node color scheme
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Generate graph
    G = nx.MultiDiGraph()
    max_depth = max([u.depth for u in record.used])+1
    depths = [[] for _ in range(max_depth)]
    isolated_node = None
    for t in record.used:
        if not t.used:
            continue

        # Determine the isolated node name
        if t.isolate:
            isolated_node = t.name

        # Skip if trace is past tmax
        if xmax is not None and t.ts_step > xmax:
            continue

        # Add node to graph
        name = f"{t.name}({t.tick})"
        edgecolor = ecolor[t.name] if t.used else oc.ecolor.excluded
        facecolor = fcolor[t.name] if t.used else oc.fcolor.excluded
        alpha = 1.0 if t.used else 0.5
        G.add_node(name, trace=t, name=t.name, used=t.used, tick=t.tick, ts_step=t.ts_step, edgecolor=edgecolor,
                   facecolor=facecolor, alpha=alpha)

        # Add depth
        depths[t.depth].append(t.name)

        # Add (initial) position
        pos[name] = (t.depth*max_dt, y[t.name])

        # Add fixed position (not used)
        fixed_pos[name] = True

        # Add upstream dependencies as edges
        for d in t.upstream:
            if not d.used:
                continue
            # Do not draw links between stateless nodes
            if not draw_stateless and not t.stateful and t.static and d.source.name == d.target.name:
                continue
            is_rerouted = True if d.target.rerouted.name != '' else False
            alpha = 1.0
            color = oc.ecolor.used
            linestyle = "--" if is_rerouted else "-"
            source_name = f"{d.source.name}({d.source.tick})"
            target_name = f"{d.target.name}({d.target.tick})"

            G.add_edge(source_name, target_name, dependency=d, used=d.used, ts=d.target.ts, rerouted=d.target.rerouted,
                       is_rerouted=is_rerouted, color=color, linestyle=linestyle, alpha=alpha)

    if draw_excess:
        for i, depth in enumerate(depths):
            # We do not have excess step calls for the isolated node (if any)
            if isolated_node in depth:
                continue
            # Add excess step calls
            ypos = [(name, pos) for name, pos in y.items() if name not in depth]
            if len(ypos) == len(y.keys()):
                continue
            for name, yy in ypos:
                if name in record.pruned:
                    continue
                if name == isolated_node:
                    continue
                # Add node to graph
                name = f"{name}_excl({i})"
                edgecolor = oc.ecolor.excluded
                facecolor = oc.fcolor.excluded
                alpha = 0.5
                G.add_node(name, trace=None, name=name, used=False, tick="", ts_step="", edgecolor=edgecolor,
                           facecolor=facecolor, alpha=alpha)

                # Add (initial) position
                pos[name] = (i * max_dt, yy)

                # Add fixed position (not used)
                fixed_pos[name] = True

    # Get edge and node properties
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data['color'] for u, v, data in edges]
    edge_alpha = [data['alpha'] for u, v, data in edges]
    edge_style = [data['linestyle'] for u, v, data in edges]
    node_alpha = [data['alpha'] for n, data in nodes]
    node_ecolor = [data['edgecolor'] for n, data in nodes]
    node_fcolor = [data['facecolor'] for n, data in nodes]

    # Get labels
    edge_labels = {(u, v): f"{data['ts']:.3f}" for u, v, data in edges}
    if node_labeltype == "tick":
        node_labels = {n: data["tick"] for n, data in nodes}
    elif node_labeltype == "ts":
        node_labels = {n: f"{data['ts_step']:.3f}" for n, data in nodes if data["ts_step"] != ""}
    else:
        raise NotImplementedError("label_type must be 'tick' or 'ts'")

    # Draw graph
    nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                           node_size=node_size, linewidths=node_linewidth)
    nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                           arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                           width=edge_linewidth, node_size=node_size)

    # Draw labels
    if draw_nodelabels:
        nx.draw_networkx_labels(G, pos, node_labels, font_size=node_fontsize)
    if draw_edgelabels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, rotate=True, bbox=edge_bbox, font_size=edge_fontsize)

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.used, label="dep")
    ax.plot([], [], color=oc.ecolor.used, label="dep (rerouted)", linestyle="--")
    for (name, e), (_, f) in zip(ecolor.items(), fcolor.items()):
        ax.scatter([], [], edgecolor=e, facecolor=f, label=name)

    if draw_excess:
        ax.scatter([], [], edgecolor=oc.ecolor.excluded, facecolor=oc.fcolor.excluded, alpha=0.5, label="excess step call")

    # Set ticks
    yticks = ax.get_yticks().tolist()
    [yticks.append(i) for _, i in y.items()]
    ax.set_yticks(yticks)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def plot_graph(ax: "matplotlib.Axes",
               record: log_pb2.EpisodeRecord,
               cscheme: Dict[str, str] = None,
               pos: Dict[str, Tuple[float, float]] = None,
               node_size: int = 2000,
               node_fontsize=10,
               edge_linewidth=3.0,
               node_linewidth=2.0,
               arrowsize=10,
               arrowstyle="->",
               connectionstyle="arc3,rad=0.2"):
    import rex.open_colors as oc

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
        G.add_node(n.info.name, name=name, rate=n.info.rate, advance=n.info.advance, phase=n.info.phase, delay=n.info.delay,
                   edgecolor=edgecolor,
                   facecolor=facecolor, alpha=1.0)
        for i in n.inputs:
            linestyle = "-" if i.info.blocking else "--"
            color = oc.ecolor.skip if i.info.skip else oc.ecolor.normal
            G.add_edge(i.info.output, n.info.name, name=i.info.name, blocking=i.info.blocking, skip=i.info.skip,
                       delay=i.info.delay,
                       window=i.info.window, jitter=i.info.jitter, phase=i.info.phase, color=color, linestyle=linestyle,
                       alpha=1.0)

    # Get edge and node properties
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    edge_color = [data['color'] for u, v, data in edges]
    edge_alpha = [data['alpha'] for u, v, data in edges]
    edge_style = [data['linestyle'] for u, v, data in edges]
    node_alpha = [data['alpha'] for n, data in nodes]
    node_ecolor = [data['edgecolor'] for n, data in nodes]
    node_fcolor = [data['facecolor'] for n, data in nodes]

    # Get labels
    # edge_labels = {(u, v): f"{data['delay']:.3f}" for u, v, data in edges}
    node_labels = {n: data["name"] for n, data in nodes}

    # Get position
    pos = nx.spring_layout(G, pos=pos, fixed=fixed_pos)

    # Draw graph
    nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                           node_size=node_size, linewidths=node_linewidth, node_shape="s")
    nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                           arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                           width=edge_linewidth, node_size=node_size)

    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=node_fontsize)
    # if draw_edgelabels:
    # 	nx.draw_networkx_edge_labels(G, pos, edge_labels, rotate=True, bbox=edge_bbox, font_size=edge_fontsize)

    # Add empty plot with correct color and label for each node
    ax.plot([], [], color=oc.ecolor.normal, label="blocking")
    ax.plot([], [], color=oc.ecolor.skip, label="skip")
    ax.plot([], [], color=oc.ecolor.normal, label="non-blocking", linestyle="--")
