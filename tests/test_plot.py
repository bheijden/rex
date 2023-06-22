import dill as pickle
import jumpy
import time
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import rex.utils as utils
from rex.constants import SILENT, DEBUG, INFO, WARN, SIMULATED, WALL_CLOCK
from rex.tracer import get_network_record
from scripts.dummy import build_dummy_env

utils.set_log_level(WARN)


def test_plot():
    env, nodes = build_dummy_env()

    # Simulate
    tstart = time.time()
    graph_state, obs, info = env.reset(jumpy.random.PRNGKey(0))
    steps = 0
    while True:
        steps += 1
        graph_state, obs, reward, truncated, done, info = env.step(graph_state, None)
        if done:
            tend = time.time()
            env.stop()
            print(f"agent_steps={steps} | t={(tend - tstart): 2.4f} sec | fps={steps / (tend - tstart): 2.4f}")
            break

    # Gather the records
    record = env.graph.get_episode_record()
    d = {n.info.name: n for n in record.node}

    from rex.plot import plot_input_thread, plot_event_thread
    import rex.open_colors as oc

    # Create new plots
    fig, ax = plt.subplots()
    xlim = [-0.001, 0.3]
    ax.set(ylim=[-18, 95], xlim=xlim, yticks=[], facecolor=oc.ccolor("gray"))
    ystart, dy, margin = 90, -10, 4

    # Plot all thread traces
    ystart = plot_input_thread(ax, d["world"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["world"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["sensor"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["sensor"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["observer"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["observer"], ystart=ystart, dy=dy)
    ystart = plot_input_thread(ax, d["observer"].inputs[1], ystart=ystart, dy=dy / 2, name="")

    ystart = plot_input_thread(ax, d["agent"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["agent"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["actuator"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["actuator"], ystart=ystart, dy=dy)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))

    from rex.plot import plot_grouped

    # Create new plot
    fig, ax = plt.subplots()
    ax.set(ylim=xlim, xlim=xlim, yticks=[], facecolor=oc.ccolor("gray"))

    # Function arguments
    # plot_grouped(ax, d["observer"], "agent")
    plot_grouped(ax, d["actuator"])

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))

    from rex.plot import plot_delay

    # Create new plots
    fig, axes = plt.subplots(ncols=2)
    [ax.set(facecolor=oc.ccolor("gray")) for ax in axes.flatten().tolist()]

    # Plot delays
    plot_delay(axes[0], d["agent"], clock=WALL_CLOCK)
    plot_delay(axes[1], d["actuator"].inputs[0], clock=SIMULATED)

    # axes[0].set(xlim=[0, 0.0025])
    axes[1].set(xlim=[0, 0.01])

    # Plot legend
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    axes[1].legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
                   bbox_to_anchor=(1.0, 0.50))

    from rex.plot import plot_step_timing

    # Create new plots
    for combinations in (["delayed", "advanced", "ontime"], ["delayed"], ["advanced"], ["ontime"]):
        fig, ax = plt.subplots()
        ax.set(facecolor=oc.ccolor("gray"))

        # Plot step timing variation
        plot_step_timing(ax, d["actuator"], combinations)

        # Plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=1, loc='center left', fancybox=True, shadow=False, bbox_to_anchor=(1.0, 0.50))

    # Trace steps
    # Set actuator to be stateless
    d["actuator"].info.stateful = False
    # d['actuator'].inputs[0].info.window = 2

    # Trace record
    root = "agent"
    order = ["world", "sensor", "observer", "agent", "actuator"]
    cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}
    split_mode = "topological"
    supergraph_mode = "MCS"
    record_network, MCS, lst_full, lst_subgraphs = get_network_record(record, root, -1, split_mode=split_mode, supergraph_mode=supergraph_mode, log_level=WARN)
    G = lst_full[0]

    from rex.plot import plot_computation_graph, plot_topological_order, plot_depth_order
    from matplotlib.ticker import MaxNLocator

    # Create new plot
    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches(12, 10)
    axes[0].set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])
    axes[1].set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])

    # Plot graph
    plot_computation_graph(axes[0], G, root=root, order=order, cscheme=cscheme, xmax=0.6, node_size=200, draw_pruned=True,
                           draw_nodelabels=True, node_labeltype="seq")
    plot_computation_graph(axes[1], G, order=order, cscheme=cscheme, xmax=0.6, node_size=200, draw_pruned=False,
                           draw_nodelabels=True, node_labeltype="ts")

    # Plot legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    axes[0].legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
                   bbox_to_anchor=(1.0, 0.50))

    # Create new plot
    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches(12, 10)
    axes[0].set(facecolor=oc.ccolor("gray"), xlabel="Depth order", yticks=[], xlim=[-1, 10])
    axes[1].set(facecolor=oc.ccolor("gray"), xlabel="Depth order", yticks=[], xlim=[-1, 10])
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}

    plot_depth_order(axes[0], G, root=root, MCS=MCS, split_mode=split_mode, supergraph_mode=supergraph_mode, xmax=0.6, cscheme=cscheme, node_labeltype="seq", draw_excess=True)
    plot_depth_order(axes[1], G, root=root, MCS=MCS, split_mode=split_mode, supergraph_mode=supergraph_mode, xmax=0.6, cscheme=cscheme, node_labeltype="ts", draw_excess=False)

    # Plot legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    axes[0].legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))

    # Create new plot
    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches(12, 10)
    axes[0].set(facecolor=oc.ccolor("gray"), xlabel="Topological order", yticks=[], xlim=[-1, 20])
    axes[1].set(facecolor=oc.ccolor("gray"), xlabel="Topological order", yticks=[], xlim=[-1, 20])
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plot_topological_order(axes[0], G, root=root, xmax=0.6, cscheme=cscheme, node_labeltype="seq", draw_excess=True, draw_root_excess=False)
    plot_topological_order(axes[1], G, root=root, xmax=0.6, cscheme=cscheme, node_labeltype="ts", draw_excess=False)

    # Plot legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    axes[0].legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
                   bbox_to_anchor=(1.0, 0.50))

    from rex.plot import plot_graph

    # Create new plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax.set(facecolor=oc.ccolor("gray"), yticks=[], xticks=[])

    # Draw graph
    pos = {"world": (0, 0), "sensor": (1.5, 0), "observer": (3, 0), "agent": (4.5, 0)}
    cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}
    plot_graph(ax, record, cscheme=cscheme, pos=pos)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))

if __name__ == "__main__":
    test_plot()