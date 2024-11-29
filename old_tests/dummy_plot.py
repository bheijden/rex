import matplotlib
matplotlib.use("TkAgg")
from rex.constants import SIMULATED


def plot_threads(d):
    # Create new plots
    try:
        import seaborn as sns
        sns.set()
    except ImportError:
        print("Seaborn not installed, plots will not be as pretty.")
    import matplotlib.pyplot as plt
    from rex.plot import plot_input_thread, plot_event_thread
    import rex.open_colors as oc

    fig, ax = plt.subplots()
    fig.set_size_inches(8.4, 4.8)
    ax.set(ylim=[-38, 95], xlim=[-0.001, 0.3], yticks=[], facecolor=oc.ccolor("gray"))
    ystart, dy, margin = 90, -10, 4

    # Plot all thread traces
    ystart = plot_input_thread(ax, d["world"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["world"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["sensor"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["sensor"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["observer"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["observer"], ystart=ystart, dy=dy)
    ystart = plot_input_thread(ax, d["observer"].inputs[1], ystart=ystart, dy=dy / 2, name="")

    ystart = plot_input_thread(ax, d["root"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["root"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["actuator"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["actuator"], ystart=ystart, dy=dy)

    ystart = plot_input_thread(ax, d["world"].inputs[0], ystart=ystart - margin, dy=dy / 2, name="")
    ystart = plot_event_thread(ax, d["world"], ystart=ystart, dy=dy)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))
    plt.show()


def plot_grouped(d):
    # Create new plots
    try:
        import seaborn as sns
        sns.set()
    except ImportError:
        print("Seaborn not installed, plots will not be as pretty.")
        pass
    import matplotlib.pyplot as plt
    import rex.open_colors as oc
    from rex.plot import plot_grouped

    # Create new plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8.4, 4.8)
    ax.set(ylim=[-0.001, 0.3], xlim=[-0.001, 0.3], yticks=[], facecolor=oc.ccolor("gray"))

    # Function arguments
    # plot_grouped(ax, d["observer"], "root")
    plot_grouped(ax, d["actuator"])

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))


def plot_delay(d):
    # Create new plots
    try:
        import seaborn as sns
        sns.set()
    except ImportError:
        print("Seaborn not installed, plots will not be as pretty.")
        pass
    import matplotlib.pyplot as plt
    import rex.open_colors as oc
    from rex.plot import plot_delay

    # Create new plot
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(12.4, 4.8)
    [ax.set(facecolor=oc.ccolor("gray")) for ax in axes.flatten().tolist()]

    # Plot delays
    plot_delay(axes[0], d["root"], clock=SIMULATED)
    plot_delay(axes[1], d["actuator"].inputs[0], clock=SIMULATED)

    # axes[0].set(xlim=[0, 0.0025])
    axes[1].set(xlim=[0, 0.01])

    # Plot legend
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    axes[1].legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
                   bbox_to_anchor=(1.0, 0.50))


def plot_graph(traceback):
    # Create new plots
    try:
        import seaborn as sns
        sns.set()
    except ImportError:
        print("Seaborn not installed, plots will not be as pretty.")
        pass
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import rex.open_colors as oc
    from rex.plot import plot_computation_graph, plot_topological_order

    # Create new plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])

    # Plot graph
    order = ["world", "sensor", "observer", "root", "actuator"]
    cscheme = {"sensor": "grape", "observer": "pink", "root": "teal", "actuator": "indigo"}
    plot_computation_graph(ax, traceback, order=order, cscheme=cscheme, xmax=0.6, node_size=200, draw_excluded=True,
                           draw_stateless=False, draw_edgelabels=False, draw_nodelabels=True)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))

    # Create new plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 2)
    ax.set(facecolor=oc.ccolor("gray"), xlabel="Topological order", yticks=[], xlim=[-1, 20])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plot_topological_order(ax, traceback, xmax=0.6, cscheme=cscheme)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))