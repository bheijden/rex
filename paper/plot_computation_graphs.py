import matplotlib.pyplot as plt
import seaborn as sns

import rex.supergraph as sg
from rex.proto import log_pb2
import rex.plot

import envs.vx300s.planner.rex as planner

if __name__ == "__main__":
    # Setup sns plotting
    sns.set(style="whitegrid", font_scale=1.5)
    scaling = 5
    MUST_BREAK = False
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 6 * scaling
    plt.rcParams['legend.fontsize'] = 5 * scaling
    plt.rcParams['font.size'] = 7 * scaling
    plt.rcParams['xtick.labelsize'] = 5 * scaling
    plt.rcParams['ytick.labelsize'] = 5 * scaling
    plt.rcParams['xtick.major.pad'] = -0.0 * scaling
    plt.rcParams['ytick.major.pad'] = -0.0 * scaling
    plt.rcParams['lines.linewidth'] = 0.65 * scaling
    plt.rcParams['lines.markersize'] = 4.0 * scaling
    plt.rcParams['axes.xmargin'] = 0.0
    plt.rcParams['axes.ymargin'] = 0.0
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Rescale figures
    width_points = 245.71811 * scaling
    width_inches = width_points / plt.rcParams['figure.dpi']
    default_figsize = plt.rcParams['figure.figsize']
    rescaled_width = width_inches
    rescaled_height = width_inches * default_figsize[1] / default_figsize[0]
    rescaled_figsize = [rescaled_width, rescaled_height]
    fullwidth_figsize = [c*s for c, s in zip([1, 0.52], rescaled_figsize)]
    thirdwidth_figsize = [c * s for c, s in zip([1 / 3, 0.5], rescaled_figsize)]
    twothirdwidth_figsize = [c * s for c, s in zip([2 / 3, 0.5], rescaled_figsize)]
    sixthwidth_figsize = [c * s for c, s in zip([1 / 6, 0.5], rescaled_figsize)]
    print("Default figsize:", default_figsize)
    print("Rescaled figsize:", rescaled_figsize)
    print("Fullwidth figsize:", fullwidth_figsize)
    print("Thirdwidth figsize:", thirdwidth_figsize)
    print("Twothirdwidth figsize:", twothirdwidth_figsize)
    print("Sixthwidth figsize:", sixthwidth_figsize)

    # Plot computation graphs
    LOG_PATH = "/home/r2ci/rex/paper/logs"

    # Load proto experiment record
    PENDULUM_DIR = f"{LOG_PATH}/real-0.5Q-20hz-async-sbx-eps10-disc-pendulum-envs.pendulum.real.world-wall-clock-real-time-frequency-latest-2023-08-16-1201"
    record_pre = log_pb2.ExperimentRecord()
    with open(f"{PENDULUM_DIR}/record_pre.pb", "rb") as f:
        record_pre.ParseFromString(f.read())
    G = sg.create_graph(record_pre.episode[0])

    # Remove nodes
    rm_kinds = ["world"]
    rm_nodes = [u for u, data in G.nodes(data=True) if data["kind"] in rm_kinds]
    G.remove_nodes_from(rm_nodes)

    # Plot graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)

    # Pendulum
    order = ["actuator", "agent", "sensor"]
    cscheme = {"world": "gray", "sensor": "grape", "agent": "teal", "actuator": "indigo", "render": "yellow", "estimator": "orange"}

    rex.plot.plot_computation_graph(ax, G, order=order, cscheme=cscheme, xmax=10.0, node_size=200,
                                    connectionstyle="arc3,rad=0.1", draw_nodelabels=False, draw_pruned=False)
    ax.set_xlim(0, 3)
    plt.show()

    # Load proto experiment record
    BOX_DIR = f"{LOG_PATH}/2023-12-12-1123_real_rex_randomeps_MCS_recorded_VarHz_3iter_vx300s"
    record_pre = log_pb2.ExperimentRecord()
    with open(f"{BOX_DIR}/record_pre.pb", "rb") as f:
        record_pre.ParseFromString(f.read())
    G = sg.create_graph(record_pre.episode[0])

    # Remove nodes
    rm_kinds = ["supervisor", "viewer", "world"]
    rm_nodes = [u for u, data in G.nodes(data=True) if data["kind"] in rm_kinds]
    G.remove_nodes_from(rm_nodes)

    # Plot graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)

    # Pendulum
    # order = ["world", "sensor", "agent", "actuator"]
    # cscheme = {"world": "gray", "sensor": "grape", "agent": "teal", "actuator": "indigo", "render": "yellow", "estimator": "orange"}

    order = ["armactuator", "controller", "planner", "armsensor", "boxsensor"]
    cscheme = {"world": "gray", "armsensor": "grape", "boxsensor": "grape", "supervisor": "teal", "viewer": "teal",
               "planner": "indigo", "controller": "orange", "armactuator": "orange", "cost": "yellow"}

    rex.plot.plot_computation_graph(ax, G, order=order, cscheme=cscheme, xmax=2.0, node_size=200,
                                    connectionstyle="arc3,rad=0.1", draw_nodelabels=False, draw_pruned=False)
    ax.set_xlim(1, 1.5)
    plt.show()