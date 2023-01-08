import rex.open_colors as oc
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set()
except ImportError:
    print("Seaborn not installed, plots will not be as pretty.")
from rex.proto import log_pb2
from rex.tracer import trace
from rex.plot import plot_computation_graph, plot_topological_order

with open("/home/r2ci/rex/scripts/record_1.pb", "rb") as f:
    record = log_pb2.EpisodeRecord()
    record.ParseFromString(f.read())

# Convert to dict
rdict = {n.info.name: n for n in record.node}

# Set actuator to be stateless
rdict["actuator"].info.stateful = False
rdict["agent"].info.stateful = True
rdict['actuator'].inputs[0].info.window = 2

# Trace record
traceback = trace(record, "agent", -1, static=True)

# Create new plot
fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
ax.set(facecolor=oc.ccolor("gray"), xlabel="time (s)", yticks=[], xlim=[-0.01, 0.3])

# Plot graph
order = ["world", "sensor", "observer", "agent", "actuator"]
cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}
plot_computation_graph(ax, traceback, order=order, cscheme=cscheme, xmax=0.6, draw_excluded=True, draw_stateless=False,
                       draw_edgelabels=False, draw_nodelabels=True)

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
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

plot_topological_order(ax, traceback, xmax=0.6, cscheme=cscheme)

# Plot legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
by_label = dict(sorted(by_label.items()))
ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
          bbox_to_anchor=(1.0, 0.50))

plt.show()
