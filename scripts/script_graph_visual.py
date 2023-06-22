import seaborn as sns

from rex.plot import plot_graph
import rex.open_colors as oc
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, SYNC, ASYNC, REAL_TIME, FAST_AS_POSSIBLE, FREQUENCY, \
	PHASE, SIMULATED, WALL_CLOCK
from scripts.dummy import build_dummy_graph

# Function imports
from rex.proto import log_pb2
import matplotlib.pyplot as plt


if __name__ == "__main__":
	sns.set()

	# Create dummy graph
	nodes = build_dummy_graph()

	# Get record
	record = log_pb2.EpisodeRecord()
	[record.node.append(node.record()) for node in nodes.values()]

	# Create new plot
	fig, ax = plt.subplots()
	fig.set_size_inches(12, 5)
	ax.set(facecolor=oc.ccolor("gray"),yticks=[], xticks=[])

	# Draw graph
	pos = {"world": (0, 0), "sensor": (1.5, 0), "observer": (3, 0), "root": (4.5, 0)}
	cscheme = {"sensor": "grape", "observer": "pink", "root": "teal", "actuator": "indigo"}
	plot_graph(ax, record, cscheme=cscheme, pos=pos)

	# Plot legend
	handles, labels = ax.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	by_label = dict(sorted(by_label.items()))
	ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
	          bbox_to_anchor=(1.0, 0.50))
	plt.show()
