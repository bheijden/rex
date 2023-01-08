import seaborn as sns

from rex.plot import plot_graph
import rex.open_colors as oc
from rex.distributions import Gaussian
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, SYNC, ASYNC, REAL_TIME, FAST_AS_POSSIBLE, FREQUENCY, \
	PHASE, SIMULATED, WALL_CLOCK
from scripts.dummy import DummyNode, DummyAgent

# Function imports
from rex.proto import log_pb2
import matplotlib.pyplot as plt


if __name__ == "__main__":
	sns.set()

	# Create nodes
	world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000), color="magenta")
	sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(0.007), color="yellow")
	observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016), color="cyan")
	agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(0.005, 0.001), color="blue", advance=True)
	actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), color="green", advance=False)
	nodes = [world, sensor, observer, agent, actuator]

	# Connect
	sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST, name="testworld")
	observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
	observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
	agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
	actuator.connect(agent, blocking=False, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
	world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)

	# Get record
	record = log_pb2.EpisodeRecord()
	[record.node.append(node.record) for node in nodes]

	# Create new plot
	fig, ax = plt.subplots()
	fig.set_size_inches(12, 5)
	ax.set(facecolor=oc.ccolor("gray"),yticks=[], xticks=[])

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
	plt.show()
