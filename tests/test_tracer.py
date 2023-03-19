import pytest
import time
import jumpy
from rex.constants import WARN
import rex.tracer as tracer
from scripts.dummy import build_dummy_env


@pytest.mark.parametrize("split_mode, supergraph_mode", [("topological", "MCS"),
                                                         ("generational", "MCS"),
                                                         ("topological", "topological")])
def test_tracer(split_mode, supergraph_mode):
	env, nodes = build_dummy_env()

	# Simulate
	tstart = time.time()
	graph_state, obs = env.reset(jumpy.random.PRNGKey(0))
	steps = 0
	while True:
		steps += 1
		graph_state, obs, reward, done, info = env.step(graph_state, None)
		if done:
			tend = time.time()
			env.stop()
			print(f"agent_steps={steps} | t={(tend - tstart): 2.4f} sec | fps={steps / (tend - tstart): 2.4f}")
			break

	# Gather the records
	record = env.graph.get_episode_record()

	# Trace the graph
	root = "agent"
	order = ["world", "sensor", "observer", "agent", "actuator"]
	cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}
	record_network, MCS, lst_full, lst_subgraphs = tracer.get_network_record(record, root, -1, split_mode=split_mode,
		                                                                     supergraph_mode=supergraph_mode, log_level=WARN,
	                                                                         cscheme=cscheme, order=order, validate=True, )

	# Get timings
	timings = tracer.get_timings_from_network_record(record_network, log_level=WARN)
	buffer = tracer.get_graph_buffer(MCS, timings, nodes)
	outputs = tracer.get_outputs_from_timings(MCS, timings, nodes)
	timings_chron = tracer.get_chronological_timings(MCS, timings, eps=0)
	seqs_step, updated_step = tracer.get_step_seqs_mapping(MCS, timings, buffer)

	G = lst_full[0]
	G_subgraphs = lst_subgraphs[0]

	# Convert to topological subgraph
	if supergraph_mode == "topological":
		G_subgraphs_topo = tracer.as_topological_subgraphs(G_subgraphs)
