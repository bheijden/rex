import pytest
import time
import jax
from rex.constants import WARN
import rex.supergraph as tracer
from tests.dummy import build_dummy_env


@pytest.mark.parametrize("supergraph_mode", ["topological", "MCS", "generational"])
def test_tracer(supergraph_mode):
	env, nodes = build_dummy_env()

	# Simulate
	for eps in range(5):
		graph_state, obs, info = env.reset(jax.random.PRNGKey(0))
		tstart = time.time()
		steps = 0
		while True:
			steps += 1
			graph_state, obs, reward, terminated, truncated, info = env.step(graph_state, None)
			done = terminated | truncated
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
	record_network, S, S_init_to_S, Gs, Gs_monomorphism = tracer.get_network_record(record, root, -1,
		                                                                     supergraph_mode=supergraph_mode, progress_bar=True,
	                                                                         cscheme=cscheme, order=order, validate=True)

	# Get timings
	timings = tracer.get_timings_from_network_record(record_network, progress_bar=True)
	buffer = tracer.get_graph_buffer(S, timings, nodes)
	outputs = tracer.get_outputs_from_timings(S, timings, nodes)
	timings_chron = tracer.get_chronological_timings(S, timings, eps=0)
	seqs_step, updated_step = tracer.get_step_seqs_mapping(S, timings, buffer)

	G = Gs[0]


if __name__ == "__main__":
	test_tracer("MCS")
	test_tracer("generational")
	test_tracer("topological")
