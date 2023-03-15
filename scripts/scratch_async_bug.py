import dill as pickle
import time
import jumpy
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import jax

import rex.utils as utils
from rex.constants import SILENT, DEBUG, INFO, WARN

from scripts.dummy import build_dummy_env


if __name__ == "__main__":
	# todo: set agent rate < 15 Hz for bug to appear
	# Set log level
	utils.set_log_level(DEBUG)

	# Build dummy environment
	env, nodes = build_dummy_env()

	# Set
	utils.set_log_level(SILENT, node=nodes["world"], color="cyan")
	utils.set_log_level(SILENT, node=nodes["sensor"], color="blue")
	utils.set_log_level(WARN, node=nodes["observer"], color="yellow")
	utils.set_log_level(WARN, node=nodes["agent"], color="red")
	utils.set_log_level(SILENT, node=nodes["actuator"], color="white")

	# Evaluate
	t_slp = 0.001
	T = 1000
	rng = jumpy.random.PRNGKey(jp.int32(0))
	tstart = time.time()
	eps_steps = 0
	done = False
	print(f"PRE_RESET")
	graph_state, obs = env.reset(rng)
	print(f"POST_RESET")
	time.sleep(t_slp)
	while not done:
		print(f"START STEP {eps_steps}")
		graph_state, obs, reward, done, info = env.step(graph_state, None)
		time.sleep(t_slp)
		print(f"END STEP {eps_steps}")
		eps_steps += 1
		if done or eps_steps > T:
			print("STOPPING")
			env.stop()
			tend = time.time()
			print(
				f"agent_steps={eps_steps} | chunk_index={graph_state.step} | t={(tend - tstart): 2.4f} sec | t_s={(tend - tend): 2.4f} sec | fps={eps_steps / (tend - tstart): 2.4f}")
			break
