import dill as pickle
import cv2
import jax.tree_util
import yaml
import os

import rex
import rex.supergraph as sg
import rex.open_colors as oc
from rex.proto import log_pb2

import envs.vx300s as vx300s
import experiments as exp


if __name__ == "__main__":
	PATH_VX300S = os.path.dirname(vx300s.__file__)
	LOG_PATH = "/home/r2ci/rex/paper/logs"
	XML_PATH = f"{PATH_VX300S}/assets/vx300s_cem_brax.xml"
	MAKE_VIDEO = True
	EXP_DIRS = {
		"deterministic_var_image": f"{LOG_PATH}/2023-12-14-1159_real_2ndcalib_brax_VarHz_3iter_record_image_vx300s",
		"mcs_var_image": f"{LOG_PATH}/2023-12-14-1058_real_2ndcalib_rex_randomeps_MCS_recorded_VarHz_3iter_record_imagevx300s",
		"topological_var_image": f"{LOG_PATH}/2023-12-14-1141_real_2ndcalib_rex_randomeps_topological_recorded_VarHz_3iter_record_imagevx300s",
		"generational_var_image": f"{LOG_PATH}/2023-12-14-1131_real_2ndcalib_rex_randomeps_generational_recorded_VarHz_3iter_record_imagevx300s",
		# "mcs_var": f"{LOG_PATH}/2023-12-12-1636_real_2ndcalib_rex_randomeps_MCS_recorded_VarHz_3iter_vx300s",
		# "topological_var": f"{LOG_PATH}/2023-12-12-1718_real_2ndcalib_rex_randomeps_topological_recorded_VarHz_3iter_vx300s",
		# "generational_var": f"{LOG_PATH}/2023-12-12-1708_real_2ndcalib_rex_randomeps_generational_recorded_VarHz_3iter_vx300s",
		# "deterministic_var": f"{LOG_PATH}/2023-12-12-1734_real_2ndcalib_brax_VarHz_3iter_vx300s",
		# "mcs_fixed": f"{LOG_PATH}/2023-12-12-1746_real_2ndcalib_rex_randomeps_MCS_recorded_3Hz_3iter_vx300s",
		# "topological_fixed": f"{LOG_PATH}/2023-12-12-1824_real_2ndcalib_rex_randomeps_topological_recorded_3Hz_3iter_vx300s",
		# "generational_fixed": f"{LOG_PATH}/2023-12-12-1814_real_2ndcalib_rex_randomeps_generational_recorded_3Hz_3iter_vx300s",
		# "deterministic_fixed": f"{LOG_PATH}/2023-12-12-1834_real_2ndcalib_brax_3Hz_3iter_vx300s",
	}
	RECORDS = {}
	PARAMS = {}
	EFFICIENCY = {}
	DELAY = {}
	for name, LOG_DIR in EXP_DIRS.items():
		with open(f"{LOG_DIR}/params.yaml", "r") as f:
			params = yaml.safe_load(f)
		PARAMS[name] = params
		RECORDS[name] = log_pb2.ExperimentRecord()
		with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
			RECORDS[name].ParseFromString(f.read())

		delays_sim = exp.load_distributions(params["DIST_FILE"], module=vx300s.dists)

		# Prepare environment
		DELAY_FN = lambda d: d.quantile(0.99) * int(params["USE_DELAYS"])
		ENV_FN = vx300s.brax.world.build_vx300s
		# params["CONFIG"]["planner"]["num_cost_mpc"] = 5
		# params["CONFIG"]["planner"]["num_cost_est"] = 5
		env = vx300s.make_env(delays_sim, DELAY_FN, params["RATES"], params["CONFIG"],
		                      win_planner=params["WIN_PLANNER"],
		                      delay_planner=params["DELAY_PLANNER"], env_fn=ENV_FN, name=params["ENV"],
		                      scheduling=params["SCHEDULING"],
		                      jitter=params["JITTER"], clock=params["CLOCK"], real_time_factor=params["RTF"],
		                      max_steps=params["MAX_STEPS"], use_delays=params["USE_DELAYS"], viewer=False
		                      )

		# Get graphs
		planner: vx300s.planner.rex.RexCEMPlanner = env.nodes["planner"]
		if not isinstance(planner, vx300s.planner.rex.RexCEMPlanner):
			EFFICIENCY[name] = 100
		else:
			graph_est = planner.graph_est
			graph_mpc = planner.graph_mpc
			graphs = {"est": graph_est, "mpc": graph_mpc}
			not_masked, total = 0, 0
			for name_graph, graph in graphs.items():
				# timings = [eps, step, slot, gen]
				timings = graph._default_timings
				slots_to_kinds = {n: data["kind"] for n, data in graph.S.nodes(data=True)}
				supergraph_size = 0
				for gen in timings:
					for slot, t in gen.items():
						if slots_to_kinds[slot] in graph._skip:
							continue
						supergraph_size += 1
						total += t["run"].size
						not_masked += t["run"].sum()
				efficiency = 100 * not_masked / total
				print(f"{name} | Efficiency {name_graph}: {efficiency:.2f}% | supergraph size: {supergraph_size}")
			EFFICIENCY[name] = efficiency

	DATA, TIMESTAMPS, DELAYS, DATA_INTERP, TIMESTAMPS_INTERP = vx300s.process_box_pushing_performance_data(RECORDS, XML_PATH)

	if MAKE_VIDEO:
		for name, LOG_DIR in EXP_DIRS.items():
			boxsensor = DATA[name]["boxsensor"]["outputs"]
			if not hasattr(boxsensor, "image"):
				continue

			images = boxsensor.image
			height, width, layers = images.shape[-3:]
			for eps, eps_images in enumerate(images):
				video_name = f"{LOG_DIR}/video_{eps}.avi"
				video = cv2.VideoWriter(video_name, 0, 10, (width, height))
				for img in eps_images:
					video.write(img)

	# Save data
	for name, LOG_DIR in EXP_DIRS.items():
		supergraph_type = PARAMS[name]["CONFIG"]["planner"]["supergraph_mode"] if PARAMS[name]["CONFIG"]["planner"]["type"] == "rex" else "deterministic"
		if supergraph_type == "MCS":
			supergraph_type = "mcs"
		EVAL_DATA = {"supergraph_type": supergraph_type,
			         "efficiency": EFFICIENCY[name],
					 "delay": DELAYS[name]["step"]["planner"].mean(),
		             "rate": PARAMS[name]["RATES"]["planner"],
		             "cm": DATA_INTERP[name]["cm"],
		             "timestamps": TIMESTAMPS_INTERP}
		with open(f"{LOG_DIR}/eval_data.pkl", "wb") as f:
			pickle.dump(EVAL_DATA, f)
		# Test unpickle
		with open(f"{LOG_DIR}/eval_data.pkl", "rb") as f:
			EVAL_DATA = pickle.load(f)




