import experiments as exp


if __name__ == "__main__":
	DIST_FILES = [
		"2023-12-11-1809_real_brax_0.15phase_3Hz_3iter_vx300s",
		"2023-12-12-0740_real_rex_randomeps_topological_smallS_3Hz_3iter_vx300s",
		"2023-12-12-0731_real_rex_randomeps_generational_smallS_3Hz_3iter_vx300s",
		"2023-12-12-0700_real_rex_randomeps_MCS_smallS_3Hz_3iter_vx300s",
	]
	q_50 = []
	q_80 = []
	q_99 = []
	for df in DIST_FILES:
		DIST_FILE = f"/home/r2ci/rex/logs/{df}/distributions.pkl"
		dists = exp.load_distributions(DIST_FILE)
		q_50.append(dists["step"]["planner"].quantile(0.5))
		q_80.append(dists["step"]["planner"].quantile(0.8))
		q_99.append(dists["step"]["planner"].quantile(0.99))

	# Calculate percentile increase with respect to the first one
	import numpy as onp
	q_50 = onp.array(q_50)
	q_80 = onp.array(q_80)
	q_99 = onp.array(q_99)
	print(q_50)
	print(q_80)
	print(q_99)
	print(100*(q_50 - q_50[2]) / q_50[2])
	print(100*(q_80 - q_80[2]) / q_80[2])
	print(100*(q_99 - q_99[2]) / q_99[2])
