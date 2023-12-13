import os
import time
import tempfile
import datetime
import tqdm
import yaml

import jax.numpy as jnp
import jumpy.numpy as jp
import jumpy.random
import numpy as onp
import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

jnp.set_printoptions(precision=2, suppress=True)

import rex
import rex.jumpy as rjp
from rex.utils import timer, make_put_output_on_device
import rex.utils as utils
from rex.base import StepState
from rex.supergraph import create_graph
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
    ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, INFO, DEBUG
import experiments as exp
import envs.vx300s as vx300s

RECORD_SETTINGS = {"planner": dict(node=False, outputs=True, rngs=False, states=False, params=False, step_states=True),
                   "world": dict(node=False, outputs=False, rngs=True, states=False, params=False, step_states=False),
                   "armactuator": dict(node=False, outputs=True, rngs=False, states=False, params=False, step_states=False),
                   "controller": dict(node=False, outputs=True, rngs=False, states=False, params=False, step_states=False),
                   "armsensor": dict(node=False, outputs=True, rngs=False, states=False, params=False, step_states=False),
                   "boxsensor": dict(node=False, outputs=True, rngs=False, states=False, params=False, step_states=False),
                   "supervisor": dict(node=False, outputs=False, rngs=False, states=True, params=False, step_states=False),
                   "viewer": dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False)}
PATH_VX300S = os.path.dirname(vx300s.__file__)

GRAPH_DIR = "2023-12-12-0820_real_brax_3Hz_3iter_vx300s"  # todo: experiment
CONFIG = {"mjx": {"xml_path": f"{PATH_VX300S}/assets/vx300s_mjx.xml"},
          "brax": {"xml_path": f"{PATH_VX300S}/assets/vx300s_brax.xml"},
          "real": {"cam_trans": [0.8699584189419002,
                                 0.7588910164656042,
                                 0.44003629704333735],
                   "cam_rot": [0.31350890708252344,
                               0.7593333938361413,
                               -0.5259166755043375,
                               -0.22031026442650153], "cam_idx": 0,  # todo: experiment
                   "block_until_detection": True, "block_between_episodes": True,  # todo: experiment
                   "cam_intrinsics": f"{PATH_VX300S}/assets/logitech_c170.yaml"},
          "planner": {"type": "brax",  # todo: experiment
                      "use_estimator": True, "randomize_eps": True, "num_cost_mpc": 1, "num_cost_est": 1,  # todo: experiment
                      "brax_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_brax.xml",
                      "rex_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_brax.xml",
                      "mjx_xml_path": f"{PATH_VX300S}/assets/vx300s_cem_mjx.xml",
                      # "rex_graph_path": "/home/r2ci/rex/logs/real_rex_randomeps_largeS_mock_10eps_vx300s_2023-12-08-1653/record_pre.pb",  # todo: experiment
                      # "rex_graph_path": "/home/r2ci/rex/logs/real_rex_randomeps_smallS_mock_10eps_vx300s_2023-12-08-1811/record_pre.pb",  # todo: experiment
                      # "rex_graph_path": "/home/r2ci/rex/logs/2023-12-11-1614_real_rex_smallS_prevgraph_3Hz_3iter_vx300s/record_pre.pb",  # todo: experiment USE THIS REX
                      "rex_graph_path": f"/home/r2ci/rex/logs/{GRAPH_DIR}/record_pre.pb",
                      "dist_path":      f"/home/r2ci/rex/logs/{GRAPH_DIR}/distributions.pkl",
                      "supergraph_mode": "topological",  # todo: experiment
                      "z_fixed": 0.051,  # todo: experiment
                      "overwrite_planner_rate": False,  # todo: experiment
                      # "u_max": [0.25*3.14, 0.1*3.14, 0.1*3.14, 0.1*3.14, 0.35*3.14, 0.35*3.14],
                      "u_max": [0.25*3.14, 0.25*3.14, 0.25*3.14, 0.25*3.14, 0.25*3.14, 0.25*3.14],
                      "horizon": 2, "dt": 0.15, "dt_substeps": 0.015, "num_samples": 75, "max_iter": 3},
          "viewer": {"xml_path": f"{PATH_VX300S}/assets/vx300s_cem_mjx.xml"}}

if __name__ == "__main__":
    # todo: set advance=true && LATEST for armsensor and boxsensor --> why does planner have num_msgs=2 for t=0?
    # Environment
    ENV = "vx300s"  # "disc_pendulum"
    DIST_FILE = CONFIG["planner"]["dist_path"]
    # DIST_FILE = "real_rex_randomeps_largeS_mock_10eps_vx300s_2023-12-08-1653.pkl"  # todo: experiment
    # DIST_FILE = "2023-12-11-1614_real_rex_smallS_prevgraph_3Hz_3iter_vx300s"  # todo: experiment USE THIS REX
    # DIST_FILE = "real_rex_randomeps_smallS_mock_10eps_vx300s_2023-12-08-1811.pkl"  # todo: experiment
    # DIST_FILE = "2023-12-11-1554_real_rex_prevgraph_3Hz_3iter_vx300s.pkl"  # todo: experiment
    JITTER = BUFFER
    SCHEDULING = FREQUENCY

    ENV_FN = vx300s.real.build_vx300s  # todo: experiment
    CLOCK = WALL_CLOCK  # todo: experiment
    RTF = REAL_TIME  # todo: experiment
    RATES = dict(world=80, supervisor=8, planner=3, controller=20, armactuator=20, armsensor=80, boxsensor=10, viewer=20)  # todo: experiment
    # RATES = dict(world=25, planner=25, controller=25, armactuator=25, armsensor=25, boxsensor=25)
    MAX_STEPS = int(25 * RATES["supervisor"])  # todo: experiment
    WIN_PLANNER = 3
    DELAY_PLANNER = None  # 0.15
    USE_DELAYS = True
    DELAY_FN = lambda d: d.quantile(0.99)*int(USE_DELAYS)

    # Logging
    # NAME = f"real_2ndcalib_rex_randomeps_topological_recorded_3Hz_3iter_{ENV}"  # todo: experiment
    NAME = f"real_2ndcalib_brax_3Hz_3iter_{ENV}"  # todo: experiment
    LOG_DIR = os.path.dirname(rex.__file__) + f"/../logs/{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}_{NAME}"
    PROGRESS_BAR = True
    MUST_LOG = True  # todo: experiment
    MUST_DIST = True  # todo: experiment
    MUST_PLOT = True
    SHOW_PLOTS = True

    # Training
    SEED = 0
    RNG = jumpy.random.PRNGKey(SEED)
    NUM_EVAL = 10  # todo: experiment

    # Load distributions
    delays_sim = exp.load_distributions(DIST_FILE, module=vx300s.dists) if DIST_FILE is not None else vx300s.get_default_distributions()
    # delays_sim = vx300s.get_nodelay_distributions()
    fig_sim_step, fig_sim_inputs = vx300s.plot_dists(delays_sim) if MUST_DIST and MUST_PLOT else (None, None)

    # overwrite delay
    if CONFIG["planner"]["overwrite_planner_rate"]:
        RATES["planner"] = float(1/(1.2*delays_sim["step"]["planner"].quantile(0.5)))
        print(f"Overwriting planner rate to {RATES['planner']} Hz")

    # Prepare environment
    env = vx300s.make_env(delays_sim, DELAY_FN, RATES, CONFIG, win_planner=WIN_PLANNER, delay_planner=DELAY_PLANNER,
                          env_fn=ENV_FN, name=ENV, scheduling=SCHEDULING, jitter=JITTER, clock=CLOCK, real_time_factor=RTF,
                          max_steps=MAX_STEPS, use_delays=USE_DELAYS)

    # Jit functions
    env.graph.init = jax.jit(env.graph.init, static_argnames=["order"], device=cpu_device)
    env._get_cost = jax.jit(env._get_cost, device=cpu_device)
    env.graph.nodes_and_root["world"].step = jax.jit(env.graph.nodes_and_root["world"].step, device=cpu_device)
    env.graph.nodes_and_root["planner"].step = make_put_output_on_device(jax.jit(env.graph.nodes_and_root["planner"].step, device=gpu_device), cpu_device)
    env.graph.nodes_and_root["controller"].step = jax.jit(env.graph.nodes_and_root["controller"].step, device=cpu_device)
    env.graph.nodes_and_root["armsensor"].step = jax.jit(env.graph.nodes_and_root["armsensor"].step, device=cpu_device)
    env.graph.nodes_and_root["boxsensor"].step = jax.jit(env.graph.nodes_and_root["boxsensor"].step, device=cpu_device)
    env.graph.nodes_and_root["armactuator"].step = jax.jit(env.graph.nodes_and_root["armactuator"].step, device=cpu_device)
    if "viewer" in env.graph.nodes:
        env.graph.nodes_and_root["viewer"].step = jax.jit(env.graph.nodes_and_root["viewer"].step, device=cpu_device)

    # Warmup
    with timer(f"warmup[graph_state]", log_level=100):
        graph_state = env.graph.init(RNG, order=("supervisor", "world"))
    with timer(f"eval[graph_state]", log_level=100):
        _ = env.graph.init(RNG, order=("supervisor", "world"))
    for name, node in env.graph.nodes.items():
        with timer(f"warmup[{name}]", log_level=100):
            ss, o = node.step(graph_state.nodes[name])
        _, _ = node.step(graph_state.nodes[name])
        t = timer(f"eval[{name}]", log_level=0)
        with t:
            num_evals = 4
            for _ in range(num_evals):
                _, _ = node.step(graph_state.nodes[name])
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
        print(f"{t.name} | Elapsed: {t.duration / num_evals: .4f}")
    with timer(f"warmup[cost]", log_level=100):
        cost, info = env._get_cost(graph_state)
    with timer(f"eval[cost]", log_level=100):
        _, _ = env._get_cost(graph_state)
    with timer(f"warmup[dist]", log_level=100):
        [n.warmup(graph_state) for n in env.graph.nodes_and_root.values()]

    # Evaluate
    # _, _ = vx300s.eval_env(env, lambda step_state: 1, 2, progress_bar=PROGRESS_BAR, record_settings=RECORD_SETTINGS, seed=SEED)  # todo: experiment
    record_pre, rwds = vx300s.eval_env(env, lambda step_state: 1, NUM_EVAL, progress_bar=PROGRESS_BAR, record_settings=RECORD_SETTINGS, seed=SEED)
    print("Done with eval.")

    # Save html
    # todo: brax timestamps are not correct.
    # todo: modify timestep.
    if "viewer" in env.graph.nodes:
        env.graph.nodes["viewer"].close()
    # rollout = [ss.state.pipeline_state for ss in env.world._record_step_states]
    # env.world.view_rollout(rollout=rollout, path="./vx300s_render.html")

    # Plot computation graph
    g = create_graph(record_pre.episode[-1])
    fig_gr, _ = vx300s.show_graph(record_pre.episode[-1]) if MUST_PLOT else (None, None)
    fig_cg, _ = vx300s.show_computation_graph(g, root="supervisor", xmax=2.0) if MUST_PLOT else (None, None)
    for i, e in enumerate(record_pre.episode[:-1]):
        if i > 2:
            break
        fig_com, _ = vx300s.show_communication(e) if MUST_PLOT else (None, None)
    fig_com, _ = vx300s.show_communication(record_pre.episode[-1]) if MUST_PLOT else (None, None)

    # Plot experiment
    fig_ee, fig_jpos, fig_cost = vx300s.show_box_pushing_experiment(record_pre.episode[-1], xml_path=CONFIG["brax"]["xml_path"]) if MUST_PLOT else (None, None, None)
    fig_perf = vx300s.show_box_pushing_performance(record_pre, xml_path=CONFIG["brax"]["xml_path"]) if MUST_PLOT else None

    # Fit distributions
    data, info, est, dist = vx300s.make_delay_distributions(record_pre, num_steps=500, num_components=4, step_size=0.05,
                                                            seed=SEED) if MUST_DIST else (None, None, None, None)
    fig_step, fig_inputs = vx300s.plot_dists(dist, data, info, est) if MUST_DIST and MUST_PLOT else (None, None)

    # Only show
    plt.show() if SHOW_PLOTS else None

    # Log
    if MUST_LOG:
        ENV_FN = ENV_FN.__module__ + "." + ENV_FN.__name__
        DELAY_FN = DELAY_FN.__module__ + "." + DELAY_FN.__name__
        del RNG
        os.mkdir(LOG_DIR)
        # Identify all capitalized variables & save them to file
        capitalized_vars = {k: v for k, v in globals().items() if k.isupper()}
        with open(f"{LOG_DIR}/params.yaml", 'w') as file:
            yaml.dump(capitalized_vars, file)
        # Save envs
        # env.unwrapped.save(f"{LOG_DIR}/{env.unwrapped.name}.pkl")
        # cenv.unwrapped.save(f"{LOG_DIR}/{cenv.unwrapped.name}.pkl")
        # Save pre-train record to file
        with open(LOG_DIR + "/record_pre.pb", "wb") as f:
            f.write(record_pre.SerializeToString())
        # Save plots
        fig_gr.savefig(LOG_DIR + "/robotic_system.png") if fig_gr is not None else None
        fig_cg.savefig(LOG_DIR + "/computation_graph.png") if fig_gr is not None else None
        fig_com.savefig(LOG_DIR + "/communication.png") if fig_gr is not None else None
        # fig_grp.savefig(LOG_DIR + "/grouped_agent_sensor.png") if fig_gr is not None else None
        fig_sim_step.savefig(LOG_DIR + "/delay_sim_step.png") if fig_sim_step is not None else None
        fig_sim_inputs.savefig(LOG_DIR + "/delay_sim_inputs.png") if fig_sim_inputs is not None else None
        fig_step.savefig(LOG_DIR + "/delay_step.png") if fig_step is not None else None
        fig_inputs.savefig(LOG_DIR + "/delay_inputs.png") if fig_inputs is not None else None
        fig_ee.savefig(LOG_DIR + "/ee.png") if fig_ee is not None else None
        fig_jpos.savefig(LOG_DIR + "/jpos.png") if fig_jpos is not None else None
        fig_cost.savefig(LOG_DIR + "/cost.png") if fig_cost is not None else None
        fig_perf.savefig(LOG_DIR + "/performance.png") if fig_perf is not None else None

        # Save to file
        import dill as pickle

        if MUST_DIST:
            with open(LOG_DIR + "/distributions.pkl", "wb") as f:
                pickle.dump(dist, f)
