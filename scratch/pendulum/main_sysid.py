import dill as pickle
import tqdm
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
JAX_USE_CACHE = False
# if JAX_USE_CACHE:
    # os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

# Cache settings
if JAX_USE_CACHE:
    # More info: https://github.com/google/jax/pull/22271/files?short_path=71526fb#diff-71526fb9807ead876cbde1c3c88a868e56d49888023dd561e6705d403ab026c0
    jax.config.update("jax_compilation_cache_dir", "./cache-rl")
    # -1: disable the size restriction and prevent overrides.
    # 0: Leave at default (0) to allow for overrides.
    #    The override will typically ensure that the minimum size is optimal for the file system being used for the cache.
    # > 0: the actual minimum size desired; no overrides.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # A computation will only be written to the persistent cache if the compilation time is longer than the specified value.
    # It is defaulted to 1.0 second.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_explain_cache_misses", False)  # True --> results in error

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
import envs.pendulum.systems as psys

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    # todo: Methodology:
    #       0. Gather real-data and construct computation graph
    #       1. Estimate simulator parameters + delays (fit simulator to reconstruct real-data) (non-causal)
    #       2. Learn estimator parameters (Causally use real-data to reconstruct simulator state) (causal)
    #       3. Learn estimation error model (Fit error model to difference between estimated states of real-data vs. simulator state) (non-causal)
    #       4. Learn controller parameters (Use simulator + error model provides state estimates to fit controller) (causal)
    # todo: Real pendulum:
    #   - Evaluate throttle in actuator until action.ts - actuator_delay --> This does influence the measured computation delay of the actuator.
    #   - Test with sensor that has delay of camera + detector.
    #   - rectify realsense image with intrinsic calibration.
    #   - Add throttling in actuator, make sure it is reflected in training graph as well.
    # todo: RL
    #   - Speed up if replacing trainable distributions with fixed ones
    #       - Replace delay_dist with static one before generating augmented graphs
    #       - Only replace delay_dist in init_inputs if they are equivalent.
    #   - Speed up if only copying state, rng instead of complete step_state
    # todo: System identification:
    #   - Use multiple episodes with short length for system identification
    #   - Try and identify camera parameters as well.
    #   - ZOH instead of linear interpolation for control inputs
    #   - Camera at lower rate. Separate thread maybe?
    #   - Double check UKF implementation
    #   - optimize for dt_future during system parameter identification.
    #   - what to do with initial th_world for seq=-1?
    #   - how to assign outputs (e.g. use images, detector)
    #   - initialize detector.state.(mincum, maxcum) in a non-causal way.
    #   - double check ts offsets (real vs. sim vs. compiled)
    #   - ts_recv for world is not correct. Set delay to rate.
    #   - Computation delay of world is equal to the rate of the world (currently set to 99% of the rate, should be 100%)
    # todo: Refactor:
    #   - partition_runner.py
    #       - INTERMEDIATE_UPDATE
    #       - only update params? Speed comparison
    #       - replace trainable distributions with fixed ones
    #   - How to deal with advance=True for nodes without inputs (& no blocking connections)?
    #       - Essentially, how to model "polling" nodes? They run as_fast_as_possible in the real_world, but not in the simulator.
    #   - Redefine ukf with pytrees (and use flatten/unflatten), and define cov as a pytree at the leaf of the pytree state.
    #   - Add .throttle method to BaseNode
    #   - Check weaktypes and recompilation & how to compile step function --> leads to more latency (maybe not if jit compiled)?

    # Make sysid nodes
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    EPS_IDX = -1
    WORLD_RATE = 100.
    ID_CAM = False  # Use images to identify camera parameters
    USE_CAM = True  # Use camera instead of sensor in estimator
    USE_BRAX = True  # Use brax for simulation  # todo: change to brax
    SUPERVISOR = "actuator"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    RECORD_FILE = f"{LOG_DIR}/data_sysid_pose2.pkl"
    PARAMS_FILE = f"{LOG_DIR}/sysid_params_pose2.pkl"  # todo: change to brax
    # ORDER = ["camera", "sensor", "actuator", "controller", "estimator", "supervisor"]
    # CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
    #            "actuator": "green", "supervisor": "indigo"}

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Gather outputs
    outputs_sysid = {name: n.steps.output for name, n in record.nodes.items()}
    cam = jax.tree_util.tree_map(lambda x: x[EPS_IDX], outputs_sysid["camera"])

    # Visualize detection
    if True:
        detector = jax.tree_util.tree_map(lambda x: x[0], record.nodes["camera"].params.detector)
        bgr_ellipse = detector.draw_ellipse(cam.bgr, color=(255, 0, 0))
        bgr_centroids = detector.draw_centroids(bgr_ellipse, cam.median)
        detector.play_video(bgr_centroids, fps=60)

    # Create nodes
    nodes_sysid = psys.simulated_system(record, outputs=outputs_sysid, world_rate=WORLD_RATE, use_cam=USE_CAM, id_cam=ID_CAM,
                                        use_brax=USE_BRAX)

    # Set initialization method
    nodes_sysid["supervisor"].set_init_method("parametrized")

    # Generate computation graph
    graphs_real = record.to_graph()
    graphs_aug = rexv2.artificial.augment_graphs(graphs_real, nodes_sysid, RNG)

    # Create compiled graph
    graph_sysid = rexv2.graph.Graph(nodes_sysid, nodes_sysid[SUPERVISOR], graphs_aug, supergraph=SUPERGRAPH, skip=["supervisor"])

    # Visualize graph
    if False:
        MAX_X = 1.0
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(graph_sysid.Gs):
            if i > 1:
                break  # Only plot first two episodes
            supergraph.plot_graph(G, max_x=MAX_X, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()

    # Jit functions
    base_init = jax.jit(graph_sysid.init, static_argnames=["order"], device=GPU_DEVICE)
    for name, node in graph_sysid.nodes.items():
        cpu = next(CPU_DEVICES)
        print(f"Jitting {name} on {cpu}")
        node.async_step = jax.jit(node.async_step, device=cpu)
    with timer(f"warmup[graph_state]", log_level=100):
        gs = base_init(RNG, order=("supervisor", "actuator"))
    # for name, node in graph_sysid.nodes.items():
    #     ss = gs.step_state[name]
    #     default_o = node.init_output(RNG, gs)
    #     with timer(f"warmup[{name}]", log_level=0):
    #         # with jax.log_compiles():
    #         ss, o = node.async_step(ss)
    #     _ = jutils.same_structure(default_o, o, tag=name, raise_on_mismatch=False)  # todo: set to true
    #     _ = jutils.same_structure(gs.step_state[name], ss, tag=name, raise_on_mismatch=False)  # todo: set to true
    #     with timer(f"eval[{name}]", log_level=100, repeat=10):
    #         for _ in range(10):
    #             ss, o = node.async_step(ss)

    # Place gs on GPU
    gs = jax.device_put(gs, device=GPU_DEVICE)

    # System identification
    import envs.pendulum.tasks as tasks
    if True:
        figs = []
        task = tasks.create_sysid_task(graph_sysid, gs).replace(max_steps=200)
        # jit
        task_evaluate = jax.jit(task.evaluate)
        task_solve = jax.jit(task.solve, static_argnames=("verbose",))
        # Initial guess
        init_params = task.to_extended_params(gs, task.init_params)
        with timer("jit_evaluate[init_params]", log_level=100):
            all_gs = task_evaluate(init_params, RNG, EPS_IDX)
        figs += task.plot(all_gs, identifier="init_sysid")
        # plt.show()
        # Solve
        with timer("solve", log_level=100):
            sol_state, opt_params, log_state = task_solve(gs, verbose=not JAX_USE_CACHE)
        # Plot results
        log_state.plot(f"{task.description} | {task.solver.strategy_name}", ylims=[0, 300])
        params = task.to_extended_params(gs, opt_params)
        with timer("jit_evaluate[opt_params]", log_level=100):
            all_gs = task_evaluate(params, RNG, EPS_IDX)
        figs += task.plot(all_gs, identifier="opt_sysid")
        plt.show()
        # Show detector
        num_steps = cam.ts.shape[0]
        fps = num_steps / cam.ts[-1]
        det_init = init_params["camera"].detector
        det_sysid = params["camera"].detector
        bgr_sysid = det_init.draw_ellipse(cam.bgr, color=(0, 255, 0))
        bgr_sysid = det_init.draw_centroids(bgr_sysid, cam.median)
        bgr_sysid = det_sysid.draw_ellipse(bgr_sysid, color=(0, 0, 255))
        print(f"Playing video at {fps} fps")
        print(f"Press 'q' to stop")
        det_init.play_video(bgr_sysid, fps=fps)
        # Save sysid params
        with open(PARAMS_FILE, "wb") as f:
            pickle.dump(params, f)
        print(f"Saved {PARAMS_FILE}")
        # Save figs with suptitle
        for fig in figs:
            suptitle = fig._suptitle.get_text() if fig._suptitle else "Untitled"
            fig.savefig(f"{LOG_DIR}/{suptitle}.png")
            print(f"Saved {LOG_DIR}/{suptitle}.png")
        # Overwrite base graph state
        gs = base_init(RNG, params=params, order=("supervisor", "actuator"))

    # Control task
    if False:
        # Make control graph
        nodes_ctrl = psys.simulated_system(record, outputs={}, world_rate=WORLD_RATE, use_cam=USE_CAM)
        nodes_ctrl["supervisor"].set_init_method("random")
        graph_ctrl = rexv2.graph.Graph(nodes_ctrl, nodes_ctrl[SUPERVISOR], graphs_aug, supergraph=SUPERGRAPH, skip=["supervisor"])
        # Initialize control graph
        rng_init, rng_eval, rng_solv = jax.random.split(RNG, 3)
        gs = base_init(rng_init, order=("supervisor", "actuator"))
        task = tasks.create_control_task(graph_ctrl, gs)
        # Initial guess
        # init_params = task.to_extended_params(gs, task.init_params)
        # all_gs = task.evaluate(init_params, rng_eval, EPS_IDX)
        # figs = task.plot(all_gs, identifier="init_ctrl")
        # plt.show()
        # Solve
        sol_state, opt_params, log_state = task.solve(gs, rng=rng_solv)
        log_state.plot(f"{task.description} | {task.solver.strategy_name}", ylims=[0, 1000])
        params = task.to_extended_params(gs, opt_params)
        all_gs = task.evaluate(params, rng_eval, EPS_IDX)
        figs = task.plot(all_gs, identifier="opt_ctrl")
        # Overwrite base graph state
        gs = graph_ctrl.init(rng_init, params=params, order=("supervisor", "actuator"))
        # plt.show()

    # Estimator
    if False:
        task = tasks.create_estimator_task(graph_sysid, gs)
        # Initial guess
        init_params = task.to_extended_params(gs, task.init_params)
        all_gs = task.evaluate(init_params, RNG, EPS_IDX)
        figs = task.plot(all_gs, identifier="init_est")
        # Solve
        sol_state, opt_params, log_state = task.solve(gs)
        log_state.plot(f"{task.description} | {task.solver.strategy_name}", ylims=[0, 10000])
        params = task.to_extended_params(gs, opt_params)
        graph_states = task.evaluate(params, RNG, EPS_IDX)
        figs = task.plot(graph_states, identifier="opt_est")
        # Overwrite base graph state
        gs = base_init(RNG, params=params, order=("supervisor", "actuator"))
        # plt.show()
    plt.show()