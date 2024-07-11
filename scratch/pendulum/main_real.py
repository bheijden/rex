from typing import Dict, Union, Callable, Any
import dill as pickle
import tqdm
import os
import multiprocessing
import itertools
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import distrax

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel
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
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    RNG = jax.random.PRNGKey(0)
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    ORDER = ["camera", "sensor", "world", "actuator", "controller", "estimator", "supervisor"]
    CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
               "actuator": "green", "supervisor": "indigo"}
    STD_TH = 0.003  # Overwrite std_th in estimator and camera --> None to keep default
    MODE = "evaluate"  # "delay_only", "sysid", "control"
    if MODE == "delay_only":
        NUM_EPISODES = 10
        TSIM = 5
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_delay_only.pkl"
        CTRL_FILE = f"{LOG_DIR}/controller_trained_params.pkl"
        INCL_COVARIANCE = False
        USE_CAM = True  # Use camera instead of sensor in estimator
        INCLUDE_IMAGE = False  # Include image in camera output
        USE_OPENLOOP = False
        WRAPPED = False
    elif MODE == "sysid":
        NUM_EPISODES = 1
        TSIM = 21
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_sysid.pkl"
        CTRL_FILE = f"{LOG_DIR}/controller_params.pkl"
        # CTRL_FILE = f"{LOG_DIR}/controller_trained_params.pkl"  # Workng controller?
        INCL_COVARIANCE = False
        USE_CAM = True  # Use camera instead of sensor in estimator
        INCLUDE_IMAGE = True  # Include image in camera output
        USE_OPENLOOP = True
        WRAPPED = False
    elif MODE == "control":
        TSIM = 5
        NUM_EPISODES = 10
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_control.pkl"
        CTRL_FILE = f"{LOG_DIR}/controller_params_brax.pkl"  # todo: CHANGE BRAX
        INCL_COVARIANCE = False
        USE_CAM = True  # Use camera instead of sensor in estimator
        INCLUDE_IMAGE = False  # Include image in camera output
        USE_OPENLOOP = False
        WRAPPED = True
    elif MODE == "evaluate":
        TSIM = 5
        NUM_EPISODES = 10
        PARAMS_FILE = f"{LOG_DIR}/20240710_141737_brax/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_eval_nodelay.pkl"
        CTRL_FILE = f"{LOG_DIR}/controller_trained_params_nodelay.pkl"  # todo: CHANGE BRAX
        INCL_COVARIANCE = False
        USE_CAM = True  # Use camera instead of sensor in estimator
        INCLUDE_IMAGE = True  # Include image in camera output
        USE_OPENLOOP = False
        WRAPPED = True
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    DELAYS_SIM = psys.load_distribution(f"{LOG_DIR}/dists.pkl")  # psys.get_default_distributions()
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    RATES = dict(sensor=50, camera=50, estimator=50, controller=50, actuator=50, supervisor=10)

    # Create graph
    nodes = psys.real_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER, use_cam=USE_CAM, include_image=INCLUDE_IMAGE,
                             use_openloop=USE_OPENLOOP)
    graph = rexv2.asynchronous.AsyncGraph(nodes, supervisor=nodes["supervisor"],
                                          clock=Clock.WALL_CLOCK,
                                          real_time_factor=RealTimeFactor.REAL_TIME)

    # Jit functions
    graph.init = jax.jit(graph.init, static_argnames=["order"], device=GPU_DEVICE)
    cpu_devices = itertools.cycle(jax.devices('cpu'))
    _ = next(cpu_devices)  # Skip first CPU
    # cpu_devices = itertools.cycle([jax.devices('cpu')[0]])  # Single CPU
    for name, node in graph.nodes.items():
        cpu = next(cpu_devices)
        print(f"Jitting {name} on {cpu}")
        node.step = jax.jit(node.step, device=cpu)

    # Initialize graph state
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs = graph.init(RNG, order=("supervisor", "actuator"))

    # Load params
    with open(PARAMS_FILE, "rb") as f:
        params: Dict[str, Any] = pickle.load(f)
    print(f"Params loaded from {PARAMS_FILE}")

    # Load trained params
    with open(CTRL_FILE, "rb") as f:
        ctrl_params = pickle.load(f)
    print(f"Controller params loaded from {CTRL_FILE}")
    params["controller"] = gs.params["controller"].replace(**ctrl_params.__dict__)
    params["controller"] = params["controller"].replace(incl_covariance=INCL_COVARIANCE)

    # Overwrite th
    params = params.copy()
    if STD_TH is not None:
        params["estimator"] = params["estimator"].replace(std_th=STD_TH)
        params["camera"] = params["camera"].replace(std_th=STD_TH)
        print(f"Overwriting std_th to {STD_TH}")
    eqx.tree_pprint(params, short_arrays=False)

    # Replace params in gs
    gs = gs.replace(params=FrozenDict(params))

    # Warmup & get initial graph state
    import logging
    logging.getLogger("jax").setLevel(logging.INFO)

    for name, node in graph.nodes.items():
        ss = gs.step_state[name]
        default_o = node.init_output(RNG, gs)
        with timer(f"warmup[{name}]", log_level=LogLevel.SILENT):
            with jax.log_compiles():
                ss, o = node.async_step(ss)
        _ = same_structure(default_o, o, tag=name, raise_on_mismatch=False)
        _ = same_structure(gs.step_state[name], ss, tag=name, raise_on_mismatch=False)
        with timer(f"eval[{name}]", log_level=LogLevel.WARN, repeat=10):
            for _ in range(10):
                ss, o = node.async_step(ss)
    with timer(f"warmup[dist]", log_level=LogLevel.WARN):
        graph.warmup(gs)

    # Set record settings
    for n, node in graph.nodes.items():
        node.set_record_settings(params=True, rng=False, inputs=False, state=True, output=True)
        if n in ["camera", "controller"]:
            node.set_record_settings(state=False)  # Avoid saving cummin, cummax, or network weights.

    # Set logging
    rutils.set_log_level(LogLevel.INFO)

    # Get data
    episodes = []
    success_rates = []
    init_gs = gs
    for i in range(NUM_EPISODES + 1):
        with jax.log_compiles():
            gs, _ss = graph.reset(init_gs)
            num_steps = int(RATES["supervisor"]*TSIM)if i > 0 else 1
            for j in tqdm.tqdm(range(num_steps), disable=True):
                gs, _ss = graph.step(gs)
        graph.stop()  # Stop environment

        # Get records
        if i > 0:
            r = graph.get_record()
            episodes.append(r)

            # Print upright percentage
            cos_th = jnp.cos(r.nodes["sensor"].steps.output.th)
            thdot = r.nodes["sensor"].steps.output.thdot
            is_upright = cos_th > 0.9
            is_static = jnp.abs(thdot) < 2.0
            is_valid = jnp.logical_and(is_upright, is_static)
            success_rate = is_valid.sum() / is_valid.size
            success_rates.append(success_rate)
            print(f"Episode {i}: Success rate: {success_rate:.2f} ({success_rate*100:.2f}%)")

        # Overwrite cummin, cummax for next episode
        cam_state = gs.state["camera"]
        init_cam_state = init_gs.state["camera"].replace(cummin=cam_state.cummin, cummax=cam_state.cummax)
        init_gs = eqx.tree_at(lambda _tree: _tree.state["camera"], init_gs, init_cam_state)

    # Print mean success rate
    mean_success_rate = jnp.mean(jnp.array(success_rates))
    print(f"Mean success rate: {mean_success_rate:.2f} ({mean_success_rate*100:.2f}%)")

    # Grab detector from camera
    det = gs.params["camera"].detector

    # Redo detection
    if INCLUDE_IMAGE:
        new_eps = []
        for i, ep in enumerate(episodes):
            # Estimate ellipse
            median = ep.nodes["camera"].steps.output.median.reshape(-1, 2)
            a, b, x0, y0, phi = det.estimate_ellipse(median)
            print(f"Initial guess: a={a}, b={b}, x0={x0}, y0={y0}, phi={phi}")
            det = det.replace(a=a, b=b, x0=x0, y0=y0, phi=phi)
            # Run detection
            ts, bgr = ep.nodes["camera"].steps.ts_start, ep.nodes["camera"].steps.output.bgr
            _, detections = det.noncausal_step(ts, bgr)
            cam_outputs = ep.nodes["camera"].steps.output.replace(**detections.__dict__)
            _tmp = eqx.tree_at(lambda _tree: _tree.nodes["camera"].steps.output, ep, cam_outputs)
            new_eps.append(_tmp)
        episodes = new_eps

    record = base.ExperimentRecord(episodes=episodes)

    # Convert to graph
    graphs_raw = record.to_graph()

    # Visualize the graph
    MAX_X = 3.0
    Gs = [rutils.to_networkx_graph(g, nodes=nodes, validate=True) for g in graphs_raw]
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for i, G in enumerate(Gs):
        if i > 1:
            continue
        supergraph.plot_graph(G, max_x=MAX_X, ax=axs[i])
        axs[i].set_title(f"Episode {i}")
        axs[i].set_xlabel("Time [s]")
    # plt.show()

    # Plot results
    camera = record.episodes[0].nodes["camera"].steps
    sensor = record.episodes[0].nodes["sensor"].steps
    estimator = record.episodes[0].nodes["estimator"].steps
    actuator = record.episodes[0].nodes["actuator"].steps

    # Get std
    std_vfn = jax.vmap(lambda x: jnp.diag(jnp.sqrt(x)))
    std_est = std_vfn(estimator.output.cov)

    # wrap angles
    def wrap_unwrap(x):
        if WRAPPED:
            return (x + onp.pi) % (2 * onp.pi) - onp.pi
        _wrap_unwrap = lambda o: jnp.unwrap((x + onp.pi + o) % (2 * onp.pi) - onp.pi, discont=onp.pi) - o

        x_map = jax.vmap(_wrap_unwrap)(jnp.array([0.1, 0.0, -0.1]))
        # take i where the first x_map[i,0] is closest to onp.pi
        i = jnp.argmin(jnp.abs(x_map[:, 0] - onp.pi))
        return x_map[i]
    th_sensor = wrap_unwrap(sensor.output.th)
    th_camera = wrap_unwrap(camera.output.th)
    th_est = wrap_unwrap(estimator.output.mean.th)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    axes[0].plot(actuator.ts_end, actuator.output.action[..., 0], label="action", color='purple')
    axes[0].plot(sensor.ts_start, th_sensor, label="sensor", color='orange')
    axes[0].plot(camera.output.ts, th_camera, label="camera", color='green')
    axes[0].plot(estimator.output.ts, th_est, label="estimator", color='red')
    axes[0].fill_between(estimator.output.ts, th_est - std_est[:, 0], th_est + std_est[:, 0], alpha=0.5, color='r')
    axes[0].legend()
    axes[1].plot(sensor.ts_start, sensor.output.thdot, label="sensor", color='orange')
    axes[1].plot(camera.output.ts, camera.output.thdot, label="camera", color='green')
    axes[1].plot(estimator.output.ts, estimator.output.mean.thdot, label="estimator", color='red')
    axes[1].fill_between(estimator.output.ts, estimator.output.mean.thdot - std_est[:, 1], estimator.output.mean.thdot + std_est[:, 1], alpha=0.5,
                         color='r')
    axes[1].legend()
    plt.show()

    # Show video
    if INCLUDE_IMAGE:
        # Calculate fps
        num_steps = camera.output.ts.shape[0]
        fps = num_steps / camera.output.ts[-1]

        ep = episodes[0]
        bgr_video = det.draw_ellipse(camera.output.bgr, color=(0, 255, 0))
        bgr_video = det.draw_centroids(bgr_video, camera.output.median)
        print(f"Playing video with {fps} fps")
        print(f"Press 'q' to quit")
        det.play_video(bgr_video, fps=fps)

    # Save data
    stacked = record.stack("padded")
    with open(RECORD_FILE, "wb") as f:
        pickle.dump(stacked, f)
    print(f"Data saved to {RECORD_FILE}")
    # Unpickle with
    # with open(os.path.join(LOG_DIR, "pendulum_data.pkl"), "rb") as f:
    #     data = pickle.load(f)
    #
    # # Play video
    # detections = record.episodes[0].nodes["camera"].steps.output
    # if detections.bgr is not None:
    #     bgr = detections.bgr
    #     masks = bgr
    #
    #     import cv2
    #     for i in range(bgr.shape[0]):
    #         if i >= detections.centroid.shape[0]:
    #             break
    #         # Get the center for the current frame (note that OpenCV uses x, y coordinates)
    #         center_col, center_row = detections.centroid[i].astype(int)
    #
    #         # Mark the centroid with a circle on the image
    #         cv2.circle(bgr[i], (int(center_row), int(center_col)), 3, (0, 0, 255), -1)
    #
    #     min_frames = min(bgr.shape[0], masks.shape[0])
    #     video = onp.concatenate([bgr[:min_frames], masks[:min_frames]], axis=2)
    #     play_video(video, RATES["camera"])
