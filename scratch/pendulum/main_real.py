import dill as pickle
import tqdm
from typing import Dict, Union, Callable
import os
import multiprocessing
import itertools
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


DelaySim = Dict[str, Dict[str, Union[base.DelayDistribution, Dict[str, base.DelayDistribution]]]]
Delay = Dict[str, Dict[str, Union[float, Dict[str, float]]]]


def play_video(bgr, fps):
    import cv2
    wait_time = int(1000 / fps)  # Time in ms between frames
    bgr_uint8 = onp.array(bgr).astype(onp.uint8)
    while True:
        for img_uint8 in bgr_uint8:
            cv2.imshow('Video', img_uint8)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return


def get_default_distributions() -> DelaySim:
    delays_sim = dict(step={}, inputs={})
    for n in ["world", "supervisor", "sensor", "camera", "estimator", "controller", "actuator", "viewer"]:
        delays_sim["step"][n] = base.StaticDist.create(distrax.Deterministic(0.))
        delays_sim["inputs"][n] = {}
        for m in ["world", "supervisor", "sensor", "camera", "estimator", "controller", "actuator", "viewer"]:
            delays_sim["inputs"][n][m] = base.StaticDist.create(distrax.Deterministic(0.))
    return delays_sim


def load_distribution(file: str) -> DelaySim:
    with open(file, "rb") as f:
        return pickle.load(f)


def make_real_pendulum_system_nodes(delays_sim: DelaySim,
                                    delay_fn: Callable[[base.DelayDistribution], float],
                                    rates: Dict[str, float],
                                    cscheme: Dict[str, str] = None,
                                    order: list = None,
                                    use_cam: bool = True,
                                    include_image: bool = True,
                                    ):
    """Make a real pendulum system."""
    delays = jax.tree_util.tree_map(delay_fn, delays_sim, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    # Initialize main process as Node
    import rospy
    rospy.init_node("mops_client", anonymous=True)

    # Make pendulum
    from envs.pendulum.real import Sensor, Actuator

    # Create sensor
    sensor = Sensor(name="sensor", color=cscheme["sensor"], order=order.index("sensor"),
                    rate=rates["sensor"], scheduling=Scheduling.FREQUENCY, advance=False,
                    delay=delays["step"]["sensor"], delay_dist=delays_sim["step"]["sensor"])

    # Create camera
    from envs.pendulum.realsense import D435iDetector
    camera = D435iDetector(name="camera", color=cscheme["camera"], order=order.index("camera"),
                           include_image=include_image, width=424,  height=240, fps=rates["camera"],
                           rate=rates["camera"], scheduling=Scheduling.PHASE, advance=True,
                           # This is a polling node (i.e. it runs as fast as possible) --> unable to simulate with clock=SIMULATED
                           delay=delays["step"]["camera"], delay_dist=delays_sim["step"]["camera"])

    # Create estimator
    from envs.pendulum.estimator import Estimator
    estimator = Estimator(name="estimator", color=cscheme["estimator"], order=order.index("estimator"),
                          use_cam=use_cam,
                          rate=rates["estimator"], scheduling=Scheduling.FREQUENCY, advance=False,
                          delay=delays["step"]["estimator"], delay_dist=delays_sim["step"]["estimator"])
    estimator.connect(sensor, window=1, blocking=False,
                      delay_dist=delays_sim["inputs"]["estimator"]["sensor"], delay=delays["inputs"]["estimator"]["sensor"])
    estimator.connect(camera, window=1, blocking=False,
                      delay_dist=delays_sim["inputs"]["estimator"]["camera"], delay=delays["inputs"]["estimator"]["camera"])

    # Create controller
    from envs.pendulum.controller import OpenLoopController
    controller = OpenLoopController(name="controller", color=cscheme["controller"], order=order.index("controller"),
                                    rate=rates["controller"], scheduling=Scheduling.FREQUENCY, advance=True,
                                    delay=delays["step"]["controller"], delay_dist=delays_sim["step"]["controller"])
    controller.connect(estimator, window=1, blocking=True,
                       delay_dist=delays_sim["inputs"]["controller"]["estimator"], delay=delays["inputs"]["controller"]["estimator"])
    estimator.connect(controller, window=3, blocking=True, skip=True,
                      delay_dist=delays_sim["inputs"]["estimator"]["controller"], delay=delays["inputs"]["estimator"]["controller"])

    # Create actuator
    actuator = Actuator(name="actuator", color=cscheme["actuator"], order=order.index("actuator"),
                        rate=rates["actuator"], scheduling=Scheduling.FREQUENCY, advance=True,
                        delay=delays["step"]["actuator"], delay_dist=delays_sim["step"]["actuator"])
    actuator.connect(controller, window=1, blocking=True,
                     delay_dist=delays_sim["inputs"]["actuator"]["controller"], delay=delays["inputs"]["actuator"]["controller"])

    # Create supervisor
    from envs.pendulum.supervisor import Supervisor
    supervisor = Supervisor(name="supervisor", color=cscheme["supervisor"], order=order.index("supervisor"),
                            rate=rates["supervisor"], scheduling=Scheduling.FREQUENCY, advance=False,
                            delay=delays["step"]["supervisor"], delay_dist=delays_sim["step"]["supervisor"])
    supervisor.connect(estimator, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["supervisor"]["estimator"], delay=delays["inputs"]["supervisor"]["estimator"])

    nodes = dict(sensor=sensor,
                 camera=camera,
                 estimator=estimator,
                 controller=controller,
                 actuator=actuator,
                 supervisor=supervisor
                 )
    return nodes


if __name__ == "__main__":
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    RNG = jax.random.PRNGKey(0)
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    ORDER = ["camera", "sensor", "actuator", "controller", "estimator", "supervisor"]
    CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
               "actuator": "green", "supervisor": "indigo"}
    FILE_NAME = "pendulum_data.pkl"  # todo: If recording for delays, set to "pendulum_data_delay_only.pkl"
    INCLUDE_IMAGE = True  # Include image in camera output # todo: If recording for delays, set to False
    USE_CAM = False  # Use camera instead of sensor in estimator
    NUM_EPISODES = 1  # todo: set to 1 for sysid?
    TSIM = 21
    DELAYS_SIM = load_distribution(f"{LOG_DIR}/dists.pkl")  # get_default_distributions()
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    RATES = dict(sensor=30, camera=30, estimator=30, controller=30, actuator=30, supervisor=10)
    nodes = make_real_pendulum_system_nodes(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER, use_cam=USE_CAM, include_image=INCLUDE_IMAGE)
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

    # Load trained params
    with open("controller_trained_params.pkl", "rb") as f:
        ctrl_params = pickle.load(f)
    ctrl_params = gs.params["controller"].replace(**ctrl_params.__dict__)
    gs = gs.replace(params=gs.params.copy({"controller": ctrl_params}))

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

    # Set record settings # todo: TURN ON TO RECORD IMAGES ETC...
    for n, node in graph.nodes.items():
        node.set_record_settings(params=True, rng=False, inputs=False, state=True, output=True)
        if n in ["camera", "controller"]:
            node.set_record_settings(state=False)

    # Set logging
    rutils.set_log_level(LogLevel.INFO)

    # Get data
    episodes = []
    for i in range(NUM_EPISODES + 1):
        print(f"Episode {i}")
        with jax.log_compiles():
            gs, _ss = graph.reset(gs)
            num_steps = int(RATES["supervisor"]*TSIM)if i > 0 else 1
            for j in tqdm.tqdm(range(num_steps), disable=True):
                gs, _ss = graph.step(gs)
        graph.stop()  # Stop environment

        # Get records
        if i > 0:
            r = graph.get_record()
            episodes.append(r)
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

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(sensor.ts_start, sensor.output.th, label="sensor", color='r')
    axes[0].plot(estimator.output.ts, estimator.output.mean.th, label="estimator", color='g')
    axes[0].fill_between(estimator.output.ts, estimator.output.mean.th - std_est[:, 0], estimator.output.mean.th + std_est[:, 0], alpha=0.5,
                         color='g')
    axes[0].plot(camera.output.ts, camera.output.th, label="camera", color='b')
    # axes[0].plot(est_fut.ts, est_fut.state.mu[:, 0], label="fut", color='b')
    # axes[0].fill_between(est_fut.ts, est_fut.state.mu[:, 0] - std_fut[:, 0], est_fut.state.mu[:, 0] + std_fut[:, 0], alpha=0.5,
    #                      color='b')
    # axes[0].plot(measurements.ts, measurements.th, 'o', label="y", markersize=2)
    axes[0].legend()

    axes[1].plot(sensor.ts_start, sensor.output.thdot, label="sensor", color='r')
    axes[1].plot(estimator.output.ts, estimator.output.mean.thdot, label="estimator", color='g')
    axes[1].fill_between(estimator.output.ts, estimator.output.mean.thdot - std_est[:, 1], estimator.output.mean.thdot + std_est[:, 1], alpha=0.5,
                         color='g')
    axes[1].plot(camera.output.ts, camera.output.thdot, label="camera", color='b')
    # axes[1].plot(est_fut.ts, est_fut.state.mu[:, 1], label="fut", color='b')
    # axes[1].fill_between(est_fut.ts, est_fut.state.mu[:, 1] - std_fut[:, 1], est_fut.state.mu[:, 1] + std_fut[:, 1], alpha=0.5,
    #                      color='b')
    axes[1].legend()
    # axes[2].plot(actions.ts, actions.action, label="action", color='r')
    # axes[2].legend()
    plt.show()

    # Save data
    stacked = record.stack("padded")
    with open(os.path.join(LOG_DIR, FILE_NAME), "wb") as f:
        pickle.dump(stacked, f)
    print(f"Data saved to {os.path.join(LOG_DIR, FILE_NAME)}")
    # Unpickle with
    # with open(os.path.join(LOG_DIR, "pendulum_data.pkl"), "rb") as f:
    #     data = pickle.load(f)

    # Play video
    detections = record.episodes[0].nodes["camera"].steps.output
    if detections.bgr is not None:
        bgr = detections.bgr
        masks = bgr

        import cv2
        for i in range(bgr.shape[0]):
            if i >= detections.centroid.shape[0]:
                break
            # Get the center for the current frame (note that OpenCV uses x, y coordinates)
            center_col, center_row = detections.centroid[i].astype(int)

            # Mark the centroid with a circle on the image
            cv2.circle(bgr[i], (int(center_row), int(center_col)), 3, (0, 0, 255), -1)

        min_frames = min(bgr.shape[0], masks.shape[0])
        video = onp.concatenate([bgr[:min_frames], masks[:min_frames]], axis=2)
        play_video(video, RATES["camera"])
