import dill as pickle
import tqdm
from typing import Dict, Union, Callable, Any
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
import equinox as eqx

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


def make_simulated_pendulum_system_nodes(record: base.EpisodeRecord,
                                         outputs: Dict[str, Any],
                                         world_rate: float = 100.,
                                         use_cam: bool = True,
                                         ):
    # Make pendulum
    from envs.pendulum.ode import World, Sensor, Actuator

    # Create sensor
    sensor = Sensor.from_info(record.nodes["sensor"].info, outputs=outputs.get("sensor", None))

    # Create camera
    from envs.pendulum.realsense import SimD435iDetector
    camera = SimD435iDetector.from_info(record.nodes["camera"].info, outputs=outputs.get("camera", None))

    # Create estimator
    from envs.pendulum.estimator import Estimator
    estimator = Estimator.from_info(record.nodes["estimator"].info, use_cam=use_cam)

    # Create controller
    from envs.pendulum.controller import PPOAgent
    controller = PPOAgent.from_info(record.nodes["controller"].info, outputs=outputs.get("controller", None))

    # Create actuator
    actuator = Actuator.from_info(record.nodes["actuator"].info, outputs=outputs.get("actuator", None))

    # Create supervisor
    from envs.pendulum.supervisor import Supervisor
    supervisor = Supervisor.from_info(record.nodes["supervisor"].info)

    nodes = dict(sensor=sensor, camera=camera, estimator=estimator, controller=controller, actuator=actuator, supervisor=supervisor)

    # Connect from info
    [n.connect_from_info(record.nodes[name].info.inputs, nodes) for name, n in nodes.items()]

    # Simulation specific nodes
    world = World(name="world", rate=world_rate, scheduling=Scheduling.FREQUENCY, advance=False,
                  delay_dist=base.StaticDist.create(distrax.Deterministic(0.999/world_rate)))
    nodes["world"] = world

    # Connect according to delays
    sensor_delay = base.TrainableDist.create(alpha=0., min=0.0, max=1 / sensor.rate)
    camera_delay = base.TrainableDist.create(alpha=0., min=0.0, max=2*1 / camera.rate)
    actuator_delay = base.TrainableDist.create(alpha=0., min=0.0, max=1 / actuator.rate)
    world.connect(actuator, window=1, blocking=False, skip=True, jitter=Jitter.LATEST, delay_dist=actuator_delay, delay=0.)
    sensor.connect(world, window=1, blocking=False, jitter=Jitter.LATEST, delay_dist=sensor_delay, delay=0.)
    camera.connect(world, window=1, blocking=False, jitter=Jitter.LATEST, delay_dist=camera_delay, delay=0.)
    return nodes


if __name__ == "__main__":
    # todo: Methodology:
    #       0. Gather real-data and construct computation graph
    #       1. Estimate simulator parameters + delays (fit simulator to reconstruct real-data) (non-causal)
    #       2. Learn estimator parameters (Causally use real-data to reconstruct simulator state) (causal)
    #       3. Learn estimation error model (Fit error model to difference between estimated states of real-data vs. simulator state) (non-causal)
    #       4. Learn controller parameters (Use simulator + error model provides state estimates to fit controller) (causal)
    # todo: Real pendulum:
    #   [DONE] Why does the actuator have such a big delay? --> seemed to be caused by using a single device for all nodes.
    #   [DONE] Improve graph:
    #       [DONE] move actuator_delay to actuator_params --> change init order with actuator before world
    #       [DONE] merge camera & detector in simulation
    #       [DONE] connect estimator to actuator with larger window
    #       [DONE] throttle in actuator until action.ts - actuator_delay --> This does influence the measured computation delay of the actuator.
    #   - provide sigma as input to policy
    #   - Test with sensor that has delay of camera + detector.
    #   - Identify host cpu, and don't use it for jitting step functions.
    #   - adjust action sequence to slow down if making full rotation (i.e. zero action if fully rotating).
    #   - rectify realsense image with intrinsic calibration.
    #   [DONE] visualize delay distributions
    #   [DONE] implement real
    #   [DONE] gather RW data
    #   [DONE] implement ODE
    #   [DONE] train policy w/o camera
    #   [DONE] implement estimator
    #   - train policy with delays
    #   - system identification with data
    #   - train policy with camera
    #   - Replace detector.LP filter with the one in crazyflie  (probably not necessary)
    # todo: System identification:
    #   - We are not feeding back the actions from the controller in the system identification task.
    #   [DONE] Are parameters correctly set? (e.g. actuator_delay in world, ode params in estimator)
    #   - infer dt_future from world.inputs["actuator"] --> how to have access to this data when real does not have a world node?
    #   - verify that min, max delays of params.delay are the same as the ones of ss.inputs.delay_sysid
    #   - get policy trained on zero-delay pendulum and use it to control the real pendulum (and gather data).
    #   - add params of dynamics model that can update and predict state.
    #   - optimize for dt_future during system parameter identification.
    #   - what to do with initial th_world for seq=-1?
    #   - how to assign outputs (e.g. use images, detector)
    #   - initialize detector.state.(mincum,maxcum) in a non-causal way.
    #   [DONE] replace delay distributions with trainable ones
    #   - double check ts offsets (real vs. sim vs. compiled)
    #   - ts_recv for world is not correct. Set delay to rate.
    #   - Computation delay of world is equal to the rate of the world (currently set to 99% of the rate, should be 100%)
    # todo: Refactor:
    #   [DONE] Refactor tfd to distrax
    #   - How to deal with advance=True for nodes without inputs (& no blocking connections)?
    #       - Essentially, how to model "polling" nodes? They run as_fast_as_possible in the real_world, but not in the simulator.
    #   - Redefine ukf with pytrees (and use flatten/unflatten), and define cov as a pytree at the leaf of the pytree state.
    #   - Add throttle to BaseNode
    #   - Check weaktypes and recompilation & how to compile step function --> leads to more latency (maybe not if jit compiled)?
    # Make sysid nodes
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    EPS_IDX = -1
    WORLD_RATE = 50.
    USE_CAM = True
    SUPERVISOR = "actuator"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    RECORD_FILE = f"{LOG_DIR}/pendulum_data.pkl"
    # ORDER = ["camera", "sensor", "actuator", "controller", "estimator", "supervisor"]
    # CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
    #            "actuator": "green", "supervisor": "indigo"}

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Create nodes
    outputs_sysid = {name: n.steps.output for name, n in record.nodes.items()}
    outputs_sysid = eqx.tree_at(lambda x: x["camera"].bgr, outputs_sysid, None)  # don't use bgr for now.
    nodes_sysid = make_simulated_pendulum_system_nodes(record, outputs=outputs_sysid, world_rate=WORLD_RATE, use_cam=USE_CAM)
    nodes_ctrl = make_simulated_pendulum_system_nodes(record, outputs={}, world_rate=WORLD_RATE, use_cam=USE_CAM)

    # Set initialization method
    nodes_sysid["supervisor"].set_init_method("parametrized")
    nodes_ctrl["supervisor"].set_init_method("random")

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

    # Control task
    import envs.pendulum.tasks as tasks
    if False:
        # Make control graph
        graph_ctrl = rexv2.graph.Graph(nodes_ctrl, nodes_ctrl[SUPERVISOR], graphs_aug, supergraph=SUPERGRAPH,skip=["supervisor"])
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

    # System identification
    if True:
        task = tasks.create_sysid_task(graph_sysid, gs).replace(max_steps=80)  # todo: remove
        # Initial guess
        init_params = task.to_extended_params(gs, task.init_params)
        all_gs = task.evaluate(init_params, RNG, EPS_IDX)
        figs = task.plot(all_gs, identifier="init_sysid")
        # plt.show()
        # Solve
        sol_state, opt_params, log_state = task.solve(gs)
        log_state.plot(f"{task.description} | {task.solver.strategy_name}", ylims=[0, 300])
        params = task.to_extended_params(gs, opt_params)
        all_gs = task.evaluate(params, RNG, EPS_IDX)
        figs = task.plot(all_gs, identifier="opt_sysid")
        plt.show()
        # Overwrite base graph state
        gs = base_init(RNG, params=params, order=("supervisor", "actuator"))

    # Estimator
    if True:
        # todo: black-box/ukf estimator
        # todo: How to train? Only real observations, or also with simulated data (incl. disturbances?)
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