import dill as pickle
import tqdm
import functools
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

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
from rexv2 import artificial

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

DelaySim = Dict[str, Dict[str, Union[base.DelayDistribution, Dict[str, base.DelayDistribution]]]]
Delay = Dict[str, Dict[str, Union[float, Dict[str, float]]]]


def load_distribution(file: str) -> DelaySim:
    with open(file, "rb") as f:
        return pickle.load(f)


def get_default_distributions() -> DelaySim:
    delays_sim = dict(step={}, inputs={})
    for n in ["world", "supervisor", "sensor", "camera", "estimator", "controller", "actuator", "viewer"]:
        delays_sim["step"][n] = base.StaticDist.create(distrax.Deterministic(0.))
        delays_sim["inputs"][n] = {}
        for m in ["world", "supervisor", "sensor", "camera", "estimator", "controller", "actuator", "viewer"]:
            delays_sim["inputs"][n][m] = base.StaticDist.create(distrax.Deterministic(0.))
    return delays_sim


def make_sensoronly_pendulum_system_nodes(delays_sim: DelaySim,
                                          delay_fn: Callable[[base.DelayDistribution], float],
                                          rates: Dict[str, float],
                                          cscheme: Dict[str, str] = None,
                                          order: list = None,
                                          ):
    """Make a nodelay pendulum system."""
    delays_sim["step"]["world"] = base.StaticDist.create(distrax.Deterministic(0.99 / rates["world"]))  # todo: This is required
    delays_sim["step"]["actuator"] = base.StaticDist.create(distrax.Deterministic(0.001))  # todo: This is required
    delays_sim["inputs"]["world"] = {}
    delays_sim["inputs"]["sensor"] = {}
    delays_sim["inputs"]["world"]["actuator"] = base.StaticDist.create(distrax.Deterministic(0.0))
    delays_sim["inputs"]["sensor"]["world"] = base.StaticDist.create(distrax.Deterministic(0.0))
    delays_sim["inputs"]["controller"]["sensor"] = delays_sim["inputs"]["estimator"]["sensor"]
    delays = jax.tree_util.tree_map(delay_fn, delays_sim, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    # Make pendulum
    from envs.pendulum.ode import World, Sensor, Actuator

    # Create sensor
    sensor = Sensor(name="sensor", color=cscheme["sensor"], order=order.index("sensor"),
                    rate=rates["sensor"], scheduling=Scheduling.FREQUENCY, advance=False,
                    delay=delays["step"]["sensor"], delay_dist=delays_sim["step"]["sensor"])

    # Create controller
    from envs.pendulum.controller import PPOAgent
    controller = PPOAgent(name="controller", color=cscheme["controller"], order=order.index("controller"),
                          rate=rates["controller"], scheduling=Scheduling.FREQUENCY, advance=False,
                          delay=delays["step"]["controller"], delay_dist=delays_sim["step"]["controller"])
    controller.connect(sensor, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["controller"]["sensor"], delay=delays["inputs"]["controller"]["sensor"])

    # Create actuator
    actuator = Actuator(name="actuator", color=cscheme["actuator"], order=order.index("actuator"),
                        rate=rates["actuator"], scheduling=Scheduling.FREQUENCY, advance=False,
                        delay=delays["step"]["actuator"], delay_dist=delays_sim["step"]["actuator"])
    actuator.connect(controller, window=1, blocking=False,
                     delay_dist=delays_sim["inputs"]["actuator"]["controller"],
                     delay=delays["inputs"]["actuator"]["controller"])

    # Create supervisor
    from envs.pendulum.supervisor import Supervisor
    supervisor = Supervisor(name="supervisor", color=cscheme["supervisor"], order=order.index("supervisor"),
                            rate=rates["supervisor"], scheduling=Scheduling.FREQUENCY, advance=False,
                            delay=delays["step"]["supervisor"], delay_dist=delays_sim["step"]["supervisor"])
    supervisor.connect(controller, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["supervisor"]["controller"],
                       delay=delays["inputs"]["supervisor"]["controller"])

    nodes = dict(sensor=sensor, controller=controller, actuator=actuator, supervisor=supervisor)

    # Simulation specific nodes
    world = World(name="world", rate=rates["world"], scheduling=Scheduling.FREQUENCY, advance=False,
                  delay_dist=delays_sim["step"]["world"], delay=delays["step"]["world"])
    nodes["world"] = world

    # Connect according to delays
    sensor_delay = base.TrainableDist.create(alpha=0., min=0.0, max=1 / 30)
    actuator_delay = base.TrainableDist.create(alpha=0., min=0.0, max=1 / 30)
    world.connect(actuator, window=1, blocking=False, skip=True, jitter=Jitter.LATEST, delay_dist=actuator_delay, delay=delays["inputs"]["world"]["actuator"])
    sensor.connect(world, window=1, blocking=False, jitter=Jitter.LATEST, delay_dist=sensor_delay, delay=delays["inputs"]["sensor"]["world"])
    return nodes


def make_real_sensoronly_pendulum_system_nodes(delays_sim: DelaySim,
                                               delay_fn: Callable[[base.DelayDistribution], float],
                                               rates: Dict[str, float],
                                               cscheme: Dict[str, str] = None,
                                               order: list = None,
                                               ):
    """Make a nodelay pendulum system."""
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

    # Create controller
    from envs.pendulum.controller import PPOAgent
    controller = PPOAgent(name="controller", color=cscheme["controller"], order=order.index("controller"),
                          rate=rates["controller"], scheduling=Scheduling.FREQUENCY, advance=False,
                          delay=delays["step"]["controller"], delay_dist=delays_sim["step"]["controller"])
    controller.connect(sensor, window=1, blocking=True,
                       delay_dist=delays_sim["inputs"]["controller"]["sensor"], delay=delays["inputs"]["controller"]["sensor"])

    # Create actuator
    actuator = Actuator(name="actuator", color=cscheme["actuator"], order=order.index("actuator"),
                        rate=rates["actuator"], scheduling=Scheduling.FREQUENCY, advance=False,
                        delay=delays["step"]["actuator"], delay_dist=delays_sim["step"]["actuator"])
    actuator.connect(controller, window=1, blocking=True,
                     delay_dist=delays_sim["inputs"]["actuator"]["controller"],
                     delay=delays["inputs"]["actuator"]["controller"])

    # Create supervisor
    from envs.pendulum.supervisor import Supervisor
    supervisor = Supervisor(name="supervisor", color=cscheme["supervisor"], order=order.index("supervisor"),
                            rate=rates["supervisor"], scheduling=Scheduling.FREQUENCY, advance=False,
                            delay=delays["step"]["supervisor"], delay_dist=delays_sim["step"]["supervisor"])
    supervisor.connect(controller, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["supervisor"]["controller"],
                       delay=delays["inputs"]["supervisor"]["controller"])

    nodes = dict(sensor=sensor, controller=controller, actuator=actuator, supervisor=supervisor)
    return nodes


def make_pendulum_system_nodes(record: base.EpisodeRecord,
                               outputs: Dict[str, Any],
                               world_rate: float = 100.,
                               include_image: bool = True,
                               ):
    """Make a real pendulum system."""
    # Make pendulum
    from envs.pendulum.ode import World, Sensor, Actuator

    # Create sensor
    sensor = Sensor.from_info(record.nodes["sensor"].info, outputs=outputs.get("sensor", None))

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
                           include_image=include_image, width=424, height=240, fps=rates["camera"],
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
                       delay_dist=delays_sim["inputs"]["controller"]["estimator"],
                       delay=delays["inputs"]["controller"]["estimator"])
    estimator.connect(controller, window=3, blocking=True, skip=True,
                      delay_dist=delays_sim["inputs"]["estimator"]["controller"],
                      delay=delays["inputs"]["estimator"]["controller"])

    # Create actuator
    actuator = Actuator(name="actuator", color=cscheme["actuator"], order=order.index("actuator"),
                        rate=rates["actuator"], scheduling=Scheduling.FREQUENCY, advance=True,
                        delay=delays["step"]["actuator"], delay_dist=delays_sim["step"]["actuator"])
    actuator.connect(controller, window=1, blocking=True,
                     delay_dist=delays_sim["inputs"]["actuator"]["controller"],
                     delay=delays["inputs"]["actuator"]["controller"])

    # Create supervisor
    from envs.pendulum.supervisor import Supervisor
    supervisor = Supervisor(name="supervisor", color=cscheme["supervisor"], order=order.index("supervisor"),
                            rate=rates["supervisor"], scheduling=Scheduling.FREQUENCY, advance=False,
                            delay=delays["step"]["supervisor"], delay_dist=delays_sim["step"]["supervisor"])
    supervisor.connect(estimator, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["supervisor"]["estimator"],
                       delay=delays["inputs"]["supervisor"]["estimator"])

    nodes = dict(sensor=sensor,
                 camera=camera,
                 estimator=estimator,
                 controller=controller,
                 actuator=actuator,
                 supervisor=supervisor
                 )
    return nodes


if __name__ == "__main__":
    # Make sysid nodes
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    SUPERVISOR = "controller"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    RECORD_FILE = f"{LOG_DIR}/pendulum_data.pkl"
    ORDER = ["camera", "sensor", "actuator", "controller", "estimator", "supervisor"]
    CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
               "actuator": "green", "supervisor": "indigo"}
    RATES = dict(sensor=30, camera=30, estimator=30, controller=30, actuator=30, supervisor=10)
    DELAYS_SIM = load_distribution(f"{LOG_DIR}/dists.pkl")  # get_default_distributions()
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    TS_MAX = 5.0

    # Evaluate in real-world
    if False:
        raise NotImplementedError("Real-world evaluation is not yet updated with new graph.")
        # Make real-world nodes
        nodes_real = make_real_sensoronly_pendulum_system_nodes(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER)
        # Create real-time graph
        graph_real = rexv2.asynchronous.AsyncGraph(nodes_real, nodes_real[SUPERVISOR],
                                                   clock=Clock.WALL_CLOCK, real_time_factor=RealTimeFactor.REAL_TIME)
        # Load trained params
        with open("controller_trained_params.pkl", "rb") as f:
            ctrl_params = pickle.load(f)
        # Get initial graph state
        gs = graph_real.init(RNG, params={"controller": ctrl_params}, order=("supervisor", "actuator"))
        # Jit functions
        for name, node in graph_real.nodes.items():
            cpu = next(CPU_DEVICES)
            print(f"Jitting {name} on {cpu}")
            node.step = jax.jit(node.step, device=cpu)
        # Warmup
        for name, node in graph_real.nodes.items():
            ss = gs.step_state[name]
            with timer(f"warmup[{name}]", log_level=LogLevel.SILENT):
                ss, o = node.async_step(ss)
            with timer(f"eval[{name}]", log_level=LogLevel.WARN, repeat=10):
                for _ in range(10):
                    ss, o = node.async_step(ss)
        # Evaluate the controller
        E = 5
        N = int(TS_MAX * RATES[SUPERVISOR])
        for e in range(E):
            for _ in range(N):
                graph_real.run(gs)
            graph_real.stop()
        exit()

    # Create simulation nodes
    nodes = make_sensoronly_pendulum_system_nodes(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER)

    # Generate computation graphs
    graphs_gen = artificial.generate_graphs(nodes, ts_max=TS_MAX, num_episodes=1)

    # Create graph
    graph = rexv2.graph.Graph(nodes, nodes[SUPERVISOR], graphs_gen, supergraph=SUPERGRAPH)

    # Visualize the graph
    if True:
        Gs = graph.Gs
        # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(Gs):
            supergraph.plot_graph(G, max_x=1.0, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()

    # Get initial graph state
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs = graph.init(RNG, params={}, order=("supervisor", "actuator"))

    # Jit functions & warmup
    if False:
        for name, node in graph.nodes.items():
            cpu = next(CPU_DEVICES)
            print(f"Jitting {name} on {cpu}")
            node.step = jax.jit(node.step, device=cpu)

        # Warmup & get initial graph state
        # import logging
        # logging.getLogger("jax").setLevel(logging.INFO)

        for name, node in graph.nodes.items():
            ss = gs.step_state[name]
            default_o = node.init_output(RNG, gs)
            with timer(f"warmup[{name}]", log_level=LogLevel.SILENT):
                with jax.log_compiles():
                    ss, o = node.async_step(ss)
            _ = same_structure(default_o, o, tag=name, raise_on_mismatch=False) # todo: Turn on raise_on_mismatch
            _ = same_structure(gs.step_state[name], ss, tag=name, raise_on_mismatch=False)
            with timer(f"eval[{name}]", log_level=LogLevel.WARN, repeat=10):
                for _ in range(10):
                    ss, o = node.async_step(ss)

    from envs.pendulum.env import Environment
    env = Environment(graph, order=("supervisor", "world"), randomize_eps=True)

    # Evaluate in simulation
    if False:
        # Load trained params
        with open("controller_trained_params.pkl", "rb") as f:
            ctrl_params = pickle.load(f)

        # Evaluate the controller
        env_eval = Environment(graph, params={"controller": ctrl_params}, order=("supervisor", "world"), randomize_eps=True)
        env_reset = jax.jit(env_eval.reset)
        env_step = jax.jit(env_eval.step)
        get_action = jax.jit(ctrl_params.get_action)
        done = False
        cum_reward = 0
        history = []
        gs, obs, info = env_reset(RNG)
        while not done:
            print(f"seq: {gs.seq['controller']}")
            action = get_action(obs)
            gs, obs, reward, terminated, truncated, info = env_step(gs, action)
            done = terminated or truncated
            cum_reward += reward
            history.append(gs.state["world"].th)
        print(f"cum_reward: {cum_reward}")
        plt.plot(onp.array(history))
        plt.show()

    # Test env API
    # obs_space = env.observation_space(gs)
    # act_space = env.action_space(gs)
    # _ = env.get_observation(gs)
    # _ = env.get_truncated(gs)
    # _ = env.get_terminated(gs)
    # _ = env.get_reward(gs, jnp.zeros(act_space.low.shape))
    # _ = env.get_info(gs, jnp.zeros(act_space.low.shape))
    # _ = env.get_output(gs, jnp.zeros(act_space.low.shape))
    # _ = env.update_graph_state_pre_step(gs, jnp.zeros(act_space.low.shape))

    _ = env.reset()

    config = rexv2.ppo.Config(
        LR=1e-4,
        NUM_ENVS=64,
        NUM_STEPS=32,  # increased from 16 to 32 (to solve approx_kl divergence)
        TOTAL_TIMESTEPS=20e6,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        GAMMA=0.99,
        GAE_LAMBDA=0.95,
        CLIP_EPS=0.2,
        ENT_COEF=0.01,
        VF_COEF=0.5,
        MAX_GRAD_NORM=0.5,  # or 0.5?
        NUM_HIDDEN_LAYERS=2,
        NUM_HIDDEN_UNITS=64,
        KERNEL_INIT_TYPE="xavier_uniform",
        HIDDEN_ACTIVATION="tanh",
        STATE_INDEPENDENT_STD=True,
        SQUASH=True,
        ANNEAL_LR=False,
        NORMALIZE_ENV=True,
        FIXED_INIT=True,
        NUM_EVAL_ENVS=20,
        EVAL_FREQ=100,
        VERBOSE=True,
        DEBUG=False,
    )
    train = functools.partial(rexv2.ppo.train, env)
    train = jax.jit(train)
    with timer("train"):
        res = train(config, jax.random.PRNGKey(2))
    print("Training done!")

    # Initialize agent params
    model_params = res["runner_state"][0].params["params"]
    ctrl_params = gs.params["controller"].replace(act_scaling=res["act_scaling"], obs_scaling=res["norm_obs"],
                                                  model=model_params, hidden_activation=config.HIDDEN_ACTIVATION,
                                                  stochastic=False)
    ctrl_params_onp = jax.tree_util.tree_map(lambda x: onp.array(x), ctrl_params)

    # Save agent params
    if True:
        with open("controller_trained_params.pkl", "wb") as f:
            pickle.dump(ctrl_params_onp, f)
        print("Controller params saved!")

