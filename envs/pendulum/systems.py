from typing import Dict, Union, Callable, Any
import dill as pickle
import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

from rexv2 import base
from rexv2.constants import Scheduling, Jitter


DelaySim = Dict[str, Dict[str, Union[base.DelayDistribution, Dict[str, base.DelayDistribution]]]]
Delay = Dict[str, Dict[str, Union[float, Dict[str, float]]]]


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


def simulated_system(record: base.EpisodeRecord,
                     outputs: Dict[str, Any] = None,
                     world_rate: float = 100.,
                     id_cam: bool = False,
                     use_cam: bool = True,
                     use_brax: bool = False,
                     use_ukf: bool = True,
                     ):
    outputs = outputs or {}

    # Load world
    if use_brax:
        from envs.pendulum.brax import World
    else:
        from envs.pendulum.ode import World

    # Make pendulum
    from envs.pendulum.ode import Sensor, Actuator

    # Create sensor
    sensor = Sensor.from_info(record.nodes["sensor"].info, outputs=outputs.get("sensor", None))

    # Create camera
    from envs.pendulum.realsense import SimD435iDetector
    outputs_cam = outputs.get("camera", None)
    outputs_cam = outputs_cam if outputs_cam is None or id_cam else outputs_cam.replace(bgr=None)
    camera = SimD435iDetector.from_info(record.nodes["camera"].info, outputs=outputs_cam, width=424,  height=240, fps=60,)

    # Create estimator
    from envs.pendulum.estimator import Estimator
    estimator = Estimator.from_info(record.nodes["estimator"].info, use_cam=use_cam, use_ukf=use_ukf)

    # Create controller
    from envs.pendulum.controller import PPOAgent
    controller = PPOAgent.from_info(record.nodes["controller"].info, outputs=outputs.get("controller", None))

    # Create actuator
    actuator = Actuator.from_info(record.nodes["actuator"].info, outputs=outputs.get("actuator", None))

    # Create supervisor
    from envs.pendulum.supervisor import Supervisor
    supervisor = Supervisor.from_info(record.nodes["supervisor"].info)

    nodes = dict(sensor=sensor, camera=camera, estimator=estimator, controller=controller, actuator=actuator,
                 supervisor=supervisor)

    # Connect from info
    [n.connect_from_info(record.nodes[name].info.inputs, nodes) for name, n in nodes.items()]

    # Simulation specific nodes
    world = World(name="world", rate=world_rate, scheduling=Scheduling.FREQUENCY, advance=False, order=-1,
                  delay_dist=base.StaticDist.create(distrax.Deterministic(0.999 / world_rate)))
    nodes["world"] = world

    # Connect according to delays
    sensor_delay = base.TrainableDist.create(delay=0., min=0.0, max=0.05)
    camera_delay = base.TrainableDist.create(delay=0., min=0.0, max=0.05)
    actuator_delay = base.TrainableDist.create(delay=0., min=0.0, max=0.05)
    world.connect(actuator, window=1, blocking=False, skip=True, jitter=Jitter.LATEST, delay_dist=actuator_delay, delay=0.)
    sensor.connect(world, window=1, blocking=False, jitter=Jitter.LATEST, delay_dist=sensor_delay, delay=0.)
    camera.connect(world, window=1, blocking=False, jitter=Jitter.LATEST, delay_dist=camera_delay, delay=0.)
    return nodes


def no_delay_system(rates: Dict[str, float],
                    cscheme: Dict[str, str] = None,
                    order: list = None,
                    use_brax: bool = False,
                    ):
    """Make a nodelay pendulum system."""
    order = ["supervisor", "sensor", "controller", "actuator", "world"] if order is None else order
    delays_sim = get_default_distributions()
    delays_sim["step"]["world"] = base.StaticDist.create(distrax.Deterministic(0.99 / rates["world"]))  # todo: This is required
    delays_sim["inputs"]["world"] = {}
    delays_sim["inputs"]["sensor"] = {}
    delays_sim["inputs"]["world"]["actuator"] = base.StaticDist.create(distrax.Deterministic(0.0))
    delays_sim["inputs"]["sensor"]["world"] = base.StaticDist.create(distrax.Deterministic(0.0))
    delays_sim["inputs"]["controller"]["sensor"] = delays_sim["inputs"]["estimator"]["sensor"]
    delays = jax.tree_util.tree_map(lambda d: d.quantile(0.85), delays_sim, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    # Load world
    if use_brax:
        from envs.pendulum.brax import World
    else:
        from envs.pendulum.ode import World

    # Make pendulum
    from envs.pendulum.ode import Sensor, Actuator

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
    world.connect(actuator, window=1, blocking=False, skip=True, jitter=Jitter.LATEST, delay_dist=delays_sim["inputs"]["world"]["actuator"], delay=delays["inputs"]["world"]["actuator"])
    sensor.connect(world, window=1, blocking=False, jitter=Jitter.LATEST, delay_dist=delays_sim["inputs"]["sensor"]["world"], delay=delays["inputs"]["sensor"]["world"])
    return nodes


def real_system(delays_sim: DelaySim,
                delay_fn: Callable[[base.DelayDistribution], float],
                rates: Dict[str, float],
                cscheme: Dict[str, str] = None,
                order: list = None,
                use_cam: bool = True,
                include_image: bool = True,
                use_openloop: bool = True,
                use_pred: bool = True,
                use_ukf: bool = True,
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
                           include_image=include_image, width=424,  height=240, fps=60,
                           rate=rates["camera"], scheduling=Scheduling.PHASE, advance=True,
                           # This is a polling node (i.e. it runs as fast as possible) --> unable to simulate with clock=SIMULATED
                           delay=delays["step"]["camera"], delay_dist=delays_sim["step"]["camera"])

    # Create estimator
    from envs.pendulum.estimator import Estimator
    estimator = Estimator(name="estimator", color=cscheme["estimator"], order=order.index("estimator"),
                          use_cam=use_cam, use_pred=use_pred, use_ukf=use_ukf,
                          rate=rates["estimator"], scheduling=Scheduling.FREQUENCY, advance=False,
                          delay=delays["step"]["estimator"], delay_dist=delays_sim["step"]["estimator"])
    estimator.connect(sensor, window=1, blocking=False,
                      delay_dist=delays_sim["inputs"]["estimator"]["sensor"], delay=delays["inputs"]["estimator"]["sensor"])
    estimator.connect(camera, window=1, blocking=False,
                      delay_dist=delays_sim["inputs"]["estimator"]["camera"], delay=delays["inputs"]["estimator"]["camera"])

    # Create controller
    from envs.pendulum.controller import OpenLoopController, PPOAgent
    Controller = OpenLoopController if use_openloop else PPOAgent
    controller = Controller(name="controller", color=cscheme["controller"], order=order.index("controller"),
                            rate=rates["controller"], scheduling=Scheduling.FREQUENCY, advance=True,
                            delay=delays["step"]["controller"], delay_dist=delays_sim["step"]["controller"])
    controller.connect(estimator, window=1, blocking=True,
                       delay_dist=delays_sim["inputs"]["controller"]["estimator"], delay=delays["inputs"]["controller"]["estimator"])
    estimator.connect(controller, window=4, blocking=True, skip=True,
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