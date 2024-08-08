from typing import Dict, Union, Callable, Any, List
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


def get_default_distributions(nodes: List[str]) -> DelaySim:
    delays_sim = dict(step={}, inputs={})
    for n in nodes:
        delays_sim["step"][n] = base.StaticDist.create(distrax.Deterministic(0.))
        delays_sim["inputs"][n] = {}
        for m in nodes:
            delays_sim["inputs"][n][m] = base.StaticDist.create(distrax.Deterministic(0.0))
    return delays_sim


def load_distribution(file: str) -> DelaySim:
    with open(file, "rb") as f:
        return pickle.load(f)


def simulated_system(record: base.EpisodeRecord,
                     outputs: Dict[str, Any] = None,
                     world_rate: float = 100.,
                     use_ukf: bool = True,
                     ):
    outputs = outputs or {}

    # Make pendulum
    from envs.crazyflie.ode import World, MoCap

    # Create sensor
    mocap = MoCap.from_info(record.nodes["mocap"].info, outputs=outputs.get("mocap", None))

    # Create estimator
    from envs.crazyflie.estimator import Estimator
    estimator = Estimator.from_info(record.nodes["estimator"].info, use_ukf=use_ukf)

    # Create controller
    from envs.crazyflie.agent import PPOAgent
    agent = PPOAgent.from_info(record.nodes["agent"].info, outputs=outputs.get("agent", None))

    # Create pid
    from envs.crazyflie.pid import PID
    pid = PID.from_info(record.nodes["pid"].info, outputs=outputs.get("pid", None))

    # Create supervisor
    from envs.crazyflie.supervisor import Supervisor
    supervisor = Supervisor.from_info(record.nodes["supervisor"].info)

    nodes = dict(mocap=mocap, estimator=estimator, agent=agent, pid=pid, supervisor=supervisor)

    # Connect from info
    [n.connect_from_info(record.nodes[name].info.inputs, nodes) for name, n in nodes.items()]

    # Create world
    from envs.crazyflie.ode import World
    world = World(name="world", color="gray", order=1,
                  rate=world_rate, scheduling=Scheduling.FREQUENCY, advance=False,
                  delay=None, delay_dist=base.StaticDist.create(distrax.Deterministic(0.999 / world_rate)))
    nodes["world"] = world

    # TODO: DEBUG: TURN ON TRAINABLE.
    # mocap_delay = base.StaticDist.create(distrax.Deterministic(0.))
    # pid_delay = base.StaticDist.create(distrax.Deterministic(0.))
    # world_delay = base.StaticDist.create(distrax.Deterministic(0.))
    mocap_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05, interp="linear_real_only")  # Because world is continuous
    pid_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05, interp="zoh")  # Because actions are discrete
    world_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05, interp="linear_real_only")  # Because world is continuous
    pid.connect(world, window=1, blocking=False, jitter=Jitter.LATEST,
                delay_dist=world_delay,
                delay=0.)
    world.connect(pid, window=1, blocking=False, skip=True, jitter=Jitter.LATEST,
                  delay_dist=pid_delay,
                  delay=0.)
    mocap.connect(world, window=1, blocking=False, jitter=Jitter.LATEST,
                  delay_dist=mocap_delay,
                  delay=0.)
    return nodes


def mock_system(delays_sim: DelaySim,
                delay_fn: Callable[[base.DelayDistribution], float],
                rates: Dict[str, float],
                cscheme: Dict[str, str] = None,
                order: list = None,
                use_pred: bool = True,
                use_ukf: bool = True,
                ):
    """Make a real pendulum system."""
    delays = jax.tree_util.tree_map(delay_fn, delays_sim, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    # Create mocap
    from envs.crazyflie.ode import MoCap
    mocap = MoCap(name="mocap", color=cscheme["mocap"], order=order.index("mocap"),
                  rate=rates["mocap"], scheduling=Scheduling.FREQUENCY, advance=False,
                  delay=delays["step"]["mocap"], delay_dist=delays_sim["step"]["mocap"])

    # Create estimator
    from envs.crazyflie.estimator import Estimator
    estimator = Estimator(name="estimator", color=cscheme["estimator"], order=order.index("estimator"),
                          use_pred=use_pred, use_ukf=use_ukf,
                          rate=rates["estimator"], scheduling=Scheduling.FREQUENCY, advance=False,
                          delay=delays["step"]["estimator"], delay_dist=delays_sim["step"]["estimator"])
    estimator.connect(mocap, window=1, blocking=False,
                      delay_dist=delays_sim["inputs"]["estimator"]["mocap"], delay=delays["inputs"]["estimator"]["mocap"])

    # Create agent
    from envs.crazyflie.agent import PPOAgent
    agent = PPOAgent(name="agent", color=cscheme["agent"], order=order.index("agent"),
                     rate=rates["agent"], scheduling=Scheduling.FREQUENCY, advance=True,
                     delay=delays["step"]["agent"], delay_dist=delays_sim["step"]["agent"])
    agent.connect(estimator, window=1, blocking=True,
                  delay_dist=delays_sim["inputs"]["agent"]["estimator"], delay=delays["inputs"]["agent"]["estimator"])
    estimator.connect(agent, window=4, blocking=True, skip=True,  # Skip to avoid algebraic loop, window=4 to compensate for delay
                      delay_dist=delays_sim["inputs"]["estimator"]["agent"], delay=delays["inputs"]["estimator"]["agent"])

    # Create pid
    from envs.crazyflie.pid import PID
    pid = PID(name="pid", color=cscheme["pid"], order=order.index("pid"),
              rate=rates["pid"], scheduling=Scheduling.FREQUENCY, advance=False,
              delay=delays["step"]["pid"], delay_dist=delays_sim["step"]["pid"])
    pid.connect(agent, window=1, blocking=False,
                delay_dist=delays_sim["inputs"]["pid"]["agent"], delay=delays["inputs"]["pid"]["agent"])

    # Create supervisor
    from envs.crazyflie.supervisor import Supervisor
    supervisor = Supervisor(name="supervisor", color=cscheme["supervisor"], order=order.index("supervisor"),
                            rate=rates["supervisor"], scheduling=Scheduling.FREQUENCY, advance=False,
                            delay=delays["step"]["supervisor"], delay_dist=delays_sim["step"]["supervisor"])
    supervisor.connect(estimator, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["supervisor"]["estimator"], delay=delays["inputs"]["supervisor"]["estimator"])

    # Create world
    from envs.crazyflie.ode import World
    world = World(name="world", color=cscheme["world"], order=order.index("world"),
                  rate=rates["world"], scheduling=Scheduling.FREQUENCY, advance=False,
                  delay=None, delay_dist=base.StaticDist.create(distrax.Deterministic(0.999 / rates["world"])))

    mocap_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05, interp="linear_real_only")  # Because world is continuous
    pid_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05, interp="zoh")  # Because actions are discrete
    world_delay = base.TrainableDist.create(alpha=0., min=0.0, max=0.05, interp="linear_real_only")  # Because world is continuous
    pid.connect(world, window=1, blocking=False, jitter=Jitter.LATEST,
                delay_dist=world_delay,
                # delay_dist=delays_sim["inputs"]["pid"]["world"],
                delay=0.)
    world.connect(pid, window=1, blocking=False, skip=True, jitter=Jitter.LATEST,
                  delay_dist=pid_delay,
                  # delay_dist=delays_sim["inputs"]["world"]["pid"],
                  delay=0.)
    mocap.connect(world, window=1, blocking=False, jitter=Jitter.LATEST,
                  delay_dist=mocap_delay,
                  # delay_dist=delays_sim["inputs"]["mocap"]["world"],
                  delay=0.)
    nodes = dict(mocap=mocap,
                 estimator=estimator,
                 agent=agent,
                 pid=pid,
                 supervisor=supervisor,
                 world=world,
                 )
    return nodes


def real_system(delays_sim: DelaySim,
                delay_fn: Callable[[base.DelayDistribution], float],
                rates: Dict[str, float],
                cscheme: Dict[str, str] = None,
                order: list = None,
                feedthrough: bool = True,
                copilot_name: str = "cf",
                mock_copilot: bool = False,
                use_pred: bool = True,
                use_ukf: bool = True,
                ):
    delays = jax.tree_util.tree_map(delay_fn, delays_sim, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    # Create mocap
    from envs.crazyflie.real import MoCap
    mocap = MoCap(name="mocap", color=cscheme["mocap"], order=order.index("mocap"),
                  rate=rates["mocap"], scheduling=Scheduling.FREQUENCY, advance=False,
                  copilot_name=copilot_name, mock=mock_copilot,
                  delay=delays["step"]["mocap"], delay_dist=delays_sim["step"]["mocap"])

    # Create estimator
    from envs.crazyflie.estimator import Estimator
    estimator = Estimator(name="estimator", color=cscheme["estimator"], order=order.index("estimator"),
                          use_pred=use_pred, use_ukf=use_ukf,
                          rate=rates["estimator"], scheduling=Scheduling.FREQUENCY, advance=False,
                          delay=delays["step"]["estimator"], delay_dist=delays_sim["step"]["estimator"])
    estimator.connect(mocap, window=1, blocking=False,
                      delay_dist=delays_sim["inputs"]["estimator"]["mocap"], delay=delays["inputs"]["estimator"]["mocap"])

    # Create agent
    from envs.crazyflie.agent import PPOAgent
    agent = PPOAgent(name="agent", color=cscheme["agent"], order=order.index("agent"),
                     rate=rates["agent"], scheduling=Scheduling.FREQUENCY, advance=True,
                     delay=delays["step"]["agent"], delay_dist=delays_sim["step"]["agent"])
    agent.connect(estimator, window=1, blocking=True,
                  delay_dist=delays_sim["inputs"]["agent"]["estimator"], delay=delays["inputs"]["agent"]["estimator"])
    estimator.connect(agent, window=4, blocking=True, skip=True,  # Skip to avoid algebraic loop, window=4 to compensate for delay
                      delay_dist=delays_sim["inputs"]["estimator"]["agent"], delay=delays["inputs"]["estimator"]["agent"])

    # Create pid
    from envs.crazyflie.real import PID
    pid = PID(name="pid", color=cscheme["pid"], order=order.index("pid"),
              rate=rates["pid"], scheduling=Scheduling.FREQUENCY, advance=False,
              copilot_name=copilot_name, mock=mock_copilot, feedthrough=feedthrough,
              delay=delays["step"]["pid"], delay_dist=delays_sim["step"]["pid"])
    pid.connect(agent, window=1, blocking=False,
                delay_dist=delays_sim["inputs"]["pid"]["agent"], delay=delays["inputs"]["pid"]["agent"])

    # Create supervisor
    from envs.crazyflie.supervisor import Supervisor
    supervisor = Supervisor(name="supervisor", color=cscheme["supervisor"], order=order.index("supervisor"),
                            rate=rates["supervisor"], scheduling=Scheduling.FREQUENCY, advance=False,
                            delay=delays["step"]["supervisor"], delay_dist=delays_sim["step"]["supervisor"])
    supervisor.connect(estimator, window=1, blocking=False,
                       delay_dist=delays_sim["inputs"]["supervisor"]["estimator"], delay=delays["inputs"]["supervisor"]["estimator"])

    nodes = dict(mocap=mocap,
                 estimator=estimator,
                 agent=agent,
                 pid=pid,
                 supervisor=supervisor,
                 )
    return nodes
