from scripts.dummy import DummyNode, DummyAgent
from rex.distributions import Distribution, Gaussian, GMM
from rex.constants import LATEST, BUFFER, WARN

def test_node_api():
    # Create nodes
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000), log_level=WARN, color="magenta")
    sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(0.007), log_level=WARN, color="yellow")
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016), log_level=WARN, color="cyan")
    agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(0.005, 0.001), log_level=WARN, color="blue", advance=True)
    actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), log_level=WARN, color="green", advance=False,
                         stateful=True)
    nodes = [world, sensor, observer, agent, actuator]

    # Connect
    sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST)
    observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    actuator.connect(agent, blocking=True, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
    world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)

    # Re-initialize nodes
    reload_world = DummyNode.from_info(world.info, log_level=WARN, color="magenta")
    reload_sensor = DummyNode.from_info(sensor.info, log_level=WARN, color="yellow")
    reload_observer = DummyNode.from_info(observer.info, log_level=WARN, color="cyan")
    reload_agent = DummyAgent.from_info(agent.info, log_level=WARN, color="blue")
    reload_actuator = DummyNode.from_info(actuator.info, log_level=WARN, color="green")
    reload_nodes = [world, sensor, observer, agent, actuator]
    reload_nodes = {n.name: n for n in nodes}

    # Re-initialize connections
    reload_sensor.connect_from_info(sensor.inputs[0].info, world)
    reload_observer.connect_from_info(observer.inputs[0].info, sensor)
    reload_observer.connect_from_info(observer.inputs[0].info, agent)
    reload_agent.connect_from_info(agent.inputs[0].info, observer)
    reload_actuator.connect_from_info(actuator.inputs[0].info, agent)
    reload_world.connect_from_info(world.inputs[0].info, actuator)

    # API test
    reload_sensor._phase_dist = reload_sensor.phase_dist
    _ =reload_sensor.phase_dist

    try:
        agent.connect(actuator, blocking=True, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
        agent.phase
    except RecursionError as e:
        pass

