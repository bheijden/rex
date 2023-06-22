from scripts.dummy import DummyNode, DummyAgent
from rex.distributions import Distribution, Gaussian, GMM
from rex.constants import LATEST, BUFFER, WARN, DEBUG, ERROR, READY
import rex.utils as utils
import pickle


def test_node_api():
    # Create nodes
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000))
    sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(0.007))
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016))
    agent = DummyAgent("root", rate=45, delay_sim=Gaussian(0.005, 0.001), advance=True)
    actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), advance=False,
                         stateful=True)
    nodes = [world, sensor, observer, agent, actuator]
    nodes = {n.name: n for n in nodes}

    # Connect
    sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST)
    observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    actuator.connect(agent, blocking=True, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
    world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)

    # Re-initialize Basenode with info (loses subclass info)
    reload_world = DummyNode.from_info(world.info)
    reload_sensor = DummyNode.from_info(sensor.info)
    reload_observer = DummyNode.from_info(observer.info)
    reload_agent = DummyAgent.from_info(agent.info)
    reload_actuator = DummyNode.from_info(actuator.info)
    reload_nodes = [reload_world, reload_sensor, reload_observer, reload_agent, reload_actuator]
    reload_nodes = {n.name: n for n in reload_nodes}

    # Re-initialize connections with info
    [n.connect_from_info(nodes[name].info.inputs, reload_nodes) for name, n in reload_nodes.items() if name != "root"]

    # Re-initialize root connection with info
    reload_nodes["root"].connect_from_info(nodes["root"].info.inputs[0], reload_nodes)

    try:
        [n.connect_from_info(nodes[name].info.inputs, reload_nodes) for name, n in reload_nodes.items()]
    except AssertionError as e:
        print(f"should fail here: {e}")

    # API test
    reload_sensor._phase_dist = reload_sensor.phase_dist
    _ = reload_sensor.phase_dist

    try:
        agent.connect(actuator, blocking=True, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
        agent.phase
    except RecursionError as e:
        pass


def test_node_pickle_reload():

    # Create nodes
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000))
    sensor = DummyNode("sensor", rate=20, delay_sim=GMM([Gaussian(0.007)], [1.0]))
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016))
    agent = DummyAgent("root", rate=45, delay_sim=Gaussian(0.005, 0.001), advance=True)
    actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), advance=False,
                         stateful=True)
    nodes = [world, sensor, observer, agent, actuator]

    # Connect
    sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST)
    observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    actuator.connect(agent, blocking=True, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
    world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)

    # Re-initialize nodes with pickle
    reload_world = pickle.loads(pickle.dumps(world))
    reload_sensor = pickle.loads(pickle.dumps(sensor))
    reload_observer = pickle.loads(pickle.dumps(observer))
    reload_agent = pickle.loads(pickle.dumps(agent))
    reload_actuator = pickle.loads(pickle.dumps(actuator))
    reload_nodes = [reload_world, reload_sensor, reload_observer, reload_agent, reload_actuator]
    reload_nodes = {n.name: n for n in reload_nodes}

    # Re-initialize connections with pickle
    [n.unpickle(reload_nodes) for n in reload_nodes.values()]
    [n.unpickle(reload_nodes) for n in reload_nodes.values()]
    assert all(n.unpickled for n in reload_nodes.values()), "Not all nodes unpickled"


def test_executor_errors():
    utils.set_log_level(DEBUG)

    world = DummyNode("world", rate=20)
    sensor = DummyNode("sensor", rate=20)
    sensor.connect(world, blocking=False, skip=False, jitter=LATEST)

    # Test that submitted work is skipped if state is not ready or running
    world._submit(lambda: print("hello"))
    sensor.inputs[0]._submit(lambda: print("hello"))

    # Test that traceback is printed if exception is raised
    world._state = READY
    sensor.inputs[0]._state = READY

    def raise_exception():
        raise Exception("Raise test exception.")

    f = world._submit(raise_exception)
    ff = sensor.inputs[0]._submit(raise_exception)

    try:
        f.result()
    except Exception:
        pass

    try:
        ff.result()
    except Exception:
        pass
    utils.set_log_level(WARN)