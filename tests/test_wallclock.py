import time
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import jax.random as rnd

import rex.utils as utils
from rex.proto import log_pb2
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, SYNC, ASYNC, REAL_TIME, FAST_AS_POSSIBLE, FREQUENCY, PHASE, SIMULATED, WALL_CLOCK
from rex.distributions import Gaussian, GMM
from rex.base import GraphState, StepState
from scripts.dummy import DummyNode
from rex.asynchronous import AsyncGraph

utils.set_log_level(WARN)


def test_wallclock():
    # Create nodes
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000))
    sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(0.007))
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016))
    agent = DummyNode("root", rate=45, delay_sim=Gaussian(0.005, 0.001), advance=True)
    actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), advance=False,
                         stateful=True)
    nodes = [world, sensor, observer, agent, actuator]

    # Connect
    sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST)
    observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    actuator.connect(agent, blocking=False, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05)
    world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)

    # Get graph state
    def reset_node(node, rng):
        rng_params, rng_state, rng_inputs, rng_step, rng_reset = rnd.split(rng, num=5)
        params = node.default_params(rng_params)
        state = node.default_state(rng_state)
        inputs = node.default_inputs(rng_inputs)
        node.reset(rng_reset)
        return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

    # Get PRNG
    seed = rnd.PRNGKey(0)
    rngs = rnd.split(seed, num=len(nodes))
    node_ss = {n.name: reset_node(n, rng) for n, rng in zip(nodes, rngs)}

    # Set the run modes
    # real_time_factor, clock = FAST_AS_POSSIBLE, SIMULATED
    real_time_factor, clock = 20, WALL_CLOCK
    graph = AsyncGraph({n.name: n for n in nodes if n.name != agent.name}, root=agent, real_time_factor=real_time_factor, clock=clock)
    graph_state = graph.init(step_states=node_ss, randomize_eps=True)

    # Warmup nodes (pre-compile jitted functions)
    [n.warmup(graph_state) for n in nodes]

    # Run
    num_steps = 500
    tstart = time.time()
    graph_state = graph.start(graph_state)
    for _ in range(500):
        graph_state = graph.run(graph_state)
    graph.stop()
    tend = time.time()

    print(f"agent_steps={num_steps} | node_steps={[n._i for n in nodes]} | t={(tend - tstart): 2.4f} sec | fps={num_steps / (tend - tstart): 2.4f}")

    # Gather the records
    record = log_pb2.EpisodeRecord()
    [record.node.append(node.record()) for node in nodes]
    d = {n.info.name: n for n in record.node}
