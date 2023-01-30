import jumpy
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
from scripts.dummy import DummyNode, DummyAgent

utils.set_log_level(WARN)


def test_wallclock():
    # Create nodes
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000))
    sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(0.007))
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016))
    agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(0.005, 0.001), advance=True)
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

    # Warmup nodes (pre-compile jitted functions)
    [n.warmup() for n in nodes]

    # Get PRNG
    seed = rnd.PRNGKey(0)
    rngs = rnd.split(seed, num=len(nodes))

    # Set the run modes
    # real_time_factor, scheduling, clock, sync = FAST_AS_POSSIBLE, FREQUENCY, SIMULATED, SYNC
    real_time_factor, scheduling, clock, sync = 20, PHASE, WALL_CLOCK, ASYNC

    # Get graph state
    def reset_node(node, rng):
        rng_params, rng_state, rng_inputs, rng_step, rng_reset = jumpy.random.split(rng, num=5)
        params = node.default_params(rng_params)
        state = node.default_state(rng_state)
        inputs = node.default_inputs(rng_inputs)
        node.reset(rng_reset)
        return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

    node_ss = {n.name: reset_node(n, rng) for n, rng in zip(nodes, rngs)}
    graph_state = GraphState(step=0, nodes=node_ss)

    # Reset nodes (Allows setting the run mode)
    [n._reset(graph_state, real_time_factor=real_time_factor, clock=clock) for rng, n in
     zip(rngs, nodes)]

    # An additional reset is required when running async (futures, etc..)
    agent._agent_reset()

    # Check that all nodes have the same episode counter
    assert len({n.eps for n in nodes}) == 1, "All nodes must have the same episode counter"

    # Start nodes (provide same starting timestamp to every node)
    num_steps = 500
    start = time.time()
    [n._start(start=start) for n in nodes]

    # Simulate
    tstart = time.time()
    ts_step, step_state = agent.observation.popleft().result()  # Retrieve first obs
    for _ in range(num_steps):
        action = agent.default_output(seed)  # NOTE! Re-using the seed here.
        agent.action[-1].set_result((step_state, action))  # The set result must be the action of the agent.
        ts_step, step_state = agent.observation.popleft().result()  # Retrieve observation
    tend = time.time()

    # Initiate reset
    agent.action[-1].cancel()  # Cancel to stop the actions being sent by the agent node.

    # Stop all ndoes
    fs = [n._stop() for n in nodes]

    # Wait for nodes to have stopped
    [f.result() for f in fs]

    print(f"agent_steps={num_steps} | node_steps={[n._i for n in nodes]} | t={(tend - tstart): 2.4f} sec | fps={num_steps / (tend - tstart): 2.4f}")

    # Gather the records
    record = log_pb2.EpisodeRecord()
    [record.node.append(node.record()) for node in nodes]
    d = {n.info.name: n for n in record.node}