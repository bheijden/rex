import jax.numpy as jnp
import pytest
from distrax import Deterministic
from test_utils import Node, Output

import rex.constants as const
from rex.asynchronous import AsyncGraph


@pytest.mark.parametrize(
    "clock, real_time_factor, scheduling",
    [
        (const.Clock.SIMULATED, const.RealTimeFactor.FAST_AS_POSSIBLE, const.Scheduling.FREQUENCY),
        (const.Clock.SIMULATED, 10, const.Scheduling.PHASE),
        (const.Clock.WALL_CLOCK, const.RealTimeFactor.REAL_TIME, const.Scheduling.FREQUENCY),
    ],
)
def test_reset_step_routine(clock, real_time_factor, scheduling):
    # Create nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), scheduling=scheduling)
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), scheduling=scheduling)
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), scheduling=scheduling)
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), scheduling=scheduling, advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, name="node2", delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=False, jitter=const.Jitter.BUFFER)
    node3.connect(node4, window=2, name="node4", delay_dist=Deterministic(0.01), blocking=True)
    node4.connect(node1, window=3, name="node1", delay_dist=Deterministic(0.01), blocking=True, skip=True)

    # Create graph
    graph = AsyncGraph(nodes=nodes, supervisor=node1, clock=clock, real_time_factor=real_time_factor)

    # Initialize graph
    gs = graph.init()

    # Warmup graph (Only once)
    graph.warmup(gs)

    # Run for one episode
    # Optionally re-init the graph_state here.
    gs, ss = graph.reset(gs)
    for i in range(10):
        if i < 5:
            gs, ss = graph.step(gs)
        else:
            gs, ss = graph.step(gs, step_state=ss, output=Output(jnp.array([99.0])))
    graph.stop()


def test_run_and_recording_api():
    # Create nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01))
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01))
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01))
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, name="node2", delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=False)
    node3.connect(node4, window=2, name="node4", delay_dist=Deterministic(0.01), blocking=True)
    node4.connect(node1, window=3, name="node1", delay_dist=Deterministic(0.01), blocking=True, skip=True)

    # Create graph
    graph = AsyncGraph(
        nodes=nodes, supervisor=node1, clock=const.Clock.SIMULATED, real_time_factor=const.RealTimeFactor.FAST_AS_POSSIBLE
    )

    # Test API
    _ = graph.max_eps
    _ = graph.max_steps

    # Set recording settings
    graph.set_record_settings(params=True, rng=True, state=True, inputs=True, output=True, max_records=5)

    # Initialize graph
    gs = graph.init()

    # Warmup graph (Only once)
    graph.warmup(gs, profile=True)  # Profiles the .step() method of each node (10 repetitions)
    for _ in range(10):
        gs = graph.run(gs)
    graph.stop()

    # Get records
    _record = graph.get_record()
