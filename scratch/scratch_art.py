import pytest
from typing import Tuple

import flax.struct as struct
import jax
import jax.numpy as jnp
from distrax import Deterministic

import rex.constants as const
from rex.artificial import augment_graphs, generate_graphs
from rex.asynchronous import AsyncGraph
from rex.base import GraphState, StepState, Base, Graph
from rex.node import BaseNode


@struct.dataclass
class Params(Base):
    """Arbitrary dataclass."""
    a: jax.Array


@struct.dataclass
class State(Base):
    """Arbitrary dataclass."""
    a: jax.Array


@struct.dataclass
class Output(Base):
    """Arbitrary dataclass."""
    a: jax.Array


class Node(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> Params:
        return Params(jnp.array([1.0]))

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> State:
        return State(jnp.array([1.0]))

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> Output:
        return Output(jnp.array([1.0]))

    def step(self, step_state: StepState) -> Tuple[StepState, Output]:
        return step_state, Output(jnp.array([1.0]))


def _test_artificial_api():
    # Create nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01))
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01))
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01))
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, name="node2", delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=True)
    node4.connect(node1, window=3, name="node1", delay_dist=Deterministic(0.01), blocking=True, skip=True)

    # Create graph
    graph = AsyncGraph(nodes=nodes, supervisor=node1, clock=const.Clock.SIMULATED, real_time_factor=const.RealTimeFactor.FAST_AS_POSSIBLE)

    # Initialize graph
    gs = graph.init()

    # Warmup graph (Only once)
    graph.warmup(gs)

    # Get records for 2 episodes
    records = []
    for _ in range(2):
        for _ in range(10):
            gs = graph.run(gs)
        graph.stop()

        # Get records
        record = graph.get_record()
        records.append(record)


def _test_generate_graphs():
    # Create and connect nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), advance=False)
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), advance=False)
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, name="node2", delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=False)
    node3.connect(node4, window=2, name="node4", delay_dist=Deterministic(0.01), blocking=False)
    node4.connect(node1, window=3, name="node1", delay_dist=Deterministic(0.01), blocking=False, skip=True)

    # Generate graphs
    cgraphs = generate_graphs(nodes, 10.0, num_episodes=2)
    assert len(cgraphs) == 2

    # Test graph stacking
    graphs_raw = [cgraphs[0], cgraphs[1]]
    _cgraphs = Graph.stack(graphs_raw)

    # Test graph filtering
    fnode1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    fnode2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), advance=False)
    fnode3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), advance=False)
    fnodes = {n.name: n for n in [fnode1, fnode2, fnode3]}
    # Connect nodes
    fnode1.connect(fnode2, window=1, name="node2", delay_dist=Deterministic(0.01), blocking=False)
    # Filter graph
    cgraphs_nf = cgraphs.filter(nodes=fnodes, filter_edges=False)
    cgraphs_f = cgraphs.filter(nodes=fnodes, filter_edges=True)
    assert ("node3", "node2") in cgraphs_nf.edges.keys()
    assert ("node3", "node2") not in cgraphs_f.edges.keys()


if __name__ == "__main__":
    _test_generate_graphs()
    _test_artificial_api()