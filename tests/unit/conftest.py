from typing import Dict

import pytest
from distrax import Deterministic

from rex.artificial import generate_graphs
from rex.base import Graph as CGraph, TrainableDist
from rex.graph import Graph
from tests.unit.test_utils import Node


@pytest.fixture(scope="module")  # This is a module-scoped fixture (i.e., it runs once per module/file)
def nodes() -> Dict[str, Node]:
    # Create and connect nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node2 = Node(name="node2", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node3 = Node(name="node3", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node4 = Node(name="node4", rate=10, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, blocking=False, delay_dist=Deterministic(0.01))
    node2.connect(node3, window=2, blocking=False, delay_dist=TrainableDist.create(0.005, 0.0, 0.01))
    node3.connect(node4, window=3, blocking=False, delay_dist=Deterministic(0.01))
    node4.connect(node1, window=4, blocking=False, skip=True, delay_dist=Deterministic(0.01))
    return nodes


@pytest.fixture(scope="module")  # This is a module-scoped fixture (i.e., it runs once per module/file)
def cgraphs(nodes) -> CGraph:
    # Generate graphs
    cgraphs = generate_graphs(nodes, 2.0, num_episodes=2)
    return cgraphs


@pytest.fixture(scope="module")
def graph(nodes, cgraphs) -> Graph:
    return Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs)
