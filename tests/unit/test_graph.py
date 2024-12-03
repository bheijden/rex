from typing import Dict

import pytest
from distrax import Deterministic

import rex.constants as const
from rex.artificial import augment_graphs, generate_graphs
from rex.asynchronous import AsyncGraph
from rex.base import ExperimentRecord, Graph, TrainableDist
from rex.utils import get_subplots, plot_graph, plot_system, to_networkx_graph
from tests.unit.test_utils import Node


@pytest.fixture()
def nodes_mixed() -> Dict[str, Node]:
    """
    Called nodes_mixed to avoid conflict with nodes fixture in conftest.py.
    "mixed" as in both blocking and non-blocking connections.
    """
    # Create and connect nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), advance=False)
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), advance=False)
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, delay_dist=Deterministic(0.01), blocking=False)
    node3.connect(node4, window=2, delay_dist=Deterministic(0.01), blocking=True)
    node4.connect(node1, window=3, delay_dist=Deterministic(0.01), blocking=True, skip=True)
    return nodes


@pytest.fixture()
def exp_record(nodes_mixed) -> ExperimentRecord:
    """Fixture to create nodes, set up the graph, warm up, and generate records."""
    nodes = nodes_mixed
    node1 = nodes["node1"]

    # Create and initialize graph
    graph = AsyncGraph(
        nodes=nodes,
        supervisor=node1,
        clock=const.Clock.SIMULATED,
        real_time_factor=const.RealTimeFactor.FAST_AS_POSSIBLE,
    )
    gs = graph.init()

    # Warmup graph
    graph.warmup(gs)

    # Generate records
    episodes = []
    for _ in range(2):
        for _ in range(10):
            gs = graph.run(gs)
        graph.stop()

        # Get records
        record = graph.get_record()
        episodes.append(record)

    # Convert to ExperimentRecord
    exp_record = ExperimentRecord(episodes=episodes)
    return exp_record


def test_filtering(nodes_mixed: Dict[str, Node], exp_record: ExperimentRecord):
    nodes = nodes_mixed
    # Test graph filtering with a subset of nodes and connections
    fnode1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    fnode2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), advance=False)
    fnode3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), advance=False)
    fnodes = {n.name: n for n in [fnode1, fnode2, fnode3]}
    # Connect nodes
    fnode1.connect(fnode2, window=1, delay_dist=Deterministic(0.01), blocking=False)

    # Filter record
    exp_record_nf = exp_record.filter(nodes=fnodes, filter_connections=False)
    exp_record_f = exp_record.filter(nodes=fnodes, filter_connections=True)
    assert "node3" in exp_record_nf.episodes[0].nodes["node2"].inputs.keys()
    assert "node3" not in exp_record_f.episodes[0].nodes["node2"].inputs.keys()

    # Filter graph
    cgraphs = exp_record.to_graph()
    cgraphs_nf = cgraphs.filter(nodes=fnodes, filter_edges=False)
    cgraphs_f = cgraphs.filter(nodes=fnodes, filter_edges=True)
    assert ("node3", "node2") in cgraphs_nf.edges.keys()
    assert ("node3", "node2") not in cgraphs_f.edges.keys()


def test_generate_graphs():
    # Create and connect nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), advance=False)
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), advance=False)
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, delay_dist=Deterministic(0.01), blocking=False)
    node3.connect(node4, window=2, delay_dist=Deterministic(0.01), blocking=False)
    node4.connect(node1, window=3, delay_dist=Deterministic(0.01), blocking=False, skip=True)

    # Generate graphs
    cgraphs = generate_graphs(nodes, 10.0, num_episodes=2)
    assert len(cgraphs) == 2

    # Test graph stacking
    graphs_raw = [cgraphs[0], cgraphs[1]]
    _cgraphs = Graph.stack(graphs_raw)


def test_graph_plotting():
    # Create and connect nodes
    node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
    node2 = Node(name="node2", rate=11, delay_dist=Deterministic(0.01), advance=False)
    node3 = Node(name="node3", rate=12, delay_dist=Deterministic(0.01), advance=False)
    node4 = Node(name="node4", rate=13, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, delay_dist=Deterministic(0.01), blocking=False)
    node3.connect(node4, window=2, delay_dist=Deterministic(0.01), blocking=False)
    node4.connect(node1, window=3, delay_dist=Deterministic(0.01), blocking=False, skip=True)

    # Generate graphs
    cgraphs = generate_graphs(nodes, 2.0, num_episodes=2)
    assert len(cgraphs) == 2

    # Test graph plotting
    G = to_networkx_graph(cgraphs[0], nodes, validate=True)
    _ax = plot_graph(G, max_x=1.0)

    # Test system plotting
    infos = {k: v.info for k, v in nodes.items()}
    plot_system(infos)


@pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
def test_subplots(num):
    some_tree = list(range(num))
    major = "row" if num % 2 == 0 else "col"
    fig, tree_axes = get_subplots(some_tree)


@pytest.mark.parametrize("batch", ["single", "multiple"])
def test_augment_graphs(batch, nodes_mixed: Dict[str, Node], exp_record: ExperimentRecord):
    nodes = nodes_mixed
    # Test for single episode or multiple episodes
    if batch == "single":
        records = exp_record.episodes[0]
    else:
        records = exp_record.stack(method="padded")
        _ = records[0]  # Test __getitem__ API

    # Convert records to cgraph
    cgraphs = records.to_graph()

    # Test API
    _ = len(cgraphs)
    try:
        _ = cgraphs[0]
    except ValueError:
        if batch == "single":
            assert True
        else:
            assert False

    # Add a node
    nodes["node5"] = Node(name="node5", rate=14, delay_dist=Deterministic(0.01), advance=False)
    nodes["node1"].connect(nodes["node5"], window=1, delay_dist=Deterministic(0.01), blocking=False)
    nodes["node2"].connect(nodes["node5"], window=1, delay_dist=TrainableDist.create(0.01, 0.0, 0.01), blocking=False)

    # Augment graphs
    cgraphs_aug = augment_graphs(cgraphs, nodes)
