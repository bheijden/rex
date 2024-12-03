import pytest
from typing import Dict

import jax
import jax.numpy as jnp

import rex.constants as const
from rex.utils import plot_supergraph, check_generations_uniformity
from rex.graph import Graph
from rex.base import Graph as CGraph
from tests.unit.test_utils import Node, Output


@pytest.mark.parametrize("supergraph", [const.Supergraph.MCS, const.Supergraph.GENERATIONAL, const.Supergraph.TOPOLOGICAL])
def test_basic_api(supergraph: const.Supergraph, nodes: Dict[str, Node], cgraphs: CGraph):
    # Create graph
    graph = Graph(
        nodes=nodes,
        supervisor=nodes["node1"],
        graphs_raw=cgraphs[0],  # Test: Single graph
        skip=None,
        supergraph=supergraph,
        prune=False,  # Test: No pruning
        debug=False,
        progress_bar=True,
        buffer_sizes=dict(node1=5),  # Test: Buffer sizes
        extra_padding=1,  # Test: Extra padding
    )

    # Test uniformity
    is_uniform = check_generations_uniformity(graph.timings.to_generation()[:-1])
    if supergraph in [const.Supergraph.TOPOLOGICAL, const.Supergraph.GENERATIONAL]:
        assert is_uniform

    # Test properties
    _ = graph.S
    _ = graph.Gs
    _ = graph.graphs_raw
    _ = graph.graphs
    _ = graph.timings
    _ = graph.max_eps
    _ = graph.max_steps


def test_supergraph_plotting(nodes: Dict[str, Node], cgraphs: CGraph):
    # Create graph
    graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs)

    # Plot supergraph
    _ax = plot_supergraph(graph.S)


def test_reset_step_routine(nodes: Dict[str, Node], cgraphs: CGraph):
    # Create graph
    graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs)

    # Initialize graph
    gs = graph.init(randomize_eps=True)

    # Jit compile
    graph.step = jax.jit(graph.step)
    graph.reset = jax.jit(graph.reset)

    # Run for one episode
    # Optionally re-init the graph_state here.
    gs, ss = graph.reset(gs)
    for i in range(10):
        if i < 5:
            gs, ss = graph.step(gs)
        else:
            gs, ss = graph.step(gs, step_state=ss, output=Output(jnp.array([99.0])))


def test_run_routine(nodes: Dict[str, Node], cgraphs: CGraph):
    # Create graph
    graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs)

    # Initialize graph
    gs = graph.init(randomize_eps=True)

    # Jit compile
    graph.run = jax.jit(graph.run)

    # Run for one episode
    for i in range(10):
        gs = graph.run(gs)


def test_recording(nodes: Dict[str, Node], cgraphs: CGraph):
    # Create graph
    graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs)

    # Initialize graph
    gs = graph.init(randomize_eps=True)

    # Initialize recording
    gs = graph.init_record(gs, params=True, rng=True, inputs=True, state=True, output=True)

    # Jit compile
    graph.run = jax.jit(graph.run)

    # Run for one episode
    for i in range(10):
        gs = graph.run(gs)


def test_rollout_and_slot_skipping(nodes: Dict[str, Node], cgraphs: CGraph):
    skip = [k for k in nodes.keys() if k != "node1"]

    # Create graph
    graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs, skip=skip)

    # Initialize graph
    gs = graph.init(randomize_eps=True)

    # Test carry only
    _carry = graph.rollout(gs, carry_only=True)

    # Test full rollout
    _rollout = graph.rollout(gs, carry_only=False)


def test_uniform_supergraph(nodes: Dict[str, Node], cgraphs: CGraph):
    # Create graph
    graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs, supergraph=const.Supergraph.GENERATIONAL)

    # Test uniformity
    is_uniform = check_generations_uniformity(graph.timings.to_generation()[:-1])
    assert is_uniform

    # Initialize graph
    gs = graph.init(randomize_eps=True)

    # Run for one step
    gs = graph.run(gs)
