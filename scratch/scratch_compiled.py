import pytest
from typing import Dict

import jax
import jax.numpy as jnp
from distrax import Deterministic

import rex.constants as const
from rex.utils import plot_supergraph, check_generations_uniformity
from rex.graph import Graph
from rex.artificial import generate_graphs
from rex.asynchronous import AsyncGraph
from rex.base import Graph as CGraph, TrainableDist
from tests.unit.test_utils import Node, Output

# Create and connect nodes
node1 = Node(name="node1", rate=10, delay_dist=Deterministic(0.01), advance=False)
node2 = Node(name="node2", rate=10, delay_dist=Deterministic(0.01), advance=False)
node3 = Node(name="node3", rate=10, delay_dist=Deterministic(0.01), advance=False)
node4 = Node(name="node4", rate=10, delay_dist=Deterministic(0.01), advance=False)
nodes = {n.name: n for n in [node1, node2, node3, node4]}

# Connect nodes
node1.connect(node2, window=1, blocking=False, delay_dist=Deterministic(0.01))
node2.connect(node3, window=2, blocking=False, delay_dist=TrainableDist.create(0.005, 0., 0.01))
node3.connect(node4, window=3, blocking=False, delay_dist=Deterministic(0.01))
node4.connect(node1, window=4, blocking=False, skip=True, delay_dist=Deterministic(0.01))

# Create asynchronous graph
async_graph = AsyncGraph(nodes=nodes, supervisor=nodes["node1"],
                         clock=const.Clock.SIMULATED,
                         real_time_factor=const.RealTimeFactor.FAST_AS_POSSIBLE)
gs = async_graph.init()
async_graph.set_record_settings(params=True, rng=True, inputs=True, state=True, output=True)
async_graph.warmup(gs)
# Run for one episode
for i in range(10):
    gs = async_graph.run(gs)
async_graph.stop()
record_async = async_graph.get_record()

# Generate graphs
cgraphs = generate_graphs(nodes, 10.0, num_episodes=2)

# Create graph
graph = Graph(nodes=nodes, supervisor=nodes["node1"], graphs_raw=cgraphs)

# Initialize graph
gs = graph.init(randomize_eps=True)

# Initialize recording
gs = graph.init_record(gs, params=True, rng=True, inputs=True, state=True, output=True)

# Run for one episode
for i in range(10):
    gs = graph.run(gs)