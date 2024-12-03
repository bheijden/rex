from typing import Tuple

import flax.struct as struct
import jax
import jax.numpy as jnp
from distrax import Deterministic

import rex.constants as const
from rex.utils import set_log_level
from rex.asynchronous import AsyncGraph
from rex.base import GraphState, StepState, Base
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


def graph_api(clock, real_time_factor):
    # Create nodes
    node1 = Node(name="node1", rate=10, color="pink", order=0, delay_dist=Deterministic(0.01))
    node2 = Node(name="node2", rate=11, color="teal", order=1, delay_dist=Deterministic(0.01))
    node3 = Node(name="node3", rate=12, color="blue", order=2, delay_dist=Deterministic(0.01))
    node4 = Node(name="node4", rate=13, color="blue", order=3, delay_dist=Deterministic(0.01), advance=False)
    nodes = {n.name: n for n in [node1, node2, node3, node4]}

    # Connect nodes
    node1.connect(node2, window=1, name="node2", delay_dist=Deterministic(0.01), blocking=False)
    node2.connect(node3, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=False, jitter=const.Jitter.BUFFER)
    node3.connect(node4, window=2, name="node3", delay_dist=Deterministic(0.01), blocking=True)
    node4.connect(node1, window=3, name="node1", delay_dist=Deterministic(0.01), blocking=True, skip=True)

    # Create graph
    graph = AsyncGraph(nodes=nodes, supervisor=node1, clock=clock, real_time_factor=real_time_factor)

    # Initialize graph
    gs = graph.init()

    # Warmup graph (Only once)
    graph.warmup(gs)

    # Set log level
    set_log_level(const.LogLevel.DEBUG)

    # Run for one episode
    # Optionally re-init the graph_state here.
    gs, ss = graph.reset(gs)
    for _ in range(10):
        gs, ss = graph.step(gs)
    graph.stop()


if __name__ == "__main__":
    graph_api(const.Clock.SIMULATED, const.RealTimeFactor.FAST_AS_POSSIBLE)