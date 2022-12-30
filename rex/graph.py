import time
import abc
from typing import Any, Dict, List, Tuple, Union
import jumpy as jp
import jax.numpy as jnp
import numpy as onp
from flax.core import FrozenDict

from rex.agent import Agent
from rex.constants import SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE
from rex.base import StepState, GraphState
from rex.proto import log_pb2
from rex.node import Node


float32 = Union[jnp.float32, onp.float32]


class BaseGraph:
    def __init__(self, agent: Agent):
        self.agent = agent

    @abc.abstractmethod
    def reset(self, graph_state: GraphState) -> Tuple[GraphState, float32, StepState]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, float32, StepState]:
        raise NotImplementedError

    def stop(self, timeout: float = None):
        pass

    def start(self):
        pass


class Graph(BaseGraph):
    def __init__(
        self,
        nodes: Dict[str, "Node"],
        agent: Agent,
        sync: int = SYNC,
        clock: int = SIMULATED,
        scheduling: int = PHASE,
        real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
    ):
        self.nodes = nodes
        self._nodes_and_agent = {**nodes, agent.name: agent}
        self.sync = sync
        self.clock = clock
        self.scheduling = scheduling
        self.real_time_factor = real_time_factor
        super().__init__(agent=agent)

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, float32, Any]:
        # Stop first, if we were previously running.
        self.stop()

        # An additional reset is required when running async (futures, etc..)
        self.agent._agent_reset()

        # Reset async backend of every node
        for node in self._nodes_and_agent.values():
            node._reset(
                graph_state,
                sync=self.sync,
                clock=self.clock,
                scheduling=self.scheduling,
                real_time_factor=self.real_time_factor,
            )

        # Check that all nodes have the same episode counter
        assert len({n.eps for n in self._nodes_and_agent.values()}) == 1, "All nodes must have the same episode counter."

        # Start nodes (provide same starting timestamp to every node)
        start = time.time()
        [n._start(start=start) for n in self._nodes_and_agent.values()]

        # Retrieve first obs
        next_ts_step, next_step_state = self.agent.observation.popleft().result()

        # Create the next graph state
        nodes = {name: node._step_state for name, node in self._nodes_and_agent.items()}
        nodes[self.agent.name] = next_step_state
        next_graph_state = GraphState(step=jp.int32(0), nodes=FrozenDict(nodes))
        return next_graph_state, next_ts_step, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, float32, StepState]:
        # Set the result to be the step_state and output (action)  of the agent.
        self.agent.action[-1].set_result((step_state, output))

        # Retrieve the first obs
        next_ts_step, next_step_state = self.agent.observation.popleft().result()

        # Create the next graph state
        nodes = {name: node._step_state for name, node in self._nodes_and_agent.items()}
        nodes[self.agent.name] = next_step_state
        next_graph_state = GraphState(step=graph_state.step + 1, nodes=FrozenDict(nodes))
        return next_graph_state, next_ts_step, next_step_state

    def stop(self, timeout: float = None):
        # Initiate stop (this unblocks the agent's step, that is waiting for an action).
        if len(self.agent.action) > 0:
            self.agent.action[-1].cancel()

        # Stop all nodes
        fs = [n._stop(timeout=timeout) for n in self._nodes_and_agent.values()]

        # Wait for all nodes to stop
        [f.result() for f in fs]




