import time
import abc
from typing import Any, Dict, List, Tuple, Union
import jumpy.numpy as jp
import jax.numpy as jnp
import numpy as onp
from flax.core import FrozenDict

from rex.agent import Agent
from rex.constants import SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE
from rex.base import StepState, GraphState
from rex.proto import log_pb2
from rex.node import BaseNode


# float32 = Union[jnp.float32, onp.float32]


class BaseGraph:
    def __init__(self, agent: Agent, nodes: Dict[str, BaseNode]):
        _assert = len([n for n in nodes.values() if n.name == agent.name]) == 0
        assert _assert, "The agent should be provided separately, so not inside the `nodes` dict"
        self.agent = agent
        self.nodes = nodes
        self.nodes_and_agent = {**nodes, agent.name: agent}

    def __getstate__(self):
        args, kwargs = (), dict(agent=self.agent, nodes=self.nodes)
        return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state

        # Unpickle nodes
        nodes, agent = kwargs["nodes"], kwargs["agent"]
        nodes_and_agent = {**nodes, agent.name: agent}
        for node in nodes_and_agent.values():
            node.unpickle(nodes_and_agent)

        self.__init__(*args, **kwargs)

    @abc.abstractmethod
    def reset(self, graph_state: GraphState) -> Tuple[GraphState, StepState]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, StepState]:
        raise NotImplementedError

    def stop(self, timeout: float = None):
        pass

    def start(self):
        pass


class Graph(BaseGraph):
    def __init__(
        self,
        nodes: Dict[str, "BaseNode"],
        agent: Agent,
        clock: int = SIMULATED,
        real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
    ):
        super().__init__(agent=agent, nodes=nodes)
        self.clock = clock
        self.real_time_factor = real_time_factor

    def __getstate__(self):
        args, kwargs = (), dict(nodes=self.nodes, agent=self.agent, clock=self.clock, real_time_factor=self.real_time_factor)
        return args, kwargs

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, StepState]:
        # Stop first, if we were previously running.
        self.stop()

        # An additional reset is required when running async (futures, etc..)
        self.agent._agent_reset()

        # Reset async backend of every node
        for node in self.nodes_and_agent.values():
            node._reset(graph_state, clock=self.clock, real_time_factor=self.real_time_factor)

        # Check that all nodes have the same episode counter
        assert len({n.eps for n in self.nodes_and_agent.values()}) == 1, "All nodes must have the same episode counter."

        # Start nodes (provide same starting timestamp to every node)
        start = time.time()
        [n._start(start=start) for n in self.nodes_and_agent.values()]

        # Retrieve first obs
        next_step_state = self.agent.observation.popleft().result()

        # Create the next graph state
        nodes = {name: node._step_state for name, node in self.nodes_and_agent.items()}
        nodes[self.agent.name] = next_step_state
        next_graph_state = GraphState(step=jp.int32(0), nodes=FrozenDict(nodes))
        return next_graph_state, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, StepState]:
        # Set the result to be the step_state and output (action)  of the agent.
        self.agent.action[-1].set_result((step_state, output))

        # Retrieve the first obs
        next_step_state = self.agent.observation.popleft().result()

        # Create the next graph state
        nodes = {name: node._step_state for name, node in self.nodes_and_agent.items()}
        nodes[self.agent.name] = next_step_state
        next_graph_state = GraphState(step=graph_state.step + 1, nodes=FrozenDict(nodes))
        return next_graph_state, next_step_state

    def stop(self, timeout: float = None):
        # Initiate stop (this unblocks the agent's step, that is waiting for an action).
        if len(self.agent.action) > 0:
            self.agent.action[-1].cancel()

        # Stop all nodes
        fs = [n._stop(timeout=timeout) for n in self.nodes_and_agent.values()]

        # Wait for all nodes to stop
        [f.result() for f in fs]




