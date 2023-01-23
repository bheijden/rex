from typing import Any, Tuple, Dict, Union, Optional
import jumpy.numpy as jp
import abc

from rex.spaces import Space
from rex.utils import log, NODE_COLOR, NODE_LOG_LEVEL
from rex.node import Node
from rex.graph import Graph
from rex.compiled import CompiledGraph
from rex.base import GraphState, Params
from rex.proto import log_pb2
from rex.constants import SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, INTERPRETED, VECTORIZED, SEQUENTIAL, BATCHED, WARN
from rex.agent import Agent


class BaseEnv:
    def __init__(self,
                 nodes: Dict[str, "Node"],
                 agent: Agent,
                 max_steps: int = 200,
                 sync: int = SYNC,
                 clock: int = SIMULATED,
                 scheduling: int = PHASE,
                 real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
                 graph_type: int = INTERPRETED,
                 trace: log_pb2.TraceRecord = None,
                 name: str = "env",
                 ):
        self.name = name
        self.graph_type = graph_type
        self.max_steps = 100 if max_steps is None else max_steps
        assert self.max_steps > 0, "max_steps must be a positive integer"

        # Check that the agent is of the correct type
        assert isinstance(agent, Agent), "The agent must be an instance of Agent"
        assert len([n for n in nodes.values() if n.name == agent.name]) == 0, "The agent should be provided separately, so not inside the `nodes` dict"

        # Initialize graph
        if graph_type in [VECTORIZED, SEQUENTIAL, BATCHED]:
            assert trace is not None, "Compiled graphs require a trace"
            self._cgraph = CompiledGraph(nodes, trace, agent, graph_type)
            assert self._cgraph.max_steps >= self.max_steps, f"max_steps ({self.max_steps}) must be smaller than the max number of compiled steps in the graph ({self._cgraph.max_steps})"
        elif graph_type == INTERPRETED:
            if trace is not None:
                self.log("WARNING", "trace is ignored. Set `graph` to a compiled setting (.e.g SEQUENTIAL) to use it.", log_level=WARN)
            self._graph = Graph(nodes, agent, sync, clock, scheduling, real_time_factor)
        else:
            raise ValueError(f"Unknown graph mode: {graph_type}")

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def graph(self):
        if self.graph_type in [VECTORIZED, SEQUENTIAL, BATCHED]:
            return self._cgraph
        elif self.graph_type == INTERPRETED:
            return self._graph
        else:
            raise ValueError(f"Unknown graph mode: {self.graph_type}")

    @abc.abstractmethod
    def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> Tuple[GraphState, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, graph_state: GraphState, action: Any) -> Tuple[GraphState, Any, float, bool, Dict]:
        raise NotImplementedError

    def close(self):
        self.stop()

    def stop(self):
        return self.graph.stop()

    def render(self):
        raise NotImplementedError

    def action_space(self, params: Params = None) -> Space:
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params: Params = None) -> Space:
        """Observation space of the environment."""
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self

    def env_is_wrapped(self, wrapper_class, indices=None):
        return False

    @property
    def log_level(self):
        return NODE_LOG_LEVEL.get(self, WARN)

    @property
    def color(self):
        return NODE_COLOR.get(self, "green")

    def log(self, id: str, value: Optional[Any] = None, log_level: Optional[int] = None):
        log_level = log_level if isinstance(log_level, int) else self.log_level
        log(self.name, self.color, min(log_level, self.log_level), id, value)