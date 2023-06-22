import time
import abc
from typing import Any, Dict, List, Tuple, Union
import jumpy
import jumpy.numpy as jp
import jax.numpy as jnp
import numpy as onp
from flax.core import FrozenDict

from rex.agent import Agent
from rex.constants import SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE
from rex.base import StepState, GraphState, InputState
from rex.proto import log_pb2
from rex.node import BaseNode, Node


class BaseGraph:
    def __init__(self, root: Agent, nodes: Dict[str, BaseNode]):
        # Exclude the node for which this environment is a drop-in replacement (i.e. the root)
        nodes = {node.name: node for _, node in nodes.items() if node.name != root.name}
        _assert = len([n for n in nodes.values() if n.name == root.name]) == 0
        assert _assert, "The root should be provided separately, so not inside the `nodes` dict"
        self.root = root
        self.nodes = nodes
        self.nodes_and_root: Dict[str, Node] = {**nodes, root.name: root}

    def __getstate__(self):
        raise NotImplementedError
        # args, kwargs = (), dict(root=self.root, nodes=self.nodes)
        # return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state

        # Unpickle nodes
        nodes, root = kwargs["nodes"], kwargs["root"]
        nodes_and_root = {**nodes, root.name: root}
        for node in nodes_and_root.values():
            node.unpickle(nodes_and_root)

        self.__init__(*args, **kwargs)

    @abc.abstractmethod
    def reset(self, graph_state: GraphState) -> Tuple[GraphState, StepState]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, graph_state: GraphState, step_state: StepState, action: Any) -> Tuple[GraphState, StepState]:
        raise NotImplementedError

    def get_episode_record(self) -> log_pb2.EpisodeRecord:
        raise NotImplementedError

    def max_eps(self, graph_state: GraphState = None):
        raise NotImplementedError

    def max_steps(self, graph_state: GraphState = None) -> int:
        raise NotImplementedError

    def max_starting_step(self, max_steps: int, graph_state: GraphState = None) -> int:
        raise NotImplementedError

    def stop(self, timeout: float = None):
        pass

    def start(self):
        pass


class Graph(BaseGraph):
    def __init__(
        self,
        nodes: Dict[str, "BaseNode"],
        root: Agent,
        clock: int = SIMULATED,
        real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
    ):
        super().__init__(root=root, nodes=nodes)
        self.clock = clock
        self.real_time_factor = real_time_factor

    def __getstate__(self):
        args, kwargs = (), dict(nodes=self.nodes, root=self.root, clock=self.clock, real_time_factor=self.real_time_factor)
        return args, kwargs

    def _default_inputs(self, graph_state: GraphState):
        # Prepare outputs
        rngs_new, outputs = {}, {}
        for name, node in self.nodes_and_root.items():
            rng_new, rng_out = jumpy.random.split(graph_state.nodes[name].rng, num=2)
            rngs_new[name] = rng_new
            outputs[name] = node.default_output(rng_out, graph_state)

        # Prepare inputs
        new_nodes = {}
        for name, node in self.nodes_and_root.items():
            inputs = {}
            for i in node.inputs:
                window = i.window
                seq = 0 * jp.arange(-window, 0, dtype=jp.int32) - 1
                ts_sent = 0 * jp.arange(-window, 0, dtype=jp.float32)
                ts_recv = 0 * jp.arange(-window, 0, dtype=jp.float32)
                _msgs = [outputs[i.output.name]] * window
                inputs[i.input_name] = InputState.from_outputs(seq, ts_sent, ts_recv, _msgs)
            new_nodes[name] = graph_state.nodes[name].replace(rng=rngs_new[name], inputs=FrozenDict(inputs))
        return graph_state.replace(nodes=FrozenDict(new_nodes))

    def reset(self, graph_state: GraphState) -> Tuple[GraphState, StepState]:
        # Stop first, if we were previously running.
        self.stop()

        # An additional reset is required when running async (futures, etc..)
        self.root._agent_reset()

        # Prepare inputs
        graph_state = self._default_inputs(graph_state)

        # Reset async backend of every node
        for node in self.nodes_and_root.values():
            node._reset(graph_state, clock=self.clock, real_time_factor=self.real_time_factor)

        # Check that all nodes have the same episode counter
        assert len({n.eps for n in self.nodes_and_root.values()}) == 1, "All nodes must have the same episode counter."

        # Start nodes (provide same starting timestamp to every node)
        start = time.time()
        [n._start(start=start) for n in self.nodes_and_root.values()]

        # Retrieve first obs
        next_step_state = self.root.observation.popleft().result()

        # Create the next graph state
        nodes = {name: node._step_state for name, node in self.nodes_and_root.items()}
        nodes[self.root.name] = next_step_state
        next_graph_state = GraphState(step=jp.int32(0), nodes=FrozenDict(nodes))
        return next_graph_state, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState, output: Any) -> Tuple[GraphState, StepState]:
        # Set the result to be the step_state and output (action)  of the root.
        self.root.action[-1].set_result((step_state, output))

        # Retrieve the first obs
        next_step_state = self.root.observation.popleft().result()

        # Create the next graph state
        nodes = {name: node._step_state for name, node in self.nodes_and_root.items()}
        nodes[self.root.name] = next_step_state
        next_graph_state = GraphState(step=graph_state.step + 1, nodes=FrozenDict(nodes))
        return next_graph_state, next_step_state

    def stop(self, timeout: float = None):
        # Initiate stop (this unblocks the root's step, that is waiting for an action).
        if len(self.root.action) > 0:
            self.root.action[-1].cancel()

        # Stop all nodes
        fs = [n._stop(timeout=timeout) for n in self.nodes_and_root.values()]

        # Wait for all nodes to stop
        [f.result() for f in fs]

    def get_episode_record(self) -> log_pb2.EpisodeRecord:
        record = log_pb2.EpisodeRecord()
        [record.node.append(node.record()) for node in self.nodes_and_root.values()]
        return record

    def max_eps(self, graph_state: GraphState = None):
        return 1

    def max_steps(self, graph_state: GraphState = None) -> int:
        return jp.inf

    def max_starting_step(self, max_steps: int, graph_state: GraphState = None) -> int:
        return 0
