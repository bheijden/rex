import time
from typing import Tuple, Deque, Dict, Union, List
from collections import deque
from concurrent.futures import Future, CancelledError

from flax.core import FrozenDict

import jax
import jax.numpy as jnp
from rex.constants import SIMULATED, FAST_AS_POSSIBLE
from rex.graph import BaseGraph

from rex.node import Node
from rex.base import StepState, Output, GraphState
from rex.proto import log_pb2


class _Synchronizer:
    def __init__(self, root: Node):
        self._root = root
        self._root._step = self._step
        self._must_reset: bool
        self._f_act: Future
        self._f_obs: Future
        self._q_act: Deque[Future] = deque()
        self._q_obs: Deque[Future]

    @property
    def action(self) -> Deque[Future]:
        return self._q_act

    @property
    def observation(self) -> Deque[Future]:
        return self._q_obs

    def reset(self):
        self._must_reset = False
        self._q_act: Deque[Future] = deque()
        self._q_obs: Deque[Future] = deque()
        self._f_obs = Future()
        self._q_obs.append(self._f_obs)

    def _step(self, step_state: StepState) -> Tuple[StepState, Output]:
        self._f_act = Future()
        self._q_act.append(self._f_act)

        # Prepare new obs future
        _new_f_obs = Future()
        self._q_obs.append(_new_f_obs)

        # Set observations as future result
        self._f_obs.set_result(step_state)
        self._f_obs = _new_f_obs

        # Wait for action future's result to be set with action
        if not self._must_reset:
            try:
                step_state, output = self._f_act.result()
                self._q_act.popleft()
                return step_state, output
            except CancelledError:  # If cancelled is None, we are going to reset
                self._q_act.popleft()
                self._must_reset = True
        return None, None  # Do not return anything if we must reset


class AsyncGraph(BaseGraph):
    def __init__(
        self,
        nodes: Dict[str, "Node"],
        root: Node,
        clock: int = SIMULATED,
        real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
        skip: List[str] = None,
    ):
        super().__init__(root=root, nodes=nodes, skip=skip)
        self.clock = clock
        self.real_time_factor = real_time_factor
        self._synchronizer = _Synchronizer(root)
        self._initial_step = True

    def __getstate__(self):
        args, kwargs = (), dict(nodes=self.nodes, root=self.root, clock=self.clock, real_time_factor=self.real_time_factor)
        return args, kwargs

    def start(self, graph_state: GraphState, timeout: float = None) -> GraphState:
        # Stop first, if we were previously running.
        self.stop(timeout=timeout)

        # An additional reset is required when running async (futures, etc..)
        self._synchronizer.reset()

        # Prepare inputs
        no_inputs = {k: (k in graph_state.nodes and graph_state.nodes[k].inputs) is not None for k, v in
                     graph_state.nodes.items()}
        assert all(no_inputs.values()), "No inputs provided to all entries in graph_state. Use graph.init()."

        # Reset async backend of every node
        for node in self.nodes_and_root.values():
            node._reset(graph_state, clock=self.clock, real_time_factor=self.real_time_factor)

        # Check that all nodes have the same episode counter
        assert len({n.eps for n in self.nodes_and_root.values()}) == 1, "All nodes must have the same episode counter."

        # Start nodes (provide same starting timestamp to every node)
        start = time.time()
        for node in self.nodes_and_root.values():
            node._start(start=start)
        return graph_state

    def stop(self, timeout: float = None):
        # Initiate stop (this unblocks the root's step, that is waiting for an action).
        if len(self._synchronizer.action) > 0:
            self._synchronizer.action[-1].cancel()

        # Stop all nodes
        fs = [n._stop(timeout=timeout) for n in self.nodes_and_root.values()]

        # Wait for all nodes to stop
        [f.result() for f in fs]

        # Toggle
        self._initial_step = True

    def run_until_root(self, graph_state: GraphState) -> GraphState:
        # Retrieve obs
        next_step_state = self._synchronizer.observation.popleft().result()
        self._initial_step = False
        nodes = {name: node._step_state for name, node in self.nodes_and_root.items()}
        nodes[self.root.name] = next_step_state
        next_graph_state = GraphState(nodes=FrozenDict(nodes))
        return next_graph_state

    def run_root(self, graph_state: GraphState, step_state: StepState = None, output: Output = None) -> GraphState:
        assert (step_state is None) == (output is None), "Either both step_state and output must be None or both must be not None."
        # If run_root is run before run_until_root, we skip.
        if self._initial_step:
            return graph_state

        # Get next step state and output from root node
        if step_state is None and output is None:  # Run root node
            ss = self.root.get_step_state(graph_state)
            new_ss, new_output = self.root.step(ss)
        else:  # Override step_state and output
            new_ss, new_output = step_state, output

        # Update step_state (increment sequence number)
        next_step_state = new_ss.replace(seq=new_ss.seq + 1)

        # Set the result to be the step_state and output (action)  of the root.
        self._synchronizer.action[-1].set_result((new_ss, new_output))

        # Get graph_state
        nodes = {name: node._step_state for name, node in self.nodes_and_root.items()}
        nodes[self.root.name] = next_step_state
        next_graph_state = GraphState(nodes=FrozenDict(nodes))
        return next_graph_state

    def get_episode_record(self) -> log_pb2.EpisodeRecord:
        record = log_pb2.EpisodeRecord()
        [record.node.append(node.record()) for node in self.nodes_and_root.values()]
        return record

    def max_eps(self, graph_state: GraphState = None):
        return 1

    def max_steps(self, graph_state: GraphState = None) -> Union[int, jax.Array]:
        return jnp.inf

    def max_starting_step(self, max_steps: int, graph_state: GraphState = None) -> Union[int, jax.Array]:
        return 0
