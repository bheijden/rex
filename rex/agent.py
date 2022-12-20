from typing import Tuple, Deque, Dict
from collections import deque
from concurrent.futures import Future, CancelledError
import jumpy as jp

from rex.base import StepState, InputState, GraphState, Output, Params, State
from rex.node import Node


class Agent(Node):
    def __init__(self, *args, **kwargs):
        self._must_reset: bool
        self._f_act: Future
        self._f_obs: Future
        self._q_act: Deque[Future] = deque()
        self._q_obs: Deque[Future]
        super().__init__(*args, **kwargs)

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Params:
        """Default params of the node."""
        raise NotImplementedError

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
        """Default state of the node."""
        raise NotImplementedError

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
        """Default output of the node."""
        raise NotImplementedError

    def default_inputs(self, rng: jp.ndarray, graph_state: GraphState = None) -> Dict[str, InputState]:
        """Default inputs of the node."""
        return super().default_inputs(rng, graph_state)

    def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
        """Reset the agent."""
        raise NotImplementedError

    def get_step_state(self, graph_state: GraphState) -> StepState:
        """Get the step state of the agent."""
        return graph_state.nodes[self.name]

    @property
    def action(self) -> Deque[Future]:
        return self._q_act

    @property
    def observation(self) -> Deque[Future]:
        return self._q_obs

    def _agent_reset(self):
        self._must_reset = False
        self._q_act: Deque[Future] = deque()
        self._q_obs: Deque[Future] = deque()
        self._f_obs = Future()
        self._q_obs.append(self._f_obs)

    def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Output]:
        self._f_act = Future()
        self._q_act.append(self._f_act)

        # Prepare new obs future
        _new_f_obs = Future()
        self._q_obs.append(_new_f_obs)

        # Set observations as future result
        self._f_obs.set_result((ts, step_state))
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

