import dill as pickle
from typing import Any, Tuple, Dict, Union, Optional
import abc
import jax

from rex.spaces import Space
from rex.utils import log, NODE_COLOR, NODE_LOG_LEVEL
from rex.graph import BaseGraph
from rex.base import GraphState, Params, RexResetReturn, RexStepReturn
from rex.constants import WARN, INFO


class BaseEnv:
    def __init__(self, graph: BaseGraph, max_steps: int, name: str = "env", render_mode: str = None):
        self.name = name
        self.graph = graph
        self.max_steps = max_steps
        self.render_mode = render_mode
        assert self.max_steps > 0, "max_steps must be a positive integer"

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @abc.abstractmethod
    def reset(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> RexResetReturn:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, graph_state: GraphState, action: Any) -> RexStepReturn:
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

    def save(self, path: str):
        """Save the environment to a file using pickle."""
        # Append pkl extension
        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)
            self.log("Environment", f"Saved environment `{self.name}` to {path}.", INFO)

    @staticmethod
    def load(path: str):
        """Load the environment from a file using pickle."""
        # Append pkl extension
        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, "rb") as f:
            env = pickle.load(f)
            env.log("Environment", f"Loaded environment `{env.name}` from {path}.", INFO)
        return env
