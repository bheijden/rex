from typing import Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct

from rex.jax_utils import tree_take
from rex.base import GraphState, StepState, Base
from rex.node import BaseNode


@struct.dataclass
class SupervisorState(Base):
    init_th: Union[float, jax.typing.ArrayLike]
    init_thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SupervisorParams(Base):
    init_states: SupervisorState
    max_th: Union[float, jax.typing.ArrayLike]
    min_th: Union[float, jax.typing.ArrayLike]
    max_thdot: Union[float, jax.typing.ArrayLike]
    min_thdot: Union[float, jax.typing.ArrayLike]
    max_torque: Union[float, jax.typing.ArrayLike]
    gamma: Union[float, jax.typing.ArrayLike]
    tmax: Union[float, jax.typing.ArrayLike]
    # init_method: str = struct.field(pytree_node=False)  # Can also be random


@struct.dataclass
class SupervisorOutput(Base):
    pass


class Supervisor(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_method = "downward"

    def set_init_method(self, init_method: str):
        self._init_method = init_method

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SupervisorParams:
        """Default params of the root."""
        init_states = SupervisorState(init_th=jnp.array([jnp.pi]), init_thdot=jnp.array([0.0]))  # Add episode dimension
        return SupervisorParams(init_states=init_states, max_th=jax.numpy.pi, min_th=-jax.numpy.pi, max_thdot=9.0, min_thdot=-9.0,
                                max_torque=2.0, gamma=0.99, tmax=3.0)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SupervisorState:
        """Default state of the root."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        if self._init_method == "downward":
            init_th = jnp.pi
            init_thdot = 0.0
            state = SupervisorState(init_th=init_th, init_thdot=init_thdot)
        elif self._init_method == "parametrized":
            eps = graph_state.eps % params.init_states.init_th.shape[0]
            state = tree_take(params.init_states, eps, axis=0)
        elif self._init_method == "random":
            init_th = jax.random.uniform(rng, shape=(), minval=params.min_th, maxval=params.max_th)
            init_thdot = jax.random.uniform(rng, shape=(), minval=params.min_thdot, maxval=params.max_thdot)
            state = SupervisorState(init_th=init_th, init_thdot=init_thdot)
        else:
            raise ValueError(f"Unknown init method: {self._init_method}")
        return state

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SupervisorOutput:
        """Default output of the root."""
        return SupervisorOutput()

    def step(self, step_state: StepState) -> Tuple[StepState, SupervisorOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        output: SupervisorOutput = self.init_output(step_state.rng)
        return new_step_state, output
