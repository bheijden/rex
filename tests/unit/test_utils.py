from typing import Tuple, Union

import jax
from flax import struct as struct
from jax import numpy as jnp

import rex.rl as rl
from rex.base import Base, GraphState, StepState
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
        return Params(jnp.array([[1.0]]))

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> State:
        return State(jnp.array([1.0]))

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> Output:
        return Output(jnp.array([1.0]))

    def step(self, step_state: StepState) -> Tuple[StepState, Output]:
        return step_state, Output(jnp.array([1.0]))


class Env(rl.Environment):
    def observation_space(self, graph_state: GraphState) -> rl.Box:
        return rl.Box(jnp.array([0.0]), jnp.array([1.0]))

    def action_space(self, graph_state: GraphState) -> rl.Box:
        return rl.Box(jnp.array([0.0]), jnp.array([1.0]))

    def get_observation(self, graph_state: GraphState) -> jax.Array:
        return jnp.array([0.5])

    def get_output(self, graph_state: GraphState, action: jax.Array) -> Output:
        return Output(a=action)

    def get_truncated(self, graph_state: GraphState) -> Union[bool, jax.Array]:
        return False

    def get_terminated(self, graph_state: GraphState) -> Union[bool, jax.Array]:
        return graph_state.seq[self.graph.supervisor.name] >= 10

    def get_reward(self, graph_state: GraphState, action: jax.Array) -> Union[float, jax.Array]:
        return action[0]
