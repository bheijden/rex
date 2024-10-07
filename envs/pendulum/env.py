from typing import Dict, Union, Any, Tuple
import jax
import jax.numpy as jnp
import numpy as onp

from rex.graph import Graph
import rex.base as base
import rex.rl as rl

import envs.pendulum.controller as ctrl
from envs.pendulum.base import ActuatorOutput


class Environment(rl.Environment):
    def __init__(self, graph: Graph, params: Dict[str, base.Params] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):

        super().__init__(graph, params, only_init, starting_eps, randomize_eps, order)

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return rl.Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        params = graph_state.params["supervisor"]
        high = jnp.array([params.max_torque], dtype=jnp.float32)
        low = -high
        return rl.Box(low, high, shape=high.shape, dtype=high.dtype)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = self.get_step_state(graph_state)
        obs = ss.params.get_observation(ss)
        return obs

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ss = self.get_step_state(graph_state)
        truncated = ss.ts >= graph_state.params["supervisor"].tmax
        return truncated

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        terminated = False
        return terminated  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        # Get the reward from the world
        reward_step = -graph_state.state["world"].loss_task

        # Determine if the episode is truncated
        truncated = self.get_truncated(graph_state)
        gamma = graph_state.params["supervisor"].gamma
        reward_final = truncated * (1/(1-gamma)) * reward_step  # Assumes that the reward is constant after truncation

        # Add the final reward
        reward = reward_step + reward_final
        return reward

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        return {}

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> ActuatorOutput:
        ss = self.get_step_state(graph_state)
        output = ss.params.to_output(ss, action)
        return output

    def update_graph_state_pre_step(self, graph_state: base.GraphState, action: jax.Array) -> base.GraphState:
        # Update agent state
        ss = self.get_step_state(graph_state)
        new_state = ss.params.update_state(ss, action)
        # Reset task reward
        new_state_world = graph_state.state["world"].replace(loss_task=0.0)
        # Update graph state
        graph_state = graph_state.replace(state=graph_state.state.copy({"world": new_state_world,
                                                                        self.graph.supervisor.name: new_state}))
        return graph_state

