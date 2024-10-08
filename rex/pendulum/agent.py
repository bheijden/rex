from typing import Union, Tuple

import jax
from flax import struct
from flax.core import FrozenDict
from jax import numpy as jnp
from rex import base
from rex.base import StepState, GraphState
from rex.node import BaseNode
from rex.ppo import Policy


@struct.dataclass
class AgentOutput(base.Base):
    """Agent's output"""
    action: jax.typing.ArrayLike  # Torque to apply to the pendulum


@struct.dataclass
class AgentParams(base.Base):
    # Policy
    policy: Policy
    # Observations
    num_act: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Action history length
    num_obs: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Observation history length
    # Action
    max_torque: Union[float, jax.typing.ArrayLike]
    # Initial state
    init_method: str = struct.field(pytree_node=False)  # "random", "parametrized"
    parametrized: jax.typing.ArrayLike
    max_th: Union[float, jax.typing.ArrayLike]
    max_thdot: Union[float, jax.typing.ArrayLike]
    # Train
    gamma: Union[float, jax.typing.ArrayLike]
    tmax: Union[float, jax.typing.ArrayLike]

    @staticmethod
    def process_inputs(inputs: FrozenDict[str, base.InputState]) -> jax.Array:
        th, thdot = inputs["sensor"][-1].data.th, inputs["sensor"][-1].data.thdot
        obs = jnp.array([jnp.cos(th), jnp.sin(th), thdot])
        return obs

    @staticmethod
    def get_observation(step_state: StepState) -> jax.Array:
        # Unpack StepState
        inputs, state = step_state.inputs, step_state.state

        # Convert inputs to single observation
        single_obs = AgentParams.process_inputs(inputs)

        # Concatenate with previous observations
        obs = jnp.concatenate([single_obs, state.history_obs.flatten(), state.history_act.flatten()])
        return obs

    @staticmethod
    def update_state(step_state: StepState, action: jax.Array) -> "AgentState":
        # Unpack StepState
        state, params, inputs = step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        single_obs = AgentParams.process_inputs(inputs)

        # Update obs history
        if params.num_obs > 0:
            history_obs = jnp.roll(state.history_obs, shift=1, axis=0)
            history_obs = history_obs.at[0].set(single_obs)
        else:
            history_obs = state.history_obs

        # Update act history
        if params.num_act > 0:
            history_act = jnp.roll(state.history_act, shift=1, axis=0)
            history_act = history_act.at[0].set(action)
        else:
            history_act = state.history_act

        new_state = state.replace(history_obs=history_obs, history_act=history_act)
        return new_state

    @staticmethod
    def to_output(action: jax.Array) -> AgentOutput:
        return AgentOutput(action=action)


@struct.dataclass
class AgentState(base.Base):
    history_act: jax.typing.ArrayLike  # History of actions
    history_obs: jax.typing.ArrayLike  # History of observations
    init_th: Union[float, jax.typing.ArrayLike] # Initial angle of the pendulum
    init_thdot: Union[float, jax.typing.ArrayLike]  # Initial angular velocity of the pendulum


class Agent(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> AgentParams:
        return AgentParams(
            policy=None,  # Policy must be set by the user
            num_act=4,  # Number of actions to keep in history
            num_obs=4,  # Number of observations to keep in history
            max_torque=2.0,  # Maximum torque that can be applied to the pendulum
            init_method="parametrized",  # "random" or "parametrized"
            parametrized=jnp.array([jnp.pi, 0.0]),  # [th, thdot]
            max_th=jnp.pi,  # Maximum initial angle of the pendulum
            max_thdot=9.0,  # Maximum initial angular velocity of the pendulum
            gamma=0.99,  # Discount factor  (used during training)
            tmax=3.0,  # Maximum time for an episode (used during training)
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> AgentState:
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        history_act = jnp.zeros((params.num_act, 1), dtype=jnp.float32)  # [torque]
        history_obs = jnp.zeros((params.num_obs, 3), dtype=jnp.float32)  # [cos(th), sin(th), thdot]

        # Set the initial state of the pendulum
        if params.init_method == "parametrized":
            init_th, init_thdot = params.parametrized
        elif params.init_method == "random":
            rng = rng if rng is not None else jax.random.PRNGKey(0)
            rngs = jax.random.split(rng, num=2)
            init_th = jax.random.uniform(rngs[0], shape=(), minval=-params.max_th, maxval=params.max_th)
            init_thdot = jax.random.uniform(rngs[1], shape=(), minval=-params.max_thdot, maxval=params.max_thdot)
        else:
            raise ValueError(f"Invalid init_method: {params.init_method}")
        return AgentState(history_act=history_act, history_obs=history_obs, init_th=init_th, init_thdot=init_thdot)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> AgentOutput:
        """Default output of the node."""
        rng = jax.random.PRNGKey(0) if rng is None else rng
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        action = jax.random.uniform(rng, shape=(1,), minval=-params.max_torque, maxval=params.max_torque)
        return AgentOutput(action=action)

    def step(self, step_state: StepState) -> Tuple[StepState, AgentOutput]:
        """Step the node."""
        # Unpack StepState
        rng, params = step_state.rng, step_state.params

        # Prepare output
        rng, rng_net = jax.random.split(rng)
        if params.policy is not None:  # Use policy to get action
            obs = AgentParams.get_observation(step_state)
            action = params.policy.get_action(obs, rng=None)  # Supply rng for stochastic policies
        else:  # Random action if no policy is set
            action = jax.random.uniform(rng_net, shape=(1,), minval=-params.max_torque, maxval=params.max_torque)
        output = AgentParams.to_output(action)  # Convert action to output message

        # Update step_state (observation and action history)
        new_state = params.update_state(step_state, action)  # Update state
        new_step_state = step_state.replace(rng=rng, state=new_state)  # Update step_state
        return new_step_state, output
