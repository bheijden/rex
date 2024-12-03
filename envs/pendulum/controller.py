from math import ceil
from typing import Union, Tuple, Dict, Any, List

import jax
from flax import struct
from flax.core import FrozenDict
import flax.linen as nn
import distrax
from jax import numpy as jnp
import numpy as onp

from rex import base
from rex.base import GraphState, StepState, Base, InputState
from rex.node import BaseNode
from rex.rl import NormalizeVec, SquashState
from rex.jax_utils import tree_dynamic_slice
from envs.pendulum.base import ActuatorOutput


@struct.dataclass
class PPOAgentParams(Base):
    num_act: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Action history length
    num_obs: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Observation history length
    act_scaling: SquashState = struct.field(default=None)
    obs_scaling: NormalizeVec = struct.field(default=None)
    model: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Any]]] = struct.field(default=None)
    hidden_activation: str = struct.field(pytree_node=False, default="tanh")
    output_activation: str = struct.field(pytree_node=False, default="gaussian")
    stochastic: bool = struct.field(pytree_node=False, default=False)
    incl_covariance: bool = struct.field(pytree_node=False, default=True)
    incl_thdot: bool = struct.field(pytree_node=False, default=True)

    def apply_actor(self, x: jax.typing.ArrayLike, rng: jax.Array = None) -> jax.Array:
        # Get parameters
        actor_params = self.model["actor"]
        num_layers = sum(["Dense" in k in k for k in actor_params.keys()])

        # Apply hidden layers
        ACTIVATIONS = dict(tanh=nn.tanh, relu=nn.relu, gelu=nn.gelu, softplus=nn.softplus)
        for i in range(num_layers-1):
            hl = actor_params[f"Dense_{i}"]
            num_output_units = hl["kernel"].shape[-1]
            if x is None:
                obs_dim = hl["kernel"].shape[-2]
                x = jnp.zeros((obs_dim,), dtype=float)
            x = nn.Dense(num_output_units).apply({"params": hl}, x)
            x = ACTIVATIONS[self.hidden_activation](x)

        # Apply output layer
        hl = actor_params[f"Dense_{num_layers-1}"]  # Index of final layer
        num_output_units = hl["kernel"].shape[-1]
        x_mean = nn.Dense(num_output_units).apply({"params": hl}, x)
        if self.output_activation == "gaussian":
            if rng is not None:
                log_std = actor_params["log_std"]
                pi = distrax.MultivariateNormalDiag(x_mean, jnp.exp(log_std))
                x = pi.sample(seed=rng)
            else:
                x = x_mean
        else:
            raise NotImplementedError("Gaussian output not implemented yet")
        return x

    def get_action(self, obs: jax.typing.ArrayLike) -> jax.Array:
        # Normalize observation
        norm_obs = self.obs_scaling.normalize(obs, clip=True, subtract_mean=True)
        # Get action
        action = self.apply_actor(norm_obs)
        # Scale action
        action = self.act_scaling.unsquash(action)
        return action

    def inputs_to_obs(self, inputs: FrozenDict[str, InputState]) -> jax.Array:
        # Get observation
        if "estimator" in inputs:
            state_estimate = inputs["estimator"][-1].data
            th, thdot = state_estimate.mean.th, state_estimate.mean.thdot
            obs = jnp.array([jnp.cos(th), jnp.sin(th), thdot]) if self.incl_thdot else jnp.array([jnp.cos(th), jnp.sin(th)])
            if self.incl_covariance:
                var = state_estimate.cov.diagonal()
                covar = state_estimate.cov[0, 1]
                obs_covar = jnp.array([var[0], var[1], covar])
                obs = jnp.concatenate([obs, obs_covar]) if self.incl_thdot else jnp.concatenate([obs, var[0]])
        elif "sensor" in inputs:
            assert self.incl_covariance is False, "Covariance not implemented for sensor input."
            th, thdot = inputs["sensor"][-1].data.th, inputs["sensor"][-1].data.thdot
            obs = jnp.array([jnp.cos(th), jnp.sin(th), thdot]) if self.incl_thdot else jnp.array([jnp.cos(th), jnp.sin(th)])
        else:
            assert self.incl_covariance is False, "Covariance not implemented for sensor input."
            assert "world" in inputs, "Either 'estimator', 'sensor', or 'world' must be in the inputs."
            th, thdot = inputs["world"][-1].data.th, inputs["world"][-1].data.thdot
            obs = jnp.array([jnp.cos(th), jnp.sin(th), thdot]) if self.incl_thdot else jnp.array([jnp.cos(th), jnp.sin(th)])
        return obs

    def get_observation(self, step_state: StepState) -> jax.Array:
        """Get observation for the policy.

        :return: Flattened observation of the policy
        """
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        obs = self.inputs_to_obs(inputs)

        # Convert to observation with history
        obs_history = jnp.concatenate([obs, state.history_obs.flatten(), state.history_act.flatten()])
        return obs_history

    def update_state(self, step_state: StepState, action: jax.Array) -> StepState:
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        obs = self.inputs_to_obs(inputs)

        # Update obs history
        if params.num_obs > 0:
            history_obs = jnp.roll(state.history_obs, shift=1, axis=0)
            history_obs = history_obs.at[0].set(obs)
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
    def to_output(step_state: StepState, action: jax.Array) -> ActuatorOutput:
        # Get estimator output
        inputs = step_state.inputs
        if "estimator" in inputs:
            state_estimate = inputs["estimator"][-1].data
        elif "sensor" in inputs:
            state_estimate = None
        else:
            assert "world" in inputs, "Either 'estimator', 'sensor', or 'world' must be in the inputs."
            state_estimate = None
        return ActuatorOutput(action=action, state_estimate=state_estimate)


@struct.dataclass
class PPOAgentState(base.Base):
    history_act: jax.typing.ArrayLike
    history_obs: jax.typing.ArrayLike


class PPOAgent(BaseNode):
    def __init__(self, *args, outputs: ActuatorOutput = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> PPOAgentParams:
        """Default params of the node."""
        return PPOAgentParams(num_act=0, num_obs=0, act_scaling=None, obs_scaling=None, model=None)

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentState:
        """Default state of the node."""
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        history_act = jnp.zeros((params.num_act, 1), dtype=jnp.float32)  # [torque]
        obs_dim = 3 if params.incl_thdot else 2
        history_obs = jnp.zeros((params.num_obs, obs_dim), dtype=jnp.float32)  # [cos(th), sin(th), thdot]
        return PPOAgentState(history_act=history_act, history_obs=history_obs)

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        action = jnp.array([0.0], dtype=jnp.float32)
        state_estimate = self.inputs["estimator"].output_node.init_output(rng, graph_state) if "estimator" in self.inputs else None
        output = ActuatorOutput(action=action, state_estimate=state_estimate)
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, ActuatorOutput]:
        """Step the node."""
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Grab estimator output
        assert len(self.inputs) > 0, "No estimator connected to controller"
        est_output = inputs["estimator"][-1].data if "estimator" in inputs else None

        # Get action from dataset or use passed through.
        if self._outputs is not None:
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            output = jax.tree_util.tree_map(lambda _o: _o[0, 0], output)
            output = output.replace(state_estimate=est_output)
            return step_state, output

        # Get observation for policy
        obs = params.get_observation(step_state)

        # Get action
        rng, rng_policy = jax.random.split(rng, num=2)
        if not (params.model is None or params.act_scaling is None or params.obs_scaling is None):
            # Apply actor
            action = params.get_action(obs)
        else:
            # action = jax.random.uniform(rng_policy, shape=(1,), dtype=jnp.float32, minval=-2.0, maxval=2.0)  # Random action
            action = jnp.array([0.3], dtype=jnp.float32)  # Fixed action

        # Update step_state (observation and action history)
        new_state = params.update_state(step_state, action)
        new_step_state = step_state.replace(rng=rng, state=new_state)

        # Convert to output
        output = params.to_output(new_step_state, action)
        return new_step_state, output


@struct.dataclass
class OpenLoopControllerParams(PPOAgentParams):
    """Pendulum root param definition"""
    max_torque_sysid: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=3)
    num_steps: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=6)
    dt_steps: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=2)
    num_rnd: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=10)
    dt_rnd: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=0.5)


@struct.dataclass
class OpenLoopControllerState(PPOAgentState):
    actions: jax.typing.ArrayLike


class OpenLoopController(PPOAgent):
    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> OpenLoopControllerParams:
        params = super().init_params(rng, graph_state)
        params = OpenLoopControllerParams(**params.__dict__)
        return params

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> OpenLoopControllerState:
        rng_policy, rng_step, rng_random = jax.random.split(rng, num=3)
        state = super().init_state(rng_policy, graph_state)

        dt = 1 / self.rate
        params = graph_state.params[self.name]

        step_repeats = ceil(params.dt_steps / dt)
        random_repeats = ceil(params.dt_rnd / dt)

        # Step actions
        step_actions = jax.random.uniform(rng_step, shape=(params.num_steps,), minval=-params.max_torque_sysid, maxval=params.max_torque_sysid)
        step_actions = step_actions.at[:3].set(onp.array([3.0, -2.5, 2.5]))

        # Random actions
        random_actions = jax.random.uniform(rng_random, shape=(params.num_rnd,), minval=-params.max_torque_sysid, maxval=params.max_torque_sysid)

        # Repeat actions in zero-order hold
        step_actions = jnp.repeat(step_actions, step_repeats, axis=0)
        random_actions = jnp.repeat(random_actions, random_repeats, axis=0)

        actions = jnp.concatenate([step_actions, random_actions], axis=0)
        return OpenLoopControllerState(**state.__dict__, actions=actions)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""
        # Get policy output
        new_step_state, output_policy = super().step(step_state)
        action_policy = output_policy.action

        # Whether to use openloop or not
        num_actions = step_state.state.actions.shape[0]
        action_openloop = step_state.state.actions[step_state.seq % num_actions][None]
        use_openloop = step_state.seq < num_actions
        action = jnp.where(use_openloop, action_openloop, action_policy)
        output = output_policy.replace(action=action)
        return new_step_state, output