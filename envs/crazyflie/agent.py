from math import ceil
from typing import Union, Tuple, Dict, Any, List

import jax
from flax import struct
from flax.core import FrozenDict
import flax.linen as nn
import distrax
from jax import numpy as jnp
import numpy as onp

from rexv2 import base
from rexv2.base import GraphState, StepState, Base, InputState
from rexv2.node import BaseNode
from rexv2.rl import NormalizeVec, SquashState
from rexv2.jax_utils import tree_dynamic_slice
from envs.crazyflie.ode import rpy_to_R, R_to_rpy, MoCapOutput
from envs.crazyflie.pid import PIDOutput


@struct.dataclass
class PPOAgentParams(base.Base):
    num_act: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Action history length
    num_obs: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Observation history length
    act_scaling: SquashState = struct.field(default=None)
    obs_scaling: NormalizeVec = struct.field(default=None)
    model: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Any]]] = struct.field(default=None)
    hidden_activation: str = struct.field(pytree_node=False, default="tanh")
    output_activation: str = struct.field(pytree_node=False, default="gaussian")
    stochastic: bool = struct.field(pytree_node=False, default=False)
    action_dim: int = struct.field(pytree_node=False, default=3)
    mapping: List[str] = struct.field(pytree_node=False, default=None)

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
        norm_obs = self.obs_scaling.normalize(obs, clip=True, subtract_mean=True) if self.obs_scaling is not None else obs
        # Get action
        action = self.apply_actor(norm_obs) if self.model is not None else jnp.zeros((self.action_dim,), dtype=jnp.float32)
        # Scale action
        action = self.act_scaling.unsquash(action) if self.act_scaling is not None else action
        return action

    def inputs_to_obs(self, inputs: FrozenDict[str, InputState], state: "PPOAgentState") -> jax.Array:
        # Get observation
        mocap: MoCapOutput = inputs["estimator"][-1].data
        mocap_ia = mocap.in_agent_frame(state.center)
        obs = jnp.concatenate([mocap_ia.pos, mocap_ia.vel, mocap_ia.att])
        return obs

    def get_observation(self, step_state: StepState) -> jax.Array:
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        obs = self.inputs_to_obs(step_state.inputs, state)

        # Convert to observation with history
        obs_history = jnp.concatenate([obs, state.history_obs.flatten(), state.history_act.flatten(), jnp.array([state.radius])])
        return obs_history

    def update_state(self, step_state: StepState, action: jax.Array) -> StepState:
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        obs = self.inputs_to_obs(inputs, state)

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

    def to_output(self, step_state: StepState, action: jax.Array) -> PIDOutput:
        # Map outputs
        assert self.mapping is not None, "Mapping not provided"
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = None
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0)
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0)
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0)
        # Determine z_ref (if not provided)
        z_ref = step_state.state.center[2]
        z = step_state.inputs["estimator"][-1].data.pos[2]
        z_ref_clip = jnp.clip(z_ref, z-0.75, z+0.75)
        actions_mapped["z_ref"] = actions_mapped.get("z_ref", z_ref_clip)
        # Get estimator output
        inputs = step_state.inputs
        if "estimator" in inputs:
            state_estimate = inputs["estimator"][-1].data
        elif "sensor" in inputs:
            state_estimate = None
        else:
            assert "world" in inputs, "Either 'estimator', 'sensor', or 'world' must be in the inputs."
            state_estimate = None
        output = PIDOutput(state_estimate=state_estimate, **actions_mapped)
        return output


@struct.dataclass
class PPOAgentState(base.Base):
    history_act: jax.typing.ArrayLike
    history_obs: jax.typing.ArrayLike
    prev_act: jax.typing.ArrayLike
    radius: jax.typing.ArrayLike
    center: jax.typing.ArrayLike


@struct.dataclass
class AgentOutput(base.Base):
    action: jax.typing.ArrayLike  # between -1 and 1 --> [pwm/thrust/zref, phi_ref, theta_ref, psi_ref]


class PPOAgent(BaseNode):
    def __init__(self, *args, outputs: PIDOutput = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentParams:
        graph_state = graph_state or GraphState()
        params_sup = graph_state.params.get("supervisor")
        mapping = params_sup.ctrl_mapping
        action_dim = params_sup.action_dim
        return PPOAgentParams(
            num_act=0,
            num_obs=0,
            act_scaling=None,
            obs_scaling=None,
            model=None,
            mapping=mapping,
            action_dim=action_dim,
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentState:
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        history_act = jnp.zeros((params.num_act, params.action_dim), dtype=jnp.float32)  # [torque]
        history_obs = jnp.zeros((params.num_obs, 9), dtype=jnp.float32)  # [pos, vel, att]
        prev_act = jnp.zeros((params.action_dim,), dtype=jnp.float32)
        # Get radius & radius of path from supervisor
        state_sup = graph_state.state.get("supervisor")
        radius = state_sup.radius
        params_sup = graph_state.params.get("supervisor")
        center = params_sup.center
        return PPOAgentState(history_act=history_act, history_obs=history_obs, prev_act=prev_act, radius=radius, center=center)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> PIDOutput:
        """Default output of the node."""
        state_estimate = self.inputs["estimator"].output_node.init_output(rng, graph_state) if "estimator" in self.inputs else None
        output = PIDOutput(
            pwm_ref=None,
            phi_ref=0.0,
            theta_ref=0.0,
            psi_ref=0.0,
            z_ref=0.0,
            state_estimate=state_estimate,
        )
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, PIDOutput]:
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        # Grab estimator output
        assert len(self.inputs) > 0, "No estimator connected to controller"
        est_output = inputs["estimator"][-1].data if "estimator" in inputs else None
        # Get action from dataset or use passed through.
        if self._outputs is not None:
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            output = output.replace(state_estimate=est_output)
            return step_state, output  # Return output from dataset
        # Get observation for policy
        obs = params.get_observation(step_state)
        # Get action
        rng, rng_policy = jax.random.split(rng, num=2)
        if not (params.model is None or params.act_scaling is None or params.obs_scaling is None):
            # Apply actor
            action = params.get_action(obs)
        else:
            action = jax.random.uniform(rng_policy, shape=(params.action_dim,), dtype=jnp.float32, minval=-1.0, maxval=1.0)  # Random action
        # Update step_state (observation and action history)
        new_state = params.update_state(step_state, action)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        # Convert to output
        output = params.to_output(new_step_state, action)
        return new_step_state, output