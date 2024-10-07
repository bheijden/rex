from typing import Union, Tuple, Dict, Any, List

import jax
from flax import struct
from flax.core import FrozenDict
import flax.linen as nn
import distrax
from jax import numpy as jnp
import numpy as onp

from rex import base
from rex.graph import Graph
import rex.rl as rl
from rex.base import GraphState, StepState, Base, InputState
from rex.node import BaseNode
import rex.ppo as ppo
from rex.rl import NormalizeVec, SquashState
from rex.jax_utils import tree_dynamic_slice
from envs.crazyflie.ode import MoCapOutput
from envs.crazyflie.pid import PIDOutput


@struct.dataclass
class PPOAgentParams(base.Base):
    # PPO
    act_scaling: SquashState
    obs_scaling: NormalizeVec
    model: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Any]]]
    hidden_activation: str = struct.field(pytree_node=False)
    output_activation: str = struct.field(pytree_node=False)
    stochastic: bool = struct.field(pytree_node=False)
    # Observations
    num_act: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Action history length
    num_obs: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Observation history length
    # Ctrl limits
    mapping: List[str] = struct.field(pytree_node=False)
    action_dim: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)
    z_max: Union[float, jax.typing.ArrayLike]
    phi_max: Union[float, jax.typing.ArrayLike]
    theta_max: Union[float, jax.typing.ArrayLike]
    psi_max: Union[float, jax.typing.ArrayLike]
    # Crazyflie state
    init_cf: str = struct.field(pytree_node=False)
    fixed_position: jax.typing.ArrayLike
    x_range: jax.typing.ArrayLike
    y_range: jax.typing.ArrayLike
    z_range: jax.typing.ArrayLike
    clip_pos: jax.typing.ArrayLike
    clip_vel: jax.typing.ArrayLike
    # Circular path state
    init_path: str = struct.field(pytree_node=False)
    fixed_radius: Union[float, jax.typing.ArrayLike]  # Fixed radius
    radius_range: jax.typing.ArrayLike
    center: jax.typing.ArrayLike  # Center of the platform
    # Train
    gamma: Union[float, jax.typing.ArrayLike]
    tmax: Union[float, jax.typing.ArrayLike]
    # Domain randomization
    use_noise: Union[bool, jax.typing.ArrayLike]
    use_dr: Union[bool, jax.typing.ArrayLike]

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
        # Has landed
        has_landed = False
        output = PIDOutput(state_estimate=state_estimate, has_landed=has_landed, **actions_mapped)
        return output


@struct.dataclass
class PPOAgentState(base.Base):
    # State of the agent
    history_act: jax.typing.ArrayLike
    history_obs: jax.typing.ArrayLike
    prev_act: jax.typing.ArrayLike
    # Initial state of Crazyflie and task
    init_pos: jax.typing.ArrayLike
    init_vel: jax.typing.ArrayLike
    init_att: jax.typing.ArrayLike
    init_ang_vel: jax.typing.ArrayLike
    radius: jax.typing.ArrayLike
    center: jax.typing.ArrayLike


class PPOAgent(BaseNode):
    def __init__(self, *args, outputs: PIDOutput = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentParams:
        return PPOAgentParams(
            act_scaling=None,
            obs_scaling=None,
            model=None,
            hidden_activation="tanh",
            output_activation="gaussian",
            stochastic=False,
            # Observation
            num_act=0,  # Action history length
            num_obs=0,  # Observation history length
            # Action
            mapping=["theta_ref", "phi_ref", "z_ref", "psi_ref"],
            action_dim=2,
            z_max=2.0,
            phi_max=onp.pi / 5,
            theta_max=onp.pi / 5,
            psi_max=0.,  # No yaw (or onp.pi?)
            # init crazyflie
            init_cf="random",  # random, fixed
            fixed_position=jnp.array([0.0, 0.0, 1.75]),  # Above the platform
            x_range=jnp.array([-4.0, 4.0]),
            y_range=jnp.array([-4.0, 4.0]),
            z_range=jnp.array([0.0, 2.0]),
            clip_pos=jnp.array([2.1, 2.1, 2.1]),
            clip_vel=jnp.array([20., 20., 20.]),
            # Circular path
            init_path="random",  # random, fixed
            fixed_radius=1.5,  # Fixed radius
            radius_range=jnp.array([0.5, 2.0]),
            center=jnp.array([0.0, 0.0, 1.0]),  # Center of the platform
            # Train settings
            tmax=5.0,  # Maximum time for the episode
            gamma=0.99,  # Discount factor (add other reward terms)
            # Domain randomization
            use_noise=True,  # Whether to add noise to the measurements & perform domain randomization.
            use_dr=True,  # Whether to use domain randomization.
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentState:
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        history_act = jnp.zeros((params.num_act, params.action_dim), dtype=jnp.float32)  # [torque]
        history_obs = jnp.zeros((params.num_obs, 9), dtype=jnp.float32)  # [pos, vel, att]
        prev_act = jnp.zeros((params.action_dim,), dtype=jnp.float32)
        # Initial crazyflie position
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num=7)
        if params.init_cf == "random":
            init_x = jax.random.uniform(rngs[1], shape=(), minval=params.x_range[0], maxval=params.x_range[1])
            init_y = jax.random.uniform(rngs[2], shape=(), minval=params.y_range[0], maxval=params.y_range[1])
            init_z = jax.random.uniform(rngs[3], shape=(), minval=params.z_range[0], maxval=params.z_range[1])
            init_pos = jnp.array([init_x, init_y, init_z])
        elif params.init_cf == "fixed":
            init_pos = params.fixed_position
        else:
            raise ValueError(f"Unknown start position method: {params.init_cf}")
        # Circular path
        if params.init_path == "random":
            radius = jax.random.uniform(rngs[5], shape=(), minval=params.radius_range[0], maxval=params.radius_range[1])
        elif params.init_path == "fixed":
            radius = params.fixed_radius
        else:
            raise ValueError(f"Unknown inclination method: {params.init_path}")
        return PPOAgentState(
            # History
            history_act=history_act,
            history_obs=history_obs,
            prev_act=prev_act,
            # Initial crazyflie state
            init_pos=init_pos,
            init_vel=jnp.array([0.0, 0.0, 0.0]),
            init_att=jnp.array([0.0, 0.0, 0.0]),
            init_ang_vel=jnp.array([0.0, 0.0, 0.0]),
            # Initial path state
            radius=radius,  # Radius of circular path
            center=params.center,
        )

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
            has_landed=False,
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


class Environment(rl.Environment):

    def __init__(self, graph: Graph, params: Dict[str, base.Base] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        super().__init__(graph, params, only_init, starting_eps, randomize_eps, order)

    @property  # todo: Currently just takes the max_steps from the graph.
    def max_steps(self) -> Union[int, jax.typing.ArrayLike]:
        return self.graph.max_steps

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return rl.Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        params_agent: PPOAgentParams = graph_state.params.get("agent")
        high_mapping = dict(
            phi_ref=params_agent.phi_max,
            theta_ref=params_agent.theta_max,
            psi_ref=params_agent.psi_max,
            z_ref=params_agent.z_max
        )
        high = jnp.array([high_mapping[k] for a, k in zip(range(params_agent.action_dim), params_agent.mapping)], dtype=float)
        return rl.Box(-high, high, shape=high.shape, dtype=float)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        obs = params.get_observation(ss)
        return obs

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ts = self.get_step_state(graph_state).ts
        params: PPOAgentParams = graph_state.params["agent"]
        return ts >= params.tmax

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        terminated = False
        return terminated  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        # Get current state
        ss = self.get_step_state(graph_state)
        # pos = last_mocap.pos
        # x, y, z = pos
        # vx, vy, vz = last_mocap.vel

        # Get transformed
        center = graph_state.state["agent"].center
        radius = graph_state.state["agent"].radius
        last_mocap: MoCapOutput = ss.inputs["estimator"][-1].data
        last_mocap_ia = last_mocap.in_agent_frame(center)
        radial, theta, z = last_mocap_ia.pos
        vrad, vel_on, vz = last_mocap_ia.vel
        pos_off = jnp.linalg.norm(jnp.array([radial - radius, z]))
        vel_off = jnp.linalg.norm(jnp.array([vrad, vz]))

        # Get denormalized action
        output = self.get_output(graph_state, action)
        prev_output = self.get_output(graph_state, ss.state.prev_act)
        dz_ref = output.z_ref - prev_output.z_ref
        dphi_ref = output.phi_ref - prev_output.phi_ref
        dtheta_ref = output.theta_ref - prev_output.theta_ref
        dpsi_ref = output.psi_ref - prev_output.psi_ref

        # Reward for following the path (higher when closer to the path)
        rwd_vel_on = 0.1*vel_on / jnp.clip(pos_off, 0.001, None)

        # Control cost
        cost_dangle = dphi_ref**2 + dtheta_ref**2 + dpsi_ref**2
        cost_decouple = jnp.abs(dphi_ref) * jnp.abs(dtheta_ref)
        cost_dz_ref = jnp.sqrt(dz_ref**2)
        cost_z_ref = jnp.abs(output.z_ref - center[2])

        # Calculate rewards
        rwd_eps = 0.4*rwd_vel_on - 0.4*vel_off - pos_off - 10.0*cost_dangle - 0.*cost_decouple - 0.2*cost_dz_ref - 5*cost_z_ref
        rwd_term = 0.0
        rwd_final = rwd_eps

        # Get termination conditions
        terminated = self.get_terminated(graph_state)
        truncated = self.get_truncated(graph_state)

        # Calculate final reward
        gamma = graph_state.params["agent"].gamma
        done = jnp.logical_or(terminated, truncated)
        rwd = rwd_eps + done * ((1 - terminated) * (1 / (1 - gamma)) + terminated) * rwd_final + done * terminated * rwd_term
        return rwd

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        # Get current state
        ss = self.get_step_state(graph_state)
        center = graph_state.state["agent"].center
        radius = graph_state.state["agent"].radius
        last_mocap: MoCapOutput = ss.inputs["estimator"][-1].data
        last_mocap_ia = last_mocap.in_agent_frame(center)
        radial, theta, z = last_mocap_ia.pos
        vrad, vel_on, vz = last_mocap_ia.vel

        # Cost for not following or being on the path (vel, pos)
        pos_off = jnp.linalg.norm(jnp.array([radial - radius, z]))
        vel_off = jnp.linalg.norm(jnp.array([vrad, vz]))

        # Reward for following the path (higher when closer to the path)
        rwd_vel_on = 0.01*jnp.clip(vel_on / jnp.clip(pos_off, 0.001, None), 0, None)

        return {
            "vel_on": vel_on,
            "pos_off": pos_off,
            "vel_off": vel_off,
            "rwd_vel_on": rwd_vel_on,
            "z": z,
        }

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> PIDOutput:
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        output = params.to_output(ss, action)
        return output

    def update_graph_state_pre_step(self, graph_state: base.GraphState, action: jax.Array) -> base.GraphState:
        # Update agent state
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        new_state = params.update_state(ss, action)
        # Update graph state
        graph_state = graph_state.replace(state=graph_state.state.copy({
            self.graph.supervisor.name: new_state
        }))
        return graph_state

    def update_graph_state_post_step(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        """Override this method if you want to update the graph state after graph.step(...).

        Note: This method is called after the graph has been stepped (before returning from .step()),
              or before returning the initial observation from .reset().
        :param graph_state: The current graph state.
        :param action: The action taken. Is None when called from .reset().
        """
        # Clip state
        center = graph_state.state["agent"].center
        radius = graph_state.state["agent"].radius
        radius_pos = jnp.array([radius, radius, 0.])
        clip_pos = graph_state.params["agent"].clip_pos
        clip_vel = graph_state.params["agent"].clip_vel

        # Clip position
        pos = graph_state.state["world"].pos
        high = center + clip_pos + radius_pos
        low = center - clip_pos - radius_pos
        pos_clip = jnp.clip(pos, low, high)

        # Clip velocity
        vel = graph_state.state["world"].vel
        vel_clip = jnp.clip(vel, -clip_vel, clip_vel)
        new_world = graph_state.state["world"].replace(pos=pos_clip, vel=vel_clip)

        # Update action
        ss = self.get_step_state(graph_state)
        action = action if action is not None else ss.state.prev_act

        # Update graph_state
        graph_state = graph_state.replace(state=graph_state.state.copy({
            self.graph.supervisor.name: ss.state.replace(prev_act=action),
            "world": new_world
        }))
        return graph_state


@struct.dataclass
class PathFollowingConfig(ppo.Config):
    def EVAL_METRICS_JAX_CB(self, total_steps, diagnostics: ppo.Diagnostics, eval_transitions: ppo.Transition = None) -> Dict:
        metrics = super().EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_transitions)
        total_done = eval_transitions.done.sum()
        done = eval_transitions.done
        info = eval_transitions.info
        metrics["eval/vel_on"] = (jnp.roll(info["vel_on"], shift=1, axis=-2) * done).sum() / total_done
        metrics["eval/pos_off"] = (jnp.roll(info["pos_off"], shift=1, axis=-2) * done).sum() / total_done
        metrics["eval/z"] = (jnp.roll(info["z"], shift=1, axis=-2) * done).sum() / total_done
        metrics["eval/vel_off"] = (jnp.roll(info["vel_off"], shift=1, axis=-2) * done).sum() / total_done
        metrics["eval/rwd_vel_on"] = (jnp.roll(info["rwd_vel_on"], shift=1, axis=-2) * done).sum() / total_done
        return metrics

    def EVAL_METRICS_HOST_CB(self, metrics: Dict):
        # Standard metrics
        global_step = metrics["train/total_steps"]
        mean_approxkl = metrics["train/mean_approxkl"]
        mean_return = metrics["eval/mean_returns"]
        std_return = metrics["eval/std_returns"]
        mean_length = metrics["eval/mean_lengths"]
        std_length = metrics["eval/std_lengths"]
        total_episodes = metrics["eval/total_episodes"]

        # Extra metrics
        # if "eval/vel_on" in metrics:
        vel_on = metrics["eval/vel_on"]
        pos_off = metrics["eval/pos_off"]
        z = metrics["eval/z"]
        vel_off = metrics["eval/vel_off"]
        rwd_vel_on = metrics["eval/rwd_vel_on"]

        if self.VERBOSE:
            print(f"train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                  f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f} | "
                  f"vel_on={vel_on:.2f} | pos_off={pos_off:.2f} | z={z:.2f} | "
                  f"vel_off={vel_off:.2f} | rwd_vel_on={rwd_vel_on:.2f}"
                  )


# COPIED from Multi-inclination (noise, mass variation, vary initial x, y, z, azimuth)
ppo_config = PathFollowingConfig(
    LR=9.23e-4,
    NUM_ENVS=128,
    NUM_STEPS=64,
    TOTAL_TIMESTEPS=10e6,
    UPDATE_EPOCHS=16,
    NUM_MINIBATCHES=8,
    GAMMA=0.9844,
    GAE_LAMBDA=0.939,
    CLIP_EPS=0.131,
    ENT_COEF=0.01,
    VF_COEF=0.756,
    MAX_GRAD_NORM=0.76,
    NUM_HIDDEN_LAYERS=2,
    NUM_HIDDEN_UNITS=64,
    KERNEL_INIT_TYPE="xavier_uniform",
    HIDDEN_ACTIVATION="tanh",
    STATE_INDEPENDENT_STD=True,
    SQUASH=True,
    ANNEAL_LR=False,
    NORMALIZE_ENV=True,
    DEBUG=False,
    VERBOSE=True,
    FIXED_INIT=False,
    OFFSET_STEP=True,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
)