import os
from typing import Union, Tuple, Dict, Any, List

import jax
from flax import struct
from flax.core import FrozenDict
import flax.linen as nn
import distrax
from jax import numpy as jnp
import numpy as onp

from rexv2 import base
from rexv2.graph import Graph
import rexv2.rl as rl
from rexv2.base import GraphState, StepState, Base, InputState
from rexv2.node import BaseNode
import rexv2.ppo as ppo
from rexv2.rl import NormalizeVec, SquashState
from rexv2.jax_utils import tree_dynamic_slice
from envs.crazyflie.ode import WorldState, PlatformState, MoCapOutput, PlatformOutput, rpy_to_spherical, spherical_to_rpy, rpy_to_R, R_to_rpy, rpy_to_wxyz
from envs.crazyflie.pid import PIDOutput


try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.io import mjcf

    BRAX_INSTALLED = True
except ModuleNotFoundError:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False

try:
    import mujoco
    import mujoco.viewer
    from mujoco import mjx
    MUJOCO_INSTALLED = True
except ModuleNotFoundError:
    print("Mujoco or mujoco-mjx not installed. Install it with `pip install mujoco` or `pip install mujoco-mjx`")
    MUJOCO_INSTALLED = False


def get_qpos(pos, att, pos_plat, att_plat):
    cf_quat = rpy_to_wxyz(att)
    cf_qpos = jnp.concatenate([pos, cf_quat, jnp.array([0])])

    plat_quat = rpy_to_wxyz(att_plat)
    pitch_plat = jnp.array([0])
    platform_qpos = jnp.concatenate([pos_plat, plat_quat, pitch_plat])

    # Set corrected platform state
    polar_cor, azimuth_cor = rpy_to_spherical(att_plat)
    att_cor = jnp.array([0., 0, azimuth_cor])
    quat_cor = rpy_to_wxyz(att_cor)
    pitch_cor = jnp.array([polar_cor])
    platform_cor_qpos = jnp.concatenate([pos_plat, quat_cor, pitch_cor])
    return platform_qpos, platform_cor_qpos, cf_qpos


def contact_distance(mjx_model, mjx_data, pos, att, pos_plat, att_plat, threshold):
    R_is2w = rpy_to_R(att_plat)
    pos_plat_threshold = pos_plat + R_is2w @ jnp.array([0, 0, threshold])
    platform_qpos, platform_cor_qpos, cf_qpos = get_qpos(pos, att, pos_plat_threshold, att_plat)
    if mjx_data.qpos.shape[0] == 24:
        qpos = jnp.concatenate([platform_qpos, platform_cor_qpos, cf_qpos])
    elif mjx_data.qpos.shape[0] == 16:
        qpos = jnp.concatenate([platform_cor_qpos, cf_qpos])
    else:
        raise ValueError(f"Unsupported qpos shape: {mjx_data.qpos.shape}")
    mjx_d = mjx_data.replace(qpos=qpos)
    mjx_d = mjx.forward(mjx_model, mjx_d)
    dmin = mjx_d.contact.dist.min()
    return dmin


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
    pos_offset: jax.typing.ArrayLike
    # Init platform
    init_plat: str = struct.field(pytree_node=False)
    fixed_inclination: Union[float, jax.typing.ArrayLike]  # Fixed inclination (25.7 degrees)
    azimuth_max: Union[float, jax.typing.ArrayLike]
    polar_range: jax.typing.ArrayLike
    vel_plat_max: Union[float, jax.typing.ArrayLike]  # Max constant velocity of platform
    vel_land: Union[float, jax.typing.ArrayLike]
    contact_threshold: Union[float, jax.typing.ArrayLike] # Contact threshold
    # Train
    gamma: Union[float, jax.typing.ArrayLike]
    tmax: Union[float, jax.typing.ArrayLike]
    # Domain randomization
    use_noise: Union[bool, jax.typing.ArrayLike]
    use_dr: Union[bool, jax.typing.ArrayLike]
    # Mujoco
    mjx_model: mjx.Model = struct.field(pytree_node=False)
    mjx_data: mjx.Data = struct.field(pytree_node=False)
    # Reward params
    k1: Union[float, jax.typing.ArrayLike]
    k2: Union[float, jax.typing.ArrayLike]
    k3: Union[float, jax.typing.ArrayLike]
    k4: Union[float, jax.typing.ArrayLike]
    f1: Union[float, jax.typing.ArrayLike]
    f2: Union[float, jax.typing.ArrayLike]
    fp: Union[float, jax.typing.ArrayLike]
    pos_perfect: Union[float, jax.typing.ArrayLike]
    att_perfect: Union[float, jax.typing.ArrayLike]
    vel_perfect: Union[float, jax.typing.ArrayLike]

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
        plat: PlatformOutput = inputs["platform"][-1].data
        mocap: MoCapOutput = inputs["estimator"][-1].data

        # offset cf_pos based on pos_offset in local frame
        R_cf2w = rpy_to_R(mocap.att)
        pos_cfiw = mocap.pos + R_cf2w @ self.pos_offset
        att_cf2w = mocap.att
        vel_cfiw = mocap.vel
        pos_isiw = plat.pos
        att_is2w = plat.att
        vel_isiw = plat.vel

        # a=agent frame
        # w=world frame
        # is=inclined surface frame
        # cf=crazyflie frame

        # Make rotation matrix
        Rz = jnp.array([[jnp.cos(att_cf2w[2]), -jnp.sin(att_cf2w[2]), 0],
                        [jnp.sin(att_cf2w[2]), jnp.cos(att_cf2w[2]), 0],
                        [0, 0, 1]])
        R_a2w = Rz

        # Make is=inclined surface rotation matrix
        R_is2w = rpy_to_R(att_is2w)

        # Agent frame
        H_w2a = jnp.eye(4)
        H_w2a = H_w2a.at[:3, :3].set(R_a2w.T)
        H_w2a = H_w2a.at[:3, 3].set(-R_a2w.T @ pos_cfiw)

        # Transform cf to agent frame
        R_cf2w = rpy_to_R(att_cf2w)
        att_cf2a = R_to_rpy(R_a2w.T @ R_cf2w)
        roll_cf2a, pitch_cf2a, yaw_cf2a = att_cf2a
        rp_cf2a = jnp.array([roll_cf2a, pitch_cf2a])

        # Transform is position to agent frame
        # pos_isia = H_w2a @ jnp.concatenate([pos_isiw, jnp.array([1.0])])
        pos_isia = jnp.dot(R_a2w.T, pos_isiw) - jnp.dot(R_a2w.T, pos_cfiw)
        pos_isia = pos_isia[:3]
        att_is2a = R_to_rpy(R_a2w.T @ R_is2w)
        polar_is2a, is_azimuth_a = rpy_to_spherical(att_is2a)
        inclination = polar_is2a

        # Transform cf velocity to is frame
        vel_cfia = R_a2w.T @ vel_cfiw
        vel_isia = R_a2w.T @ vel_isiw

        obs = jnp.concatenate([rp_cf2a, vel_cfia, pos_isia, jnp.array([inclination, is_azimuth_a]), vel_isia])
        return obs

    def get_observation(self, step_state: StepState) -> jax.Array:
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Convert inputs to observation
        obs = self.inputs_to_obs(step_state.inputs, state)

        # Convert to observation with history
        obs_history = jnp.concatenate([obs, state.history_obs.flatten(), state.history_act.flatten()])
        return obs_history

    def update_state_history(self, step_state: StepState, action: jax.Array) -> StepState:
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

    def update_state_has_landed(self, state: "PPOAgentState", mocap, platform) -> "PPOAgentState":
        # Unpack StepState
        # rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Determine if landed
        # plat: PlatformOutput = inputs["platform"][-1].data
        # mocap: MoCapOutput = inputs["estimator"][-1].data
        dmin = self.contact_distance(mocap.pos, mocap.att, platform.pos, platform.att)
        has_landed = jnp.logical_or(state.has_landed, dmin < 0.0)  # Landed if contact distance is negative or was already landed

        new_state = state.replace(has_landed=has_landed)
        return new_state

    def update_state_prev_action(self, step_state: StepState, action: jax.Array) -> StepState:
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        new_state = state.replace(prev_act=action)
        return new_state

    def to_output(self, step_state: StepState, action: jax.Array) -> PIDOutput:
        # Get platform height
        z_plat = step_state.inputs["platform"][-1].data.pos[2]
        # Map outputs
        assert self.mapping is not None, "Mapping not provided"
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = None
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0)
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0)
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0)
        actions_mapped["z_ref"] = actions_mapped.get("z_ref", 0.0) + z_plat
        # Get estimator output
        inputs = step_state.inputs
        if "estimator" in inputs:
            state_estimate = inputs["estimator"][-1].data
        elif "sensor" in inputs:
            state_estimate = None
        else:
            assert "world" in inputs, "Either 'estimator', 'sensor', or 'world' must be in the inputs."
            state_estimate = None
        # Determine if landed
        has_landed = step_state.state.has_landed
        # Make output
        output = PIDOutput(state_estimate=state_estimate, has_landed=has_landed, **actions_mapped)
        return output

    def contact_distance(self, pos, att, pos_plat, att_plat) -> Union[float, jax.typing.ArrayLike]:
        return contact_distance(self.mjx_model, self.mjx_data, pos, att, pos_plat, att_plat, self.contact_threshold)


@struct.dataclass
class PPOAgentState(base.Base):
    # State of the agent
    history_act: jax.typing.ArrayLike
    history_obs: jax.typing.ArrayLike
    prev_act: jax.typing.ArrayLike
    has_landed: jax.typing.ArrayLike
    # Initial state of Crazyflie and task
    init_pos: jax.typing.ArrayLike
    init_vel: jax.typing.ArrayLike
    init_att: jax.typing.ArrayLike
    init_ang_vel: jax.typing.ArrayLike
    # Initialize platform state
    init_pos_plat: jax.typing.ArrayLike
    init_vel_plat: jax.typing.ArrayLike
    init_att_plat: jax.typing.ArrayLike


class PPOAgent(BaseNode):
    def __init__(self, *args, outputs: PIDOutput = None, xml_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._outputs = outputs

        if not MUJOCO_INSTALLED:
            raise ImportError("Mujoco not installed. Install it with `pip install mujoco` or `pip install mujoco-mjx`")

        # Load model
        CRAZYFLIE_XML = os.path.join(os.path.dirname(__file__), "cf2_inclined_lw.xml")
        self._xml_path = CRAZYFLIE_XML if xml_path is None else xml_path

        # Initialize system
        self._mj_m = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_d = mujoco.MjData(self._mj_m)
        try:
            self._mjx_m = mjx.device_put(self._mj_m)
            self._mjx_d = mjx.device_put(self._mj_d)
        except AttributeError:
            # Newer mjx version has different API.
            self._mjx_m = mjx.put_model(self._mj_m)
            self._mjx_d = mjx.put_data(self._mj_m, self._mj_d)

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
            mapping=["z_ref", "theta_ref", "phi_ref", "psi_ref"],
            action_dim=3,
            z_max=2.0,
            phi_max=onp.pi / 6,
            theta_max=onp.pi / 6,
            psi_max=0.,  # No yaw (or onp.pi?)
            # init crazyflie
            init_cf="random",  # random, fixed
            fixed_position=jnp.array([0.0, 0.0, 1.75]),  # Above the platform
            x_range=jnp.array([-2.0, 2.0]),
            y_range=jnp.array([-0.5, 0.5]),
            z_range=jnp.array([-0.5, 2.0]),
            clip_pos=jnp.array([2.1, 2.1, 2.1]),
            clip_vel=jnp.array([20., 20., 20.]),
            pos_offset=jnp.array([0.0, 0.0, 0.0]),
            # Init platform
            init_plat="random",  # random, fixed
            fixed_inclination=onp.pi / 7,  # Fixed inclination (25.7 degrees)
            azimuth_max=0.5,
            polar_range=jnp.array([0., onp.pi / 7]),  # Max inclination (25.7 degrees)
            vel_plat_max=1.0,  # todo: Set back to 1.0 # Max constant velocity of platform
            vel_land=0.3,  # Landing velocity (m/s)
            contact_threshold=0.0,  # Contact threshold
            # Train
            tmax=6.0,  # Maximum time for the episode
            gamma=0.99,  # Discount factor (add other reward terms)
            # Domain randomization
            use_noise=True,  # Whether to add noise to the measurements & perform domain randomization.
            use_dr=True,  # Whether to use domain randomization.
            # Mujoco
            mjx_model=self._mjx_m,
            mjx_data=self._mjx_d,
            # Reward params
            k1=0.3242625604682307,  # Weights att_error
            k2=0.4129516217140342,  # Weights vyz_error
            k3=1.415367037157159,  # Weights vx*theta
            k4=1.6040170382473968,  # Weights act_att_error
            f1=7.393182659952971,  # Weights final att_error
            f2=1.183511931649486,  # Weights final vel_error
            fp=356.7628687436355,  # Weights final perfect reward
            # p=0.075,  # Landing threshold
            pos_perfect=0.1,  # todo: used to be 0.05
            att_perfect=0.1,
            vel_perfect=0.5,
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentState:
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        history_act = jnp.zeros((params.num_act, params.action_dim), dtype=jnp.float32)  # [torque]
        history_obs = jnp.zeros((params.num_obs, 13), dtype=jnp.float32)  # [pos, vel, att]
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
        # Inclination
        if params.init_plat == "random":
            vx, vy = jax.random.uniform(rngs[4], shape=(2,),
                                        minval=-params.vel_plat_max,
                                        maxval=params.vel_plat_max)
            init_vel_plat = jnp.array([vx, vy, 0.0])
            polar = jax.random.uniform(rngs[5], shape=(), minval=params.polar_range[0], maxval=params.polar_range[1])
            azimuth = jax.random.uniform(rngs[6], shape=(), minval=-params.azimuth_max, maxval=params.azimuth_max)
        elif params.init_plat == "fixed":
            init_vel_plat = jnp.array([0., 0., 0.])
            polar = params.fixed_inclination
            azimuth = 0.0
        else:
            raise ValueError(f"Unknown inclination method: {params.init_plat}")
        init_att_plat = spherical_to_rpy(polar, azimuth)

        return PPOAgentState(
            # History
            history_act=history_act,
            history_obs=history_obs,
            prev_act=prev_act,
            has_landed=False,
            # Initial crazyflie state
            init_pos=init_pos,
            init_vel=jnp.array([0.0, 0.0, 0.0]),
            init_att=jnp.array([0.0, 0.0, 0.0]),
            init_ang_vel=jnp.array([0.0, 0.0, 0.0]),
            # Initialize platform state
            init_pos_plat=jnp.array([0.0, 0.0, 0.0]),  # Always start at origin
            init_att_plat=init_att_plat,
            init_vel_plat=init_vel_plat,
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
        new_state = params.update_state_history(step_state, action)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        # Update step_state (has_landed)
        plat: PlatformOutput = inputs["platform"][-1].data
        mocap: MoCapOutput = inputs["estimator"][-1].data
        new_state = params.update_state_has_landed(new_step_state.state, mocap, plat)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        # Update step_state (has_landed)
        new_state = params.update_state_prev_action(new_step_state, action)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        # Convert to output
        output = params.to_output(new_step_state, action)
        return new_step_state, output


class Environment(rl.Environment):
    def __init__(self, graph: Graph, params: Dict[str, base.Base] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        super().__init__(graph, params, only_init, starting_eps, randomize_eps, order)

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
        plat: PlatformOutput = graph_state.state["platform"]
        world: WorldState = graph_state.state["world"]
        params: PPOAgentParams = graph_state.params["agent"]
        dmin = params.contact_distance(world.pos, world.att, plat.pos, plat.att)
        has_landed = dmin < 0.0
        # Check if has landed
        # has_landed = graph_state.state["agent"].has_landed
        return has_landed

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> PIDOutput:
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        output = params.to_output(ss, action)
        return output

    def update_graph_state_history(self, graph_state: base.GraphState, action: jax.Array) -> base.GraphState:
        # Update agent state
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        new_state = params.update_state_history(ss, action)
        # Update graph state
        graph_state = graph_state.replace(state=graph_state.state.copy({
            self.graph.supervisor.name: new_state
        }))
        return graph_state

    def update_graph_state_clip_has_landed(self, graph_state: base.GraphState) -> base.GraphState:
        """Override this method if you want to update the graph state after graph.step(...).

        Note: This method is called after the graph has been stepped (before returning from .step()),
              or before returning the initial observation from .reset().
        :param graph_state: The current graph state.
        :param action: The action taken. Is None when called from .reset().
        """
        # Clip state
        clip_pos = graph_state.params["agent"].clip_pos
        clip_vel = graph_state.params["agent"].clip_vel

        # Clip position
        pos = graph_state.state["world"].pos
        pos_plat = graph_state.state["platform"].pos
        pos_clip = jnp.clip(pos - pos_plat, -clip_pos, clip_pos) + pos_plat

        # Clip velocity
        vel = graph_state.state["world"].vel
        vel_clip = jnp.clip(vel, -clip_vel, clip_vel)
        new_world = graph_state.state["world"].replace(pos=pos_clip, vel=vel_clip)

        # Update action
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params # todo: Untested
        new_state = params.update_state_has_landed(ss.state, mocap=graph_state.state["world"], platform=graph_state.state["platform"])

        # Update graph_state
        graph_state = graph_state.replace(state=graph_state.state.copy({
            self.graph.supervisor.name: new_state,
            "world": new_world
        }))
        return graph_state

    def update_graph_state_post_reward(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        # Update action
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        action = action if action is not None else ss.state.prev_act
        new_state = params.update_state_prev_action(ss, action)

        # Update graph_state
        graph_state = graph_state.replace(state=graph_state.state.copy({
            self.graph.supervisor.name: new_state,
        }))
        return graph_state

    def get_reward(self, graph_state: base.GraphState, action: jax.Array):
        # Unpack
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        state: PPOAgentState = ss.state

        # Get denormalized action
        output = params.to_output(ss, action)
        z_ref = output.z_ref
        theta_ref = output.theta_ref
        phi_ref = output.phi_ref
        psi_ref = output.psi_ref
        att_ref = jnp.array([phi_ref, theta_ref, psi_ref])

        # Penalize delta actions
        prev_output = params.to_output(ss, state.prev_act)
        dz_ref = z_ref - prev_output.z_ref
        dphi_ref = phi_ref - prev_output.phi_ref
        dtheta_ref = theta_ref - prev_output.theta_ref
        dpsi_ref = psi_ref - prev_output.psi_ref

        # Decouple error
        error_dangle_decouple = jnp.abs(dphi_ref) * jnp.abs(dtheta_ref)

        # Delta action cost
        error_dangle_ref = 7*jnp.sqrt(dphi_ref**2 + dtheta_ref**2 + dpsi_ref**2)  # todo: Remove harcoded scaling parameter
        error_dz_ref = 7*jnp.sqrt(dz_ref**2)  # todo: Remove harcoded scaling parameter
        # error_daction = (0.2 * error_dz_ref + 0.6 * error_dangle_ref)  # Penalize delta action more.
        # act_z_error = z_ref ** 2

        # Get current state
        world_state: WorldState = graph_state.state["world"]
        plat_state: PlatformState = graph_state.state["platform"]
        vel_rel = world_state.vel - plat_state.vel  # Calculate relative velocity vector
        pos_rel = world_state.pos - plat_state.pos  # Calculate relative position vector

        # Calculate position error
        pos_error = jnp.linalg.norm(pos_rel)

        # Get rotation matrices
        R_cf2w_ref = rpy_to_R(att_ref)
        R_cf2w = rpy_to_R(world_state.att)
        R_is2w = rpy_to_R(plat_state.att)
        z_cf_ref = R_cf2w_ref[:, 2]
        z_cf = R_cf2w[:, 2]
        z_is = R_is2w[:, 2]  # Final attitude target

        # Calculate attitude error
        att_error = jnp.arccos(jnp.clip(jnp.dot(z_cf, z_is), -1, 1))  # Minimize angle between two z-axis vectors
        act_att_error = jnp.arccos(jnp.clip(jnp.dot(z_cf_ref, z_is), -1, 1))  # Minimize angle between two z-axis vectors

        # Calculate components of the landing velocity
        vel_land_ref = -z_is * params.vel_land  # target is landing velocity in negative direction of platform z-axis
        vel_land_error = jnp.linalg.norm(vel_land_ref-vel_rel)
        vxyz_error = jnp.linalg.norm(vel_rel - jnp.dot(z_is, vel_rel) * z_is)

        cf_is_align = 7*jnp.clip(jnp.dot(z_cf, z_is), 0, None)  # todo: remove hardcoded scaling parameter
        vel_cf_align = jnp.clip(jnp.dot(z_cf, vel_rel), None, 0)
        proximity = pos_error < 0.3
        vel_underact = cf_is_align * vel_cf_align * (pos_rel[2] > 0.) * proximity

        # Penalize being on the negative side of the platform
        pos_cfiis = R_is2w.T @ pos_rel
        z_cfiis = pos_cfiis[2]
        z_underplat = jnp.clip(z_cfiis, None, 0)  # Only penalize if below the platform
        mag_xy_cfiis = jnp.clip(jnp.linalg.norm(pos_cfiis[:2]), 0.01, None)  # Get magnitude of x and y components
        error_underplat = z_underplat ** 2 * 10/mag_xy_cfiis  # If further away in xy-plane, less penalty

        # @struct.dataclass
        # class RewardParams:
        #     k1: float = 0.3242625604682307  # Weights att_error
        #     k2: float = 0.4129516217140342  # Weights vyz_error
        #     k3: float = 1.415367037157159  # Weights vx*theta
        #     k4: float = 1.6040170382473968  # Weights act_att_error
        #     f1: float = 7.393182659952971  # Weights final att_error
        #     f2: float = 1.183511931649486  # Weights final vel_error
        #     fp: float = 356.7628687436355  # Weights final perfect reward
        #     p: float = 0.05

        # running cost
        k1, k2, k3, k4 = params.k1, params.k2, params.k3, params.k4
        f1, f2 = params.f1, params.f2
        fp = params.fp
        pos_perfect = (pos_error < params.pos_perfect)  # 0.05
        att_perfect = (att_error < params.att_perfect)  # 0.1
        vel_perfect = (vel_land_error < params.vel_perfect)  # 0.5
        is_perfect = pos_perfect * att_perfect * vel_perfect
        # cost_eps = pos_error + k1*att_error + k2*vxyz_error + k3*vel_underact + k1*act_att_error + k4*act_z_error
        cost_eps = pos_error + k1*att_error + k2*vxyz_error + k3*vel_underact + k1*error_dangle_ref + k4*error_dz_ref + 10 * error_underplat + 20 * error_dangle_decouple
        cost_final = pos_error + f1*att_error + f2*vel_land_error
        cost_perfect = -fp * is_perfect

        # Get termination conditions
        gamma = params.gamma
        terminated = self.get_terminated(graph_state)
        truncated = self.get_truncated(graph_state)
        done = jnp.logical_or(terminated, truncated)
        cost = cost_eps + done * ((1-terminated) * (1/(1-gamma)) + terminated) * cost_final + done * terminated * cost_perfect

        info = {
            "is_perfect": is_perfect,
            "pos_perfect": pos_perfect,
            "att_perfect": att_perfect,
            "vel_perfect": vel_perfect,
            "pos_error": pos_error,
            "att_error": att_error,
            "vel_error": vel_land_error,
        }

        return -cost, truncated, terminated, info

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        return {
            # "new_rwd": 0.,
            "is_perfect": False,
            "pos_perfect": False,
            "att_perfect": False,
            "vel_perfect": False,
            "pos_error": 0.,
            "att_error": 0.,
            "vel_error": 0.,
        }

    def step(self, graph_state: base.GraphState, action: jax.Array) -> rl.StepReturn:
        """
        Step the environment.
        Can be overridden to provide custom step behavior.

        :param graph_state: The current graph state.
        :param action: The action to take.
        :return: Tuple of (graph_state, observation, reward, terminated, truncated, info)
        """
        # Convert action to output
        output = self.get_output(graph_state, action)
        # Pre-step update of graph_state
        gs = self.update_graph_state_history(graph_state, action)
        # Step the graph
        gs, _ = self.graph.step(gs, self.get_step_state(gs), output)
        # Post-step update of graph_state
        gs = self.update_graph_state_clip_has_landed(gs)
        # Get reward, done flags, and some info
        reward, truncated, terminated, info_rwd = self.get_reward(gs, action)
        # Post-reward update of graph_state
        # todo: reward must be retuned when turning this back one
        # gs = self.update_graph_state_post_reward(gs, action)
        # Get observation
        obs = self.get_observation(gs)
        # Get info
        info = self.get_info(gs, action)
        info.update(info_rwd)
        return gs, obs, reward, terminated, truncated, info


@struct.dataclass
class InclinedLandingConfig(ppo.Config):
    def EVAL_METRICS_JAX_CB(self, total_steps, diagnostics: ppo.Diagnostics, eval_transitions: ppo.Transition = None) -> Dict:
        metrics = super().EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_transitions)
        total_done = eval_transitions.done.sum()
        done = eval_transitions.done
        info = eval_transitions.info

        metrics["eval/is_perfect"] = (jnp.roll(info["is_perfect"], shift=1, axis=-1) * done).sum() / total_done
        metrics["eval/pos_perfect"] = (jnp.roll(info["pos_perfect"], shift=1, axis=-1) * done).sum() / total_done
        metrics["eval/att_perfect"] = (jnp.roll(info["att_perfect"], shift=1, axis=-1) * done).sum() / total_done
        metrics["eval/vel_perfect"] = (jnp.roll(info["vel_perfect"], shift=1, axis=-1) * done).sum() / total_done
        pos_error_done = jnp.roll(info["pos_error"], shift=1, axis=-2) * done
        att_error_done = jnp.roll(info["att_error"], shift=1, axis=-2) * done
        vel_error_done = jnp.roll(info["vel_error"], shift=1, axis=-2) * done
        metrics["eval/mean_pos_error"] = pos_error_done.sum() / total_done
        metrics["eval/std_pos_error"] = jnp.sqrt(((pos_error_done - metrics["eval/mean_pos_error"]) ** 2 * done).sum() / total_done)
        metrics["eval/mean_att_error"] = att_error_done.sum() / total_done
        metrics["eval/std_att_error"] = jnp.sqrt(((att_error_done - metrics["eval/mean_att_error"]) ** 2 * done).sum() / total_done)
        metrics["eval/mean_vel_error"] = vel_error_done.sum() / total_done
        metrics["eval/std_vel_error"] = jnp.sqrt(((vel_error_done - metrics["eval/mean_vel_error"]) ** 2 * done).sum() / total_done)
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
        is_perfect = metrics["eval/is_perfect"]
        pos_perfect = metrics["eval/pos_perfect"]
        att_perfect = metrics["eval/att_perfect"]
        vel_perfect = metrics["eval/vel_perfect"]

        if self.VERBOSE:
            print(f"train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                  f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f} | "
                  f"is_perfect={is_perfect:.2f} | pos_perfect={pos_perfect:.2f} | "
                  f"att_perfect={att_perfect:.2f} | vel_perfect={vel_perfect:.2f}"
                  )


# Multi-inclination (noise, mass variation, vary initial x, y, z, azimuth)
ppo_config = InclinedLandingConfig(
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