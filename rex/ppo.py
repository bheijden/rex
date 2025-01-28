from typing import Any, Dict, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy as onp


try:
    import optax
except ImportError as e:
    raise ImportError(f"Failed to import optax: {e}. Please install it with `pip install optax`.")

import distrax
from flax import struct
from flax.training.train_state import TrainState

from rex.actor_critic import Actor, ActorCritic, Critic
from rex.base import Base, GraphState
from rex.rl import (
    AutoResetWrapper,
    BaseEnv,
    Environment,
    LogWrapper,
    NormalizeVec,
    NormalizeVecObservationWrapper,
    NormalizeVecReward,
    SquashActionWrapper,
    SquashState,
    VecEnvWrapper,
)


@struct.dataclass
class Transition(Base):
    done: Union[bool, jax.typing.ArrayLike]
    action: jax.typing.ArrayLike
    value: Union[float, jax.typing.ArrayLike]
    reward: Union[float, jax.typing.ArrayLike]
    log_prob: Union[float, jax.typing.ArrayLike]
    obs: jax.typing.ArrayLike
    info: Dict[str, jax.typing.ArrayLike]


@struct.dataclass
class Diagnostics(Base):
    total_loss: Union[float, jax.typing.ArrayLike]
    value_loss: Union[float, jax.typing.ArrayLike]
    policy_loss: Union[float, jax.typing.ArrayLike]
    entropy_loss: Union[float, jax.typing.ArrayLike]
    approxkl: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class Config(Base):
    """
    Configuration for PPO.

    Inherit from this class and override the `EVAL_METRICS_JAX_CB` and `EVAL_METRICS_HOST_CB` methods to customize the
    evaluation metrics and the host-side callback for the evaluation metrics.

    Attributes:
        LR: The learning rate.
        NUM_ENVS: The number of parallel environments.
        NUM_STEPS: The number of steps to run in each environment per update.
        TOTAL_TIMESTEPS: The total number of timesteps to run.
        UPDATE_EPOCHS: The number of epochs to run per update.
        NUM_MINIBATCHES: The number of minibatches to split the data into.
        GAMMA: The discount factor.
        GAE_LAMBDA: The Generalized Advantage Estimation (GAE) parameter.
        CLIP_EPS: The clipping parameter for the ratio in the policy loss.
        ENT_COEF: The coefficient of the entropy regularizer.
        VF_COEF: The value function coefficient.
        MAX_GRAD_NORM: The maximum gradient norm.
        NUM_HIDDEN_LAYERS: The number of hidden layers (same for actor and critic).
        NUM_HIDDEN_UNITS: The number of hidden units per layer (same for actor and critic).
        KERNEL_INIT_TYPE: The kernel initialization type (same for actor and critic).
        HIDDEN_ACTIVATION: The hidden activation function (same for actor and critic).
        STATE_INDEPENDENT_STD: Whether to use state-independent standard deviation for the actor.
        SQUASH: Whether to squash the action output of the actor.
        ANNEAL_LR: Whether to anneal the learning rate.
        NORMALIZE_ENV: Whether to normalize the environment (observations and rewards), actions are always normalized.
        FIXED_INIT: Whether to use fixed initial states for each parallel environment.
        OFFSET_STEP: Whether to offset the step counter for each parallel environment to break temporal correlations.
        NUM_EVAL_ENVS: The number of evaluation environments.
        EVAL_FREQ: The number of evaluations to run per run of training.
        VERBOSE: Whether to print verbose output.
        DEBUG: Whether to print debug output per step.
    """

    # Learning rate
    LR: float = struct.field(default=5e-4)
    # Number of parallel environments
    NUM_ENVS: int = struct.field(pytree_node=False, default=64)
    # Number of steps to run in each environment per update
    NUM_STEPS: int = struct.field(pytree_node=False, default=16)
    # Total number of timesteps to run
    TOTAL_TIMESTEPS: int = struct.field(pytree_node=False, default=1e6)
    # Number of epochs to run per update
    UPDATE_EPOCHS: int = struct.field(pytree_node=False, default=4)
    # Number of minibatches to split the data into
    NUM_MINIBATCHES: int = struct.field(pytree_node=False, default=4)
    # Discount factor
    GAMMA: float = struct.field(default=0.99)
    # Generalized Advantage Estimation (GAE) parameter
    GAE_LAMBDA: float = struct.field(default=0.95)
    # Clipping parameter for the ratio in the policy loss
    CLIP_EPS: float = struct.field(default=0.2)
    # Coefficient of the entropy regularizer
    ENT_COEF: float = struct.field(default=0.01)
    # Value function coefficient
    VF_COEF: float = struct.field(default=0.5)
    # Maximum gradient norm
    MAX_GRAD_NORM: float = struct.field(default=0.5)
    # Number of hidden layers (same for actor and critic)
    NUM_HIDDEN_LAYERS: int = struct.field(pytree_node=False, default=2)
    # Number of hidden units per layer (same for actor and critic)
    NUM_HIDDEN_UNITS: int = struct.field(pytree_node=False, default=64)
    # Kernel initialization type (same for actor and critic)
    KERNEL_INIT_TYPE: str = struct.field(pytree_node=False, default="xavier_uniform")
    # Hidden activation function (same for actor and critic)
    HIDDEN_ACTIVATION: str = struct.field(pytree_node=False, default="tanh")
    # Whether to use state-independent standard deviation for the actor
    STATE_INDEPENDENT_STD: bool = struct.field(pytree_node=False, default=True)
    # Whether to squash the action output of the actor
    SQUASH: bool = struct.field(pytree_node=False, default=True)
    # Whether to anneal the learning rate
    ANNEAL_LR: bool = struct.field(pytree_node=False, default=False)
    # Whether to normalize the environment (observations and rewards), actions are always normalized
    NORMALIZE_ENV: bool = struct.field(pytree_node=False, default=False)
    # Whether to use fixed initial states for each parallel environment
    # If True, states are sampled once and then fixed for the entire run per parallel environment
    FIXED_INIT: bool = struct.field(pytree_node=False, default=True)
    # Whether to offset the step counter for each parallel environment
    # If True, parallel environments will start at different steps to avoid temporal correlations
    OFFSET_STEP: bool = struct.field(pytree_node=False, default=False)
    # Number of evaluation environments
    NUM_EVAL_ENVS: int = struct.field(pytree_node=False, default=20)
    # Number of evaluations to run per run of training
    EVAL_FREQ: int = struct.field(pytree_node=False, default=10)
    # Whether to print verbose output
    VERBOSE: bool = struct.field(pytree_node=False, default=True)
    # Whether to print debug output per step
    DEBUG: bool = struct.field(pytree_node=False, default=False)

    @property
    def NUM_UPDATES(self):
        """Number of updates to run"""
        return self.TOTAL_TIMESTEPS // self.NUM_STEPS // self.NUM_ENVS

    @property
    def NUM_UPDATES_PER_EVAL(self):
        """Number of updates to run per evaluation"""
        return self.NUM_UPDATES // self.EVAL_FREQ

    @property
    def NUM_TIMESTEPS(self):
        """Number of timesteps to run per evaluation"""
        return self.NUM_UPDATES_PER_EVAL * self.NUM_STEPS * self.NUM_ENVS * self.EVAL_FREQ

    @property
    def MINIBATCH_SIZE(self):
        """Size of the minibatch"""
        return self.NUM_ENVS * self.NUM_STEPS // self.NUM_MINIBATCHES

    def EVAL_METRICS_JAX_CB(
        self, total_steps: Union[int, jax.Array], diagnostics: Diagnostics, eval_transitions: Transition = None
    ) -> Dict:
        """
        Compute evaluation metrics for the PPO algorithm.

        Args:
            total_steps: The total number of steps run.
            diagnostics: The diagnostics from the training process.
            eval_transitions: The transitions from the evaluation process.

        Returns:
            Dict: A dictionary containing the evaluation metrics.
        """
        returns_done = eval_transitions.info["returned_episode_returns"] * eval_transitions.done
        lengths_done = eval_transitions.info["returned_episode_lengths"] * eval_transitions.done
        total_eps = eval_transitions.done.sum()
        clip_done = jnp.clip(eval_transitions.done.sum(axis=0), 1, None).sum()
        mean_returns = returns_done.sum() / clip_done
        std_returns = jnp.sqrt(((returns_done - mean_returns) ** 2 * eval_transitions.done).sum() / clip_done)
        mean_lengths = lengths_done.sum() / clip_done
        std_lengths = jnp.sqrt(((lengths_done - mean_lengths) ** 2 * eval_transitions.done).sum() / clip_done)

        metrics = {}
        metrics["train/total_steps"] = total_steps
        metrics["train/mean_approxkl"] = diagnostics.approxkl.mean()
        metrics["train/std_approxkl"] = diagnostics.approxkl.std()
        metrics["eval/clip_done"] = clip_done
        metrics["eval/mean_returns"] = mean_returns
        metrics["eval/std_returns"] = std_returns
        metrics["eval/mean_lengths"] = mean_lengths
        metrics["eval/std_lengths"] = std_lengths
        metrics["eval/total_episodes"] = total_eps
        return metrics

    def EVAL_METRICS_HOST_CB(self, metrics: Dict) -> None:
        """
        Evaluate the evaluation metrics for the PPO algorithm on the host.

        Can be used for printing or logging the evaluation metrics on the host as this is side-effectful.

        Args:
            metrics: The evaluation metrics.
        """
        # Standard metrics
        global_step = metrics["train/total_steps"]
        mean_approxkl = metrics["train/mean_approxkl"]
        mean_return = metrics["eval/mean_returns"]
        std_return = metrics["eval/std_returns"]
        mean_length = metrics["eval/mean_lengths"]
        std_length = metrics["eval/std_lengths"]
        total_episodes = metrics["eval/total_episodes"]

        if self.VERBOSE:
            warn = ""
            if total_episodes == 0:
                warn = "WARNING: No eval. episodes returned | "
            print(
                f"{warn}train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f}"
            )


@struct.dataclass
class Policy(Base):
    """
    Represents the policy model.

    Attributes:
        act_scaling: The action scaling parameters.
        obs_scaling: The observation scaling parameters.
        model: The model parameters.
        hidden_activation: The hidden activation function.
        output_activation: The output activation function.
        state_independent_std: Whether the standard deviation of the actor is state-independent
    """

    act_scaling: SquashState
    obs_scaling: NormalizeVec
    model: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Any]]]
    hidden_activation: str = struct.field(pytree_node=False)
    output_activation: str = struct.field(pytree_node=False)
    state_independent_std: bool = struct.field(pytree_node=False)

    def apply_actor(self, norm_obs: jax.typing.ArrayLike, rng: jax.Array = None) -> jax.Array:
        """
        Apply the actor model to the normalized observation

        Args:
            norm_obs: The normalized observation
            rng: Random number generator key

        Returns:
            The unscaled action
        """
        x = norm_obs  # Rename for clarity

        # Get parameters
        actor_params = self.model["actor"]
        num_layers = sum(["Dense" in k in k for k in actor_params.keys()])

        # Apply hidden layers
        ACTIVATIONS = dict(tanh=nn.tanh, relu=nn.relu, gelu=nn.gelu, softplus=nn.softplus)
        for i in range(num_layers - 1):
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

    def get_action(self, obs: jax.typing.ArrayLike, rng: jax.Array = None) -> jax.Array:
        """
        Get the action from the policy model

        Args:
            obs: The observation
            rng: Random number generator key

        Returns:
            The action, scaled to the action space.
        """
        # Normalize observation
        norm_obs = self.obs_scaling.normalize(obs, clip=True, subtract_mean=True) if self.obs_scaling is not None else obs
        # Get action
        action = (
            self.apply_actor(norm_obs, rng=rng) if self.model is not None else jnp.zeros((self.action_dim,), dtype=jnp.float32)
        )
        # Scale action
        action = self.act_scaling.unsquash(action) if self.act_scaling is not None else action
        return action


@struct.dataclass
class RunnerState(Base):
    """
    Represents the state of the runner during training.

    Attributes:
        train_state: The state of the training process.
        env_state: The state of the environment.
        last_obs: The last observation.
        rng: Random number generator key
    """

    train_state: TrainState
    env_state: GraphState
    last_obs: jax.typing.ArrayLike
    rng: jax.Array


@struct.dataclass
class PPOResult(Base):
    """
    Represents the result of the PPO training process.

    Attributes:
        config: Configuration for the PPO algorithm.
        runner_state: The state of the runner after training.
        metrics: Dictionary containing various metrics collected during training.
    """

    config: Config
    runner_state: RunnerState
    metrics: Dict[str, Any]

    @property
    def obs_scaling(self) -> SquashState:
        """Returns the observation scaling parameters."""
        return self.runner_state.env_state.aux.get("norm_obs", None)

    @property
    def act_scaling(self) -> SquashActionWrapper:
        """Returns the action scaling parameters."""
        # Note, we changed from x[0] to x[..., 0, :] as env_state has [..., NUM_ENVS, ACTION_DIM] shape
        # and we only want the shape of a single environment.
        return jax.tree_util.tree_map(lambda x: x[..., 0, :], self.runner_state.env_state.aux.get("act_scaling", None))

    @property
    def rwd_scaling(self) -> NormalizeVec:
        """Returns the reward scaling parameters."""
        return self.runner_state.env_state.aux.get("norm_reward", None)

    @property
    def policy(self) -> Policy:
        """Returns the policy model."""
        return Policy(
            act_scaling=self.act_scaling,
            obs_scaling=self.obs_scaling,
            model=self.runner_state.train_state.params["params"],
            hidden_activation=self.config.HIDDEN_ACTIVATION,
            output_activation="gaussian",
            state_independent_std=self.config.STATE_INDEPENDENT_STD,
        )


def train(env: Union[BaseEnv, Environment], config: Config, rng: jax.Array) -> PPOResult:
    """
    Train the PPO model.

    PPO implementation based on the PPO implementation from purejaxrl:
    https://github.com/luchris429/purejaxrl

    Args:
        env: The environment to train on.
        config: Configuration for the PPO algorithm.
        rng: Random number generator key.

    Returns:
        PPOResult: The result of the training process.
    """
    # INIT TRAIN ENV
    env = AutoResetWrapper(env, fixed_init=config.FIXED_INIT)
    env = LogWrapper(env)
    env = SquashActionWrapper(env, squash=config.SQUASH)
    env = VecEnvWrapper(env)
    vec_env = env
    if config.NORMALIZE_ENV:
        env = NormalizeVecObservationWrapper(env)
        env = NormalizeVecReward(env, config.GAMMA)

    def linear_schedule(count):
        frac = 1.0 - (count // (config.NUM_MINIBATCHES * config.UPDATE_EPOCHS)) / config.NUM_UPDATES
        return config.LR * frac

    # INIT VECTORIZED ENV
    rng, rng_reset = jax.random.split(rng)
    rngs_reset = jax.random.split(rng_reset, config.NUM_ENVS)
    gsv, obsv, vinfo = env.reset(rngs_reset)
    gsv_excl_aux = gsv.replace(aux={})  # Some data in aux is not vectorized
    gs = jax.tree_util.tree_map(lambda x: x[0], gsv_excl_aux)  # Grab single gs
    env_params = gs

    # OFFSET STEP
    if config.OFFSET_STEP:
        max_steps = env.max_steps
        offset = (onp.arange(config.NUM_ENVS) * (env.max_steps / config.NUM_ENVS)).astype(int) % max_steps
        gsv = gsv.replace(step=gsv.step + offset)

    # INIT ACTOR NETWORK
    actor = Actor(
        env.action_space(gs).shape[0],
        num_hidden_units=config.NUM_HIDDEN_UNITS,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
        hidden_activation=config.HIDDEN_ACTIVATION,
        kernel_init_type=config.KERNEL_INIT_TYPE,
        state_independent_std=config.STATE_INDEPENDENT_STD,
    )

    # INIT CRITIC NETWORK
    critic = Critic(
        num_hidden_units=config.NUM_HIDDEN_UNITS,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
        hidden_activation=config.HIDDEN_ACTIVATION,
        kernel_init_type=config.KERNEL_INIT_TYPE,
    )

    # INIT NETWORK
    network = ActorCritic(actor=actor, critic=critic)
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_x)
    if config.ANNEAL_LR:
        tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    else:
        tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM), optax.adam(config.LR, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    env_state = gsv

    # UPDATE LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            env_state, obsv, reward, terminated, truncated, info = env.step(env_state, action)
            done = jnp.logical_or(terminated, truncated)  # todo: handle truncation correctly.
            transition = Transition(done, action, value, reward, log_prob, last_obs, info)
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.NUM_STEPS)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng = runner_state
        _, last_val = network.apply(train_state.params, last_obs)

        def _calculate_gae(traj_batch, last_val):
            """https://nn.labml.ai/rl/ppo/gae.html"""

            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config.GAMMA * next_value * (1 - done) - value
                gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # gae:=advantages in this context
                    # targets:=advantages + traj_batch.value
                    # RERUN NETWORK
                    pi, value = network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.CLIP_EPS, config.CLIP_EPS)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    approxkl = (
                        (ratio - 1) - logratio
                    ).mean()  # Approximate KL estimators: http://joschu.net/blog/kl-approx.html
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)  # Advantage normalization (optional, but True in cleanrl)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.CLIP_EPS,
                            1.0 + config.CLIP_EPS,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    # CALCULATE TOTAL LOSS
                    total_loss = loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy

                    # RETURN DIAGNOSTICS
                    d = Diagnostics(total_loss, value_loss, loss_actor, entropy, approxkl)
                    return total_loss, d

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                # todo: return value_loss, loss_actor, entropy_loss
                #       - How to calculate approx_kl?
                (total_loss, d), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, d

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
            assert (
                batch_size == config.NUM_STEPS * config.NUM_ENVS
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            train_state, diagnostics = jax.lax.scan(_update_minbatch, train_state, minibatches)
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, diagnostics

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, diagnostics = jax.lax.scan(_update_epoch, update_state, None, config.UPDATE_EPOCHS)
        train_state = update_state[0]
        metric = traj_batch.info
        metric["diagnostics"] = diagnostics
        rng = update_state[-1]

        # PRINT METRICS
        if config.DEBUG:

            def callback(info):
                return_values = info["returned_episode_returns"][info["returned_episode"]]
                timesteps = info["timestep"][info["returned_episode"]] * config.NUM_ENVS
                if len(timesteps) > 0:
                    global_step = timesteps[-1]
                    mean_return = np.mean(return_values)
                    std_return = np.std(return_values)
                    min_return = np.min(return_values)
                    max_return = np.max(return_values)
                    print(
                        f"global step={global_step} | mean return={mean_return:.2f} +- {std_return:.2f} | min return={min_return:.2f} | max return={max_return:.2f}"
                    )

            jax.debug.callback(callback, metric)

        runner_state = (train_state, env_state, last_obs, rng)
        return runner_state, metric

    # TRAIN LOOP
    def _update_and_eval(runner_state, xs):
        (rng_eval, idx_eval) = xs

        # RUN TRAIN UPDATES
        runner_state, train_metrics = jax.lax.scan(_update_step, runner_state, None, config.NUM_UPDATES_PER_EVAL)

        # GRAB METRICS
        total_steps = idx_eval * config.NUM_UPDATES_PER_EVAL * config.NUM_STEPS * config.NUM_ENVS
        diagnostics = train_metrics["diagnostics"]
        eval_traj_batch = None

        # EVALUATE
        if config.NUM_EVAL_ENVS > 0:
            rngs_eval = jax.random.split(rng_eval, config.NUM_EVAL_ENVS + env.max_steps)
            eval_train_state = runner_state[0]
            init_env_state, init_obs, _ = vec_env.reset(rngs_eval[: config.NUM_EVAL_ENVS])

            # Properly normalize the observations
            if config.NORMALIZE_ENV:
                norm_obs = runner_state[1].aux["norm_obs"]

            def _evaluate_env_step(__runner_state, _rng):
                last_env_state, last_obs = __runner_state
                if config.NORMALIZE_ENV:
                    last_obs = norm_obs.normalize(last_obs, clip=True, subtract_mean=True)

                pi, value = network.apply(eval_train_state.params, last_obs)
                action = pi.mean()
                next_env_state, next_obsv, reward, terminated, truncated, info = vec_env.step(last_env_state, action)
                done = jnp.logical_or(terminated, truncated)
                transition = Transition(done, action, value, reward, None, next_obsv, info)
                next_runner_state = (next_env_state, next_obsv)
                return next_runner_state, transition

            init_runner_state = (init_env_state, init_obs)
            _, eval_traj_batch = jax.lax.scan(_evaluate_env_step, init_runner_state, rngs_eval[config.NUM_EVAL_ENVS :])

        # GENERATE METRICS
        metrics = config.EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_traj_batch)

        # CALL METRICS CALLBACK
        if config.VERBOSE:
            jax.debug.callback(config.EVAL_METRICS_HOST_CB, metrics)
        return runner_state, metrics

    rng, rng_update, rng_eval = jax.random.split(rng, num=3)
    rngs_eval = jax.random.split(rng_eval, config.EVAL_FREQ)
    idx_eval = jnp.arange(1, config.EVAL_FREQ + 1)
    runner_state = (train_state, env_state, obsv, rng_update)
    runner_state, metrics = jax.lax.scan(_update_and_eval, runner_state, (rngs_eval, idx_eval))

    # Before returning
    ret = PPOResult(
        config=config,
        runner_state=RunnerState(
            train_state=runner_state[0], env_state=runner_state[1], last_obs=runner_state[2], rng=runner_state[3]
        ),
        metrics=metrics,
    )
    return ret
