import yaml
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3.common.utils import safe_mean
import sys
import wandb
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
import time
from sbx.common.policies import BaseJaxPolicy


def unknown_constructor(loader, tag_suffix, node):
    print(f"Skipping unknown tag: {node.tag}")
    return None


class SafeLoader(yaml.FullLoader):
    pass


SafeLoader.add_multi_constructor('!', unknown_constructor)
SafeLoader.add_multi_constructor('tag:yaml.org,2002:', unknown_constructor)


class RolloutWrapper(object):
    def __init__(self, env, model):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env = env
        self.model = model
        self.num_env_steps = self.env.max_steps

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout(self, rng_eval, policy_state):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_state)

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout(self, rng_input, policy_state):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_policy, rng_episode = jax.random.split(rng_input, num=3)
        state, obs, info = self.env.reset(rng_reset)

        assert policy_state is not None, "Policy state must be provided for rollout."

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_state, rng, cum_reward, valid_mask = state_input
            rng, rng_net = jax.random.split(rng, 2)
            scaled_action = BaseJaxPolicy.select_action(policy_state, obs)
            action = self.model.policy.unscale_action(scaled_action)
            next_state, next_obs, reward, terminated, truncated, info = self.env.step(state, action)
            done = jnp.logical_or(terminated, truncated)
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_state,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by root in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state, info = self.env.reset(rng, self.env_params)
        return obs.shape


class WandbCallback(callbacks.BaseCallback):
    """
    When using multiple environments, each call to  ``env.step()`` will effectively correspond
    to ``n_envs`` steps. To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    """

    def __init__(self, eval_env, n_eval_episodes: int = 5, log_freq: int = 10, eval_freq: int = 10000,
                 model_dir: str = ".", save_model: bool = False, verbose=0):
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.model_dir = model_dir
        self.save_model = save_model
        self._best_reward = None
        self.rw: RolloutWrapper = None
        super().__init__(verbose)

    def _init_callback(self) -> None:
        self.rw = RolloutWrapper(self.eval_env, self.model)

    def _on_step(self) -> bool:
        continue_training = True
        if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
            metrics = {}
            time_elapsed = max((time.time_ns() - self.model.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self.model._num_timesteps_at_start) / time_elapsed)
            metrics["train/episodes"] = self.model._episode_num
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                metrics["train/ep_rew_mean"] = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                metrics["train/ep_len_mean"] = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            metrics["train/fps"] = fps
            metrics["train/time_elapsed"] = int(time_elapsed)
            metrics["train/total_timesteps"] = self.num_timesteps
            if len(self.model.ep_success_buffer) > 0:
                metrics["train/success_rate"] = safe_mean(self.model.ep_success_buffer)
            wandb.log(metrics)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            seed = jax.random.PRNGKey(self.n_calls)
            rng = jax.random.split(seed, num=self.n_eval_episodes)
            start = time.time()
            obs, action, reward, next_obs, done, cum_return = self.rw.batch_rollout(rng, self.model.policy.actor_state)
            end = time.time()
            duration = end - start
            fps = (self.n_eval_episodes * self.eval_env.max_steps) / duration
            return_mean, return_std = cum_return.mean(), cum_return.std()
            metrics = {"fps": fps, "duration": duration, "ep_rew_mean": return_mean, "ep_rew_std": return_std}
            if self.verbose >= 1:
                print(f"fps: {fps:.2f} | duration: {duration:.2f} | return: {return_mean:.2f}+/-{return_std:.2f}")
            # Save best model
            if self._best_reward is None or return_mean > self._best_reward:
                self._best_reward = return_mean
                if self.save_model:
                    self.model.save(f"{self.model_dir}/model_best.zip")
                    if self.verbose >= 1:
                        print(f"New best model saved: rwd={self._best_reward} in '{self.model_dir}/model_best.zip'")
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        return continue_training

    def _on_training_end(self) -> None:
        seed = jax.random.PRNGKey(self.n_calls)
        rng = jax.random.split(seed, num=self.n_eval_episodes)
        start = time.time()
        obs, action, reward, next_obs, done, cum_return = self.rw.batch_rollout(rng, self.model.policy.actor_state)
        end = time.time()
        duration = end - start
        fps = (self.n_eval_episodes * self.eval_env.max_steps) / duration
        return_mean, return_std = cum_return.mean(), cum_return.std()
        metrics = {"fps": fps, "duration": duration, "ep_rew_mean": return_mean, "ep_rew_std": return_std}
        if self.verbose >= 1:
            print(f"fps: {fps:.2f} | duration: {duration:.2f} | return: {return_mean:.2f}+/-{return_std:.2f}")
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.log({f"final/eval/{k}": v for k, v in metrics.items()})


class StopTrainingOnTimeBudget(callbacks.BaseCallback):
    """
    Stop the training once a time budget has been reached.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, time_budget: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.time_budget = time_budget

    def _init_callback(self) -> None:
        self._start = time.time()

    def _on_step(self) -> bool:
        end = time.time()
        continue_training = (end - self._start) < self.time_budget

        if self.verbose >= 1 and not continue_training:
            print(f"Stopping training due to time budget of {self.time_budget} seconds being reached.")
        return continue_training


def get_PPO_params(config):
    # Training (linear_schedule comes from rl_zoo3)
    learning_rate = linear_schedule(config.learning_rate) if config.lr_schedule == "linear" else config.learning_rate
    batch_size = config.n_steps if config.batch_size > config.n_steps else config.batch_size
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[config.net_arch]
    activation_fn = {"tanh": nn.tanh, "relu": nn.relu, "elu": nn.elu, "leaky_relu": nn.leaky_relu}[config.activation_fn]
    model_params = {
        "n_steps": config.n_steps,
        "batch_size": batch_size,
        "gamma": config.gamma,
        "learning_rate": learning_rate,
        "ent_coef": config.ent_coef,
        "clip_range": config.clip_range,
        "n_epochs": config.n_epochs,
        "gae_lambda": config.gae_lambda,
        "max_grad_norm": config.max_grad_norm,
        "vf_coef": config.vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=config.ortho_init,
        ),
    }
    return model_params


def get_SAC_params(config):
    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[config.net_arch]
    model_params = {
        "gamma": config.gamma,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "buffer_size": config.buffer_size,
        "learning_starts": config.learning_starts,
        "train_freq": config.num_envs,
        "gradient_steps": config.num_envs,
        "ent_coef": config.ent_coef,
        "tau": config.tau,
        "policy_kwargs": dict(log_std_init=config.log_std_init, net_arch=net_arch, use_sde=False),
        "qf_learning_rate": config.qf_learning_rate,
    }
    return model_params


HYPERPARAMETERS_FN = {
    "PPO": get_PPO_params,
    "SAC": get_SAC_params,
}

