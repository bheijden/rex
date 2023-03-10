import jumpy
import jumpy.numpy as jp
from rex.env import BaseEnv
import rex.utils as utils
import rex.jumpy as rjp

import sbx
import stable_baselines3 as sb3
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper

import envs.double_pendulum as dpend
import experiments as exp

# Environment
ENV_FILE = "/home/r2ci/rex/logs/sim_dp_sbx_2023-02-01-1510/double_pendulum_ode_train_buffer_phase_awin2_owin3_blocking_noadvance_compiled_vectorized.pkl"

# Load models
CONTINUE = True
MODEL_CLS = sbx.SAC  # sbx.SAC sb3.SAC
MODEL_MODULE = dpend.models
MODEL_PRELOAD = "/home/r2ci/rex/logs/sim_dp_sbx_2023-02-01-1510/model.zip"
# MODEL_PRELOAD = "sbx_sac_model"  # sb_sac_pendulum

env = BaseEnv.load(ENV_FILE)

env.max_steps = 400
if not hasattr(env, "_max_starting_step"):
    env._max_starting_step = env._cgraph.max_steps - env.max_steps

model: sbx.SAC = exp.load_model(MODEL_PRELOAD, MODEL_CLS, seed=0, module=MODEL_MODULE)


import jax
import jax.numpy as jnp
from functools import partial


class RolloutWrapper(object):
    def __init__(self, env: BaseEnv, model_forward=None):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env = env
        self.model_forward: sbx.SAC = model_forward
        self.num_env_steps = self.env.max_steps

    # @partial(jax.jit, static_argnums=(0,))
    # def population_rollout(self, rng_eval, policy_params):
    # 	"""Reshape parameter vector and evaluate the generation."""
    # 	# Evaluate population of nets on gymnax task - vmap over rng & params
    # 	pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
    # 	return pop_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout(self, rng_eval):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0,))
        return batch_rollout(rng_eval)

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout(self, rng_input):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_policy, rng_episode = jumpy.random.split(rng_input, num=3)
        state, obs = self.env.reset(rng_reset)

        if self.model_forward is not None:
            policy_state = self.model_forward.policy.actor_state
            # policy_state = self.model_forward.reset(rng_policy)
        else:
            policy_state = None

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_state, rng, cum_reward, valid_mask = state_input
            rng, rng_net = jumpy.random.split(rng, 2)
            if self.model_forward is not None:
                scaled_action = self.model_forward.policy._predict(obs, deterministic=True)
                action = self.model_forward.policy.unscale_action(scaled_action)
                # action = self.model_forward.policy.select_action(policy_state, obs)
            else:
                action = self.env.action_space().sample(rng_net)
            next_state, next_obs, reward, done, info = self.env.step(state, action)
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
        carry_out, scan_out = jumpy.lax.scan(
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
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape


rw = exp.RolloutWrapper(env, model_forward=model)

nenvs = 32
seed = jumpy.random.PRNGKey(0)
rng = jumpy.random.split(seed, num=nenvs)
for i in range(5):
    seed = jumpy.random.PRNGKey(i)
    rng = jumpy.random.split(seed, num=nenvs)
    timer = utils.timer(f"{i=}", log_level=0)
    with timer:
        obs, action, reward, next_obs, done, cum_return = rw.batch_rollout(rng)
    fps = obs.shape[-2] * nenvs / timer.duration

    print(f"[{timer.name}] {obs.shape[-2]} steps/rollout | time={timer.duration:.2f} s | fps={fps:.2f} steps/s | cum_return={cum_return.mean():.2f}  +/- {cum_return.std():.2f}")

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(nrows=3)
# axes = axes.flatten()
# th, th2 = jp.arctan2(obs[:, 1], obs[:, 0]), jp.arctan2(obs[:, 3], obs[:, 2])
# axes[0].plot(th)
# axes[1].plot(th2)
# axes[2].plot(jp.pi - jp.abs(th + th2))
