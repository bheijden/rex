import jax
import jax.numpy as jnp
import jumpy
import jumpy.numpy as jp
from rex.env import BaseEnv
from rex.wrappers import Wrapper, AutoResetWrapper
from brax.envs import State
import brax.envs.wrappers as brax_wrappers


class BraxWrapper(Wrapper):
    def __init__(self, env: BaseEnv):
        super().__init__(env)
        self.env = env

    @property
    def action_size(self) -> int:
        action_space = self.env.action_space().sample(jax.random.PRNGKey(0))
        return action_space.shape[0]

    @property
    def observation_size(self) -> int:
        observation_space = self.env.observation_space().sample(jax.random.PRNGKey(0))
        return observation_space.shape[0]

    def reset(self, rng: jax.random.KeyArray) -> State:
        graph_state, obs, info = self.env.reset(rng)
        info = {}
        metrics = {}
        reward, done = jnp.array(1.0), jnp.array(False, dtype=jnp.float32)
        return State(qp=graph_state, obs=obs, reward=reward, done=done, info=info, metrics=metrics)

    def step(self, state: State, action: jax.typing.ArrayLike):
        graph_state = state.qp
        next_graph_state, obs, reward, truncated, done, info = self.env.step(graph_state, action)
        done = jnp.array(done, dtype=jnp.float32)
        reward = jnp.array(reward)
        return state.replace(qp=next_graph_state, obs=obs, reward=reward, done=done)

    def close(self):
        return self.env.close()


def wrap_for_training(env,
                      episode_length: int = 1000,
                      action_repeat: int = 1):
    """Common wrapper pattern for all training agents.

    Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step

    Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
    """
    env = AutoResetWrapper(env)
    env = BraxWrapper(env)
    env = brax_wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    batched = False
    if hasattr(env, 'custom_tree_in_axes'):
        batch_indices, _ = jax.tree_util.tree_flatten(env.custom_tree_in_axes)
        if 0 in batch_indices:
            batched = True
    if not batched:
        env = brax_wrappers.VmapWrapper(env)
    return env

# Monkey patch
brax_wrappers.wrap_for_training = wrap_for_training


if __name__ == "__main__":

    from brax.training.agents.sac import train as sac
    from brax.training.agents.ppo import train as ppo

    # Set logging level
    from absl import logging
    logging.set_verbosity(logging.DEBUG)

    # Environment
    ENV_FILE = "/home/r2ci/rex/logs/sim_dp_sbx_2023-02-01-1510/double_pendulum_ode_train_buffer_phase_awin2_owin3_blocking_noadvance_compiled_vectorized.pkl"
    env = BaseEnv.load(ENV_FILE)

    import functools
    from datetime import datetime
    import matplotlib.pyplot as plt

    #
    env_name = "double_pendulum"
    train_fn = functools.partial(sac.train, num_timesteps=6_553_600, num_evals=20, reward_scaling=1.0, episode_length=400,
                              normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4,
                              num_envs=128,
                              batch_size=512, grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=1048576,
                              min_replay_size=8192)

    max_y = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000, 'humanoidstandup': 75_000, 'reacher': 5,
             'walker2d': 5000, 'fetch': 15, 'grasp': 100, 'ur5e': 10, 'pusher': 0}.get(env_name, 800)
    min_y = {'reacher': -100, 'pusher': -150}.get(env_name, -2500)

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        # clear_output(wait=True)
        plt.xlim([0, train_fn.keywords['num_timesteps']])
        plt.ylim([min_y, max_y])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        # plt.show()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')
    plt.show()

    # with rjax.use("jax"):
    #     _, _, metrics = ppo.train(
    #         env,
    #         num_timesteps=2 ** 18,
    #         episode_length=128,
    #         num_envs=64,
    #         learning_rate=3e-4,
    #         entropy_cost=1e-2,
    #         discounting=0.95,
    #         unroll_length=5,
    #         batch_size=64,
    #         num_minibatches=8,
    #         num_updates_per_batch=4,
    #         normalize_observations=True,
    #         seed=2,
    #         reward_scaling=1,
    #         num_evals=10,
    #         normalize_advantage=False)
    #     print(metrics)
    #     # execute more evals, reduce grad_updates per step.
    #     # increase batch_size
    #     _, _, metrics = sac.train(
    #         env,
    #         num_timesteps=2**15,
    #         episode_length=50,
    #         num_envs=10,
    #         learning_rate=3e-4,
    #         discounting=0.99,
    #         batch_size=64,
    #         normalize_observations=False,
    #         reward_scaling=1,
    #         num_evals=10,
    #         grad_updates_per_step=10,
    #         seed=0)
