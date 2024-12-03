import functools

import jax
import jax.numpy as jnp
import pytest

import rex.ppo as ppo
from rex.actor_critic import Actor, ActorCritic, Critic
from rex.graph import Graph
from tests.unit.test_utils import Env


@pytest.mark.parametrize(
    "state_independent_std,hidden_activation,output_activation",
    [
        (state_independent_std, hidden_activation, output_activation)
        for state_independent_std in [True, False]
        for hidden_activation in ["relu", "tanh", "gelu", "softplus"]
        for output_activation in ["identity", "tanh", "gaussian"]
    ],
)
def test_actor_critic(state_independent_std: bool, hidden_activation: str, output_activation: str):
    actor = Actor(
        num_output_units=2,
        num_hidden_units=32,
        num_hidden_layers=2,
        hidden_activation=hidden_activation,  # relu, tanh, gelu, softplus
        output_activation=output_activation,  # identity, tanh, gaussian
        kernel_init_type="lecun_normal",
        state_independent_std=state_independent_std,  # True/False
    )

    critic = Critic(
        num_hidden_units=32,
        num_hidden_layers=2,
        hidden_activation=hidden_activation,  # relu, tanh, gelu, softplus
        kernel_init_type="lecun_normal",
    )

    # Create actor-critic
    ac = ActorCritic.create(actor, critic)

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    obs = jnp.array([0.5, 0.5])
    ac_params = ac.init(rng, obs)

    # Call
    _, _ = ac.apply(ac_params, obs)


@pytest.mark.parametrize(
    "offset_step,anneal_lr, debug, stochastic_action",
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
    ],
)
def test_train(graph: Graph, offset_step: bool, anneal_lr: bool, debug, stochastic_action: bool):
    # Create env
    env = Env(graph)

    # Create ppo config
    config = ppo.Config(
        LR=5e-4,
        NUM_ENVS=2,  # To reduce computation time
        NUM_STEPS=16,  # To reduce computation time
        TOTAL_TIMESTEPS=64,  # To reduce computation time
        UPDATE_EPOCHS=1,  # To reduce computation time
        NUM_MINIBATCHES=1,  # To reduce computation time
        GAMMA=0.99,
        GAE_LAMBDA=0.95,
        CLIP_EPS=0.2,
        ENT_COEF=0.01,
        VF_COEF=0.5,
        MAX_GRAD_NORM=0.5,
        NUM_HIDDEN_LAYERS=2,
        NUM_HIDDEN_UNITS=64,
        KERNEL_INIT_TYPE="xavier_uniform",
        HIDDEN_ACTIVATION="tanh",
        STATE_INDEPENDENT_STD=True,
        SQUASH=True,
        ANNEAL_LR=anneal_lr,
        NORMALIZE_ENV=True,
        FIXED_INIT=True,
        OFFSET_STEP=offset_step,
        NUM_EVAL_ENVS=2,  # To reduce computation time
        EVAL_FREQ=2,  # To reduce computation time
        VERBOSE=True,
        DEBUG=debug,
    )

    # Test API
    _ = config.NUM_TIMESTEPS

    # Jit the train function
    train = jax.jit(functools.partial(ppo.train, env))

    # Create PPO
    res = train(config, rng=jax.random.PRNGKey(0))

    # Test API
    _ = res.obs_scaling
    _ = res.act_scaling
    _ = res.rwd_scaling

    # Test Policy API
    policy = res.policy
    obs = env.get_observation(graph_state=None)  # In this test case, graph_state is not used
    rng = None if not stochastic_action else jax.random.PRNGKey(0)
    action = policy.get_action(obs, rng=rng)
