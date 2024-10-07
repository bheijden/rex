from typing import Callable, Union
import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
# import tensorflow_probability.substrates.jax as tfp
# tfd = tfp.distributions
import distrax


KERNEL_INIT_FN = {
    """
    These initializations are recommended based on the general behavior of each activation function. 
    Adjustments may be needed based on specific network architectures and tasks.
    
    - Sigmoid/Logistic: Use Xavier/Glorot Normal or Uniform. Suitable for activations with limited output ranges.
    - tanh: Use Xavier/Glorot Normal or Uniform. Ideal for outputs in the -1 to 1 range.
    - ReLU: Use He Normal or He Uniform. Prevents vanishing gradients by increasing initial weight variance.
    - Leaky ReLU/PReLU: Use He Normal or He Uniform. Accounts for the linear non-saturation of these activations.
    - ELU: Use He Normal or He Uniform. Despite ELU's dual nature, He initialization supports its ReLU-like behavior.
    - Softmax: Use Xavier/Glorot Normal or Uniform. Maintains consistent output variance, often used in output layers for classification.
    - Swish: Use He Normal or He Uniform. Benefits from He initialization due to its non-monotonous and smooth nature.
    - Mish: Use He Normal or He Uniform. Similar to Swish and ReLU, suitable for its unbounded, smooth, and non-monotonic properties.
    """
    "glorot_normal": jax.nn.initializers.glorot_normal,
    "glorot_uniform": jax.nn.initializers.glorot_uniform,
    "he_normal": jax.nn.initializers.he_normal,
    "he_uniform": jax.nn.initializers.he_uniform,
    "kaiming_normal": jax.nn.initializers.kaiming_normal,
    "kaiming_uniform": jax.nn.initializers.kaiming_uniform,
    "lecun_normal": jax.nn.initializers.lecun_normal,
    "lecun_uniform": jax.nn.initializers.lecun_uniform,
    "xavier_normal": jax.nn.initializers.xavier_normal,
    "xavier_uniform": jax.nn.initializers.xavier_uniform,
}


class Actor(nn.Module):
    num_output_units: int
    num_hidden_units: int = 64
    num_hidden_layers: int = 2
    hidden_activation: str = "relu"
    output_activation: str = "gaussian"
    kernel_init_type: str = "lecun_normal"
    state_independent_std: bool = True
    # squash_output: bool = True  # Whether to squash the output to [-1, 1] or not
    # low: jax.typing.ArrayLike = None   # Apply this function to unscale the output from [-1, 1] to the original range
    # high: jax.typing.ArrayLike = None
    model_name: str = "Actor"

    @nn.compact
    def __call__(self, x):
        # Initialize hidden layers
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.num_hidden_units,
                         kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                         bias_init=nn.initializers.uniform(scale=0.05))(x)
            if self.hidden_activation == "relu":
                x = nn.relu(x)
            elif self.hidden_activation == "tanh":
                x = nn.tanh(x)
            elif self.hidden_activation == "gelu":
                x = nn.gelu(x)
            elif self.hidden_activation == "softplus":
                x = nn.softplus(x)
            else:
                raise ValueError(f"Unknown hidden_activation: {self.hidden_activation}")

        # Initialize output layer
        if self.output_activation == "identity":
            # Simple affine layer
            x = nn.Dense(self.num_output_units,
                         kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                         bias_init=nn.initializers.uniform(scale=0.05))(x)
            pi = distrax.Deterministic(x)
        elif self.output_activation == "tanh":
            # Simple affine layer
            x = nn.Dense(self.num_output_units,
                         kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                         bias_init=nn.initializers.uniform(scale=0.05))(x)
            x = nn.tanh(x)
            pi = distrax.Deterministic(x)
        elif self.output_activation == "gaussian":
            if self.state_independent_std:
                x_mean = nn.Dense(self.num_output_units,
                             kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                             bias_init=nn.initializers.uniform(scale=0.05))(x)
                actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.num_output_units,))
                pi = distrax.MultivariateNormalDiag(x_mean, jnp.exp(actor_logtstd))
            else:
                x = nn.Dense(2 * self.num_output_units,
                             kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                             bias_init=nn.initializers.uniform(scale=0.05))(x)
                x_mean = x[:self.num_output_units]
                x_log_std = x[self.num_output_units:]
                pi = distrax.MultivariateNormalDiag(x_mean, jnp.exp(0.5*x_log_std))
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")
        return pi

class Critic(nn.Module):
    num_hidden_units: int = 64
    num_hidden_layers: int = 2
    hidden_activation: str = "relu"
    kernel_init_type: str = "lecun_normal"
    model_name: str = "Critic"

    @nn.compact
    def __call__(self, x) -> Union[float, jax.Array]:
        # Initialize hidden layers
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.num_hidden_units,
                         kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                         bias_init=nn.initializers.uniform(scale=0.0))(x)
            if self.hidden_activation == "relu":
                x = nn.relu(x)
            elif self.hidden_activation == "tanh":
                x = nn.tanh(x)
            elif self.hidden_activation == "gelu":
                x = nn.gelu(x)
            elif self.hidden_activation == "softplus":
                x = nn.softplus(x)
            else:
                raise ValueError(f"Unknown hidden_activation: {self.hidden_activation}")

        # Initialize output layer
        x = nn.Dense(1,
                     kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                     bias_init=nn.initializers.uniform(scale=0.0))(x)
        return jnp.squeeze(x, axis=-1)  # x[0]  # Return scalar value


class ActorCritic(nn.Module):
    actor: Actor
    critic: Critic

    @classmethod
    def create(cls, actor: Actor, critic: Critic) -> "ActorCritic":
        return cls(actor=actor, critic=critic)

    @nn.compact
    def __call__(self, x):
        return self.actor(x), self.critic(x)
