from typing import Callable
import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

KERNEL_INIT_FN = {
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
    squash_output: bool = True  # Whether to squash the output to [-1, 1] or not
    low: jax.typing.ArrayLike = None   # Apply this function to unscale the output from [-1, 1] to the original range
    high: jax.typing.ArrayLike = None
    model_name: str = "Actor"

    def scale_output(self, x) -> jax.Array:
        """Return a function that scales the input to [-1, 1]"""
        raise NotImplementedError("Double check correctness i.c.w. tanh squashing. Should inverse tanh be applied first?")
        return 2.0 * (x - self.low) / (self.high - self.low) - 1.0

    def unscale_output(self, x) -> jax.Array:
        """Return a function that unscales the input from [-1, 1] to the original range"""
        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    @nn.compact
    def __call__(self, x, rng: jax.Array = None):
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
        elif self.output_activation == "tanh":
            # Simple affine layer
            x = nn.Dense(self.num_output_units,
                         kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                         bias_init=nn.initializers.uniform(scale=0.05))(x)
            x = nn.tanh(x)
        elif self.output_activation == "gaussian":
            x = nn.Dense(2 * self.num_output_units,
                         kernel_init=KERNEL_INIT_FN[self.kernel_init_type](),
                         bias_init=nn.initializers.uniform(scale=0.05))(x)
            x_mean = x[:self.num_output_units]
            x_log_std = x[self.num_output_units:]
            x = tfd.MultivariateNormalDiag(x_mean, jnp.exp(0.5*x_log_std))
            # Sample from the distribution
            if rng is not None:
                # x = x.mean()
                x = x.sample(seed=rng)
            else:
                x = x.mean()
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")

        # Squash the output to [-1, 1]
        if self.squash_output:
            x = nn.tanh(x)
            assert (self.low is None) == (self.high is None), "Both low and high must be provided if squashing is enabled"
            if self.low is not None:
                x = self.unscale_output(x)
        return x