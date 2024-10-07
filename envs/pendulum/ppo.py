import jax.numpy as jnp
from typing import Dict
import flax.struct as struct
import rex.ppo as ppo


@struct.dataclass
class PendulumConfig(ppo.Config):
    def EVAL_METRICS_JAX_CB(self, total_steps, diagnostics: ppo.Diagnostics, eval_transitions: ppo.Transition = None) -> Dict:
        metrics = super().EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_transitions)

        # Calculate success rate
        cos_th = eval_transitions.obs[..., 0]
        thdot = eval_transitions.obs[..., 2]
        is_upright = cos_th > 0.95
        is_static = jnp.abs(thdot) < 2.0
        is_valid = jnp.logical_and(is_upright, is_static)
        success_rate = is_valid.sum() / is_valid.size
        metrics["eval/success_rate"] = success_rate
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
        success_rate = metrics["eval/success_rate"]

        if self.VERBOSE:
            print(f"train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                  f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f} | "
                  f"success_rate={success_rate:.2f}"
                  )


default_config = PendulumConfig(
    LR=1e-4,
    NUM_ENVS=64,
    NUM_STEPS=32,  # increased from 16 to 32 (to solve approx_kl divergence)
    TOTAL_TIMESTEPS=5e6,  # 30M solves the task
    UPDATE_EPOCHS=4,
    NUM_MINIBATCHES=4,
    GAMMA=0.99,
    GAE_LAMBDA=0.95,
    CLIP_EPS=0.2,
    ENT_COEF=0.01,
    VF_COEF=0.5,
    MAX_GRAD_NORM=0.5,  # or 0.5?
    NUM_HIDDEN_LAYERS=2,
    NUM_HIDDEN_UNITS=64,
    KERNEL_INIT_TYPE="xavier_uniform",
    HIDDEN_ACTIVATION="tanh",
    STATE_INDEPENDENT_STD=True,
    SQUASH=True,
    ANNEAL_LR=False,
    NORMALIZE_ENV=True,
    FIXED_INIT=True,
    OFFSET_STEP=False,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
    VERBOSE=False,  # If True, disables persistent-cache
    DEBUG=False,  # If True, disables persistent-cache
)

# Sweep for pendulum swing-up with 30Hz control rate and 0.02 std_th (very small...)
# Note: This sweep also found that covariance should not be included.
sweep_pmv2r1zf = PendulumConfig(
    LR=0.0003261962464827655,
    NUM_ENVS=128,
    NUM_STEPS=32,
    TOTAL_TIMESTEPS=5e6,
    UPDATE_EPOCHS=8,
    NUM_MINIBATCHES=16,
    GAMMA=0.9939508937435216,
    GAE_LAMBDA=0.9712149137900143,
    CLIP_EPS=0.16413213812946092,
    ENT_COEF=0.01,
    VF_COEF=0.8015258840683805,
    MAX_GRAD_NORM=0.9630061315073456,
    NUM_HIDDEN_LAYERS=2,
    NUM_HIDDEN_UNITS=64,
    KERNEL_INIT_TYPE='xavier_uniform',
    HIDDEN_ACTIVATION='tanh',
    STATE_INDEPENDENT_STD=True,
    SQUASH=True,
    ANNEAL_LR=False,
    NORMALIZE_ENV=True,
    FIXED_INIT=True,
    OFFSET_STEP=False,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
    VERBOSE=True,
    DEBUG=False
)
