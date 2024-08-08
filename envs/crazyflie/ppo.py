import jax.numpy as jnp
from typing import Dict
import flax.struct as struct
import rexv2.ppo as ppo


@struct.dataclass
class PathFollowingConfig(ppo.Config):
    def EVAL_METRICS_JAX_CB(self, total_steps, diagnostics: ppo.Diagnostics, eval_transitions: ppo.Transition = None) -> Dict:
        metrics = super().EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_transitions)
        total_done = eval_transitions.done.sum()
        done = eval_transitions.done
        info = eval_transitions.info
        metrics["eval/vel_on"] = (jnp.roll(info["vel_on"], shift=1, axis=-2) * done).sum() / total_done
        metrics["eval/pos_off"] = (jnp.roll(info["pos_off"], shift=1, axis=-2) * done).sum() / total_done
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
        vel_off = metrics["eval/vel_off"]
        rwd_vel_on = metrics["eval/rwd_vel_on"]

        if self.VERBOSE:
            print(f"train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                  f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f} | "
                  f"vel_on={vel_on:.2f} | pos_off={pos_off:.2f} | "
                  f"vel_off={vel_off:.2f} | rwd_vel_on={rwd_vel_on:.2f}"
                  )


# Fixed inclination (no noise, fixed mass, vary initial x)
# env: InclinedLanding
fixed_inclination = PathFollowingConfig(
    LR=1e-4,
    NUM_ENVS=64,
    NUM_STEPS=128,  # increased from 16 to 32 (to solve approx_kl divergence)
    TOTAL_TIMESTEPS=5e6,
    UPDATE_EPOCHS=32,
    NUM_MINIBATCHES=32,
    GAMMA=0.91,
    GAE_LAMBDA=0.97,
    CLIP_EPS=0.44,
    ENT_COEF=0.01,
    VF_COEF=0.77,
    MAX_GRAD_NORM=0.87,  # or 0.5?
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
    FIXED_INIT=True,
    OFFSET_STEP=True,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
)

# Reference tracking (no noise, fixed mass).
# env: Environment
ref_tracking = ppo.Config(
   LR=1e-4,
   NUM_ENVS=64,
   NUM_STEPS=128,  # increased from 16 to 32 (to solve approx_kl divergence)
   TOTAL_TIMESTEPS=2e6,
   UPDATE_EPOCHS=8,
   NUM_MINIBATCHES=8,
   GAMMA=0.90,
   GAE_LAMBDA=0.983,
   CLIP_EPS=0.93,
   ENT_COEF=0.03,
   VF_COEF=0.58,
   MAX_GRAD_NORM=0.44,  # or 0.5?
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
   FIXED_INIT=True,
   OFFSET_STEP=True,
   NUM_EVAL_ENVS=20,
   EVAL_FREQ=10,
)

# Multi-inclination (noise, mass variation, vary initial x, y, z)
multi_inclination = PathFollowingConfig(
    LR=5e-4,
    NUM_ENVS=128,  # todo: 128?
    NUM_STEPS=64,  # todo: 128?
    TOTAL_TIMESTEPS=2e6,  # todo: a lot.
    UPDATE_EPOCHS=16,  # todo: a lot --> 8?
    NUM_MINIBATCHES=8,
    GAMMA=0.978,
    GAE_LAMBDA=0.951,
    CLIP_EPS=0.131,
    ENT_COEF=0.01,
    VF_COEF=0.899,
    MAX_GRAD_NORM=0.87,
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

term_ref_tracking = PathFollowingConfig(
    LR=5e-4,
    NUM_ENVS=128,
    NUM_STEPS=64,
    TOTAL_TIMESTEPS=10e6,
    UPDATE_EPOCHS=16,
    NUM_MINIBATCHES=8,
    GAMMA=0.978,
    GAE_LAMBDA=0.951,
    CLIP_EPS=0.131,
    ENT_COEF=0.01,
    VF_COEF=0.899,
    MAX_GRAD_NORM=0.87,
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

# Multi-inclination (noise, mass variation, vary initial x, y, z, azimuth)
multi_inclination_azi = PathFollowingConfig(
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


# COPIED from Multi-inclination (noise, mass variation, vary initial x, y, z, azimuth)
path_following = PathFollowingConfig(
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

debug = PathFollowingConfig(
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