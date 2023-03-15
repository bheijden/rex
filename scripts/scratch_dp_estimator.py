import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import jumpy
import jumpy.numpy as jp
import jax

import rex.utils as utils
from rex.wrappers import GymWrapper
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, SYNC, ASYNC, REAL_TIME, FAST_AS_POSSIBLE, FREQUENCY, PHASE, SIMULATED, WALL_CLOCK

utils.set_log_level(WARN)

import experiments as exp
import stable_baselines3 as sb3
import envs.double_pendulum as dpend

# Environment
ENV = "double_pendulum"
DIST_FILE = f"21eps_pretrained_sbx_sac_gmms_2comps.pkl"
SPLIT_MODE = "generational"
SUPERGRAPH_MODE = "MCS"  # topo=15, 1.3M, MCS=5, 2M
JITTER = BUFFER
SCHEDULING = PHASE
MAX_STEPS = 5*80
WIN_ACTION = 2
WIN_OBS = 3
BLOCKING = True
ADVANCE = False
ENV_FN = dpend.ode.build_double_pendulum
ENV_CLS = dpend.env.DoublePendulumEnv
CLOCK = SIMULATED
RTF = REAL_TIME
RATE_ESTIMATOR = 40
RATES = dict(world=150, agent=80, actuator=80, sensor=80, render=20)
DELAY_FN = lambda d: d.high*0.75
USE_DELAYS = True   # todo: TOGGLE WITH/WITHOUT DELAYS HERE

# Load models
MODEL_CLS = sb3.SAC  # sbx.SAC
MODEL_MODULE = dpend.models
MODEL_PRELOAD = "sb_sac_model"

# Training
SEED = 0
NUM_ENVS = 10
SAVE_FREQ = 40_000
NSTEPS = 200_000
NUM_EVAL_PRE = 1
NUM_EVAL_POST = 20
HYPERPARAMS = {"learning_rate": 0.01}
CONTINUE = True

# Record settings
RECORD_SETTINGS = {"agent": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "world": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "actuator": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "sensor": dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True),
                   "render": dict(node=True, outputs=False, rngs=True, states=True, params=True, step_states=True)}

# Load distributions
delays_sim = exp.load_distributions(DIST_FILE)

# Prepare environment
env = exp.make_env(delays_sim, DELAY_FN, RATES, blocking=BLOCKING, advance=ADVANCE, win_action=WIN_ACTION, win_obs=WIN_OBS,
                   scheduling=SCHEDULING, jitter=JITTER,
                   env_fn=ENV_FN, env_cls=ENV_CLS, name=ENV, eval_env=True, clock=CLOCK, real_time_factor=RTF,
                   max_steps=MAX_STEPS, use_delays=USE_DELAYS)
gym_env = GymWrapper(env)

# Load model
model = exp.load_model(MODEL_PRELOAD, MODEL_CLS, env=gym_env, seed=SEED, module=MODEL_MODULE)

# Make policy
policy = exp.make_policy(model)

# Evaluate model
record_pre = exp.eval_env(gym_env, policy, n_eval_episodes=NUM_EVAL_PRE, verbose=True, seed=SEED, record_settings=RECORD_SETTINGS)

# Trace & compile environment
cenv = exp.make_compiled_env(env, record_pre.episode, max_steps=MAX_STEPS, eval_env=False, split_mode=SPLIT_MODE, supergraph_mode=SUPERGRAPH_MODE,log_level=50)

# Rollouts
rw = exp.RolloutWrapper(cenv)

# Start tracing
# import tensorflow
# jax.profiler.start_trace("/tmp/tensorboard")

# Rn rollouts

nenvs = 7000
seed = jumpy.random.PRNGKey(0)
rng = jumpy.random.split(seed, num=nenvs)
for i in range(10):
    seed = jumpy.random.PRNGKey(i)
    rng = jumpy.random.split(seed, num=nenvs)
    timer = utils.timer(f"{i=}", log_level=0)
    with timer:
        res = rw.batch_rollout(rng)
    fps = env.max_steps * nenvs / timer.duration
    print(f"[{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")

# jax.profiler.stop_trace()