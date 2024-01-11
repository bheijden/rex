from typing import Union
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

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
SUPERGRAPH_MODE = ["MCS", "generational", "topological"]  # envs=10, CPU: MCS=5 (0.5M), gen=10 (0.267M), topo=19 (0.123M). OLD: topo=15, 3.1M, MCS=5, 8.6M
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
RTF = FAST_AS_POSSIBLE
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
NUM_EVAL_PRE = 2
NUM_EVAL_POST = 20
HYPERPARAMS = {"learning_rate": 0.01}
CONTINUE = True
RUN_PER_STEP = True
RUN_ROLLOUTS = True
ROLLOUTS_NUM_ENVS = [1, 3500, 7000]
PLATFORM = ["gpu", "cpu"]

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

# Logging inputs
import os, datetime, itertools, wandb

MUST_LOG = True
os.environ["WANDB_SILENT"] = "true"
DATA_DIR = "/home/r2ci/supergraph/data"
PROJECT = "supergraph"
SYNC_MODE = "offline"
GROUP = f"double-pendulum-evaluation-{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"

SUPERGRAPH_MODE = ["MCS", "generational", "topological"]
ROLLOUTS_NUM_ENVS = [1, 3500]
PLATFORM = ["gpu", "cpu"]
settings = list(itertools.product(PLATFORM, SUPERGRAPH_MODE, ROLLOUTS_NUM_ENVS))  # Right-most is varied first

if MUST_LOG:
    wandb.setup()

for platform, supergraph_type, nenvs  in settings:
    cenv = exp.make_compiled_env(env, record_pre.episode, max_steps=MAX_STEPS, eval_env=False,
                                 supergraph_mode=supergraph_type, progress_bar=True)
    size = len(cenv.graph.S)

    config = dict(nenvs=nenvs, supergraph_type=supergraph_type, platform=platform, size=size, vmap=True)
    print(config)
    if MUST_LOG:
        run = wandb.init(project=PROJECT,
                         job_type=f"double-pendulum-{supergraph_type}",
                         mode=SYNC_MODE,
                         group=GROUP,
                         config=config)
    with jax.default_device(jax.devices(platform)[0]):
            # Evaluate per step jit rollout
            if RUN_PER_STEP:
                # Trace & compile environment
                # cenv = exp.make_compiled_env(env, record_pre.episode, max_steps=MAX_STEPS, eval_env=False,
                #                              supergraph_mode=supergraph_type, progress_bar=True)

                def env_reset(_rng):
                    new_rng, rng_reset = jax.random.split(_rng, 2)
                    gs, obs, info = cenv.reset(rng_reset)
                    return new_rng, gs, obs, info

                def env_step(_rng, _gs, _obs,):
                    new_rng, rng_net = jax.random.split(_rng, 2)
                    action = cenv.action_space().sample(rng_net)
                    new_gs, new_obs, reward, terminated, truncated, info = cenv.step(_gs, action)
                    done = jnp.logical_or(terminated, truncated)
                    return new_rng, new_gs, new_obs, reward, done, info

                # nenvs = ROLLOUTS_NUM_ENVS
                seed = jax.random.PRNGKey(0)
                rng = jax.random.split(seed, num=nenvs)
                env_reset = jax.jit(jax.vmap(env_reset))
                env_step = jax.jit(jax.vmap(env_step))

                for i in range(40):
                    timer = utils.timer(f"Per step | {i=}", log_level=0)
                    rng, gs, obs, info = env_reset(rng)
                    with timer:
                        for j in range(cenv.max_steps):
                            rng, gs, obs, reward, done, info = env_step(rng, gs, obs)
                        reward.block_until_ready()
                    fps = cenv.max_steps * nenvs / timer.duration
                    if MUST_LOG:
                        if i == 0:
                            print(f"[warmup][{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")
                            metrics = {"per_step/jit/fps": fps,
                                       "per_step/jit/time": timer.duration,
                                       "per_step/jit/steps": cenv.max_steps * nenvs}
                        else:
                            print(f"[{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")
                            metrics = {"per_step/computation/fps": fps,
                                       "per_step/computation/time": timer.duration,
                                       "per_step/computation/steps": cenv.max_steps * nenvs}
                        run.log(metrics)
                del env_reset, env_step

            # Evaluate rollouts
            if RUN_ROLLOUTS:
                # # Trace & compile environment
                # cenv = exp.make_compiled_env(env, record_pre.episode, max_steps=MAX_STEPS, eval_env=False, supergraph_mode=SUPERGRAPH_MODE, progress_bar=True)

                # Rollouts
                rw = exp.RolloutWrapper(cenv)

                # Rn rollouts
                # nenvs = ROLLOUTS_NUM_ENVS
                seed = jax.random.PRNGKey(0)
                for i in range(40):
                    seed = jax.random.PRNGKey(i)
                    rng = jax.random.split(seed, num=nenvs)
                    timer = utils.timer(f"Full rollout | {i=}", log_level=0)
                    with timer:
                        if nenvs > 1:
                            res = rw.batch_rollout(rng)
                        else:
                            res = rw.single_rollout(rng[0])
                        res[2].block_until_ready()
                    fps = rw.num_env_steps * nenvs / timer.duration
                    print(f"[{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")
                    if MUST_LOG:
                        if i == 0:
                            print(f"[warmup][{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")
                            metrics = {"rollout/jit/fps": fps,
                                       "rollout/jit/time": timer.duration,
                                       "rollout/jit/steps": cenv.max_steps * nenvs}
                        else:
                            print(f"[{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")
                            metrics = {"rollout/computation/fps": fps,
                                       "rollout/computation/time": timer.duration,
                                       "rollout/computation/steps": cenv.max_steps * nenvs}
                        run.log(metrics)
                del rw, res
            if MUST_LOG:
                run.finish()

exit()

# Prepare data
data = exp.RecordHelper(record_pre)

# Build estimator
from estimator import build_estimator
record, nodes, excludes_inputs = build_estimator(record_pre, rate=RATE_ESTIMATOR, data=data)

# Trace
from rex.supergraph import get_network_record

record_network, S, S_init_to_S, Gs, Gs_monomorphism = get_network_record(record.episode, "estimator",
                                                                         supergraph_mode=SUPERGRAPH_MODE,
                                                                         excludes_inputs=excludes_inputs,
                                                                         progress_bar=True)

# fig_cd, _ = exp.show_computation_graph(Gs[0], S, "estimator", "depth", xmax=2.0, supergraph_mode="MCS")

# Add Estimator node to record
import flax.struct as struct


@struct.dataclass
class Loss:
    loss: Union[float, jax.typing.ArrayLike]
    rloss: Union[float, jax.typing.ArrayLike]
    dloss: Union[float, jax.typing.ArrayLike]


alpha_dloss = nodes["world"].default_state(jax.random.PRNGKey(0)).replace(th=jp.float32(0.e0), th2=jp.float32(0.e0), thdot=jp.float32(0.e-2), thdot2=jp.float32(0.e-3))


def loss_fn(graph_state):
    """Get loss."""
    # Calculate reconstruction loss
    rloss_sensor = graph_state.nodes["sensor"].state.cum_loss
    rloss = rloss_sensor.cos_th + rloss_sensor.sin_th + rloss_sensor.sin_th2 + rloss_sensor.cos_th2
    # rloss += 1e-1*rloss_sensor.thdot
    rloss += 1e-3*rloss_sensor.thdot + 1e-3*rloss_sensor.thdot2

    # Calculate transition loss
    fwd_state = graph_state.nodes["world"].state
    # NOTE: mode="clip" disables negative indexing.
    eps, step = graph_state.eps, graph_state.step
    ws = graph_state.nodes["estimator"].params.world_states
    est_state = jax.tree_map(lambda x: jax.lax.dynamic_slice(x, [eps, step+1] + [0 * s for s in x.shape[2:]], [1, 1] + list(x.shape[2:]))[0, 0], ws)
    dloss = jax.tree_util.tree_map(lambda x, y: (x - y) ** 2, est_state, fwd_state)
    dloss = jax.tree_util.tree_map(lambda e, a: a*e, dloss, alpha_dloss)
    dloss = jax.tree_util.tree_reduce(lambda acc, l: acc + jp.sum(l), dloss, 0.)
    loss = rloss + dloss
    output = Loss(loss=loss, rloss=rloss, dloss=dloss)
    return output

# Compile env
from estimator import EstimatorEnv
from rex.compiled import CompiledGraph

graph = CompiledGraph(nodes, nodes["estimator"], S)
cenv = EstimatorEnv(graph, loss_fn=loss_fn)

# Get initial graph_state
from estimator import init_graph_state

init_gs, outputs = init_graph_state(cenv, nodes, record_network, S, Gs, Gs_monomorphism, data)

# Define initial params
p_tree = jax.tree_util.tree_map(lambda x: None, nodes["world"].default_params(jax.random.PRNGKey(0)))
# p_world = p_tree.replace(mass=jp.float32(0.3), mass2=jp.float32(0.3), K=jp.float32(1.0), J=jp.float32(0.02))
p_world = jax.tree_util.tree_map(lambda x: x * 1.0, p_tree.replace(# J=jp.float32(0.037),
                                                                   # J2=jp.float32(0.000111608131930852),
                                                                   mass=jp.float32(0.18),
                                                                   mass2=jp.float32(0.0691843934004535),
                                                                   # length=jp.float32(0.1),
                                                                   # length2=jp.float32(0.1),
                                                                   b=jp.float32(0.975872107940422),
                                                                   # b2=jp.float32(1.07098956449896e-05),
                                                                   # c=jp.float32(0.06),
                                                                   # c2=jp.float32(0.0185223578523340),
                                                                   K=jp.float32(1.09724557347983),
                                                                   ))
initial_params = {"estimator": init_gs.nodes["estimator"].params,
                  "world": p_world}


# Define prior
def make_prior_fn(guess, multiplier):
    def prior_fn(params):
        loss = jax.tree_util.tree_map(lambda x: None, params)
        if params.get("world", None) is not None:
            wloss = jax.tree_util.tree_map(lambda x, g: 1/(multiplier*(x/g)), params["world"], guess)
            loss["world"] = wloss
        return loss
    return prior_fn


guess = nodes["world"].default_params(jax.random.PRNGKey(0))
prior_fn = make_prior_fn(guess, multiplier=10000)

import optax
optimizer = optax.adam(learning_rate=5e-2)

# Define callbacks
plt.ion()
from estimator.callback import LogCallback, StateFitCallback, ParamFitCallback
targets = nodes["world"].default_params(jax.random.PRNGKey(0))
callbacks = {"log": LogCallback(visualize=True),
             "state_fit": StateFitCallback(visualize=True),
             "param_fit": ParamFitCallback(targets=targets, visualize=True)}

# Optimize
from estimator import fit
metrics, opt_params, opt_state, opt_gs = fit(cenv, initial_params, optimizer, init_gs, outputs,
                                             # num_steps=10, num_batches=50, lr=1e-2 works, 1e-3 thdot.
                                             prior_fn=prior_fn, num_batches=50, num_steps=2, num_training_steps=10_000,
                                             num_training_steps_per_epoch=200, callbacks=callbacks)