import dill as pickle
import tqdm
import functools
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
import datetime

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

import supergraph
import rex
from rex import base, jax_utils as jutils, constants
from rex.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rex.utils import timer
import rex.utils as rutils
from rex.jax_utils import same_structure
from rex import artificial
import envs.pendulum.systems as psys
import envs.pendulum.ppo as pend_ppo
import rex.rl as rl

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    jnp.set_printoptions(precision=5, suppress=True)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])

    # General settings
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    # EXP_DIR = f"{LOG_DIR}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_brax_images"
    EXP_DIR = f"{LOG_DIR}/20240710_141737_brax_norandomization_longerstack_v5_dark" # todo: CHANGE
    # EXP_DIR = f"{LOG_DIR}/20240710_141737_brax_longerstack"
    # EXP_DIR = f"{LOG_DIR}/test_main_eval"
    SEED = 0
    SAVE_FILES = True
    # System identification
    RUN_SYSID = False
    # Train policies
    RUN_RL = False
    RUN_RL_NODELAY = False
    RUN_RL_STACKED = False
    RUN_RL_STACKED_NODELAY = False
    # Evaluate policies
    RUN_REAL = True
    # Trained on full state
    RUN_REAL_NODELAY_NOCAM = True
    RUN_REAL_NODELAY_CAM = True
    RUN_REAL_NODELAY_CAM_NOPRED = True
    # Stacked
    RUN_REAL_STACKED = True
    RUN_REAL_STACKED_NODELAY = True
    RUN_REAL_STACKED_NODELAY_NOCAM = True

    # Environment Settings
    USE_BRAX = True  # Use brax for simulation

    # Sysid settings
    DATA_SYSID_FILE = f"{LOG_DIR}/data_sysid.pkl"
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"  # todo: change to brax
    ID_CAM = False  # Use images to identify camera parameters
    MAX_STEPS = 40

    # RL settings
    RANDOMIZE_EPS = False
    TOTAL_TIMESTEPS = 2_000_000
    EVAL_FREQ = 50  # Evaluate every 50 steps
    INCL_COVARIANCE = False
    DATA_CTRL_FILE = f"{LOG_DIR}/data_control.pkl"
    CONTROLLER_FILE = f"{EXP_DIR}/ctrl_controllers.pkl"  # todo: change to brax
    NUM_POLICIES = 5  # Number of policies to train
    STD_TH_RL = 0.02  # Overwrite std_th in estimator and camera --> None to keep default

    # RL (stacked) settings
    NUM_OBS = 3  # todo: used to be 2
    NUM_ACT = 3  # todo: used to be 2
    CONTROLLER_STACKED_FILE = f"{EXP_DIR}/stacked_controllers.pkl"

    # RL (no delay) settings
    CONTROLLER_STACKED_NO_DELAY_FILE = f"{EXP_DIR}/stacked_nodelay_controllers.pkl"

    # RL (no delay) settings
    CONTROLLER_NO_DELAY_FILE = f"{EXP_DIR}/nodelay_controllers.pkl"

    # Real settings
    NUM_EPISODES = 10  # todo: CHANGE BACK TO 10
    INCLUDE_IMAGES = False  # todo: CHANGE BACK TO False
    TS_MAX = 5.0
    STD_TH_REAL = 0.003  # Overwrite std_th in estimator and camera --> None to keep default
    DIST_FILE = f"{LOG_DIR}/dists.pkl"
    RATES = dict(sensor=50, camera=50, estimator=50, controller=50, actuator=50, supervisor=10, world=100)
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    ORDER = ["camera", "sensor", "world", "actuator", "controller", "estimator", "supervisor"]
    CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
               "actuator": "green", "supervisor": "indigo"}

    # Make directory if it doesn't exist
    if os.path.exists(EXP_DIR):
        print(f"Directory {EXP_DIR} already exists.")
    os.makedirs(EXP_DIR, exist_ok=True)

    # Figures
    figs = []

    # Initialize RNG
    rng = jax.random.PRNGKey(SEED)

    # Sysid
    if RUN_SYSID:
        # Load data
        with open(DATA_SYSID_FILE, "rb") as f:  # Load record
            data_sysid: base.EpisodeRecord = pickle.load(f)
        outputs_sysid = {name: n.steps.output for name, n in data_sysid.nodes.items()}
        # Create system
        nodes_sysid = psys.simulated_system(data_sysid, outputs=outputs_sysid, world_rate=RATES["world"], use_cam=True,
                                            id_cam=ID_CAM, use_brax=USE_BRAX)
        nodes_sysid["supervisor"].set_init_method("parametrized")
        graphs_real = data_sysid.to_graph()
        rng, rng_aug = jax.random.split(rng, num=2)
        graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes_sysid, rng_aug)
        graph_sysid = rex.graph.Graph(nodes_sysid, nodes_sysid["actuator"], graphs_aug, skip=["supervisor"])
        # Initialize state
        rng, rng_init = jax.random.split(rng, num=2)
        gs = graph_sysid.init(rng_init, order=("supervisor", "actuator"))
        # Create task
        import envs.pendulum.tasks as tasks
        task = tasks.create_sysid_task(graph_sysid, gs).replace(max_steps=MAX_STEPS)
        # Jit, lower, precompile
        t_solve_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
        with t_solve_jit:
            task_solve = jax.jit(task.solve)
            with timer("lower", log_level=100):
                task_solve = task_solve.lower(gs)
            with timer("compile", log_level=100):
                task_solve = task_solve.compile()
        # Solve
        t_solve = timer("solve", log_level=100)
        with t_solve:
            sol_state, opt_params, log_state = task_solve(gs)
        params_sysid = task.to_extended_params(gs, opt_params)
        # Store timings
        elapsed_sysid = dict(solve=t_solve.duration, solve_jit=t_solve_jit.duration)
        # Evaluate
        rng, rng_eval = jax.random.split(rng, num=2)
        gs_eval = task.evaluate(params_sysid, rng_eval, -1)
        figs_sysid = task.plot(gs_eval, identifier="opt_sysid")
        # Reduce size
        gs_eval = gs_eval.replace(buffer=None, timings_eps=None)
        # Save
        if SAVE_FILES:
            # Save params
            with open(f"{EXP_DIR}/sysid_params.pkl", "wb") as f:
                pickle.dump(params_sysid, f)
            print(f"Sysid params saved to {EXP_DIR}/sysid_params.pkl")
            # Save data_sysid used for sysid
            with open(f"{EXP_DIR}/sysid_data.pkl", "wb") as f:
                pickle.dump(data_sysid, f)
            print(f"Data_sysid saved to {EXP_DIR}/sysid_data.pkl")
            # Save log_state
            with open(f"{EXP_DIR}/sysid_log_state.pkl", "wb") as f:
                pickle.dump(log_state, f)
            print(f"Log_state saved to {EXP_DIR}/sysid_log_state.pkl")
            # Save sol_state
            with open(f"{EXP_DIR}/sysid_sol_state.pkl", "wb") as f:
                pickle.dump(sol_state, f)
            print(f"Sol_state saved to {EXP_DIR}/sysid_sol_state.pkl")
            # Save
            with open(f"{EXP_DIR}/sysid_elapsed.pkl", "wb") as f:
                pickle.dump(elapsed_sysid, f)
            print(f"Elapsed_sysid saved to {EXP_DIR}/sysid_elapsed.pkl")
            # Save gs_eval
            with open(f"{EXP_DIR}/sysid_gs_eval.pkl", "wb") as f:
                pickle.dump(gs_eval, f)
            print(f"gs_eval saved to {EXP_DIR}/sysid_gs_eval.pkl")
    else:
        print("NOT RUNNING SYSID")
        with open(PARAMS_FILE, "rb") as f:
            params_sysid = pickle.load(f)
        print(f"Sysid params loaded from {PARAMS_FILE}")

    # RL
    if RUN_RL:
        # Load record
        with open(DATA_CTRL_FILE, "rb") as f:  # Load record
            data_ctrl: base.EpisodeRecord = pickle.load(f)
        # Create system
        nodes_train = psys.simulated_system(data_ctrl, world_rate=RATES["world"], use_cam=True, use_brax=USE_BRAX)
        nodes_train["supervisor"].set_init_method("random")  # Set initialization method
        graphs_real = data_ctrl.to_graph()
        if True:  # Exclude sensor if using camera
            sensor = nodes_train.pop("sensor")
            [v.disconnect() for k, v in list(sensor.inputs.items())]
            [v.disconnect() for k, v in list(sensor.outputs.items())]
            graphs_real.vertices.pop("sensor")
            graphs_real.edges.pop(("sensor", "estimator"))
        rng, rng_aug = jax.random.split(rng, num=2)
        graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes_train, rng_aug)
        graph_train = rex.graph.Graph(nodes_train, nodes_train["controller"], graphs_aug)
        # Modify params if necessary
        params_train = params_sysid.copy()
        params_train["estimator"] = params_train["estimator"].replace(std_th=STD_TH_RL)
        params_train["camera"] = params_train["camera"].replace(std_th=STD_TH_RL)
        print(f"[RL] Overwriting std_th to {STD_TH_RL}")
        params_train["controller"] = params_train["controller"].replace(incl_covariance=INCL_COVARIANCE)
        # Create environment
        from envs.pendulum.env import Environment
        env = Environment(graph_train, params=params_train, order=("supervisor", "actuator"), randomize_eps=RANDOMIZE_EPS)
        # Create train function
        import rex.ppo as ppo
        ppo_config = pend_ppo.sweep_pmv2r1zf.replace(TOTAL_TIMESTEPS=TOTAL_TIMESTEPS, EVAL_FREQ=EVAL_FREQ)
        train = functools.partial(ppo.train, env)
        train_v = jax.vmap(train, in_axes=(None, 0))
        train_vjit = jax.jit(train_v)
        rng, rng_ppo = jax.random.split(rng, num=2)
        rngs_policies = jax.random.split(rng_ppo, NUM_POLICIES)
        # Jit, lower, precompile
        t_train_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
        with t_train_jit:
            with timer("lower", log_level=100):
                train_vjit = train_vjit.lower(ppo_config, rngs_policies)
            with timer("compile", log_level=100):
                train_vjit = train_vjit.compile()
        # Train
        t_train = timer("train", log_level=100)
        with t_train:
            ppo_out = train_vjit(ppo_config, rngs_policies)
        # Store timings
        elapsed_ctrl = dict(solve=t_train.duration, solve_jit=t_train_jit.duration)
        # Extract policies
        model_params = ppo_out.policy.model
        act_scaling = ppo_out.act_scaling
        obs_scaling = ppo_out.obs_scaling
        controllers = params_train["controller"].replace(act_scaling=act_scaling, obs_scaling=obs_scaling, model=model_params,
                                                         hidden_activation=ppo_config.HIDDEN_ACTIVATION, stochastic=False)
        controllers = jax.tree_util.tree_map(lambda x: onp.array(x), controllers)
        # Create evaluation system
        nodes_eval = psys.simulated_system(data_ctrl, world_rate=RATES["world"], use_cam=True, use_brax=USE_BRAX)
        if True:  # Exclude sensor if using camera
            sensor = nodes_eval.pop("sensor")
            [v.disconnect() for k, v in list(sensor.inputs.items())]
            [v.disconnect() for k, v in list(sensor.outputs.items())]
        nodes_eval["supervisor"].set_init_method("downward")  # Set initialization method
        params_eval = params_train.copy()
        params_eval["supervisor"] = params_eval["supervisor"].replace(tmax=5.0)
        graph_eval = rex.graph.Graph(nodes_eval, nodes_eval["controller"], graphs_aug)

        # Initialize state
        def evaluate(controller, _rng):
            # Replace controller
            _params = params_eval.copy()
            _params["controller"] = controller
            # Initialize graph state todo: NOT TESTED AFTER graph.rollout REFACTOR
            eps = jnp.arange(NUM_EPISODES) % graph_eval.max_eps
            init_v = jax.vmap(functools.partial(graph_eval.init, rng=_rng, params=_params, order=("supervisor", "actuator")))
            _gs_init = init_v(starting_eps=eps)
            # Evaluate
            _gs_eval = jax.vmap(functools.partial(graph_eval.rollout, carry_only=False))(_gs_init)

            # Replace buffer and timings_eps to save space
            _params = _gs_eval.params.unfreeze()
            _params.pop("controller")
            _gs_eval = _gs_eval.replace(buffer=None, timings_eps=None, params=_params)
            return _gs_eval

        rng, rng_eval = jax.random.split(rng, num=2)
        rngs_eval = jax.random.split(rng_eval, num=NUM_POLICIES)
        eval_fn = jax.jit(jax.vmap(evaluate))
        with timer("eval_fn", log_level=100):
            gs_evals = eval_fn(controllers, rngs_eval)
        # Save
        if SAVE_FILES:
            # Save params
            with open(f"{EXP_DIR}/ctrl_controllers.pkl", "wb") as f:
                pickle.dump(controllers, f)
            print(f"Controller params saved to {EXP_DIR}/ctrl_controllers.pkl")
            # Save data_ctrl used for training
            with open(f"{EXP_DIR}/ctrl_data.pkl", "wb") as f:
                pickle.dump(data_ctrl, f)
            print(f"Data_ctrl saved to {EXP_DIR}/ctrl_data.pkl")
            # Save ppo metrics
            with open(f"{EXP_DIR}/ctrl_ppo_metrics.pkl", "wb") as f:
                pickle.dump(ppo_out["metrics"], f)
            print(f"PPO metrics saved to {EXP_DIR}/ctrl_ppo_metrics.pkl")
            # Save
            with open(f"{EXP_DIR}/ctrl_elapsed.pkl", "wb") as f:
                pickle.dump(elapsed_ctrl, f)
            print(f"Elapsed_ctrl saved to {EXP_DIR}/ctrl_elapsed.pkl")
            # Save gs_evals
            with open(f"{EXP_DIR}/ctrl_gs_evals.pkl", "wb") as f:
                pickle.dump(gs_evals, f)
            print(f"gs_evals saved to {EXP_DIR}/ctrl_gs_evals.pkl")
    else:
        print("NOT RUNNING RL")
        with open(CONTROLLER_FILE, "rb") as f:
            controllers = pickle.load(f)
        print(f"Controllers loaded from {CONTROLLER_FILE}")

    # RL
    if RUN_RL_STACKED:
        # Load record
        with open(DATA_CTRL_FILE, "rb") as f:  # Load record
            data_ctrl: base.EpisodeRecord = pickle.load(f)
        # Create system
        nodes_train = psys.simulated_system(data_ctrl, world_rate=RATES["world"], use_cam=True, use_brax=USE_BRAX, use_ukf=False)
        nodes_train["supervisor"].set_init_method("random")  # Set initialization method
        graphs_real = data_ctrl.to_graph()
        if True:  # Exclude sensor if using camera
            sensor = nodes_train.pop("sensor")
            [v.disconnect() for k, v in list(sensor.inputs.items())]
            [v.disconnect() for k, v in list(sensor.outputs.items())]
            graphs_real.vertices.pop("sensor")
            graphs_real.edges.pop(("sensor", "estimator"))
        rng, rng_aug = jax.random.split(rng, num=2)
        graphs_aug = rex.artificial.augment_graphs(graphs_real, nodes_train, rng_aug)
        graph_train = rex.graph.Graph(nodes_train, nodes_train["controller"], graphs_aug)
        # Modify params if necessary
        params_train = params_sysid.copy()
        params_train["estimator"] = params_train["estimator"].replace(std_th=STD_TH_RL)
        params_train["camera"] = params_train["camera"].replace(std_th=STD_TH_RL)
        print(f"[RL] Overwriting std_th to {STD_TH_RL}")
        params_train["controller"] = params_train["controller"].replace(incl_covariance=False, incl_thdot=False,
                                                                        num_act=NUM_ACT, num_obs=NUM_OBS)
        # Create environment
        from envs.pendulum.env import Environment
        env = Environment(graph_train, params=params_train, order=("supervisor", "actuator"), randomize_eps=RANDOMIZE_EPS)
        # rng, rng_init = jax.random.split(rng)
        # gs = graph_train.init(rng_init, params=params_train, order=("supervisor", "actuator"))
        # print(f"obs_space | shape={env.observation_space(gs).shape}")  # Check observation space
        # Create train function
        import rex.ppo as ppo
        ppo_config = pend_ppo.sweep_pmv2r1zf.replace(TOTAL_TIMESTEPS=TOTAL_TIMESTEPS, EVAL_FREQ=EVAL_FREQ)
        train = functools.partial(ppo.train, env)
        train_v = jax.vmap(train, in_axes=(None, 0))
        train_vjit = jax.jit(train_v)
        rng, rng_ppo = jax.random.split(rng, num=2)
        rngs_policies = jax.random.split(rng_ppo, NUM_POLICIES)
        # Jit, lower, precompile
        t_train_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
        with t_train_jit:
            with timer("lower", log_level=100):
                train_vjit = train_vjit.lower(ppo_config, rngs_policies)
            with timer("compile", log_level=100):
                train_vjit = train_vjit.compile()
        # Train
        t_train = timer("train", log_level=100)
        with t_train:
            ppo_out = train_vjit(ppo_config, rngs_policies)
        # Store timings
        elapsed_ctrl = dict(solve=t_train.duration, solve_jit=t_train_jit.duration)
        # Extract policies
        model_params = ppo_out.policy.model
        act_scaling = ppo_out.act_scaling
        obs_scaling = ppo_out.obs_scaling
        controllers_stacked = params_train["controller"].replace(act_scaling=act_scaling, obs_scaling=obs_scaling, model=model_params,
                                                         hidden_activation=ppo_config.HIDDEN_ACTIVATION, stochastic=False)
        controllers_stacked = jax.tree_util.tree_map(lambda x: onp.array(x), controllers_stacked)
        # Create evaluation system
        nodes_eval = psys.simulated_system(data_ctrl, world_rate=RATES["world"], use_cam=True, use_brax=USE_BRAX, use_ukf=False)
        if True:  # Exclude sensor if using camera
            sensor = nodes_eval.pop("sensor")
            [v.disconnect() for k, v in list(sensor.inputs.items())]
            [v.disconnect() for k, v in list(sensor.outputs.items())]
        nodes_eval["supervisor"].set_init_method("downward")  # Set initialization method
        params_eval = params_train.copy()
        params_eval["supervisor"] = params_eval["supervisor"].replace(tmax=5.0)
        graph_eval = rex.graph.Graph(nodes_eval, nodes_eval["controller"], graphs_aug)

        # Initialize state
        def evaluate(controller, _rng):
            # Replace controller
            _params = params_eval.copy()
            _params["controller"] = controller
            # Initialize graph state todo: NOT TESTED AFTER graph.rollout REFACTOR
            eps = jnp.arange(NUM_EPISODES) % graph_eval.max_eps
            init_v = jax.vmap(functools.partial(graph_eval.init, rng=_rng, params=_params, order=("supervisor", "actuator")))
            _gs_init = init_v(starting_eps=eps)
            # Evaluate
            _gs_eval = jax.vmap(functools.partial(graph_eval.rollout, carry_only=False))(_gs_init)

            # Replace buffer and timings_eps to save space
            _params = _gs_eval.params.unfreeze()
            _params.pop("controller")
            _gs_eval = _gs_eval.replace(buffer=None, timings_eps=None, params=_params)
            return _gs_eval

        rng, rng_eval = jax.random.split(rng, num=2)
        rngs_eval = jax.random.split(rng_eval, num=NUM_POLICIES)
        eval_fn = jax.jit(jax.vmap(evaluate))
        with timer("eval_fn", log_level=100):
            gs_evals = eval_fn(controllers_stacked, rngs_eval)
        # Save
        if SAVE_FILES:
            # Save params
            with open(f"{EXP_DIR}/stacked_controllers.pkl", "wb") as f:
                pickle.dump(controllers_stacked, f)
            print(f"Controller params saved to {EXP_DIR}/stacked_controllers.pkl")
            # Save data_ctrl used for training
            with open(f"{EXP_DIR}/stacked_data.pkl", "wb") as f:
                pickle.dump(data_ctrl, f)
            print(f"Data_ctrl saved to {EXP_DIR}/stacked_data.pkl")
            # Save ppo metrics
            with open(f"{EXP_DIR}/stacked_ppo_metrics.pkl", "wb") as f:
                pickle.dump(ppo_out["metrics"], f)
            print(f"PPO metrics saved to {EXP_DIR}/stacked_ppo_metrics.pkl")
            # Save
            with open(f"{EXP_DIR}/stacked_elapsed.pkl", "wb") as f:
                pickle.dump(elapsed_ctrl, f)
            print(f"Elapsed_ctrl saved to {EXP_DIR}/stacked_elapsed.pkl")
            # Save gs_evals
            with open(f"{EXP_DIR}/stacked_gs_evals.pkl", "wb") as f:
                pickle.dump(gs_evals, f)
            print(f"gs_evals saved to {EXP_DIR}/stacked_gs_evals.pkl")
    else:
        print("NOT RUNNING RL_STACKED")
        with open(CONTROLLER_STACKED_FILE, "rb") as f:
            controllers_stacked = pickle.load(f)
        print(f"Controllers loaded from {CONTROLLER_STACKED_FILE}")

    # RL NODELAY
    if RUN_RL_STACKED_NODELAY:
        nodes_train = psys.no_delay_system(RATES, cscheme=CSCHEME, use_brax=USE_BRAX)
        nodes_train["supervisor"].set_init_method("random")  # Set initialization method
        graphs_gen = artificial.generate_graphs(nodes_train, ts_max=TS_MAX, num_episodes=1)
        graph_train = rex.graph.Graph(nodes_train, nodes_train["controller"], graphs_gen)
        params_train = params_sysid.copy()
        # params_train["estimator"] = params_train["estimator"].replace(std_th=STD_TH_RL)
        # params_train["camera"] = params_train["camera"].replace(std_th=STD_TH_RL)
        # print(f"[RL] Overwriting std_th to {STD_TH_RL}")
        params_train["controller"] = params_train["controller"].replace(incl_covariance=INCL_COVARIANCE, incl_thdot=False,
                                                                        num_act=NUM_ACT, num_obs=NUM_OBS)
        # Create environment
        from envs.pendulum.env import Environment
        env = Environment(graph_train, params=params_train, order=("supervisor", "actuator"), randomize_eps=False)
        # Create train function
        import rex.ppo as ppo
        ppo_config = pend_ppo.sweep_pmv2r1zf.replace(TOTAL_TIMESTEPS=TOTAL_TIMESTEPS, EVAL_FREQ=EVAL_FREQ)
        train = functools.partial(ppo.train, env)
        train_v = jax.vmap(train, in_axes=(None, 0))
        train_vjit = jax.jit(train_v)
        rng, rng_ppo = jax.random.split(rng, num=2)
        rngs_policies = jax.random.split(rng_ppo, NUM_POLICIES)
        # Jit, lower, precompile
        t_train_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
        with t_train_jit:
            with timer("lower", log_level=100):
                train_vjit = train_vjit.lower(ppo_config, rngs_policies)
            with timer("compile", log_level=100):
                train_vjit = train_vjit.compile()
        # Train
        t_train = timer("train", log_level=100)
        with t_train:
            ppo_out = train_vjit(ppo_config, rngs_policies)
        # Store timings
        elapsed_ctrl = dict(solve=t_train.duration, solve_jit=t_train_jit.duration)
        # Extract policies
        model_params = ppo_out.policy.model
        act_scaling = ppo_out.act_scaling
        obs_scaling = ppo_out.obs_scaling
        controllers_stacked_nodelay = params_train["controller"].replace(act_scaling=act_scaling, obs_scaling=obs_scaling, model=model_params,
                                                         hidden_activation=ppo_config.HIDDEN_ACTIVATION, stochastic=False)
        controllers_stacked_nodelay = jax.tree_util.tree_map(lambda x: onp.array(x), controllers_stacked_nodelay)
        # Create evaluation system
        nodes_eval = psys.no_delay_system(RATES, cscheme=CSCHEME, use_brax=USE_BRAX)
        nodes_eval["supervisor"].set_init_method("downward")  # Set initialization method
        params_eval = params_train.copy()
        params_eval["supervisor"] = params_eval["supervisor"].replace(tmax=5.0)
        graph_eval = rex.graph.Graph(nodes_eval, nodes_eval["controller"], graphs_gen)

        # Initialize state
        def evaluate(controller, _rng):
            # Replace controller
            _params = params_eval.copy()
            _params["controller"] = controller
            # Initialize graph state todo: NOT TESTED AFTER graph.rollout REFACTOR
            eps = jnp.arange(NUM_EPISODES) % graph_eval.max_eps
            init_v = jax.vmap(functools.partial(graph_eval.init, rng=_rng, params=_params, order=("supervisor", "actuator")))
            _gs_init = init_v(starting_eps=eps)
            # Evaluate
            _gs_eval = jax.vmap(functools.partial(graph_eval.rollout, carry_only=False))(_gs_init)

            # Replace buffer and timings_eps to save space
            _params = _gs_eval.params.unfreeze()
            _params.pop("controller")
            _gs_eval = _gs_eval.replace(buffer=None, timings_eps=None, params=_params)
            return _gs_eval

        rng, rng_eval = jax.random.split(rng, num=2)
        rngs_eval = jax.random.split(rng_eval, num=NUM_POLICIES)
        eval_fn = jax.jit(jax.vmap(evaluate))
        with timer("eval_fn", log_level=100):
            gs_evals = eval_fn(controllers, rngs_eval)
        # Save
        if SAVE_FILES:
            # Save params
            with open(f"{EXP_DIR}/stacked_nodelay_controllers.pkl", "wb") as f:
                pickle.dump(controllers, f)
            print(f"Controller params saved to {EXP_DIR}/stacked_nodelay_controllers.pkl")
            # Save ppo metrics
            with open(f"{EXP_DIR}/stacked_nodelay_ppo_metrics.pkl", "wb") as f:
                pickle.dump(ppo_out["metrics"], f)
            print(f"PPO metrics saved to {EXP_DIR}/stacked_nodelay_ppo_metrics.pkl")
            # Save
            with open(f"{EXP_DIR}/stacked_nodelay_elapsed.pkl", "wb") as f:
                pickle.dump(elapsed_ctrl, f)
            print(f"Elapsed_ctrl saved to {EXP_DIR}/stacked_nodelay_elapsed.pkl")
            # Save gs_evals
            with open(f"{EXP_DIR}/stacked_nodelay_gs_evals.pkl", "wb") as f:
                pickle.dump(gs_evals, f)
            print(f"gs_evals saved to {EXP_DIR}/stacked_nodelay_gs_evals.pkl")
    else:
        print("NOT RUNNING RL_STACKED_NODELAY")
        with open(CONTROLLER_STACKED_NO_DELAY_FILE, "rb") as f:
            controllers_stacked_nodelay = pickle.load(f)
        print(f"Controllers loaded from {CONTROLLER_STACKED_NO_DELAY_FILE}")

    # RL NODELAY
    if RUN_RL_NODELAY:
        nodes_train = psys.no_delay_system(RATES, cscheme=CSCHEME, use_brax=USE_BRAX)
        nodes_train["supervisor"].set_init_method("random")  # Set initialization method
        graphs_gen = artificial.generate_graphs(nodes_train, ts_max=TS_MAX, num_episodes=1)
        graph_train = rex.graph.Graph(nodes_train, nodes_train["controller"], graphs_gen)
        params_train = params_sysid.copy()
        # params_train["estimator"] = params_train["estimator"].replace(std_th=STD_TH_RL)
        # params_train["camera"] = params_train["camera"].replace(std_th=STD_TH_RL)
        # print(f"[RL] Overwriting std_th to {STD_TH_RL}")
        params_train["controller"] = params_train["controller"].replace(incl_covariance=INCL_COVARIANCE)
        # Create environment
        from envs.pendulum.env import Environment
        env = Environment(graph_train, params=params_train, order=("supervisor", "actuator"), randomize_eps=False)
        # Create train function
        import rex.ppo as ppo
        ppo_config = pend_ppo.sweep_pmv2r1zf.replace(TOTAL_TIMESTEPS=TOTAL_TIMESTEPS, EVAL_FREQ=EVAL_FREQ)
        train = functools.partial(ppo.train, env)
        train_v = jax.vmap(train, in_axes=(None, 0))
        train_vjit = jax.jit(train_v)
        rng, rng_ppo = jax.random.split(rng, num=2)
        rngs_policies = jax.random.split(rng_ppo, NUM_POLICIES)
        # Jit, lower, precompile
        t_train_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
        with t_train_jit:
            with timer("lower", log_level=100):
                train_vjit = train_vjit.lower(ppo_config, rngs_policies)
            with timer("compile", log_level=100):
                train_vjit = train_vjit.compile()
        # Train
        t_train = timer("train", log_level=100)
        with t_train:
            ppo_out = train_vjit(ppo_config, rngs_policies)
        # Store timings
        elapsed_ctrl = dict(solve=t_train.duration, solve_jit=t_train_jit.duration)
        # Extract policies
        model_params = ppo_out.policy.model
        act_scaling = ppo_out.act_scaling
        obs_scaling = ppo_out.obs_scaling
        controllers_nodelay = params_train["controller"].replace(act_scaling=act_scaling, obs_scaling=obs_scaling, model=model_params,
                                                         hidden_activation=ppo_config.HIDDEN_ACTIVATION, stochastic=False)
        controllers_nodelay = jax.tree_util.tree_map(lambda x: onp.array(x), controllers_nodelay)
        # Create evaluation system
        nodes_eval = psys.no_delay_system(RATES, cscheme=CSCHEME, use_brax=USE_BRAX)
        nodes_eval["supervisor"].set_init_method("downward")  # Set initialization method
        params_eval = params_train.copy()
        params_eval["supervisor"] = params_eval["supervisor"].replace(tmax=5.0)
        graph_eval = rex.graph.Graph(nodes_eval, nodes_eval["controller"], graphs_gen)

        # Initialize state
        def evaluate(controller, _rng):
            # Replace controller
            _params = params_eval.copy()
            _params["controller"] = controller
            # Initialize graph state todo: NOT TESTED AFTER graph.rollout REFACTOR
            eps = jnp.arange(NUM_EPISODES) % graph_eval.max_eps
            init_v = jax.vmap(functools.partial(graph_eval.init, rng=_rng, params=_params, order=("supervisor", "actuator")))
            _gs_init = init_v(starting_eps=eps)
            # Evaluate
            _gs_eval = jax.vmap(functools.partial(graph_eval.rollout, carry_only=False))(_gs_init)

            # Replace buffer and timings_eps to save space
            _params = _gs_eval.params.unfreeze()
            _params.pop("controller")
            _gs_eval = _gs_eval.replace(buffer=None, timings_eps=None, params=_params)
            return _gs_eval

        rng, rng_eval = jax.random.split(rng, num=2)
        rngs_eval = jax.random.split(rng_eval, num=NUM_POLICIES)
        eval_fn = jax.jit(jax.vmap(evaluate))
        with timer("eval_fn", log_level=100):
            gs_evals = eval_fn(controllers, rngs_eval)
        # Save
        if SAVE_FILES:
            # Save params
            with open(f"{EXP_DIR}/nodelay_controllers.pkl", "wb") as f:
                pickle.dump(controllers, f)
            print(f"Controller params saved to {EXP_DIR}/nodelay_controllers.pkl")
            # Save ppo metrics
            with open(f"{EXP_DIR}/nodelay_ppo_metrics.pkl", "wb") as f:
                pickle.dump(ppo_out["metrics"], f)
            print(f"PPO metrics saved to {EXP_DIR}/nodelay_ppo_metrics.pkl")
            # Save
            with open(f"{EXP_DIR}/nodelay_elapsed.pkl", "wb") as f:
                pickle.dump(elapsed_ctrl, f)
            print(f"Elapsed_ctrl saved to {EXP_DIR}/nodelay_elapsed.pkl")
            # Save gs_evals
            with open(f"{EXP_DIR}/nodelay_gs_evals.pkl", "wb") as f:
                pickle.dump(gs_evals, f)
            print(f"gs_evals saved to {EXP_DIR}/nodelay_gs_evals.pkl")
    else:
        print("NOT RUNNING RL_NODELAY")
        with open(CONTROLLER_NO_DELAY_FILE, "rb") as f:
            controllers_nodelay = pickle.load(f)
        print(f"Controllers loaded from {CONTROLLER_NO_DELAY_FILE}")

    def real_evaluate(_rng, _controllers, use_cam, use_pred, use_ukf):
        DELAYS_SIM = psys.load_distribution(DIST_FILE)  # psys.get_default_distributions()
        nodes_real = psys.real_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER,
                                      use_cam=use_cam, use_pred=use_pred, use_ukf=use_ukf,
                                      include_image=INCLUDE_IMAGES, use_openloop=False)
        nodes_real["supervisor"].set_init_method("downward")  # Set initialization method
        graph_real = rex.asynchronous.AsyncGraph(nodes_real, supervisor=nodes_real["supervisor"], clock=Clock.WALL_CLOCK,
                                                   real_time_factor=RealTimeFactor.REAL_TIME)
        # Set record settings
        for n, node in graph_real.nodes.items():
            node.set_record_settings(params=True, rng=False, inputs=False, state=True, output=True)
            if n in ["camera", "controller"]:
                node.set_record_settings(state=False)  # Avoid saving cummin, cummax, or network weights.
        # Prepare controllers
        tree_flat, treedef = jax.tree_util.tree_flatten(_controllers)
        num_controllers = tree_flat[0].shape[0]
        controllers_lst = [jax.tree_util.tree_unflatten(treedef, [c[i] for c in tree_flat]) for i in range(num_controllers)]
        # Overwrite params
        params_real = params_sysid.copy()
        params_real["estimator"] = params_real["estimator"].replace(std_th=STD_TH_REAL)
        detector = params_real["camera"].detector.replace(width=nodes_real["camera"]._width,
                                                          height=nodes_real["camera"]._height)
        params_real["camera"] = params_real["camera"].replace(std_th=STD_TH_REAL, detector=detector)
        params_real["controller"] = controllers_lst[0]  # Take first policy
        print(f"[REAL] Overwriting std_th to {STD_TH_REAL}")
        # Init functions
        _rng, rng_init = jax.random.split(_rng, num=2)
        gs_base = graph_real.init(rng_init, params=params_real, order=("supervisor", "actuator"))
        # Jit
        cpu_devices = itertools.cycle(jax.devices('cpu'))
        _ = next(cpu_devices)  # Skip first CPU
        for name, node in graph_real.nodes.items():
            cpu = next(cpu_devices)
            ss = gs_base.step_state[name]
            with timer(f"{name} | jit on {cpu} | lower | compile", log_level=LogLevel.SILENT):
                node.step = jax.jit(node.step, device=cpu).lower(ss).compile()
            with timer(f"{name} | evaluate", log_level=LogLevel.WARN, repeat=10):
                for _ in range(10):
                    ss, o = node.step(ss)
        with timer(f"warmup[dist]", log_level=LogLevel.WARN):
            graph_real.warmup(gs_base)
        # Set logging
        rutils.set_log_level(LogLevel.INFO)
        # Get data
        episodes = []
        success_rates = []
        for idx, ctrl in enumerate(controllers_lst):
            print(f"Controller {idx + 1}/{num_controllers}")
            init_gs = eqx.tree_at(lambda _tree: _tree.params["controller"], gs_base, ctrl)
            for i in range(NUM_EPISODES + 1):
                gs, _ss = graph_real.reset(init_gs)
                num_steps = int(RATES["supervisor"] * TS_MAX) if i > 0 else 1
                for j in tqdm.tqdm(range(num_steps), disable=True):
                    gs, _ss = graph_real.step(gs)
                graph_real.stop()  # Stop environment

                # Get records
                if i > 0:
                    r = graph_real.get_record()
                    episodes.append(r)

                    # Print upright percentage
                    cos_th = jnp.cos(r.nodes["sensor"].steps.output.th)
                    thdot = r.nodes["sensor"].steps.output.thdot
                    is_upright = cos_th > 0.9
                    is_static = jnp.abs(thdot) < 2.0
                    is_valid = jnp.logical_and(is_upright, is_static)
                    success_rate = is_valid.sum() / is_valid.size
                    success_rates.append(success_rate)
                    print(f"Episode {i}: Success rate: {success_rate:.2f} ({success_rate * 100:.2f}%)")

                # Overwrite cummin, cummax for next episode
                cam_state = gs.state["camera"]
                init_cam_state = init_gs.state["camera"].replace(cummin=cam_state.cummin, cummax=cam_state.cummax)
                init_gs = eqx.tree_at(lambda _tree: _tree.state["camera"], init_gs, init_cam_state)

        # Print mean success rate
        mean_success_rate = jnp.mean(jnp.array(success_rates))
        print(f"Mean success rate: {mean_success_rate:.2f} ({mean_success_rate * 100:.2f}%)")
        record = base.ExperimentRecord(episodes=episodes)
        # Shutdown camera pipeline (so that we can reinitialize it later)
        nodes_real["camera"]._shutdown()
        return record

    # REAL
    if RUN_REAL:
        print("RUNNING REAL")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers, use_cam=True, use_pred=True, use_ukf=True)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/real_data.pkl")
    else:
        print("NOT RUNNING REAL")

    # REAL STACKED
    if RUN_REAL_STACKED:
        print("RUNNING RUN_REAL_STACKED")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers_stacked, use_cam=True, use_pred=True, use_ukf=False)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/stacked_real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/stacked_real_data.pkl")
    else:
        print("NOT RUNNING RUN_REAL_STACKED")

    # REAL RUN_REAL_STACKED_NODELAY
    if RUN_REAL_STACKED_NODELAY:
        print("RUNNING RUN_REAL_STACKED_NODELAY")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers_stacked_nodelay, use_cam=True, use_pred=True, use_ukf=False)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/stacked_nodelay_real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/stacked_nodelay_real_data.pkl")
    else:
        print("NOT RUNNING RUN_REAL_STACKED_NODELAY")

    # REAL RUN_REAL_STACKED_NODELAY_NOCAM
    if RUN_REAL_STACKED_NODELAY_NOCAM:
        print("RUNNING RUN_REAL_STACKED_NODELAY_NOCAM")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers_stacked_nodelay, use_cam=False, use_pred=True, use_ukf=False)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/stacked_nodelay_nocam_real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/stacked_nodelay_nocam_real_data.pkl")
    else:
        print("NOT RUNNING RUN_REAL_STACKED_NODELAY_NOCAM")

    # REAL RUN_REAL_NODELAY_NOCAM
    if RUN_REAL_NODELAY_NOCAM:
        print("RUNNING RUN_REAL_NODELAY_NOCAM")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers_nodelay, use_cam=False, use_pred=False, use_ukf=False)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/nodelay_nocam_real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/nodelay_nocam_real_data.pkl")
    else:
        print("NOT RUNNING REAL")

    # REAL RUN_REAL_NODELAY_CAM
    if RUN_REAL_NODELAY_CAM:
        print("RUNNING RUN_REAL_NODELAY_CAM")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers_nodelay, use_cam=True, use_pred=True, use_ukf=True)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/nodelay_cam_real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/nodelay_cam_real_data.pkl")

    # REAL RUN_REAL_NODELAY_CAM
    if RUN_REAL_NODELAY_CAM_NOPRED:
        print("RUNNING RUN_REAL_NODELAY_CAM_NOPRED")
        rng, rng_real = jax.random.split(rng, num=2)
        record = real_evaluate(rng_real, controllers_nodelay, use_cam=True, use_pred=False, use_ukf=True)
        if SAVE_FILES:
            with open(f"{EXP_DIR}/nodelay_cam_nopred_real_data.pkl", "wb") as f:
                pickle.dump(record, f)
            print(f"Real record saved to {EXP_DIR}/nodelay_cam_nopred_real_data.pkl")


