import datetime
from typing import Dict, Union, Callable, Any
import dill as pickle
import tqdm
import os
import multiprocessing
import itertools
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import distrax

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
import envs.crazyflie.systems as csys
from envs.crazyflie.ode import plot_data

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    # todo: OLD
    # todo: stream vicon quaternions?
    # todo: Save best policy
    # todo: Properly handle truncated episodes (record terminal observation)

    # todo:
    #   - Reference tracking controller (how to deal with yaw offset?)
    #   - Single device (to increase latency), no domain randomization
    # todo: Real
    #   - Platform KF
    # todo: RL
    #   - Turn off Domain randomization and increase latency (single device?)
    #   - Reduce training time
    #   - If pre-loading params, ensure that supervisor params are changed in main_rl.py etc...
    #       - use_noise=False, use_dr=False are copied from sysid_params, while they should be True.
    #   - Reference tracking instead of path following? Track path variable instead of tangent speed? Then psi matters...
    # todo: Sysid
    #   - Enable yaw in dynamics

    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu'))
    CPU_DEVICE = next(CPU_DEVICES)
    RNG = jax.random.PRNGKey(0)
    ORDER = ["mocap", "world", "pid", "agent", "estimator", "platform"]
    CSCHEME = {"world": "gray", "mocap": "grape", "estimator": "violet", "agent": "lime", "pid": "green", "actuator": "indigo", "platform": "grape"}
    RATES = dict(mocap=50, world=100, pid=50, agent=25, estimator=25, supervisor=10, platform=25)
    POSITION = onp.array([0.0, 0.0, 1.0])
    CENTER = onp.array([0.0, 0.0, 1.0])
    RADIUS = 1.0
    SKIP_SEC = 3.0
    # Settings
    MOCK = "real"  # "ode", "copilot, or "real" else todo: "real"
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    # EXP_DIR = f"{LOG_DIR}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_10Hz"
    EXP_DIR = f"{LOG_DIR}/20240813_142721_no_zref_eval_sysid_retry_eval"  # todo: CHANGE
    # DELAYS_SIM = csys.load_distribution(f"{EXP_DIR}/dists.pkl") # TODO: CHANGE
    DELAYS_SIM = csys.load_distribution(f"{LOG_DIR}/dists.pkl")
    # DELAYS_SIM = csys.get_default_distributions(nodes=ORDER)
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    MODE = "sysid"  # todo: CHANGE
    if MODE == "delay_only":
        NUM_EPISODES = 5
        TSIM = 10
        # INPUT
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
        # OUTPUT
        RECORD_FILE = f"{EXP_DIR}/data_delay_only.pkl"
        SAVE_AGENT_FILE = f"{EXP_DIR}/delay_only_agent_params.pkl"
        FIG_FILE = None
        FEEDTHROUGH = False
    elif MODE == "sysid":
        NUM_EPISODES = 1
        TSIM = 15
        # INPUT
        PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"
        AGENT_FILE = f"{EXP_DIR}/delay_agent_params.pkl"  # todo: CHANGE
        # AGENT_FILE = f"{EXP_DIR}/nodelay_agent_params.pkl" # todo: CHANGE
        # OUTPUT
        RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"
        FIG_FILE = f"{EXP_DIR}/sysid_fig.png"
        SAVE_AGENT_FILE = f"{EXP_DIR}/sysid_agent_params.pkl"
        FEEDTHROUGH = True
    elif MODE == "evaluate":
        TSIM = 15
        NUM_EPISODES = 1
        # INPUT
        PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"
        AGENT_FILE = f"{EXP_DIR}/eval_agent_params.pkl" # todo: CHANGE
        # OUTPUT
        RECORD_FILE = f"{EXP_DIR}/eval_data.pkl"
        FIG_FILE = f"{EXP_DIR}/eval_fig.png"
        SAVE_AGENT_FILE = f"{EXP_DIR}/eval_agent_params.pkl"
        FEEDTHROUGH = True
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    # Make directory if it doesn't exist
    if os.path.exists(EXP_DIR):
        print(f"Directory {EXP_DIR} already exists.")
    os.makedirs(EXP_DIR, exist_ok=True)

    # Create graph
    if MOCK == "ode":
        CLOCK = Clock.SIMULATED
        REAL_TIME_FACTOR = RealTimeFactor.FAST_AS_POSSIBLE
        nodes = csys.mock_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER)
    else:
        if MOCK == "real":
            MOCK_COPILOT = False
            CLOCK = Clock.WALL_CLOCK
            REAL_TIME_FACTOR = RealTimeFactor.REAL_TIME
        elif MOCK == "copilot":
            MOCK_COPILOT = True
            CLOCK = Clock.WALL_CLOCK
            REAL_TIME_FACTOR = RealTimeFactor.REAL_TIME
        else:
            raise ValueError(f"Invalid mock system: {MOCK}")
        nodes = csys.real_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER, mock_copilot=MOCK_COPILOT, feedthrough=FEEDTHROUGH)

    # Create graph
    print(f"MOCK={MOCK}, CLOCK={CLOCK}, REAL_TIME_FACTOR={REAL_TIME_FACTOR}")
    graph = rexv2.asynchronous.AsyncGraph(nodes, supervisor=nodes["agent"], clock=CLOCK, real_time_factor=REAL_TIME_FACTOR)

    # Initialize graph state
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs_init = graph.init(RNG, order=("agent", "pid"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
    else:
        params = gs_init.params.unfreeze()
        print(f"Params not found at {PARAMS_FILE}")

    # Load trained params (if file exists)
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "rb") as f:
            agent_params = pickle.load(f)
        print(f"Agent params loaded from {AGENT_FILE}")
        params["agent"] = gs_init.params["agent"].replace(**agent_params.__dict__)
    else:
        # print(f"Agent params not found at {AGENT_FILE}")
        raise FileNotFoundError(f"Agent params not found at {AGENT_FILE}")

    # Modify supervisor params
    params["agent"] = params["agent"].replace(
        init_cf="fixed",
        fixed_position=POSITION,
        init_path="fixed",
        fixed_radius=RADIUS,
        center=CENTER,
    )

    # Replace params in gs
    gs_init = graph.init(RNG, order=("agent", "pid"), params=params)

    # Warmup devices
    rutils.set_log_level(LogLevel.WARN)
    [rutils.set_log_level(LogLevel.WARN, n) for n in graph.nodes.values()]  # Set log level
    devices_step = {k: next(CPU_DEVICES) if k != "agent" else GPU_DEVICE for k in ORDER}
    devices_dist = devices_step.copy()
    graph.warmup(gs_init, devices_step, devices_dist, jit_step=True, profile=False)

    # Set record settings
    for n, node in graph.nodes.items():
        node.set_record_settings(params=True, rng=False, inputs=False, state=True, output=True)
        # if n in ["agent"]:
        #     node.set_record_settings(state=False)

    # Warmup & get initial graph state
    import logging
    logging.getLogger("jax").setLevel(logging.INFO)

    # Evaluate
    from envs.crazyflie.ode import metrics

    start_eval = int(SKIP_SEC * RATES["mocap"])
    end_eval = start_eval + int((TSIM - SKIP_SEC) * RATES["mocap"] * 0.95)
    dummy_mocap = jax.vmap(nodes["mocap"].init_output, in_axes=(0, None))(jax.random.split(RNG, end_eval - start_eval), gs_init)
    metrics_jv = jax.jit(jax.vmap(metrics, in_axes=(0, None, None))).lower(dummy_mocap, RADIUS, CENTER).compile()
    vel_on, vel_off, pos_on, pos_off = metrics_jv(dummy_mocap, RADIUS, CENTER)
    vel_on.mean(), vel_off.mean(), pos_on.mean(), pos_off.mean()

    # Simulate
    episodes = []
    with jax.log_compiles():
        for i in range(NUM_EPISODES + 1):
            gs, _ss = graph.reset(gs_init)
            num_steps = int(RATES["agent"]*TSIM)if i > 0 else 1  # Warmup
            for j in tqdm.tqdm(range(num_steps), disable=True):
                gs, _ss = graph.step(gs)
            graph.stop()  # Stop environment

            # Get records (skip first episode)
            if i > 0:
                r = graph.get_record()

                # Pop world (if using mock system)
                if "world" in r.nodes:
                    r.nodes.pop("world")
                    r.nodes["mocap"].inputs.pop("world")
                    r.nodes["mocap"].info.inputs.pop("world")
                    r.nodes["pid"].inputs.pop("world")
                    r.nodes["pid"].info.inputs.pop("world")

                # Append record
                episodes.append(r)

                # Calculate statistics
                mocap = r.nodes["mocap"].steps.output[start_eval:end_eval]
                vel_on, vel_off, pos_on, pos_off = metrics_jv(mocap, RADIUS, CENTER)

                # Print upright percentage
                print(f"Episode {i} | vel_on={vel_on.mean():.2f} | vel_off={vel_off.mean():.2f} | pos_off={pos_off.mean():.2f}")

    # Save record
    record = base.ExperimentRecord(episodes=episodes)
    graphs_raw = record.to_graph()

    # Visualize the graph
    MAX_X = 1.0
    Gs = [rutils.to_networkx_graph(g, nodes=nodes, validate=True) for g in graphs_raw]
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for i, G in enumerate(Gs):
        if i > 1:
            break
        supergraph.plot_graph(G, max_x=MAX_X, ax=axs[i])
        axs[i].set_title(f"Episode {i}")
        axs[i].set_xlabel("Time [s]")

    # Plot
    mocap = record.episodes[0].nodes["mocap"].steps
    mocap_ia = jax.vmap(mocap.output.static_in_agent_frame, in_axes=(0, None))(mocap.output, CENTER)
    estimator = record.episodes[0].nodes["estimator"].steps
    agent = record.episodes[0].nodes["agent"].steps
    pid = record.episodes[0].nodes["pid"].steps

    fig, axes = plot_data(output={"att": mocap.output.att,
                                  "pos": mocap.output.pos,
                                  "pos_ia": mocap_ia.pos,
                                  "vel_ia": mocap_ia.vel,
                                  "pwm_ref": pid.output.pwm_ref,
                                  "phi_ref": pid.output.phi_ref,
                                  "theta_ref": pid.output.theta_ref,
                                  "z_ref": pid.output.z_ref},
                          ts={"att": mocap.ts_start,
                              "pos": mocap.ts_start,
                              "pos_ia": mocap.ts_start,
                              "vel_ia": mocap.ts_start,
                              "pwm_ref": pid.ts_end,
                              "phi_ref": pid.ts_end,
                              "theta_ref": pid.ts_end,
                              "z_ref": pid.ts_end},
                          # ts_max=3.0,
                          )
    plt.show()    # Save data

    # Save
    stacked = record.stack("padded")
    with open(RECORD_FILE, "wb") as f:
        pickle.dump(stacked, f)
    print(f"Data saved to {RECORD_FILE}")
    # with open(f"{EXP_DIR}/eval_params.pkl", "wb") as f:
    #     pickle.dump(params, f)
    # print(f"Params saved to {EXP_DIR}/sysid_params.pkl")
    with open(SAVE_AGENT_FILE, "wb") as f:
        pickle.dump(agent_params, f)
    print(f"Agent params saved to {SAVE_AGENT_FILE}")
    if FIG_FILE is not None:
        fig.savefig(FIG_FILE)
        print(f"Fig saved to {FIG_FILE}")
