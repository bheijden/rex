from collections import deque
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
import rex
from rex import base, jax_utils as jutils, constants
from rex.constants import Clock, RealTimeFactor, Scheduling, LogLevel
from rex.utils import timer
import rex.utils as rutils
from rex.jax_utils import same_structure
import envs.crazyflie.systems as csys
from envs.crazyflie.ode import plot_data

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


def _plot_inclined(mocap, pid, platform, ts_mocap, ts_pid, ts_plat):
    # done = rollout.done.reshape(-1)
    # ode_state = rollout.next_gs.nodes["mocap"].inputs["world"][:, -1].data
    # ctrl = rollout.next_gs.nodes["world"].inputs["attitude"][:, -1].data
    # ctrl = jax.tree_util.tree_map(lambda x: onp.where(done, onp.nan, onp.array(x)), ctrl)
    # att = onp.where(done[:, None], onp.nan, onp.array(ode_state.att))
    # pos = onp.where(done[:, None], onp.nan, onp.array(ode_state.pos))
    # vel = onp.where(done[:, None], onp.nan, onp.array(ode_state.vel))
    pos = mocap.pos
    pos_plat = platform.pos
    vel = mocap.vel
    vel_plat = platform.vel
    att = mocap.att
    att_plat = platform.att

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    if pid.pwm_ref is not None:
        axes[0, 0].plot(ts_pid, pid.pwm_ref, label="pwm", color="blue")
        axes[0, 0].legend()
        axes[0, 0].set_title("PWM")
    else:
        # Turn off axes
        axes[0, 0].axis("off")

    axes[0, 1].plot(ts_mocap, att[:, 0], label="cf", color="blue")
    axes[0, 1].plot(ts_plat, att_plat[:, 0], label="plat", color="green")
    axes[0, 1].plot(ts_pid, pid.phi_ref, label="ref", color="blue", linestyle="--")
    axes[0, 1].legend()
    axes[0, 1].set_title("phi")

    axes[0, 2].plot(ts_mocap, att[:, 1], label="cf", color="blue")
    axes[0, 2].plot(ts_plat, att_plat[:, 1], label="plat", color="green")
    axes[0, 2].plot(ts_pid, pid.theta_ref, label="ref", color="blue", linestyle="--")
    axes[0, 2].legend()
    axes[0, 2].set_title("theta")

    axes[1, 0].plot(ts_mocap, pos[:, 0], label="cf", color="blue")
    axes[1, 0].plot(ts_plat, pos_plat[:, 0], label="plat", color="green")
    axes[1, 0].legend()
    axes[1, 0].set_title("X")
    axes[1, 1].plot(ts_mocap, pos[:, 1], label="cf", color="blue")
    axes[1, 1].plot(ts_plat, pos_plat[:, 1], label="plat", color="green")
    axes[1, 1].legend()
    axes[1, 1].set_title("Y")
    axes[1, 2].plot(ts_mocap, pos[:, 2], label="cf", color="blue")
    axes[1, 2].plot(ts_plat, pos_plat[:, 2], label="plat", color="green")
    axes[1, 2].plot(ts_pid, pid.z_ref, label="ref", color="blue", linestyle="--") if pid.z_ref is not None else None
    axes[1, 2].legend()
    axes[1, 2].set_title("Z")

    axes[2, 0].plot(ts_mocap, vel[:, 0], label="cf", color="blue")
    axes[2, 0].plot(ts_plat, vel_plat[:, 0], label="plat", color="green")
    axes[2, 0].legend()
    axes[2, 0].set_title("Vx")
    axes[2, 1].plot(ts_mocap, vel[:, 1], label="cf", color="blue")
    axes[2, 1].plot(ts_plat, vel_plat[:, 1], label="plat", color="green")
    axes[2, 1].legend()
    axes[2, 1].set_title("Vy")
    axes[2, 2].plot(ts_mocap, vel[:, 2], label="cf", color="blue")
    axes[2, 2].plot(ts_plat, vel_plat[:, 2], label="plat", color="green")
    axes[2, 2].plot(ts_pid, pid.z_ref, label="ref", color="blue", linestyle="--") if pid.z_ref is not None else None
    axes[2, 2].legend()
    axes[2, 2].set_title("Vz")
    return fig, axes


if __name__ == "__main__":
    # todo: increase rates (especially checking for has_landed)
    # todo: add another marker (glue to legs, on top of rotor)
    # todo: Re-measure POS_OFFSET
    #   - From vicon to xml center (probably top mocap board to bottom of control board)
    #   - Tune threshold. eagerx used 0.03 m.
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu'))
    CPU_DEVICE = next(CPU_DEVICES)
    SEED = 0
    ORDER = ["mocap", "world", "pid", "agent", "estimator", "platform"]
    CSCHEME = {"world": "gray", "mocap": "grape", "estimator": "violet", "agent": "lime", "pid": "green", "actuator": "indigo", "platform": "grape"}
    RATES = dict(mocap=50, world=100, pid=80, agent=50, estimator=50, supervisor=10, platform=80)
    POSITION = onp.array([2.0, 0.3, 0.3])
    # Settings
    MOCK = "real"  # "ode", "copilot, or "real" else todo: "real"
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    # EXP_DIR = f"{LOG_DIR}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_10Hz"
    EXP_DIR = f"{LOG_DIR}/inclined_debug"  # todo: CHANGE
    # DELAYS_SIM = csys.load_distribution(f"{EXP_DIR}/dists.pkl") # TODO: CHANGE
    DELAYS_SIM = csys.load_distribution(f"{LOG_DIR}/dists.pkl")
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    MODE = "sysid"  # todo: CHANGE
    if MODE == "delay_only":
        NUM_EPISODES = 5
        TSIM = 10
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
        RECORD_FILE = f"{EXP_DIR}/data_delay_only.pkl"
        FIG_FILE = None
        FEEDTHROUGH = False
    elif MODE == "sysid":
        NUM_EPISODES = 10
        TSIM = 6
        PARAMS_FILE = f"{LOG_DIR}/inclined_debug/sysid_params.pkl"
        AGENT_FILE = f"{LOG_DIR}/inclined_debug/agent_params.pkl"
        RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"
        FIG_FILE = f"{EXP_DIR}/sysid_fig.png"
        FEEDTHROUGH = True  # todo: CHANGE
    elif MODE == "evaluate":
        TSIM = 15
        NUM_EPISODES = 1
        PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"
        AGENT_FILE = f"{EXP_DIR}/agent_params.pkl"
        RECORD_FILE = f"{EXP_DIR}/eval_data.pkl"
        FIG_FILE = f"{EXP_DIR}/eval_fig.png"
        FEEDTHROUGH = True
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    # Seed
    rng = jax.random.PRNGKey(SEED)

    # Make directory if it doesn't exist
    if os.path.exists(EXP_DIR):
        print(f"Directory {EXP_DIR} already exists.")
    os.makedirs(EXP_DIR, exist_ok=True)

    # Create graph
    if MOCK == "ode":
        POS_OFFSET = onp.array([0.0, 0.0, 0.0])  # From top mocap board to control board (corrects vicon position to cf center in xml)
        THRESHOLD = 0.0
        CLOCK = Clock.SIMULATED
        REAL_TIME_FACTOR = RealTimeFactor.FAST_AS_POSSIBLE
        nodes = csys.mock_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER, inclined_landing=True)
    else:
        POS_OFFSET = onp.array([0.0, 0.0, -0.018])  # From top mocap board to control board (corrects vicon position to cf center in xml)
        THRESHOLD = 0.03
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
        print(f"POS_OFFSET={POS_OFFSET}, THRESHOLD={THRESHOLD}")
        nodes = csys.real_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER, mock_copilot=MOCK_COPILOT, feedthrough=FEEDTHROUGH, inclined_landing=True)

    # Create graph
    print(f"MOCK={MOCK}, CLOCK={CLOCK}, REAL_TIME_FACTOR={REAL_TIME_FACTOR}")
    graph = rex.asynchronous.AsyncGraph(nodes, supervisor=nodes["agent"], clock=CLOCK, real_time_factor=REAL_TIME_FACTOR)

    # Initialize graph state
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        rng, rng_init = jax.random.split(rng)
        gs_init = graph.init(rng_init, order=("agent", "pid"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
        if "agent" in params:
            params.pop("agent")
    else:
        params = gs_init.params.unfreeze()
        print(f"Params not found at {PARAMS_FILE}")
        if "agent" in params:
            params.pop("agent")

    # Load trained params (if file exists)
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "rb") as f:
            agent_params = pickle.load(f)
        print(f"Agent params loaded from {AGENT_FILE}")
        params["agent"] = gs_init.params["agent"].replace(**agent_params.__dict__)
    else:
        # Add agent params
        rng, rng_agent = jax.random.split(rng)
        params["agent"] = nodes["agent"].init_params(rng_agent)
        params["agent"] = params["agent"].replace(
            init_cf="fixed",
            fixed_position=POSITION,
            pos_offset=POS_OFFSET,
            contact_threshold=THRESHOLD,
            init_plat="fixed",
            use_noise=True,
            use_dr=True,
        )
        print(f"Agent params not found at {AGENT_FILE}")

    # todo: DEBUG
    # AGENT_FILE_DEBUG = f"{EXP_DIR}/main_crazyflie_ss_agent.pkl"
    AGENT_FILE_DEBUG = f"/home/r2ci/rex/scratch/crazyflie/logs/ss_agent_movingSpot.pkl"
    assert os.path.exists(AGENT_FILE_DEBUG), "Agent file not found"
    with open(AGENT_FILE_DEBUG, "rb") as f:
        ss_agent = pickle.load(f)

    # todo: save agent params
    params["agent"] = params["agent"].replace(
        act_scaling=ss_agent.params.act_scaling,
        obs_scaling=ss_agent.params.obs_scaling,
        model=ss_agent.params.model,
        hidden_activation=ss_agent.params.hidden_activation,
        output_activation=ss_agent.params.output_activation,
        stochastic=ss_agent.params.stochastic,
        action_dim=ss_agent.params.action_dim,
        mapping=ss_agent.params.mapping,
    )
    # todo: END DEBUG

    # Replace params in gs
    rng, rng_init = jax.random.split(rng)
    gs_init = graph.init(rng_init, order=("agent", "pid"), params=params)

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

    # Simulate
    episodes = []
    with jax.log_compiles():
        for i in range(NUM_EPISODES + 1):
            landed_queue = deque(maxlen=int(0.1*RATES["agent"]+1))
            gs, _ss = graph.reset(gs_init)
            num_steps = int(RATES["agent"]*TSIM)if i > 0 else 1  # Warmup
            for j in tqdm.tqdm(range(num_steps), disable=True):
                gs, _ss = graph.step(gs)
                # Once landed, add to queue
                has_landed = _ss.state.has_landed
                if has_landed or len(landed_queue) > 0:
                    landed_queue.append(has_landed)
                    print(f"main | Queue: {len(landed_queue)}")
                # Break after queue is full
                if len(landed_queue) == landed_queue.maxlen:
                    break
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
                    if "platform" in r.nodes and "world" in r.nodes["platform"].inputs:
                        r.nodes["platform"].inputs.pop("world")
                        r.nodes["platform"].info.inputs.pop("world")

                # Append record
                episodes.append(r)

                # Print upright percentage
                print(f"Episode {i} | has_landed={bool(has_landed)} | ts={_ss.ts: .2f} sec")
            else:
                print(f"Warmup episode {i} |")

    # Save record
    record = base.ExperimentRecord(episodes=episodes)
    graphs_raw = record.to_graph()

    # Visualize the graph
    if False:
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
    platform = record.episodes[0].nodes["platform"].steps
    mocap = record.episodes[0].nodes["mocap"].steps
    pid = record.episodes[0].nodes["pid"].steps
    fig, axes = _plot_inclined(mocap.output, pid.output, platform.output,
                               mocap.ts_start, pid.ts_start, platform.ts_start)
    plt.show()

    # Save
    stacked = record.stack("padded")
    with open(RECORD_FILE, "wb") as f:
        pickle.dump(stacked, f)
    print(f"Data saved to {RECORD_FILE}")
    # with open(f"{EXP_DIR}/eval_params.pkl", "wb") as f:
    #     pickle.dump(params, f)
    # print(f"Params saved to {EXP_DIR}/sysid_params.pkl")
    # with open(f"{EXP_DIR}/sysid_agent_params.pkl", "wb") as f:
    #     pickle.dump(agent_params, f)
    # print(f"Agent params saved to {EXP_DIR}/agent_params.pkl")
    if FIG_FILE is not None:
        fig.savefig(FIG_FILE)
        print(f"Fig saved to {FIG_FILE}")
