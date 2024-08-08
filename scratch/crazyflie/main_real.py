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
    # todo: CF is unstable when pitching & rolling simultaneously
    #  - Penalize abrupt changes in both roll & pitch behavior
    #  - LPF Roll/pitch references.
    # todo: stream vicon quaternions?
    # todo: Save best policy
    # todo: Properly handle truncated episodes (record terminal observation)

    # todo: NEW
    #   - Z oscillations probably due to steady-state offset and penalty on (z-z_ref)?
    #   - Why is circle origin not working--> crazyflie/ode.py clipping in World.step
    #   - Fix vel_on reference?
    #   - Fix z_ref to desired z height? PIDParams.to_command for testing.
    #   - Implement real system.
    #   - If pre-loading params, make sure they match dtype of dtypes returned by .step. Else recompilation....
    #   - crazyflie.ode
    #       - Mocap noise off when sysid'ing
    #   - crazyflie.supervisor
    #       - Make sure to use fixed state (set it to recorded initial position).
    #       - turn off noise when sysid'ing
    #       - add (mapping, pwm_range, mass, pwm_constants, hover_pwm, ...)
    #   - crazyflie.pid
    #       - pwm_constants instead of pwm_base as params
    #   - crazyflie.estimator

    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu'))
    CPU_DEVICE = next(CPU_DEVICES)
    RNG = jax.random.PRNGKey(0)
    LOG_DIR = "/home/r2ci/rex/scratch/crazyflie/logs"
    ORDER = ["mocap", "world", "pid", "agent", "estimator", "supervisor"]
    CSCHEME = {"world": "gray", "mocap": "grape", "estimator": "violet", "agent": "lime", "pid": "green", "actuator": "indigo", "supervisor": "gray"}
    MOCK = "ode"  # "ode", "copilot, or "real" else todo: "real"
    CLOCK = Clock.SIMULATED  # todo: WALL_CLOCK?
    REAL_TIME_FACTOR = RealTimeFactor.REAL_TIME  # todo: REAL_TIME?
    CENTER = onp.array([0.0, 0.0, 1.5])
    RADIUS = 1.5
    MODE = "sysid"  # "delay_only", "sysid", "rl"
    if MODE == "delay_only":
        NUM_EPISODES = 5
        TSIM = 10
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_delay_only.pkl"
        AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
        FEEDTHROUGH = False
        # USE_OPENLOOP = False
    elif MODE == "sysid":
        NUM_EPISODES = 1
        TSIM = 19
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_sysid.pkl"
        AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
        FEEDTHROUGH = True
        # USE_OPENLOOP = True
    elif MODE == "rl":
        TSIM = 10
        NUM_EPISODES = 1
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_rl.pkl"
        AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
        FEEDTHROUGH = True
        # USE_OPENLOOP = False
    elif MODE == "evaluate":
        TSIM = 5
        NUM_EPISODES = 10
        PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"
        RECORD_FILE = f"{LOG_DIR}/data_evaluate.pkl"
        AGENT_FILE = f"{LOG_DIR}/agent_params.pkl"
        FEEDTHROUGH = True
        # USE_OPENLOOP = False
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    DELAYS_SIM = csys.load_distribution(f"{LOG_DIR}/dists.pkl")  # csys.get_default_distributions(nodes=ORDER)
    # DELAYS_SIM = csys.get_default_distributions(nodes=ORDER)  # csys.get_default_distributions(nodes=ORDER)
    DELAY_FN = lambda d: d.quantile(0.85)  # lambda d: 0.0
    RATES = dict(mocap=50, world=50, pid=50, agent=25, estimator=25, supervisor=10)

    # Create graph
    if MOCK =="ode":
        CLOCK = Clock.SIMULATED
        REAL_TIME_FACTOR = RealTimeFactor.FAST_AS_POSSIBLE
        nodes = csys.mock_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER)
    else:
        if MOCK == "real":
            MOCK_COPILOT = False
        elif MOCK == "copilot":
            MOCK_COPILOT = True
        else:
            raise ValueError(f"Invalid mock system: {MOCK}")
        nodes = csys.real_system(DELAYS_SIM, DELAY_FN, RATES, cscheme=CSCHEME, order=ORDER, mock_copilot=MOCK_COPILOT, feedthrough=FEEDTHROUGH)

    # Create graph
    graph = rexv2.asynchronous.AsyncGraph(nodes, supervisor=nodes["supervisor"], clock=CLOCK, real_time_factor=REAL_TIME_FACTOR)

    # Initialize graph state
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs_init = graph.init(RNG, order=("supervisor", "pid"))

    # Load params (if file exists)
    if os.path.exists(PARAMS_FILE):
        raise NotImplementedError("Make sure they match dtype of dtypes returned by .step. Else recompilation....")
        with open(PARAMS_FILE, "rb") as f:
            params: Dict[str, Any] = pickle.load(f)
        print(f"Params loaded from {PARAMS_FILE}")
    else:
        params = gs_init.params.unfreeze()
        print(f"Params not found at {PARAMS_FILE}")

    # Modify supervisor params
    params["supervisor"] = params["supervisor"].replace(
        init_cf="fixed",
        init_path="fixed",
        fixed_radius=RADIUS,
        center=CENTER,
    )

    # Load trained params (if file exists)
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "rb") as f:
            agent_params = pickle.load(f)
        print(f"Agent params loaded from {AGENT_FILE}")
        params["agent"] = gs_init.params["agent"].replace(**agent_params.__dict__)
    else:
        print(f"Agent params not found at {AGENT_FILE}")

    # Replace params in gs
    gs_init = graph.init(RNG, order=("supervisor", "pid"), params=params)

    # Warmup devices
    rutils.set_log_level(LogLevel.WARN)
    [rutils.set_log_level(LogLevel.WARN, n) for n in graph.nodes.values()]  # Set log level
    devices_step = {k: next(CPU_DEVICES) if k != "agent" else GPU_DEVICE for k in ORDER}
    devices_dist = devices_step.copy()
    graph.warmup(gs_init, devices_step, devices_dist, jit_step=True, profile=True)

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
            gs, _ss = graph.reset(gs_init)
            num_steps = int(RATES["supervisor"]*TSIM)if i > 0 else 1  # Warmup
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

                # Print upright percentage
                print(f"Episode {i}: <todo: print statistics>")

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
    estimator = record.episodes[0].nodes["estimator"].steps
    agent = record.episodes[0].nodes["agent"].steps
    pid = record.episodes[0].nodes["pid"].steps

    fig, axes = plot_data(output={"att": mocap.output.att,
                                  "pos": mocap.output.pos,
                                  "pwm_ref": pid.output.pwm_ref,
                                  "phi_ref": pid.output.phi_ref,
                                  "theta_ref": pid.output.theta_ref,
                                  "z_ref": pid.output.z_ref},
                          ts={"att": mocap.ts_start,
                              "pos": mocap.ts_start,
                              "pwm_ref": pid.ts_end,
                              "phi_ref": pid.ts_end,
                              "theta_ref": pid.ts_end,
                              "z_ref": pid.ts_end},
                          # ts_max=3.0,
                          )
    plt.show()
    # Save data
    stacked = record.stack("padded")
    with open(RECORD_FILE, "wb") as f:
        pickle.dump(stacked, f)
    print(f"Data saved to {RECORD_FILE}")
