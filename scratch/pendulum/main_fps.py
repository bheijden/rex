import dill as pickle
import tqdm
import functools
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax

import equinox as eqx

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
from rexv2 import artificial
import envs.pendulum.systems as psys


# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
# matplotlib.use("PyQt")


if __name__ == "__main__":
    # Make sysid nodes
    jnp.set_printoptions(precision=4, suppress=True)
    onp.set_printoptions(precision=4, suppress=True)
    SEED = 0
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    WORLD_RATE = 100
    MAX_ANGLE = onp.pi / 6
    MAX_STEPS = 50*20
    ACTION_DIM = 2
    SUPERVISOR = "agent"

    # Input files
    SAVE_FILE = False

    # Seed
    rng = jax.random.PRNGKey(SEED)


    #

    from rexv2.pendulum.nodes import Actuator, Sensor, OdeWorld, RandomAgent, BraxWorld

    sensor = Sensor(name="sensor", rate=50, color="pink", order=1,  # Sensor that reads the angle from the pendulum
                    delay_dist=None)
                    # delay_dist=base.StaticDist.create(distrax.Normal(loc=0.0075, scale=0.003)))
    agent = RandomAgent(name="agent", rate=50, color="teal", order=3,  # Agent that generates random actions
                        delay_dist=None)
                        # delay_dist=base.StaticDist.create(distrax.Normal(loc=0.01, scale=0.003)))
    actuator = Actuator(name="actuator", rate=50, color="orange", order=2,  # Actuator that applies the action to the pendulum
                        delay_dist=None)
                        # delay_dist=base.StaticDist.create(distrax.Normal(loc=0.0075, scale=0.003)))

    # Computation delay of the world is the world's step size (i.e. 1/rate)
    world = OdeWorld(name="world", rate=50, color="grape", order=0)  # Brax world that simulates the pendulum
    nodes = dict(world=world, sensor=sensor, agent=agent, actuator=actuator)

    # Connect nodes
    # The window determine the buffer size, i.e., the number of previous messages that are stored and can be accessed
    # in the .step() method of the node. The window should be at least 1, as the most recent message is always stored.
    # Blocking connections are synchronous, i.e., the receiving node waits for the sending node to send a message.
    # The window determines the number of messages that are stored and can be accessed in the .step() method of the node.
    agent.connect(sensor, window=1, name="sensor", blocking=False,
                  # Use the last three sensor messages as input (sync communication)
                  delay_dist=None)
                  # delay_dist=base.StaticDist.create(distrax.Normal(loc=0.002, scale=0.002)))  # Communication delay of the sensor
    actuator.connect(agent, window=1, name="agent", blocking=False,
                     # Agent receives the most recent action (sync communication)
                     delay_dist=None)
                     # delay_dist=base.StaticDist.create(distrax.Normal(loc=0.002, scale=0.002)))  # Communication delay of the agent

    # Connections below would not be necessary in a real-world system,
    # but are used to communicate the action to brax, and convert brax's state to a sensor message
    # Delay distributions are used to simulate the delays in the real-world system
    sensor_delay, actuator_delay = 0.01, 0.01
    std_delay = 0.002
    world.connect(actuator, window=1, name="actuator", skip=True,
                  # Sends the action to the brax world (skip=True to resolve circular dependency)
                  delay_dist=None)
                  # delay_dist=base.StaticDist.create(distrax.Normal(loc=actuator_delay, scale=std_delay)))
    sensor.connect(world, window=1, name="world",  # Communicate brax's state to the sensor node
                   delay_dist=None)
                   # delay_dist=base.StaticDist.create(distrax.Normal(loc=sensor_delay, scale=std_delay)))

    from rexv2.artificial import generate_graphs
    graphs_aug = generate_graphs(nodes, 30)

    # Create graph
    graph = rexv2.graph.Graph(nodes, nodes[SUPERVISOR], graphs_aug, supergraph=Supergraph.MCS)

    # Visualize the graph
    if False:
        Gs = graph.Gs
        # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        for i, G in enumerate(Gs):
            if i > 1:
                break
            supergraph.plot_graph(G, max_x=1.0, ax=axs[i])
            axs[i].set_title(f"Episode {i}")
            axs[i].set_xlabel("Time [s]")
        plt.show()

    # Get initial graph state
    rng, rng_init = jax.random.split(rng)
    with timer(f"warmup[graph_state]", log_level=LogLevel.WARN):
        gs_init = graph.init(rng_init)


    def rollout_fn(rng):
        # Initialize graph state
        # _gs = graph.init(rng, params=params, order=("supervisor", "actuator"))
        _gs_rollout = graph.rollout(gs_init, carry_only=False, max_steps=MAX_STEPS)
        return _gs_rollout.state["world"].th

    # Rollout
    rng, rng_rollout = jax.random.split(rng)
    t_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
    t_lower = timer("lower", log_level=100)
    t_compile = timer("compile", log_level=100)
    with t_jit:
        rollout_fn_j = jax.jit(rollout_fn)
        with t_lower:
            rollout_fn_j = rollout_fn_j.lower(rng_rollout)
        with t_compile:
            rollout_fn_j = rollout_fn_j.compile()
    repeat = 10
    t_run = timer("run", log_level=100, repeat=repeat)
    with t_run:
        for _ in range(repeat):
            final_states = rollout_fn_j(rng_rollout).block_until_ready()
    print(f"Final states: {final_states.mean():.2f} ± {final_states.std():.2f}")
    stats = {
        "num_steps": MAX_STEPS,
        "fps": repeat*MAX_STEPS / t_run.duration,
        "jit": t_jit.duration,
        "lower": t_lower.duration,
        "compile": t_compile.duration,
        "run": t_run.duration / repeat,
    }
    stats_str = ", ".join([f"{k}: {v:.2f}" for k, v in stats.items()])
    print(f"Stats: {stats_str}")

