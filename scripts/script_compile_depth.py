import time
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import jax
from flax import struct

from rex.jumpy import use
from rex.utils import timer
from rex.constants import WARN, SYNC, FAST_AS_POSSIBLE, PHASE, SIMULATED, VECTORIZED, SEQUENTIAL, INTERPRETED
from rex.proto import log_pb2
from rex.tracer import trace
from rex.plot import plot_depth_order

from dummy import DummyNode, DummyEnv, DummyAgent


def _plot(new_record):
    # Create new plot
    import seaborn as sns
    sns.set()
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import rex.open_colors as oc
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax.set(facecolor=oc.ccolor("gray"), xlabel="Depth order", yticks=[], xlim=[-1, 10])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    cscheme = {"sensor": "grape", "observer": "pink", "agent": "teal", "actuator": "indigo"}

    plot_depth_order(ax, new_record, xmax=0.6, cscheme=cscheme, node_labeltype="tick", draw_excess=True)

    # Plot legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items()))
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc='center left', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 0.50))
    plt.show()


if __name__ == "__main__":

    # Load protobuf trace
    with open("/home/r2ci/rex/scripts/record_1.pb", "rb") as f:
        record = log_pb2.TraceRecord()
        record.ParseFromString(f.read())
    d = {n.info.name: n for n in record.episode.node}
    inputs = {n.info.name: {i.info.name: i for i in n.inputs} for n in record.episode.node}

    # Re-initialize nodes
    world = DummyNode.from_info(d["world"].info, log_level=WARN, color="magenta")
    sensor = DummyNode.from_info(d["sensor"].info, log_level=WARN, color="yellow")
    observer = DummyNode.from_info(d["observer"].info, log_level=WARN, color="cyan")
    agent = DummyAgent.from_info(d["agent"].info, log_level=WARN, color="blue")
    actuator = DummyNode.from_info(d["actuator"].info, log_level=WARN, color="green")
    nodes = [world, sensor, observer, agent, actuator]
    nodes = {n.name: n for n in nodes}

    # Re-initialize connections
    sensor.connect_from_info(inputs["sensor"]["world"].info, world)
    observer.connect_from_info(inputs["observer"]["sensor"].info, sensor)
    observer.connect_from_info(inputs["observer"]["agent"].info, agent)
    agent.connect_from_info(inputs["agent"]["observer"].info, observer)
    actuator.connect_from_info(inputs["actuator"]["agent"].info, agent)
    world.connect_from_info(inputs["world"]["actuator"].info, actuator)

    # Split trace into chunks
    eps_record = record.episode
    new_record = trace(record.episode, "agent", -1, static=True, isolate=True)

    # Plot
    _plot(new_record)

    # Create environment
    # env = DummyEnv(nodes, agent=agent, max_steps=200, sync=SYNC, clock=SIMULATED, scheduling=PHASE, real_time_factor=FAST_AS_POSSIBLE)
    env = DummyEnv(nodes, agent=agent, max_steps=200, trace=new_record, graph=SEQUENTIAL)

    # Warmup
    # [n.warmup() for n in nodes.values()]

    # Initial graph state
    num_steps = 200000
    backend = "jax"
    use_jit = True and backend == "jax"
    with use(backend=backend):
        # Get reset and step function
        env_reset = jax.jit(env.reset) if use_jit else env.reset
        env_step = jax.jit(env.step) if use_jit else env.step

        # Get initial graph state
        seed = jp.random_prngkey(0)

        # Reset environment (warmup)
        with timer("jit reset", log_level=WARN):
            graph_state, obs = env_reset(seed)

        # Initial step (warmup)
        with timer("jit step", log_level=WARN):
            graph_state, obs, reward, done, info = env_step(graph_state, None)

        # Run environment
        tstart = time.time()
        eps_steps = 1
        for i in range(num_steps):
            graph_state, obs, reward, done, info = env_step(graph_state, None)
            eps_steps += 1
            if done:
                step = graph_state.step
                tend = time.time()
                graph_state, obs = env_reset(seed)
                treset = time.time()
                print(f"agent_steps={eps_steps} | chunk_index={step} | t={(treset - tstart): 2.4f} sec | t_r={(treset - tend): 2.4f} sec | fps={eps_steps / (tend - tstart): 2.4f} | fps={eps_steps / (treset - tstart): 2.4f} (incl. reset)")
                tstart = treset
                eps_steps = 0
