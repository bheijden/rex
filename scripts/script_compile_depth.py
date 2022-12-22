from typing import Any, Dict, List, Tuple, Generator, Callable, Union
import time
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import jax
from flax import struct

from rex.jumpy import use
from rex.utils import timer
from rex.constants import WARN, SYNC, FAST_AS_POSSIBLE, PHASE, SIMULATED
from rex.proto import log_pb2
from rex.node import Node
from rex.base import InputState, StepState, GraphState
from rex.tracer import trace
from rex.plot import plot_depth_order

from dummy import DummyNode, DummyEnv, DummyAgent


SplitOutput = Dict[str, Union[int, log_pb2.TracedStep, Dict[str, int], List[log_pb2.TracedStep]]]


def make_timings(nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, name: str):
    # todo: add depth
    num_steps = len(trace.used)
    node_timings = {n.name: dict(run=num_steps*[False],         # run:= whether the node must run,
                                 ts_step=num_steps*[0],         # ts_step:= ts trajectory,
                                 stateful=num_steps * [False],  # stateful:= whether to update the state,
                                 inputs={}                      # inputs:= inputs to the node
                                 ) for i, n in enumerate(trace.node)}
    timings = dict(nodes=node_timings,              # nodes:= node_timings
                   node_id=num_steps * [None],      # node_id:= node id that must run
                   index=num_steps * [None])        # index:= topological index of the node that must run
    for name, nt, in node_timings.items():
        for i in nodes[name].inputs:
            nt["inputs"][i.info.name] = dict(update=num_steps*[False],      # update:= whether to update the input with output,
                                             seq=num_steps*[-1],            # seq:= seq trajectory,
                                             ts_sent=num_steps*[-1.],       # ts_sent:= ts trajectory
                                             ts_recv=num_steps*[-1.])       # ts_recv:= ts trajectory

    # Populate timings
    for idx, t in enumerate(trace.used):
        # Update global timings
        timings["node_id"][idx] = t.name
        timings["index"][idx] = t.index
        # Update source node timings
        node_timings[t.name]["run"][idx] = True
        node_timings[t.name]["ts_step"][idx] = t.ts_step
        node_timings[t.name]["stateful"][idx] = t.stateful or not t.static
        # Update input timings
        for d in t.downstream:
            if d.target.name == t.name:
                continue
            target_node = d.target.name
            input_name = d.target.input_name
            node_timings[target_node]["inputs"][input_name]["update"][idx] = True
            node_timings[target_node]["inputs"][input_name]["seq"][idx] = d.source.tick
            node_timings[target_node]["inputs"][input_name]["ts_sent"][idx] = d.source.ts
            node_timings[target_node]["inputs"][input_name]["ts_recv"][idx] = d.target.ts
    print("wait")


def make_splitter(nodes: Dict[str, "Node"],  trace: log_pb2.TraceRecord, name: str) -> Generator[SplitOutput, None, None]:
    # Ticks of every node right before the step is executed.
    tick_state = {n.name: 0 for n in trace.node}
    steps = []
    chunk_index = 0
    for t in trace.used:
        # Update tick state
        tick_state[t.name] = t.tick

        if t.name == name:
            chunk = make_chunk(nodes, t, steps, chunk_index)
            yield dict(index=t.index, steptrace=t, tick_state=tick_state, steps=steps, chunk=chunk, chunk_index=chunk_index)
            chunk_index += 1
            tick_state = dict(**tick_state)  # Creates a deepcopy
            steps = []

        else:
            steps.append(t)


def make_chunk(nodes: Dict[str, "Node"], steptrace: log_pb2.TracedStep, steps: List[log_pb2.TracedStep], chunk_index: int) -> Callable[[GraphState], Tuple[GraphState, jp.float32, StepState]]:
    ts_step = steptrace.ts_step
    name = steptrace.name
    node_steps = [make_node_step(nodes, s) for s in steps]

    def _chunk(graph_state: GraphState) -> Tuple[GraphState, jp.float32, StepState]:
        # Apply chunk of sequential steps
        for step in node_steps:
            graph_state = step(graph_state)

        # Update to next chunk index
        new_graph_state = graph_state.replace(step=jp.int32(chunk_index))  # Increment here?

        # Return new graph state, timestamp and step state of the node that was used to split the trace (usually the agent's)
        return new_graph_state, ts_step, new_graph_state.nodes[name]

    return _chunk


def make_node_step(nodes: Dict[str, "Node"], trace: log_pb2.TracedStep) -> Callable[[GraphState], GraphState]:
    name = trace.name
    node = nodes[trace.name]
    ts_step = jp.float32(trace.ts_step)

    # Pushes the output into the input state of other nodes (excludes state dependency)
    push_outputs = {(d.target.name, d.target.input_name): make_push_output(d) for d in trace.downstream if d.target.name != name}
    update_graph_state = make_update_graph_state(name, push_outputs)

    def _node_step(graph_state: GraphState) -> GraphState:
        # Unpack Step State
        ss = graph_state.nodes[name]

        # Call node step
        new_ss, output = node.step(ts_step, ss)

        # Update graph state with new step_state and push output to downstream dependencies
        new_graph_state = update_graph_state(graph_state, new_ss, output)

        return new_graph_state

    return _node_step


def make_update_graph_state(name: str, push_outputs: Dict[Tuple[str, str], Callable]) -> Callable[[GraphState, StepState, Any], GraphState]:
    # @jax.jit
    def _update_graph_state(graph_state: GraphState, step_state: StepState, output: Any) -> GraphState:
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy())  # NOTE! This makes a shallow copy
        graph_state.nodes[name] = step_state  # NOTE! This updates a dict in-place
        for (node_name, input_name), push_output in push_outputs.items():
            new_inputs = push_output(graph_state, output)
            graph_state.nodes[node_name] = graph_state.nodes[node_name].replace(inputs=graph_state.nodes[node_name].inputs.copy())  # NOTE! This makes a shallow copy
            graph_state.nodes[node_name].inputs[input_name] = new_inputs  # NOTE: This updates a dict in-place
        return graph_state
    return _update_graph_state


def make_push_output(trace: log_pb2.Dependency) -> Callable[[GraphState, Any], InputState]:
    node_name = trace.target.name
    input_name = trace.target.input_name
    seq = trace.source.tick
    ts_sent = trace.source.ts
    ts_recv = trace.target.ts

    def _push_output(graph_state: GraphState, output: Any) -> InputState:
        return graph_state.nodes[node_name].inputs[input_name].push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=output)

    return _push_output


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
    new_record = trace(record.episode, "agent", -1, static=True)

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

    timings = make_timings(nodes, new_record, agent.name)
    splitter = make_splitter(nodes, new_record, agent.name)
    chunks = {i: d for i, d in enumerate(splitter)}

    exit()
    # Create environment
    # env = DummyEnv(nodes, agent=agent, max_steps=200, sync=SYNC, clock=SIMULATED, scheduling=PHASE, real_time_factor=FAST_AS_POSSIBLE)
    env = DummyEnv(nodes, agent=agent, max_steps=20, trace=record)

    # Warmup
    [n.warmup() for n in nodes.values()]

    # Initial graph state
    num_steps = 200000
    backend = "jax"
    use_jit = False and backend == "jax"
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

    # import numpy as np
    #
    #
    # def scan(f, init, xs, length=None):
    #     """Scan over the first dimension of an array.
    #     :param f: function to apply to the array
    #     :param init: initial carry value (state)
    #     :param xs: array to scan over (inputs)
    #     :param length: length of the scan (number of inputs, e.g. xs.shape[0])
    #     :return: tuple of (final carry/state, stacked y (outputs))
    #     """
    #     if xs is None:
    #         xs = [None] * length
    #     carry = init  # Sets the initial carry/state
    #     ys = []  # Holds the outputs
    #     for x in xs:  # Iterate over inputs.
    #         carry, y = f(carry, x)
    #         ys.append(y)
    #     return carry, np.stack(ys)  # final state, stacked outputs