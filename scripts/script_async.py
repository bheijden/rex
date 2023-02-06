# use JAX_LOG_COMPILES=1 to log JIT compilation.
# todo: [TESTS] Install jumpy from github on CI
# todo: [PROTO] Write API to easily extract graph_state and step_state from trace --> script_reinitialize.py
# todo: [PROTO] Record ts start (and end?) of each step.
# todo: [ASYNC] Do nodes require at least one input (only when clock=SIMULATED)?
# todo: [ASYNC] Only compiled graphs can be transformed with vmap.
# todo: [ASYNC] We reuse the initial step_state.key for seeding the inputs. Is this a problem?
# todo: [ASYNC] Remove 1.002 from node.phase. It's a hack that makes the phase always slightly greater than the BUFFER
# todo: [ASYNC] How to determine phase-shift of a node due to non-blocking connections.
# todo: [WRAPPER] Remove __getattr__ from wrapper --> leads to unexpected behavior.
# todo: [API] Add .close() method to nodes. This should be called when the node is no longer needed (e.g. in env.close()).
# todo: [API] Define transform functions with the API of scipy that can be used for input transformations.
# todo: [API] Switch `input.name` and `input.input_name` in the API.
# todo: [JIT] Is jumpy.core.is_jitted() True when in pmap?
# todo: [JIT] Test difference cond vs select (GPU, CPU, vectorized)
# todo: [JIT] Implement BATCHED graph mode.
# todo: [PLOT] Half phase bars, and place sleep behind it
# todo: [PLOT] Fix plot_step_timing bug with isfinite.
# todo: [PLOT] Allow setting a name for plot_input_thread plots
# todo: [PLOT] In grouped plot, do not scale y with x axis. Instead, use a fixed scale.
# todo: [PLOT] Optionally add tick numbers to callback blocks in even_thread plot --> to connect with graph plot
#        link: https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html
# todo: [TRACE] Add helper functions to modify traces (e.g. remove nodes, add nodes, etc.)
# todo: [DELAY] Make a distribution that samples from a pre-recorded delay sequence.

import time
import jax.random as rnd
from rex.proto import log_pb2
import rex.utils as utils
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, REAL_TIME, FAST_AS_POSSIBLE, SIMULATED, WALL_CLOCK, SYNC, ASYNC, FREQUENCY, PHASE
from rex.distributions import Gaussian, GMM
from rex.tracer import trace
from rex.base import GraphState
from dummy import DummyNode, DummyAgent
from dummy_plot import plot_threads, plot_delay, plot_graph, plot_grouped

utils.set_log_level(WARN)

# Create nodes
c = 1e3
world = DummyNode("world",           rate=c*20/1e3, delay_sim=Gaussian(0/c))
sensor = DummyNode("sensor",         rate=c*20/1e3, delay_sim=Gaussian(7/c))
observer = DummyNode("observer",     rate=c*30/1e3, delay_sim=Gaussian(16/c))
agent = DummyAgent("agent",          rate=c*45/1e3, delay_sim=Gaussian(5/c, 1/c), advance=True)
actuator = DummyNode("actuator",     rate=c*45/1e3, delay_sim=Gaussian((c/45)/c), advance=False, stateful=False)
nodes = [world, sensor, observer, agent, actuator]

# Connect
sensor.connect(world,        blocking=False, delay_sim=Gaussian(4/c), skip=False, jitter=LATEST)
observer.connect(sensor,     blocking=False, delay_sim=Gaussian(3/c), skip=False, jitter=BUFFER)
observer.connect(agent,      blocking=False, delay_sim=Gaussian(3/c), skip=True, jitter=LATEST)
agent.connect(observer,      blocking=True, delay_sim=Gaussian(3/c), skip=False, jitter=BUFFER)
actuator.connect(agent,      blocking=True, delay_sim=Gaussian(3/c, 1/c), skip=False, jitter=LATEST, delay=3/c)
world.connect(actuator,      blocking=False, delay_sim=Gaussian(4/c), skip=True, jitter=LATEST)

# Warmup nodes (pre-compile jitted functions)
[n.warmup() for n in nodes]

# Get PRNG
seed = rnd.PRNGKey(0)
rngs = rnd.split(seed, num=len(nodes))

# Start episode
num_steps = 1000
while True:
    # Sleep
    for i in range(72000):

        if not i == 0:
            # Stop all nodes
            fs = [n._stop() for n in nodes]

            # Wait for nodes to have stopped
            [f.result() for f in fs]

            # Gather record
            record = log_pb2.EpisodeRecord()
            [record.node.append(node.record()) for node in nodes]
            d = {n.info.name: n for n in record.node}

            print(f"agent_steps={num_steps} | node_steps={[n._i for n in nodes]} | t={(tend - tstart): 2.4f} sec | fps={num_steps / (tend - tstart): 2.4f}")

            # Trace
            trace_record = trace(record, "agent", -1)

            # Write record to file
            # with open(f"/home/r2ci/rex/scripts/record_{i}.pb", "wb") as f:
            #     f.write(trace_record.SerializeToString())

            # Plot progress
            plot_graph(trace_record)
            plot_delay(d)
            plot_grouped(d)
            plot_threads(d)

        # Set the run modes
        real_time_factor, scheduling, clock, sync = FAST_AS_POSSIBLE, PHASE, SIMULATED, SYNC

        # Show effect real-time factor
        real_time_factor, scheduling, clock, sync = FAST_AS_POSSIBLE, PHASE, SIMULATED, SYNC
        # real_time_factor, scheduling, clock, sync = 20, PHASE, WALL_CLOCK, ASYNC

        # Get graph state
        node_ss = {n.name: n.reset(rng) for rng, n in zip(rngs, nodes)}
        graph_state = GraphState(step=0, nodes=node_ss)

        # Reset nodes (Allows setting the run mode)
        [n._reset(graph_state, real_time_factor=real_time_factor, clock=clock) for rng, n in zip(rngs, nodes)]

        # An additional reset is required when running async (futures, etc..)
        agent._agent_reset()

        # Check that all nodes have the same episode counter
        assert len({n.eps for n in nodes}) == 1, "All nodes must have the same episode counter"
        print(f"READY | eps={i}")

        # Start nodes (provide same starting timestamp to every node)
        start = time.time()
        [n._start(start=start) for n in nodes]

        # Simulate
        tstart = time.time()
        ts_step, step_state = agent.observation.popleft().result()  # Retrieve first obs
        for _ in range(num_steps):
            action = agent.default_output(seed)  # NOTE! Re-using the seed here.
            agent.action[-1].set_result((step_state, action))  # The set result must be the action of the agent.
            ts_step, step_state = agent.observation.popleft().result()  # Retrieve observation
        tend = time.time()

        # Initiate reset
        agent.action[-1].cancel()  # Cancel to stop the actions being sent by the agent node.


