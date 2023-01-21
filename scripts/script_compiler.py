import time
import jumpy
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import jax

from rex.jumpy import use
from rex.utils import timer
from rex.constants import WARN, SYNC, FAST_AS_POSSIBLE, PHASE, SIMULATED
from rex.proto import log_pb2
from dummy import DummyNode, DummyEnv, DummyAgent


if __name__ == "__main__":
    raise NotImplementedError("This script must be refactored to use pickle.")

    # Load protobuf trace
    with open("/home/r2ci/rex/scripts/record_1.pb", "rb") as f:
        record = log_pb2.TraceRecord()
        record.ParseFromString(f.read())
    d = {n.info.name: n for n in record.episode.node}
    inputs = {n.info.name: {i.info.name: i for i in n.inputs} for n in record.episode.node}

    # Re-initialize nodes
    # todo: use pickle
    world = DummyNode.from_info(d["world"].info)
    sensor = DummyNode.from_info(d["sensor"].info)
    observer = DummyNode.from_info(d["observer"].info)
    agent = DummyAgent.from_info(d["agent"].info)
    actuator = DummyNode.from_info(d["actuator"].info)
    nodes = [world, sensor, observer, agent, actuator]
    nodes = {n.name: n for n in nodes}

    # Re-initialize connections
    # todo: use pickle
    sensor.connect_from_info(inputs["sensor"]["world"].info, world)
    observer.connect_from_info(inputs["observer"]["sensor"].info, sensor)
    observer.connect_from_info(inputs["observer"]["agent"].info, agent)
    agent.connect_from_info(inputs["agent"]["observer"].info, observer)
    actuator.connect_from_info(inputs["actuator"]["agent"].info, agent)
    world.connect_from_info(inputs["world"]["actuator"].info, actuator)

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
        seed = jumpy.random.PRNGKey(0)

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