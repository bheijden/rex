import time
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import jax

from rex.jumpy import use
from rex.tracer import trace
from rex.utils import timer
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, REAL_TIME, FAST_AS_POSSIBLE, SIMULATED, \
    WALL_CLOCK, SYNC, ASYNC, FREQUENCY, PHASE
from rex.proto import log_pb2
from rex.distributions import Gaussian
from dummy import DummyNode, DummyEnv, DummyAgent
from dummy_plot import plot_threads, plot_delay, plot_graph, plot_grouped


def evaluate(env, name: str = "env", backend: str = "numpy", use_jit: bool = False, seed: int = 0):
    use_jit = use_jit and backend == "jax"
    with use(backend=backend):
        # Get reset and step function
        env_reset = jax.jit(env.reset) if use_jit else env.reset
        env_step = jax.jit(env.step) if use_jit else env.step

        gs_lst = []
        obs_lst = []
        ss_lst = []

        # Reset environment (warmup)
        with timer(f"{name} | jit reset", log_level=WARN):
            graph_state, obs = env_reset(jp.random_prngkey(seed))
            gs_lst.append(graph_state)
            obs_lst.append(obs)
            ss_lst.append(graph_state.nodes["agent"])

        # Initial step (warmup)
        with timer(f"{name} | jit step", log_level=WARN):
            graph_state, obs, reward, done, info = env_step(graph_state, None)
            obs_lst.append(obs)
            gs_lst.append(graph_state)
            ss_lst.append(graph_state.nodes["agent"])

        # Run environment
        tstart = time.time()
        eps_steps = 1
        while True:
            # print(obs["observer"].seq)
            graph_state, obs, reward, done, info = env_step(graph_state, None)
            obs_lst.append(obs)
            gs_lst.append(graph_state)
            ss_lst.append(graph_state.nodes["agent"])
            eps_steps += 1
            if done:
                # Time env stopping
                tend = time.time()
                env.stop()
                tstop = time.time()

                # Print timings
                print(
                    f"{name=} | agent_steps={eps_steps} | chunk_index={graph_state.step} | t={(tstop - tstart): 2.4f} sec | t_s={(tstop - tend): 2.4f} sec | fps={eps_steps / (tend - tstart): 2.4f} | fps={eps_steps / (tstop - tstart): 2.4f} (incl. stop)")
                break
    return gs_lst, obs_lst, ss_lst


if __name__ == "__main__":
    world = DummyNode("world", rate=20, delay=Gaussian(0/1e3), log_level=WARN, color="magenta")
    sensor = DummyNode("sensor", rate=20, delay=Gaussian(7/1e3), log_level=WARN, color="yellow")
    observer = DummyNode("observer", rate=30, delay=Gaussian(16/1e3), log_level=WARN, color="cyan")
    agent = DummyAgent("agent", rate=45, delay=Gaussian(5/1e3, 1/1e3), log_level=WARN, color="blue", advance=True)
    actuator = DummyNode("actuator", rate=45, delay=Gaussian(1/45), log_level=WARN, color="green", advance=False, stateful=False)
    nodes = [world, sensor, observer, agent, actuator]
    nodes = {n.name: n for n in nodes}

    # Connect
    sensor.connect(world, blocking=False, delay=Gaussian(4/1e3), skip=False, jitter=LATEST)
    observer.connect(sensor, blocking=False, delay=Gaussian(3/1e3), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay=Gaussian(3/1e3), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay=Gaussian(3/1e3), skip=False, jitter=BUFFER)
    actuator.connect(agent, blocking=True, delay=Gaussian(3/1e3, 1/1e3), skip=False, jitter=LATEST, phase=3/1e3)
    world.connect(actuator, blocking=False, delay=Gaussian(4/1e3), skip=True, jitter=LATEST)

    # Warmup nodes (pre-compile jitted functions)
    [n.warmup() for n in nodes.values()]

    # Create environment
    max_steps = 200
    env = DummyEnv(nodes, agent=agent, max_steps=max_steps, sync=SYNC, clock=SIMULATED, scheduling=PHASE,
                   real_time_factor=FAST_AS_POSSIBLE)

    # Evaluate async env
    gs_async, obs_async, ss_async = evaluate(env, name="async", backend="numpy", use_jit=False, seed=0)

    # Gather record
    record = log_pb2.EpisodeRecord()
    [record.node.append(node.record) for node in nodes.values()]
    r = {n.info.name: n for n in record.node}

    # Trace
    trace_all = trace(record, "agent", -1, static=False)
    trace_opt = trace(record, "agent", -1, static=True)

    # Write record to file
    # with open(f"/home/r2ci/eagerx-dev/eagerx_paper/examples/record_{i}.pb", "wb") as f:
    #     f.write(trace_record.SerializeToString())

    # Plot progress
    must_plot = True
    if must_plot:
        plot_graph(trace_opt)
        plot_delay(r)
        plot_grouped(r)
        plot_threads(r)

    # Compile environments
    env_all = DummyEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_opt)
    env_opt = DummyEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_opt)

    # Evaluate compiled envs
    gs_all, obs_all, ss_all = evaluate(env_all, name="all", backend="numpy", use_jit=False, seed=0)
    gs_opt, obs_opt, ss_opt = evaluate(env_opt, name="opt", backend="numpy", use_jit=False, seed=0)

    # Compare
    def compare(_async, _opt, _all):
        if not isinstance(_async, (onp.ndarray, jnp.ndarray)):
            _equal_all = onp.allclose(_async, _all)
            _equal_opt = onp.allclose(_all, _opt)
            _op_all = "==" if _equal_all else "!="
            _op_opt = "==" if _equal_opt else "!="
            msg = f"{_async} {_op_all} {_all} {_op_opt} {_opt}"
            assert _equal_all, msg
            assert _equal_opt, msg
        else:
            for i in range(len(_async)):
                _equal_all = onp.allclose(_async[i], _all[i])
                _equal_opt = onp.allclose(_all[i], _opt[i])
                _op_all = "==" if _equal_all else "!="
                _op_opt = "==" if _equal_opt else "!="
                msg = f"{_async} {_op_all} {_all} {_op_opt} {_opt}"
                assert _equal_all, msg
                assert _equal_opt, msg


    obs = jp.tree_map(lambda *args: args, obs_async, obs_opt, obs_all)
    gs = jp.tree_map(lambda *args: args, gs_async, gs_opt, gs_all)
    ss = jp.tree_map(lambda *args: args, ss_async, ss_opt, ss_all)

    # Compare observations and agent step states
    jp.tree_map(compare, obs_async, obs_opt, obs_all)
    jp.tree_map(compare, ss_all, ss_opt, ss_all)
    print("finished")
