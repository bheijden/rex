import dill as pickle
import time
import jumpy
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import jax

import rex.utils as utils
import rex.jumpy as rjp
from rex.multiprocessing import new_process
from rex.tracer import trace
from rex.utils import timer
from rex.constants import LATEST, BUFFER, SILENT, DEBUG, INFO, WARN, REAL_TIME, FAST_AS_POSSIBLE, SIMULATED, \
    WALL_CLOCK, SYNC, ASYNC, FREQUENCY, PHASE, VECTORIZED, SEQUENTIAL, INTERPRETED
from rex.proto import log_pb2
from rex.distributions import Gaussian, GMM
from scripts.dummy import DummyNode, DummyEnv, DummyAgent


def evaluate(env, name: str = "env", backend: str = "numpy", use_jit: bool = False, seed: int = 0, vmap: int = 1):
    # Record
    gs_lst = []
    obs_lst = []
    ss_lst = []

    use_jit = use_jit and backend == "jax"
    with rjp.use(backend=backend):
        rng = jumpy.random.PRNGKey(jp.int32(seed))

        env_reset = rjp.vmap(env.reset)
        env_step = rjp.vmap(env.step)
        rng = jumpy.random.split(rng, num=vmap)

        # Get reset and step function
        env_reset = jax.jit(env_reset) if use_jit else env_reset
        env_step = jax.jit(env_step) if use_jit else env_step

        # Reset environment (warmup)
        with timer(f"{name} | jit reset", log_level=WARN):
            graph_state, obs = env_reset(rng)
            gs_lst.append(graph_state)
            obs_lst.append(obs)
            ss_new = graph_state.nodes["agent"]
            ss_lst.append(ss_new)

        # Initial step (warmup)
        with timer(f"{name} | jit step", log_level=WARN):
            graph_state, obs, reward, done, info = env_step(graph_state, None)
            obs_lst.append(obs)
            gs_lst.append(graph_state)
            # ss_new = graph_state.nodes["agent"].replace(rng=jp.array([0, 0], dtype=jp.int32))
            ss_new = graph_state.nodes["agent"]
            ss_lst.append(ss_new)

        # Run environment
        tstart = time.time()
        eps_steps = 1
        while True:
            graph_state, obs, reward, done, info = env_step(graph_state, None)
            obs_lst.append(obs)
            gs_lst.append(graph_state)
            # ss_new = graph_state.nodes["agent"].replace(rng=jp.array([0, 0], dtype=jp.int32))
            ss_new = graph_state.nodes["agent"]
            ss_lst.append(ss_new)
            eps_steps += 1
            if done[0]:
                # Time env stopping
                tend = time.time()
                env.stop()
                tstop = time.time()

                # Print timings
                print(
                    f"{name=} | agent_steps={eps_steps} | chunk_index={graph_state.step} | t={(tstop - tstart): 2.4f} sec | t_s={(tstop - tend): 2.4f} sec | fps={eps_steps / (tend - tstart): 2.4f} | fps={eps_steps / (tstop - tstart): 2.4f} (incl. stop)")
                break
    return gs_lst, obs_lst, ss_lst


def _plot(new_record):
    # Create new plot
    from rex.plot import plot_depth_order
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


def test_compiler():
    # Create environment
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0/1e3))
    sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(7/1e3))
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(16/1e3))
    agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(5/1e3, 1/1e3), advance=True)
    actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1/45), advance=False, stateful=False)
    nodes = [world, sensor, observer, agent, actuator]
    nodes = {n.name: n for n in nodes}

    # Set log level
    utils.set_log_level(WARN)
    utils.set_log_level(DEBUG, world, "blue")

    # Place observer step in separate process
    observer.step = new_process(observer.step, max_workers=2)

    # Connect
    sensor.connect(world, blocking=False, delay_sim=Gaussian(4/1e3), skip=False, jitter=LATEST, window=2, name="testworld")
    observer.connect(sensor, blocking=False, delay_sim=Gaussian(3/1e3), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay_sim=Gaussian(3/1e3), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay_sim=Gaussian(3/1e3), skip=False, jitter=BUFFER)
    actuator.connect(agent, blocking=True, delay_sim=Gaussian(3/1e3, 1/1e3), skip=False, jitter=LATEST, delay=5/1e3)
    world.connect(actuator, blocking=False, delay_sim=Gaussian(4/1e3), skip=True, jitter=LATEST)

    # Warmup nodes (pre-compile jitted functions)
    [n.warmup() for n in nodes.values()]

    # Create environment
    max_steps = 200
    env = DummyEnv(nodes, agent=agent, max_steps=max_steps, clock=SIMULATED, real_time_factor=FAST_AS_POSSIBLE, name="env")

    # Evaluate async env
    gs_async, obs_async, ss_async = evaluate(env, name="async", backend="numpy", use_jit=False, seed=0)

    # Gather record
    record = log_pb2.EpisodeRecord()
    [record.node.append(node.record()) for node in nodes.values()]

    # Trace
    trace_seq = trace(record, "agent", -1)
    trace_vec = trace(record, "agent", -1)

    # Plot
    # _plot(trace_vec)
    # _plot(trace_seq)

    # Plot progress
    must_plot = False
    if must_plot:
        from scripts.dummy_plot import plot_delay, plot_graph, plot_grouped, plot_threads
        r = {n.info.name: n for n in record.node}
        plot_graph(trace_seq)
        plot_delay(r)
        plot_grouped(r)
        plot_threads(r)

    # Compile environments
    env_seq = DummyEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_seq, graph=SEQUENTIAL, name="env_seq")
    env_vec = DummyEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_vec, graph=VECTORIZED, name="env_vec")

    # Test pickle
    env_seq._cgraph = pickle.loads(pickle.dumps(env_seq._cgraph))

    # Evaluate compiled envs
    gs_seq, obs_seq, ss_seq = evaluate(env_seq, name="seq-nojit-numpy", backend="numpy", use_jit=False, seed=0)
    gs_vec, obs_vec, ss_vec = evaluate(env_vec, name="vec-jit-jax", backend="jax", use_jit=True, seed=0)

    gs_seq, obs_seq, ss_seq = evaluate(env_seq, name="seq-jit-jax", backend="jax", use_jit=True, seed=0)
    gs_vec, obs_vec, ss_vec = evaluate(env_vec, name="vec-nojit-numpy", backend="numpy", use_jit=False, seed=0)

    # Compare
    def compare(_async, _seq, _vec):
        if not isinstance(_async, (onp.ndarray, jnp.ndarray)):
            _equal_vec = onp.allclose(_async, _vec)
            _equal_seq = onp.allclose(_vec, _seq)
            _op_vec = "==" if _equal_vec else "!="
            _op_seq = "==" if _equal_seq else "!="
            msg = f"{_async} {_op_vec} {_vec} {_op_seq} {_seq}"
            assert _equal_vec, msg
            assert _equal_seq, msg
        else:
            for i in range(len(_async)):
                _equal_vec = onp.allclose(_async[i], _vec[i])
                _equal_seq = onp.allclose(_vec[i], _seq[i])
                _op_vec = "==" if _equal_vec else "!="
                _op_seq = "==" if _equal_seq else "!="
                msg = f"{_async} {_op_vec} {_vec} {_op_seq} {_seq}"
                try:
                    assert _equal_vec, msg
                    assert _equal_seq, msg
                except AssertionError:
                    print("waiting for debugger...")
                    raise

    # Test InputState API
    _ = ss_vec[0].inputs["observer"][0]

    # Merge all logged obs, gs, and ss
    obs = jax.tree_map(lambda *args: args, obs_async, obs_seq, obs_vec)
    ss = jax.tree_map(lambda *args: args, ss_async, ss_seq, ss_vec)
    params = [[__ss.params for __ss in _ss] for _ss in [ss_async, ss_seq, ss_vec]]
    state = [[__ss.state for __ss in _ss] for _ss in [ss_async, ss_seq, ss_vec]]

    # Compare observations and agent step states
    print("Comparing agent.inputs...")
    jax.tree_map(compare, obs_async, obs_seq, obs_vec)
    print("agent.inputs ok")
    print("Comparing agent.params...")
    jax.tree_map(compare, *params)
    print("agent.params ok")
    print("Comparing agent.state...")
    jax.tree_map(compare, *state)
    print("agent.state ok")


if __name__ == "__main__":
    test_compiler()
