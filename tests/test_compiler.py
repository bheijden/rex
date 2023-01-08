import time
import jax.numpy as jnp
import numpy as onp
import jumpy as jp
import jax

import rex.jumpy as rjp
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
        rng = rjp.random_prngkey(jp.int32(seed))

        env_reset = rjp.vmap(env.reset)
        env_step = rjp.vmap(env.step)
        rng = jp.random_split(rng, num=vmap)

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
    world = DummyNode("world", rate=20, delay_sim=Gaussian(0/1e3), log_level=WARN, color="magenta")
    sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(7/1e3), log_level=WARN, color="yellow")
    observer = DummyNode("observer", rate=30, delay_sim=Gaussian(16/1e3), log_level=WARN, color="cyan")
    agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(5/1e3, 1/1e3), log_level=WARN, color="blue", advance=True)
    actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1/45), log_level=WARN, color="green", advance=False, stateful=False)
    nodes = [world, sensor, observer, agent, actuator]
    nodes = {n.name: n for n in nodes}

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
    env = DummyEnv(nodes, agent=agent, max_steps=max_steps, sync=SYNC, clock=SIMULATED, scheduling=PHASE, real_time_factor=FAST_AS_POSSIBLE, name="env")

    # Evaluate async env
    gs_async, obs_async, ss_async = evaluate(env, name="async", backend="numpy", use_jit=False, seed=0)

    # Gather record
    record = log_pb2.EpisodeRecord()
    [record.node.append(node.record) for node in nodes.values()]

    # Trace
    trace_opt = trace(record, "agent", -1, static=True)
    trace_all = trace(record, "agent", -1, static=False)

    # Plot
    # _plot(trace_all)
    # _plot(trace_opt)

    # Plot progress
    must_plot = False
    if must_plot:
        from scripts.dummy_plot import plot_delay, plot_graph, plot_grouped, plot_threads
        r = {n.info.name: n for n in record.node}
        plot_graph(trace_opt)
        plot_delay(r)
        plot_grouped(r)
        plot_threads(r)

    # Compile environments
    env_opt = DummyEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_opt, graph=SEQUENTIAL, name="env_opt")
    env_all = DummyEnv(nodes, agent=agent, max_steps=max_steps, trace=trace_all, graph=VECTORIZED, name="env_all")

    # Evaluate compiled envs
    gs_opt, obs_opt, ss_opt = evaluate(env_opt, name="opt", backend="numpy", use_jit=False, seed=0)
    gs_all, obs_all, ss_all = evaluate(env_all, name="all", backend="jax", use_jit=True, seed=0)

    gs_opt, obs_opt, ss_opt = evaluate(env_opt, name="opt", backend="jax", use_jit=True, seed=0)
    gs_all, obs_all, ss_all = evaluate(env_all, name="all", backend="numpy", use_jit=False, seed=0)

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
                try:
                    assert _equal_all, msg
                    assert _equal_opt, msg
                except AssertionError:
                    print("waiting for debugger...")
                    raise

    # Test InputState API
    _ = obs_all[0]["observer"][0]

    # Merge all logged obs, gs, and ss
    obs = jp.tree_map(lambda *args: args, obs_async, obs_opt, obs_all)
    ss = jp.tree_map(lambda *args: args, ss_async, ss_opt, ss_all)
    params = [[__ss.params for __ss in _ss] for _ss in [ss_async, ss_opt, ss_all]]
    state = [[__ss.state for __ss in _ss] for _ss in [ss_async, ss_opt, ss_all]]

    # Compare observations and agent step states
    print("Comparing agent.inputs...")
    jp.tree_map(compare, obs_async, obs_opt, obs_all)
    print("agent.inputs ok")
    print("Comparing agent.params...")
    jp.tree_map(compare, *params)
    print("agent.params ok")
    print("Comparing agent.state...")
    jp.tree_map(compare, *state)
    print("agent.state ok")


if __name__ == "__main__":
    test_compiler()
