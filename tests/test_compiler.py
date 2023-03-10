import dill as pickle
import time
import jumpy
import jax.numpy as jnp
import numpy as onp
import jumpy.numpy as jp
import jax

import rex.jumpy as rjp
from rex.multiprocessing import new_process
from rex.utils import timer
from rex.constants import SILENT, DEBUG, INFO, WARN
from rex.tracer_new import get_network_record, get_timings_from_network_record
from rex.compiled_new import CompiledGraph

from scripts.dummy import DummyEnv, build_dummy_env


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
            # ss_new = graph_state.nodes["agent""].replace(rng=jp.array([0, 0], dtype=jp.int32))
            ss_new = graph_state.nodes["agent"]
            ss_lst.append(ss_new)

        # Run environment
        tstart = time.time()
        eps_steps = 1
        while True:
            graph_state, obs, reward, done, info = env_step(graph_state, None)
            obs_lst.append(obs)
            gs_lst.append(graph_state)
            # ss_new = graph_state.nodes["agent""].replace(rng=jp.array([0, 0], dtype=jp.int32))
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


def test_compiler():
    env, nodes = build_dummy_env()

    # Place observer step in separate process
    # TODO: TURN ON AGAIN!
    # nodes["observer"].step = new_process(nodes["observer"].step, max_workers=2)

    # Evaluate async env
    gs_async, obs_async, ss_async = evaluate(env, name="async", backend="numpy", use_jit=False, seed=0)

    # Gather record
    record = env.graph.get_episode_record()

    # Trace
    trace_mcs, MCS, G, G_subgraphs = get_network_record(record, "agent", -1)
    timings = get_timings_from_network_record(trace_mcs, G, G_subgraphs)

    # Define graph
    graph = CompiledGraph(nodes=nodes, root=nodes["agent"], MCS=MCS, default_timings=timings)

    # Define env
    env_mcs = DummyEnv(graph=graph, max_steps=env.max_steps, name="env_mcs")

    # Plot progress
    must_plot = False
    if must_plot:
        from scripts.dummy_plot import plot_delay, plot_graph, plot_grouped, plot_threads
        r = {n.info.name: n for n in record.node}
        # plot_graph(trace_seq)
        plot_delay(r)
        plot_grouped(r)
        plot_threads(r)

    # Test pickle
    env_mcs.graph = pickle.loads(pickle.dumps(env_mcs.graph))

    # Evaluate compiled envs
    _, _, _ = evaluate(env_mcs, name="mcs-nojit-numpy", backend="numpy", use_jit=False, seed=0)
    gs_mcs, obs_mcs, ss_mcs = evaluate(env_mcs, name="mcs-jit-jax", backend="jax", use_jit=True, seed=0, vmap=1)

    # Compare
    def compare(_async, _mcs):
        if not isinstance(_async, (onp.ndarray, jnp.ndarray)):
            _equal_mcs = onp.allclose(_async, _mcs)
            _op_mcs = "==" if _equal_mcs else "!="
            msg = f"{_async} {_op_mcs} {_mcs}"
            assert _equal_mcs, msg
        else:
            for i in range(len(_async)):
                _equal_mcs = onp.allclose(_async[i], _mcs[i])
                _op_mcs = "==" if _equal_mcs else "!="
                msg = f"{_async} {_op_mcs} {_mcs}"
                try:
                    assert _equal_mcs, msg
                except AssertionError:
                    raise

    # Test InputState API
    _ = ss_mcs[0].inputs["observer"][0]

    # Merge all logged obs, gs, and ss
    # obs = jax.tree_map(lambda *args: args, obs_async, obs_mcs)
    # ss = jax.tree_map(lambda *args: args, ss_async, ss_mcs)
    params = [[__ss.params for __ss in _ss] for _ss in [ss_async, ss_mcs]]
    state = [[__ss.state for __ss in _ss] for _ss in [ss_async, ss_mcs]]

    # Compare observations and root step states
    print("Comparing agent.inputs...")
    jax.tree_map(compare, obs_async, obs_mcs)
    print("agent.inputs ok")
    print("Comparing agent.params...")
    jax.tree_map(compare, *params)
    print("agent.params ok")
    print("Comparing agent.state...")
    jax.tree_map(compare, *state)
    print("agent.state ok")


if __name__ == "__main__":
    test_compiler()
