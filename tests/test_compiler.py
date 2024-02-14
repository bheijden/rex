import dill as pickle
import time
import jax.numpy as jnp
import numpy as onp
import jax
from flax.core import FrozenDict

from rex.base import GraphState
import rex.utils as utils
from rex.constants import SILENT, DEBUG, INFO, WARN
from rex.supergraph import get_network_record, get_timings_from_network_record, get_graph_buffer
from rex.compiled import CompiledGraph

from tests.dummy import DummyEnv, build_dummy_env


def evaluate(env, name: str = "env", use_jit: bool = False, seed: int = 0, vmap: int = 1):
    # Record
    gs_lst = []
    obs_lst = []
    ss_lst = []

    # vmap functions
    env_reset = jax.vmap(env.reset) if vmap > 1 else env.reset
    env_step = jax.vmap(env.step) if vmap > 1 else env.step

    # Jit reset and step function
    env_reset = jax.jit(env_reset) if use_jit else env_reset
    env_step = jax.jit(env_step) if use_jit else env_step

    # Get random key
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, num=vmap) if vmap > 1 else rng

    # Reset environment (warmup)
    with utils.timer(f"{name} | jit reset", log_level=WARN):
        graph_state, obs, info = env_reset(rng)
        gs_lst.append(graph_state)
        obs_lst.append(obs)
        ss_lst.append(graph_state.nodes["agent"])

    # Initial step (warmup)
    with utils.timer(f"{name} | jit step", log_level=WARN):
        graph_state, obs, reward, terminated, truncated, info = env_step(graph_state, None)
        obs_lst.append(obs)
        gs_lst.append(graph_state)
        ss_lst.append(graph_state.nodes["agent"])

    # Run environment
    tstart = time.time()
    eps_steps = 1
    while True:
        graph_state, obs, reward, terminated, truncated, info = env_step(graph_state, None)
        obs_lst.append(obs)
        gs_lst.append(graph_state)
        ss_lst.append(graph_state.nodes["agent"])
        eps_steps += 1
        done = jnp.logical_or(terminated, truncated)
        if done[0] if vmap > 1 else done:
            # Time env stopping
            tend = time.time()
            env.stop()
            tstop = time.time()

            # Print timings
            print(f"{name=} | agent_steps={eps_steps} | chunk_index={ss_lst[-1].seq} | t={(tstop - tstart): 2.4f} sec | t_s={(tstop - tend): 2.4f} sec | fps={vmap*eps_steps / (tend - tstart): 2.4f} | fps={vmap*eps_steps / (tstop - tstart): 2.4f} (incl. stop)")
            break
    return gs_lst, obs_lst, ss_lst


def test_compiler():
    env, nodes = build_dummy_env()

    # # Jit node steps
    # env.graph.init = jax.jit(env.graph.init)
    # for node in nodes.values():
    #     node.step = jax.jit(node.step)

    # Evaluate async env
    gs_async, obs_async, ss_async = evaluate(env, name="async", use_jit=False, seed=0)

    # Gather record
    record = env.graph.get_episode_record()

    # Trace
    trace_mcs, S, S_init_to_S, Gs, Gs_monomorphism = get_network_record(record, "agent", -1)

    # Initialize graph state
    timings = get_timings_from_network_record(trace_mcs)
    # buffer = get_graph_buffer(S, timings, nodes)
    # init_gs = GraphState(timings=timings, buffer=buffer)

    # Test graph
    _ = env.graph.max_eps()
    _ = env.graph.max_starting_step(max_steps=10)

    # Test compiled graph
    graph = CompiledGraph(nodes=nodes, root=nodes["agent"], S=S, default_timings=timings)
    init_gs = graph.init(randomize_eps=True)
    _ = graph.max_starting_step(max_steps=10, graph_state=init_gs)
    _ = graph.max_starting_step(max_steps=10, graph_state=None)
    _ = graph.max_eps(graph_state=None)
    _ = graph.max_eps(graph_state=init_gs)

    # Define env
    env_mcs = DummyEnv(graph=graph, max_steps=env.max_steps, name="env_mcs")

    # Test graph with timings & output buffers already set
    _, obs, info = env_mcs.reset(rng=jax.random.PRNGKey(0), graph_state=init_gs)
    _, obs, info = env_mcs.reset(rng=jax.random.PRNGKey(0))

    # Plot progress
    must_plot = False
    if must_plot:
        from tests.dummy_plot import plot_delay, plot_graph, plot_grouped, plot_threads
        r = {n.info.name: n for n in record.node}
        # plot_graph(trace_seq)
        plot_delay(r)
        plot_grouped(r)
        plot_threads(r)

    # Test pickle
    env_mcs.graph = pickle.loads(pickle.dumps(env_mcs.graph))

    # Evaluate compiled envs
    gs_mcs, obs_mcs, ss_mcs = evaluate(env_mcs, name="mcs-jit", use_jit=True, seed=0, vmap=100)

    # Compare
    def compare(_async, _mcs):
        _async = _async
        _equal_mcs = onp.allclose(_async, _mcs)
        _op_mcs = "==" if _equal_mcs else "!="
        msg = f"{_async} {_op_mcs} {_mcs}"
        try:
            assert _equal_mcs, msg
        except AssertionError:
            raise
        # if not isinstance(_async, (onp.ndarray, jnp.ndarray)):
        #     _equal_mcs = onp.allclose(_async, _mcs)
        #     _op_mcs = "==" if _equal_mcs else "!="
        #     msg = f"{_async} {_op_mcs} {_mcs}"
        #     assert _equal_mcs, msg
        # else:
        #     for i in range(len(_async)):
        #         _equal_mcs = onp.allclose(_async[i], _mcs[i])
        #         _op_mcs = "==" if _equal_mcs else "!="
        #         msg = f"{_async} {_op_mcs} {_mcs}"
        #         try:
        #             assert _equal_mcs, msg
        #         except AssertionError:
        #             raise

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
