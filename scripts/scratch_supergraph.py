import time
from collections import deque
from functools import partial
import jax
import jumpy
import numpy as np
import jumpy.numpy as jp
import rex.jumpy as rjp
import jax.numpy as jnp

import rex.proto.log_pb2 as log_pb2
from rex.env import BaseEnv
from rex.graph import BaseGraph
from rex.asynchronous import AsyncGraph
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, LATEST, BUFFER
from rex.distributions import Gaussian
from rex.compiled import CompiledGraph
import rex.utils as utils
from rex.supergraph import get_network_record, get_timings_from_network_record, get_timings
import envs.supergraph
import experiments as exp

import supergraph as sg
import supergraph.evaluate
from tqdm import tqdm
import multiprocessing as mp
from itertools import product
from functools import reduce, partial
from operator import mul
import os
import yaml

# Graph inputs
TOPOLOGY_TYPE = ["unidirectional-ring"]
FREQUENCY_TYPE = ["linear"]
MAX_FREQ = [200]
WINDOW = [0]
LEAF_KIND = [0]
EPISODES = [20]
LENGTH = [100]
NUM_NODES = [4] #[2, 4, 8, 16]
SIGMA = [0]
THETA = [0.07]
SEED = [0]

# Algorithm inputs
SUPERGRAPH_TYPE = ["mcs"]#, "generational", "topological"]#, "sequential"]
COMBINATION_MODE = ["linear"]
SORT_MODE = ["arbitrary"]#, "optimal"]
BACKTRACK = [20]

# Logging inputs
MUST_LOG = False
MULTIPROCESSING = False
WORKERS = 8
DATA_DIR = "/home/r2ci/supergraph/data"

# Compilation
BUFFER_SIZE = 1
NUM_ENVS = 1000


# Define a function to generate all episodes for the given parameters
def evaluate_graph(seed, frequency_type, topology_type, theta, sigma, window, num_nodes, max_freq, episodes, length, leaf_kind):
    buffer_size = BUFFER_SIZE

    # Create a unique temporary directory
    name = supergraph.evaluate.to_graph_name(seed, frequency_type, topology_type, theta, sigma, window, num_nodes, max_freq, episodes, length, leaf_kind)
    RUN_DIR = f"{DATA_DIR}/{name}"

    # Load metadata from file and verify that it matches the params_graph
    metadata = {"seed": seed,
                "frequency_type": frequency_type,
                "topology_type": topology_type,
                "theta": theta,
                "sigma": sigma,
                "window": window,
                "num_nodes": num_nodes,
                "max_freq": max_freq,
                "episodes": episodes,
                "length": length,
                "leaf_kind": leaf_kind}
    with open(f"{DATA_DIR}/{name}/metadata.yaml", "r") as f:
        metadata_check = yaml.load(f, Loader=yaml.FullLoader)
    assert all([metadata_check[k] == v for k, v in metadata.items()])

    # Load graph from file
    Gs = []
    edges = None
    for i in range(episodes):
        if i > 1:
            continue
        EPS_DIR = f"{RUN_DIR}/{i}"
        assert os.path.exists(EPS_DIR), f"Episode directory does not exist: {EPS_DIR}"
        G_edges = np.load(f"{EPS_DIR}/G_edges.npy")
        G_ts = np.load(f"{EPS_DIR}/G_ts.npy")
        G = supergraph.evaluate.from_numpy(G_edges, G_ts)
        # todo: START [CONVERTS TO REX]
        if edges is None:
            edges = {(u, v) for (u, v) in np.unique(G_edges[:, :, 0], axis=0).tolist()}
        G_rex = supergraph.evaluate.to_rex(G.copy(as_view=False), edges=edges, window=buffer_size)
        # Gs.append(G)
        Gs.append(G_rex)
        # todo: END [CONVERTS TO REX]
    # num_nodes = sum([G.number_of_nodes() for G in Gs]) # todo: this overwrites the num_nodes parameter

    # Run evaluation for each supergraph type. Logged as separate runs to wandb.
    for supergraph_type in SUPERGRAPH_TYPE:
        S_top, S_gen = None, None
        if supergraph_type == "mcs":
            # Generate combinations
            all_algorithms = [COMBINATION_MODE, SORT_MODE, BACKTRACK]
            algorithm_combinations = list(product(*all_algorithms))

            for combination_mode, sort_mode, backtrack in algorithm_combinations:
                # Define config
                config = metadata.copy()
                config["supergraph_type"] = supergraph_type
                config["combination_mode"] = combination_mode
                config["sort_mode"] = sort_mode
                config["backtrack"] = backtrack

                # Set sort mode
                if sort_mode == "optimal":
                    # Only run optimal sort for unidirectional-ring (otherwise it is not optimal)
                    if not topology_type == "unidirectional-ring":
                        continue
                    sort_fn = supergraph.evaluate.perfect_sort
                elif sort_mode == "arbitrary":
                    sort_fn = None
                else:
                    raise ValueError(f"Invalid sort mode: {sort_mode}")

                # Define initial supergraph
                S_init, _ = sg.as_supergraph(Gs[0], leaf_kind=leaf_kind, sort=[f"{leaf_kind}_0"])

                # Run evaluation
                S_sup, _S_init_to_S, m_sup = sg.grow_supergraph(Gs, S_init, leaf_kind,
                                                                combination_mode=combination_mode,
                                                                backtrack=backtrack,
                                                                sort_fn=sort_fn,
                                                                progress_fn=None,
                                                                progress_bar=True,
                                                                validate=False)
                # Run compiled environment
                compile_env(Gs, edges, S_sup, m_sup, num_nodes, window=buffer_size, Gs_is_rex=True)
                del m_sup
        elif supergraph_type == "topological":
            # Get supergraph
            if S_top is None:
                S_top, S_gen = supergraph.evaluate.baselines_S(Gs, leaf_kind)

            # Evaluate supergraph
            m_top = sg.evaluate_supergraph(Gs, S_top, progress_bar=True, name="S_top")

            # Run compiled environment
            compile_env(Gs, edges, S_top, m_top, num_nodes, window=buffer_size, Gs_is_rex=True)
            del m_top
        elif supergraph_type == "generational":
            # Get supergraph
            if S_gen is None:
                S_top, S_gen = supergraph.evaluate.baselines_S(Gs, leaf_kind)

            # Evaluate supergraph
            m_gen = sg.evaluate_supergraph(Gs, S_gen, progress_bar=True, name="S_gen")

            # Run compiled environment
            compile_env(Gs, edges, S_gen, m_gen, num_nodes, window=buffer_size, Gs_is_rex=True)
            del m_gen
        elif supergraph_type == "sequential":
            # Get supergraph
            linear_iter = supergraph.evaluate.linear_S_iter(Gs)
            S_seq, _ = next(linear_iter)

            # Evaluate supergraph
            m_seq = sg.evaluate_supergraph(Gs, S_seq, progress_bar=False, name="S_seq")
            del m_seq


def compile_env(Gs, edges, S, Gs_monomorphism, num_nodes, window: int = 1, Gs_is_rex: bool = False):
    # Assert that all records have at least seq number of root steps.
    num_root_steps_fn = lambda G: len([i for i in range(len(G)) if f"{leaf_kind}_{i}" in G])
    Gs_num_root_steps = [num_root_steps_fn(G) for G in Gs]
    min_num_root_steps = min(Gs_num_root_steps)
    assert min_num_root_steps > 0, f"min_num_root_steps: {min_num_root_steps}"

    # Convert S to rex
    S_rex = supergraph.evaluate.to_rex_supergraph(S.copy(as_view=False), edges=edges, window=window)

    # Get monomorphisms
    timings = []
    for i, (G, G_monomorphism) in enumerate(zip(Gs, Gs_monomorphism)):
        G_rex = G if Gs_is_rex else supergraph.evaluate.to_rex(G.copy(as_view=False), edges=edges, window=window)
        t = get_timings(S_rex, G_rex, G_monomorphism, num_root_steps=min_num_root_steps, root=leaf_kind)
        timings.append(t)

    # Stack timings
    timings = jax.tree_util.tree_map(lambda *args: np.stack(args, axis=0), *timings)

    # Create nodes
    import equinox as eq
    nodes = {0: envs.supergraph.DummyAgent(name=0, rate=1)}
    # nodes[0].step = eq.internal.noinline(nodes[0].step)
    for i in range(1, num_nodes):
        nodes[i] = envs.supergraph.DummyNode(name=i, rate=1)
        # nodes[i].step = eq.internal.noinline(nodes[i].step)

    # Define compiled graph
    graph = CompiledGraph(nodes=nodes, root=nodes[0], S=S, default_timings=timings)

    # Create traced environment
    cenv = envs.supergraph.DummyEnv(graph=graph, name="dummy_env", max_steps=min_num_root_steps)

    # Rollouts
    rw = exp.RolloutWrapper(cenv)

    # Rn rollouts
    fps_history = []
    duration_history = []
    nenvs = NUM_ENVS
    for i in tqdm(range(100), desc="Rollouts"):
        seed = jumpy.random.PRNGKey(i)
        rng = jumpy.random.split(seed, num=nenvs)
        timer = utils.timer(f"Full rollout | {i=}", log_level=0)
        with timer:
            res = rw.batch_rollout(rng)
            res[2].block_until_ready()
        fps = rw.num_env_steps * nenvs / timer.duration
        fps_history.append(fps)
        duration_history.append(timer.duration)
        # print(f"[{timer.name}] time={timer.duration} sec | fps={fps:.0f} steps/sec")
    # Print mean and std fps
    print(f"jit={duration_history[0]} sec | fps = {np.mean(fps_history[10:]):.0f} ± {np.std(fps_history[10:]):.0f} steps/sec")

if __name__ == "__main__":
    # Generate combinations of all the necessary graphs
    all_graphs = [SEED, FREQUENCY_TYPE, TOPOLOGY_TYPE, THETA, SIGMA, WINDOW, NUM_NODES, MAX_FREQ, EPISODES, LENGTH, LEAF_KIND]
    graph_combinations = list(product(*all_graphs))
    all_exist = True
    for params in graph_combinations:  # Verify that all graphs are generated in DATA_DIR
        seed, freq_type, topology_type, theta, sigma, window, num_nodes, max_freq, episodes, length, leaf_kind = params
        name = supergraph.evaluate.to_graph_name(seed, freq_type, topology_type, theta, sigma, window, num_nodes, max_freq, episodes, length, leaf_kind)
        if not os.path.exists(f"{DATA_DIR}/{name}"):
            print(f"Missing graph: {name}")
            all_exist = False
    assert all_exist, "Not all graphs exist"

    # Create a multiprocessing pool
    pool = mp.Pool(WORKERS) if MULTIPROCESSING else None

    # Create a progress bar
    total = reduce(mul, [len(p) for p in all_graphs], 1)
    pbar = tqdm(total=total, desc="Graphs", disable=True)
    update = lambda *a: pbar.update()

    # Call the function for each combination of parameters using multiprocessing
    for params_graph in graph_combinations:
        if MULTIPROCESSING:
            pool.apply_async(evaluate_graph, args=params_graph, callback=update)
        else:
            evaluate_graph(*params_graph)
            update()

    # Close and join the pool
    pool.close() if MULTIPROCESSING else None
    pool.join() if MULTIPROCESSING else None

    exit()

    # Create nodes
    world = envs.supergraph.DummyNode("world", rate=1)
    sensor = envs.supergraph.DummyNode("sensor", rate=2)
    observer = envs.supergraph.DummyNode("observer", rate=3)
    agent = envs.supergraph.DummyAgent("agent", rate=45, delay_sim=Gaussian(0.005, 0.001), advance=True)
    actuator = envs.supergraph.DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), advance=False, stateful=True)
    nodes = [world, sensor, observer, agent, actuator]

    # Connect
    sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST, name="testworld", window=1)
    observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
    observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
    agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER, window=1)
    actuator.connect(agent, blocking=False, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05, window=2)
    world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)
    nodes = {n.name: n for n in nodes}

    agent: envs.supergraph.DummyAgent = nodes["agent"]  # type: ignore
    graph = AsyncGraph(nodes, root=agent, clock=SIMULATED, real_time_factor=FAST_AS_POSSIBLE)
    env_async = envs.supergraph.DummyEnv(graph=graph, max_steps=100, name="dummy_env")

    # Get spaces
    action_space = env_async.action_space()

    # # Record experiment
    # exp_record = log_pb2.ExperimentRecord()
    #
    # # Run environment
    # done, (graph_state, obs, info) = False, env_async.reset(jumpy.random.PRNGKey(0))
    # for _ in range(2):
    #     while not done:
    #         action = action_space.sample(jumpy.random.PRNGKey(0))
    #         graph_state, obs, reward, terminated, truncated, info = env_async.step(graph_state, action)
    #         done = terminated | truncated
    #     env_async.stop()  # NOTE: This is required to make the number of ticks equal to max_steps for compiled graph
    #
    #     # Save record
    #     _kwargs = dict(node=True, outputs=True, rngs=True, states=True, params=True, step_states=True)
    #     record = env_async.graph.get_episode_record()
    #     exp_record.episode.append(record)
    #
    # # Save experiment record
    # with open("scratch_supergraph.pb", "wb") as f:
    #     f.write(exp_record.SerializeToString())

    # Open experiment record
    with open("scratch_supergraph.pb", "rb") as f:
        exp_record = log_pb2.ExperimentRecord()
        exp_record.ParseFromString(f.read())

    # Trace computation graph
    SEED = 0
    NUM_ENVS = 10
    SUPERGRAPH_MODE = "MCS"  # MCS, topological, generational
    trace_mcs, S, _, Gs, Gs_monomorphism = get_network_record(exp_record.episode, root="agent", supergraph_mode=SUPERGRAPH_MODE)
    timings = get_timings_from_network_record(trace_mcs, Gs, Gs_monomorphism)
    # trace_mcs, MCS, G, G_subgraphs = get_network_record(exp_record.episode, root="agent", split_mode=SPLIT_MODE, supergraph_mode=SUPERGRAPH_MODE)
    # print(f"Number of MCS: {len(MCS)} | Number of S: {len(S)}")
    # timings = get_timings_from_network_record(trace_mcs, G, G_subgraphs)

    # Define compiled graph
    graph = CompiledGraph(nodes=nodes, root=nodes["agent"], S=S, default_timings=timings)

    # Create traced environment
    env = envs.supergraph.DummyEnv(graph=graph, max_steps=env_async.max_steps, name="dummy_env_mcs")

    # Get functions
    get_action = jax.vmap(action_space.sample, in_axes=(0,), out_axes=0) if NUM_ENVS > 1 else action_space.sample
    env_reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=0) if NUM_ENVS > 1 else env.reset
    env_step = jax.vmap(env.step, in_axes=(0, 0), out_axes=(0, 0, 0, 0, 0, 0)) if NUM_ENVS > 1 else env.step
    get_action = jax.jit(get_action)
    env_reset = jax.jit(env_reset)
    env_step = jax.jit(env_step)

    # Run environment
    rng_reset, rng_action = jumpy.random.split(jumpy.random.PRNGKey(SEED), num=2)
    action = get_action(jumpy.random.split(rng_action, num=NUM_ENVS)) if NUM_ENVS > 1 else get_action(rng_action)
    timer = utils.timer(f"jit[reset]", log_level=100)
    with timer:
        gs, obs, info = env_reset(jumpy.random.split(rng_reset, num=NUM_ENVS), None) if NUM_ENVS > 1 else env_reset(rng_reset, None)
    timer = utils.timer(f"jit[step]", log_level=100)
    with timer:
        gs, obs, reward, terminated, truncated, info = env_step(gs, action)

    t_history = deque(maxlen=100)
    for i in range(1000):
        start = time.perf_counter()
        gs, obs, reward, terminated, truncated, info = env_step(gs, action)
        t_elapsed = time.perf_counter() - start
        t_history.append(t_elapsed)
        fps_history = NUM_ENVS / np.array(t_history)
        t_total = np.sum(t_history)
        print(f"[{i}] t_elapsed={t_total:.4f} s | fps_mean={np.mean(fps_history):.5f} s | t_std={np.std(fps_history):.5f} s")
    exit()
    import experiments as exp
    rw = exp.RolloutWrapper(env)

    nenvs = 2
    for i in range(10):
        seed = jumpy.random.PRNGKey(i)
        rng = jumpy.random.split(seed, num=nenvs)
        timer = utils.timer(f"{i=}", log_level=0)
        with timer:
            obs, action, reward, next_obs, done, cum_return = rw.batch_rollout(rng)
        fps = obs.shape[-2] * nenvs / timer.duration

        print(f"[{timer.name}] {obs.shape[-2]} steps/rollout | time={timer.duration:.2f} s | fps={fps:.2f} steps/s | cum_return={cum_return.mean():.2f}  +/- {cum_return.std():.2f}")

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(nrows=3)
# axes = axes.flatten()
# th, th2 = jp.arctan2(obs[:, 1], obs[:, 0]), jp.arctan2(obs[:, 3], obs[:, 2])
# axes[0].plot(th)
# axes[1].plot(th2)
# axes[2].plot(jp.pi - jp.abs(th + th2))
