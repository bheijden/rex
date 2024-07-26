import json
import hashlib
import functools
from typing import Dict, Union, Callable, Any
import dill as pickle
import equinox
import tqdm
import os
import multiprocessing
import itertools
import datetime

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import distrax

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
from rexv2 import artificial
import rexv2.evo as evo
import envs.abstract.systems as systems


# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


def make_loss(_graph: rexv2.graph.Graph, rollout: Callable, loss_filter: base.Filter, starting_eps: jax.typing.ArrayLike, return_final_gs: bool = False) -> base.Loss:
    def _loss(opt_params: Dict[str, base.Params], args: base.LossArgs, rng: jax.Array = None) -> Union[float, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Unpack args
        trans, = args

        # Extend with base params
        params_extend = trans.apply(opt_params)

        # Get rollout
        # jax.debug.print("starting_eps={starting_eps}", starting_eps=starting_eps)
        final_gs = rollout(_graph, params_extend, rng, starting_eps)

        # Get states
        states = {name: s for name, s in final_gs.state.items()}

        # Filter states
        # tree_loss = eqx.filter(states, loss_filter)  # Here, loss_filter must be static
        tree_loss = jax.tree_util.tree_map(lambda f, x: f*x, loss_filter, states)

        # Sum losses
        leaves, _ = jax.tree_util.tree_flatten(tree_loss)
        cum_loss = jnp.array(0.) if len(leaves) == 0 else functools.reduce(jnp.add, leaves).sum()
        cum_loss = 0.5 * cum_loss.real
        if return_final_gs:
            return cum_loss, final_gs
        else:
            return cum_loss

    return _loss


def rollout_fn(_graph: rexv2.graph.Graph, params: Dict[str, base.Params], rng: jax.Array = None, starting_eps: jax.typing.ArrayLike = None, carry_only: bool = True) -> base.GraphState:
    # Initialize graph state
    gs = _graph.init(rng=rng, params=params, starting_eps=starting_eps)
    # jax.debug.print("eps={eps}", eps=gs.eps)

    # Rollout graph
    final_gs = _graph.rollout(gs, carry_only=carry_only)
    return final_gs


if __name__ == "__main__":
    # todo: Rerun experiments without pruned leaf nodes.
    # todo: Run with 6 nodes (chain-->5, tree--> 5, spares-->6 connections)
    # todo: Record reconstruction error when true parameters are used
    # todo: Reduce jitter to 0.05
    # todo: what to evaluate:
    #   - Repeat 5 times  --> can be done in the same vmap
    #   - DATA GENERATION
    #      - Different topologies (linear, tree, sparse)
    #      - Different dynamics (linear, harmonic)
    #      [OPTIONAL] Stochasticity in delays
    #   - SYSID
    #       - Delays + dynamics, delays only
    #       - All data, only leaf data (different masks --> can be done in the same vmap)
    #   - Record
    #       - Computation graph (generator + sysid)
    #       - Loss curve
    #       - Generated data (last gs with large buffer)
    #       - Sysid reconstruction (last gs with large buffer)
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=3, suppress=True)

    # General settings
    LOG_DIR = "/home/r2ci/rex/scratch/abstract/logs"
    EXP_DIR = f"{LOG_DIR}/main_4-6-12Nodes_0.05Jitter_0sup"
    # EXP_DIR = f"{LOG_DIR}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_abstract"
    SAVE_FILES = True
    SEED = 0
    NUM_REPEATS = 5  # todo: CHANGE
    # ENV PARAMS
    RATE = 10
    NUM_NODES = 5  # todo: CHANGE
    TS_MAX = 10.0  # todo: CHANGE max delay * num_nodes
    TOPOLOGY = "tree"  # "tree", "linear", "sparse
    PARAM_CLS = "linear"  # "linear" or "harmonic"
    INTERP = "zoh"
    MAX_DELAY = 1.0  # todo: increase to 2.0?
    STD_JITTER = 0.0
    # SYSID PARAMS
    SYSID_PARAMS = "delay"  # "delay" or "delay_dynamics"  # todo: iterate over
    # MASK = "all"  # "all" or "leaf"  # not used, as both are evaluated in parallel with the same vmap call
    # CMA_ES
    MAX_STEPS = 300  # todo: CHANGE
    POPSIZE = 300
    ELITE_RATIO = 0.2
    SIGMA_INIT = 0.5
    MEAN_DECAY = 0.0

    for STD_JITTER in [0.0, 0.05]:
        for NUM_NODES in [12, 4, 6]:
            for TOPOLOGY in ["sparse", "linear", "tree"]:
                for PARAM_CLS in ["linear", "harmonic"]:
                    # # TODO: DEBUG BEGIN
                    # # STD_JITTER: 0.0, NUM_NODES: 10, TOPOLOGY: sparse, PARAM_CLS: harmonic, SYSID_PARAMS: delay_dynamics
                    # STD_JITTER = 0.0
                    # NUM_NODES = 10
                    # TOPOLOGY = "linear"  # sparse
                    # PARAM_CLS = "linear"  # harmonic
                    # SYSID_PARAMS = "delay_dynamics"
                    # # TODO: DEBUG END
                    topologies = {
                        "linear": systems.linear,
                        "tree": systems.tree,
                        "sparse": systems.sparse,
                    }

                    # Initialize sed
                    rng = jax.random.PRNGKey(SEED)

                    # Generate nodes
                    nodes_lst = []
                    true_params_lst = []
                    graphs_gen_lst = []
                    buffer_sizes = None
                    for i in range(NUM_REPEATS):
                        rng, rng_graph = jax.random.split(rng)
                        nodes = topologies[TOPOLOGY](rng_graph, rate=RATE, num_nodes=NUM_NODES, param_cls=PARAM_CLS, max_delay=MAX_DELAY, std_jitter=STD_JITTER)
                        nodes_lst.append(nodes)

                        # Get true params
                        rng, rng_params = jax.random.split(rng)
                        rngs = jax.random.split(rng_params, NUM_NODES)
                        _true_params = {k: v.init_params(_rng) for (k, v), _rng in zip(nodes.items(), rngs)}
                        true_params_lst.append(_true_params)

                        # Generate graphs
                        graphs_gen = artificial.generate_graphs(nodes, ts_max=TS_MAX, num_episodes=1)
                        graphs_gen_lst.append(graphs_gen[0])

                        # Determine output buffer sizes so that no output is overwritten
                        next_sizes = {k: int(v.seq.max() + 1) for k, v in graphs_gen.vertices.items()}
                        if buffer_sizes is None:
                            buffer_sizes = next_sizes
                        else:
                            for k, v in next_sizes.items():
                                assert buffer_sizes[k] == v, f"Buffer size mismatch: {buffer_sizes[k]} vs {v}"

                    # Create graph
                    nodes = nodes_lst[0]  # Use arbitrary nodes, but make sure to grab the params for each node_lst later
                    graphs_gen = base.Graph.stack(graphs_gen_lst)
                    supervisor = f"{0}"
                    graph = rexv2.graph.Graph(nodes, nodes[supervisor], graphs_gen, buffer_sizes=buffer_sizes, prune=False)
                    print(graph._timings.get_buffer_sizes())  # Print buffer sizes.

                    # Visualize the graph
                    if False:
                        Gs = graph.Gs
                        # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
                        fig_graph, axs_graph = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
                        for i, G in enumerate(Gs):
                            supergraph.plot_graph(G, max_x=5.0, ax=axs_graph[0])
                            axs_graph[0].set_title(f"Episode {i}")
                            axs_graph[0].set_xlabel("Time [s]")
                            break
                        # plt.show()

                    # Initialize graph state & replace params
                    true_params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *true_params_lst)
                    starting_eps = jnp.arange(NUM_REPEATS)
                    rng, rng_init = jax.random.split(rng)
                    rngs = jax.random.split(rng_init, NUM_REPEATS)
                    init_gs = jax.vmap(graph.init)(rngs, true_params, starting_eps=starting_eps)

                    # Generate outputs
                    rollout = jax.vmap(functools.partial(graph.rollout, carry_only=True))(init_gs)

                    # Create sysid nodes
                    nodes_sysid = systems.sysid_system(nodes, rollout)

                    # Generate graphs
                    graphs_sysid = artificial.generate_graphs(nodes_sysid, ts_max=TS_MAX, num_episodes=NUM_REPEATS)  # todo: SAVE

                    # Create graph
                    graph_sysid = rexv2.graph.Graph(nodes_sysid, nodes_sysid[supervisor], graphs_sysid, prune=False)
                    graph_opt = rexv2.graph.Graph(nodes_sysid, nodes_sysid[supervisor], graphs_sysid, buffer_sizes=buffer_sizes, prune=False)

                    # Visualize the graph
                    if False:
                        Gs = graph_sysid.Gs
                        # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
                        if fig_graph is None:
                            fig_graph, axs_graph = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
                        for i, G in enumerate(Gs):
                            supergraph.plot_graph(G, max_x=5.0, ax=axs_graph[1])
                            axs_graph[1].set_title(f"Episode {i}")
                            axs_graph[1].set_xlabel("Time [s]")
                            break
                        plt.show()

                    # Calculate loss of true_params
                    # if True:
                    def get_true_loss(_true_params, _rng, _eps):
                        _loss_mask = {k: s.replace(x=False, loss=True) for k, s in rollout.state.items()}
                        loss_fn = make_loss(graph_sysid, rollout_fn, _loss_mask, _eps, return_final_gs=True)
                        return loss_fn(_true_params, (base.Identity.init(),), rng=_rng)
                    rng, rng_true = jax.random.split(rng)
                    rngs = jax.random.split(rng_true, NUM_REPEATS)
                    loss_true, final_gs = jax.vmap(get_true_loss)(true_params, rngs, starting_eps)

                    # Before sysid
                    for SYSID_PARAMS in ["delay", "delay_dynamics"]:
                        # Print ########## divider
                        msg = f"STD_JITTER: {STD_JITTER}, NUM_NODES: {NUM_NODES}, TOPOLOGY: {TOPOLOGY}, PARAM_CLS: {PARAM_CLS}, SYSID_PARAMS: {SYSID_PARAMS}"
                        print("\n" + "#" * len(msg))
                        print(msg)
                        print("#" * len(msg))
                        if STD_JITTER == 0.0 and (loss_true > 0.1).any():
                            print(f"WARNING! True loss: {loss_true} | Jitter: {STD_JITTER}")
                        else:
                            print(f"True loss: {loss_true} | Jitter: {STD_JITTER > 0}")

                        # Store config
                        config = {k: v for k, v in locals().items() if k.isupper()}
                        config_all = config.copy()
                        config_leaf = config.copy()
                        config_all["MASK"] = "all"
                        config_leaf["MASK"] = "leaf"
                        for k, v in config.items():
                            print(f"{k}: {v}")

                        any_exist = False
                        for c in [config_all, config_leaf]:
                            # Generate a unique identifier from the filters
                            filter_str = json.dumps(c, sort_keys=True)  # Convert dict to string in a consistent manner
                            hash_object = hashlib.md5(filter_str.encode())  # Use MD5 or another hashing algorithm
                            filter_hash = hash_object.hexdigest()  # Get the hash as a string
                            # Use the hash as the filename
                            filename = "run_" + filter_hash + ".pkl"
                            any_exist = any_exist or os.path.exists(f"{EXP_DIR}/{filename}")
                        if any_exist:
                            print("Already exists. Skipping...")
                            continue

                        # Solve sysid
                        def solve_mask(_rng, _true_params, _eps, _loss_mask):
                            def _delays_only(_p):
                                """Filters non-delay trainable params"""
                                p = jax.tree_util.tree_map(lambda x: None, _p)
                                p = {k: p[k].replace(delays=v.delays) for k, v in _p.items()}
                                return p

                            # Initialize params
                            if SYSID_PARAMS == "delay":
                                min_params = _delays_only({k: v.min() for k, v in _true_params.items()})
                                max_params = _delays_only({k: v.max() for k, v in _true_params.items()})
                            elif SYSID_PARAMS == "delay_dynamics":
                                min_params = {k: v.min() for k, v in _true_params.items()}  # Min range for sysid
                                max_params = {k: v.max() for k, v in _true_params.items()}  # Max range for sysid
                            else:
                                raise ValueError(f"Unknown SYSID_PARAMS: {SYSID_PARAMS}")

                            # Prepare transformations
                            denorm = base.Denormalize.init(min_params, max_params)
                            extend = base.Extend.init(_true_params, min_params)  # Only tree structure is used.
                            denorm_extend = base.Chain.init(denorm, extend)

                            # Make loss_fn
                            loss_fn = make_loss(graph_sysid, rollout_fn, _loss_mask, _eps)

                            # Initialize solver & logger
                            strategy_kwargs = dict(popsize=POPSIZE, elite_ratio=ELITE_RATIO, sigma_init=SIGMA_INIT, mean_decay=MEAN_DECAY)
                            solver = evo.EvoSolver.init(denorm.normalize(min_params), denorm.normalize(max_params), "CMA_ES", strategy_kwargs)
                            init_sol_state = solver.init_state(jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), min_params))
                            logger = solver.init_logger(num_generations=MAX_STEPS)

                            # Run solver
                            _rng, rng_solver = jax.random.split(_rng)
                            sol_state, log_state, losses = evo.evo(loss_fn, solver, init_sol_state, (denorm_extend,),
                                                                   max_steps=MAX_STEPS, rng=rng_solver, verbose=False, logger=logger)
                            opt_params = solver.unflatten(sol_state.best_member)
                            opt_params_extend = denorm_extend.apply(opt_params)

                            # Print results
                            def cb_pretty_print(_opt_params, _opt_params_trans_inv):
                                class _PrettyPrint:
                                    def __init__(self, xt, x, x_true, xmin, xmax):
                                        self.xt = onp.array(xt)
                                        self.x = x
                                        self.x_true = onp.array(x_true)
                                        self.xmin = xmin
                                        self.xmax = xmax
                                        self.error = onp.linalg.norm(self.xt - self.x_true) / onp.linalg.norm(self.xmax - self.xmin)

                                    def __repr__(self):
                                        """Relative to the transformed value."""
                                        # return f"{float(self.x_true):.2f} vs {float(self.xt):.2f} Error({float(self.error):.2f})"
                                        return f"{self.x_true} vs {self.xt} Error({self.error})"

                                extend = base.Extend.init(_true_params, min_params)  # Only tree structure is used.
                                _true_params_inv = extend.inv(_true_params)
                                pp = jax.tree_util.tree_map(lambda _xt, _x, _x_true, _xmin, _xmax: _PrettyPrint(_xt, _x, _x_true, _xmin, _xmax), _opt_params_trans_inv, _opt_params, _true_params_inv, min_params, max_params)
                                _ = eqx.tree_pprint(pp)
                                return jnp.array(0.)  # Dummy return

                            opt_params_trans = denorm.apply(opt_params)
                            extend = base.Extend.init(opt_params_trans, opt_params)
                            opt_params_trans_inv = extend.inv(opt_params_trans)  # Filters shared parameters
                            # jax.experimental.io_callback(cb_pretty_print, jnp.array(0.), opt_params, opt_params_trans_inv)

                            # Evaluate
                            _rng, rng_rollout = jax.random.split(_rng)
                            rollout_opt = rollout_fn(graph_opt, opt_params_extend, rng=rng_rollout, starting_eps=_eps, carry_only=True)
                            return dict(true_params=_true_params, opt_params=opt_params_trans_inv, log_state=log_state, rollout_opt=rollout_opt)

                        def solve(_rng, _true_params, _eps):
                            _rngs = jax.random.split(_rng, 2)  # Number of masks

                            # Determine what outputs to use for sysid (True if leaf of the graph, i.e. no connections use the output)
                            _loss_mask_all = {k: s.replace(x=False, loss=True) for k, s in rollout.state.items()}
                            _loss_mask_leaf = {k: s.replace(x=False, loss=len(nodes_sysid[k].outputs) == 0) for k, s in rollout.state.items()}
                            _loss_mask = jax.tree_util.tree_map(lambda x, y: jnp.stack((x, y)), _loss_mask_all, _loss_mask_leaf)

                            # Vmap
                            solve_mask_vmap = jax.vmap(solve_mask, in_axes=(0, None, None, 0))
                            res = solve_mask_vmap(_rngs, _true_params, _eps, _loss_mask)
                            mask_all = jax.tree_util.tree_map(lambda x: x[0], res)
                            mask_leaf = jax.tree_util.tree_map(lambda x: x[1], res)
                            return mask_all, mask_leaf

                        # Test solve_mask
                        if False:
                            rng, rng_solve = jax.random.split(rng)
                            loss_mask_all = {k: s.replace(x=False, loss=True) for k, s in rollout.state.items()}
                            res = solve_mask(rng_solve, true_params_lst[0], starting_eps[0], loss_mask_all)

                        # Test solve
                        if False:
                            rng, rng_solve = jax.random.split(rng)
                            _mask_all, _mask_leaf = solve(rng_solve, true_params_lst[0], starting_eps[0])

                        # Test solve_vmap
                        rng, rng_solve = jax.random.split(rng)
                        rngs = jax.random.split(rng_solve, NUM_REPEATS)
                        t_solve_jit = timer("jit | lower | compile", log_level=100)  # Makes them available outside the context manager
                        with t_solve_jit:
                            solve_jv = jax.jit(jax.vmap(solve, in_axes=(0, 0, 0)))
                            with timer("lower", log_level=100):
                                solve_jv = solve_jv.lower(rngs, true_params, starting_eps)
                            with timer("compile", log_level=100):
                                solve_jv = solve_jv.compile()
                        # Solve
                        t_solve = timer("solve", log_level=100)
                        with t_solve:
                            mask_all, mask_leaf = solve_jv(rngs, true_params, starting_eps)
                        print(mask_all["log_state"].state["log_top_1"][..., -1])
                        print(mask_leaf["log_state"].state["log_top_1"][..., -1])
                        # Store timings
                        elapsed = dict(solve=t_solve.duration, solve_jit=t_solve_jit.duration)
                        print(elapsed)

                        # # Store config
                        # config = {k: v for k, v in locals().items() if k.isupper()}
                        # for k, v in config.items():
                        #     print(f"{k}: {v}")

                        # Store other results
                        for d, _config in zip([mask_all, mask_leaf], [config_all, config_leaf]):
                            d["elapsed"] = elapsed.copy()
                            d["config"] = _config.copy()
                            d["rollout"] = rollout
                            d["graphs_sysid"] = graphs_sysid
                            d["graphs_gen"] = graphs_gen
                            d["true_loss"] = loss_true

                        # Save results
                        if SAVE_FILES:
                            # Make directory if it doesn't exist
                            if os.path.exists(EXP_DIR):
                                print(f"Directory {EXP_DIR} already exists.")
                            else:
                                print(f"Creating directory: {EXP_DIR}.")
                                os.makedirs(EXP_DIR, exist_ok=True)
                            for d in [mask_all, mask_leaf]:
                                # Generate a unique identifier from the filters
                                filter_str = json.dumps(d["config"], sort_keys=True)  # Convert dict to string in a consistent manner
                                hash_object = hashlib.md5(filter_str.encode())  # Use MD5 or another hashing algorithm
                                filter_hash = hash_object.hexdigest()  # Get the hash as a string

                                # Use the hash as the filename
                                filename = "run_" + filter_hash + ".pkl"

                                # Check if the file already exists
                                is_new = True
                                if os.path.exists(f"{EXP_DIR}/{filename}"):
                                    is_new = False

                                # Save
                                with open(f"{EXP_DIR}/{filename}", "wb") as f:
                                    pickle.dump(d, f)
                                if is_new:
                                    print(f"Saved: {filename}")
                                else:
                                    print(f"File overwritten: {filename}")
