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


def make_loss(graph: rexv2.graph.Graph, rollout: Callable, loss_filter: base.Filter) -> base.Loss:
    def _loss(opt_params: Dict[str, base.Params], args: base.LossArgs, rng: jax.Array = None) -> Union[float, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Unpack args
        trans, = args

        # Extend with base params
        params_extend = trans.apply(opt_params)

        # Get rollout
        final_gs = rollout(graph, params_extend, rng)

        # Get states
        states = {name: s for name, s in final_gs.state.items()}

        # Filter states
        tree_loss = eqx.filter(states, loss_filter)

        # Sum losses
        leaves, _ = jax.tree_util.tree_flatten(tree_loss)
        cum_loss = jnp.array(0.) if len(leaves) == 0 else functools.reduce(jnp.add, leaves).sum()
        return 0.5 * cum_loss.real

    return _loss


def rollout_fn(graph: rexv2.graph.Graph, params: Dict[str, base.Params], rng: jax.Array = None, carry_only: bool = True) -> base.GraphState:
    # Initialize graph state
    gs = graph.init(rng=rng, params=params)

    # Rollout graph
    final_gs = graph.rollout(gs, carry_only=carry_only)
    return final_gs


if __name__ == "__main__":
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=3, suppress=True)

    # General settings
    SEED = 0
    RATE = 10
    NUM_NODES = 4
    TS_MAX = 5.0
    MASK = "all"  # "all" or "leaf"
    PARAM_CLS = "linear"  # "linear" or "harmonic"
    MAX_DELAY = 1.0
    STD_JITTER = 0.0

    # Initialize sed
    rng = jax.random.PRNGKey(SEED)

    # Generate nodes
    rng, rng_graph = jax.random.split(rng)
    nodes = systems.linear_chain(rng_graph, rate=RATE, num_nodes=NUM_NODES, param_cls=PARAM_CLS, max_delay=MAX_DELAY, std_jitter=STD_JITTER)

    # Generate graphs
    graphs_gen = artificial.generate_graphs(nodes, ts_max=TS_MAX, num_episodes=1)

    # Determine output buffer sizes so that no output is overwritten
    buffer_sizes = {k: int(v.seq.max()+1) for k, v in graphs_gen.vertices.items()}

    # Create graph
    graph = rexv2.graph.Graph(nodes, nodes[f"{NUM_NODES-1}"], graphs_gen, buffer_sizes=buffer_sizes)

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

    # Initialize graph state
    init_gs = graph.init()

    # Visualize the graph
    if False:
        # Get rollout
        rollout = graph.rollout(init_gs)

        # Visualize data
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        for i in range(NUM_NODES):
            seq = int(rollout.seq[f"{i}"][-1])
            y = rollout.buffer[f"{i}"].y[-1, :seq]
            print(f"{i} | last_seq={seq} |y[0]={y[0]:.3f} y[{seq-1}]={y[seq-1]:.3}")
            ax.plot(y)
        plt.show()

    # Generate outputs
    rollout = graph.rollout(init_gs, carry_only=True)
    true_params = rollout.params.unfreeze()
    base_params = true_params
    outputs = {k: jax.tree_util.tree_map(lambda x: x[:rollout.seq[k]][None], v) for k, v in rollout.buffer.unfreeze().items()}

    # Create sysid nodes
    nodes_sysid = systems.sysid_system(nodes, rollout)# outputs, base_params)

    # Generate graphs
    graphs_sysid = artificial.generate_graphs(nodes_sysid, ts_max=TS_MAX, num_episodes=1)

    # Create graph
    graph_sysid = rexv2.graph.Graph(nodes_sysid, nodes_sysid[f"{NUM_NODES-1}"], graphs_sysid)

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

    # Prepare for sysid
    init_params = jax.tree_util.tree_map(lambda x: x, base_params)  # Initial guess for sysid
    min_params = {k: v.min() for k, v in base_params.items()}  # Min range for sysid
    max_params = {k: v.max() for k, v in base_params.items()}  # Max range for sysid

    # Prepare transformations
    denorm = base.Denormalize.init(min_params, max_params)
    extend = base.Extend.init(base_params, init_params)
    denorm_extend = base.Chain.init(denorm, extend)

    # Determine what outputs to use for sysid (True if leaf of the graph, i.e. no connections use the output)
    if MASK == "all":
        loss_mask = {k: s.replace(x=False, loss=True) for k, s in rollout.state.items()}
    elif MASK == "leaf":
        loss_mask = {k: s.replace(x=False, loss=len(nodes_sysid[k].outputs) == 0) for k, s in rollout.state.items()}
    else:
        raise ValueError(f"Invalid mask")

    # Make loss_fn
    loss_fn = make_loss(graph_sysid, rollout_fn, loss_mask)

    # Calculate loss of true_params
    true_params_norm = denorm_extend.inv(true_params)
    true_params_recon = denorm_extend.apply(true_params_norm)
    rng, rng_true = jax.random.split(rng)
    loss_true = loss_fn(true_params_norm, (denorm_extend,), rng=rng_true)
    print(f"True loss: {loss_true}")

    # Initialize solver & logger
    strategies = {
        "OpenES": dict(popsize=300, use_antithetic_sampling=True, opt_name="adam",
                       lrate_init=0.125, lrate_decay=0.999, lrate_limit=0.125,
                       sigma_init=0.05, sigma_decay=0.999, sigma_limit=0.01, mean_decay=0.0),
        "CMA_ES": dict(popsize=300, elite_ratio=0.2, sigma_init=0.5, mean_decay=0.),
    }
    max_steps = 300
    strategy = "CMA_ES"
    solver = evo.EvoSolver.init(denorm.normalize(min_params), denorm.normalize(max_params), strategy, strategies[strategy])
    init_sol_state = solver.init_state(denorm_extend.inv(init_params))
    logger = solver.init_logger(num_generations=max_steps)

    # Run solver
    sol_state, log_state, losses = evo.evo(loss_fn, solver, init_sol_state, (denorm_extend,),
                                           max_steps=max_steps, rng=rng, verbose=True, logger=logger)
    opt_params = solver.unflatten(sol_state.best_member)

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
                return f"{self.x_true:.2f} vs {self.xt:.2f} Error({self.error:.2f})"
                # return f"{self.xt} Rel({self.x:.2f})"

        pp = jax.tree_util.tree_map(lambda _xt, _x, _x_true, _xmin, _xmax: _PrettyPrint(_xt, _x, _x_true, _xmin, _xmax), _opt_params_trans_inv, _opt_params, true_params, min_params, max_params)
        _ = eqx.tree_pprint(pp)
        return jnp.array(0.)  # Dummy return

    opt_params_trans = denorm.apply(opt_params)
    extend = base.Extend.init(opt_params_trans, opt_params)
    opt_params_trans_inv = extend.inv(opt_params_trans)  # Filters shared parameters
    jax.experimental.io_callback(cb_pretty_print, jnp.array(0.), opt_params, opt_params_trans_inv)

    # Plot loss
    log_state.plot("loss")

    # Visualize output
    graph_recon = rexv2.graph.Graph(nodes_sysid, nodes_sysid[f"{NUM_NODES - 1}"], graphs_sysid, buffer_sizes=buffer_sizes)
    rollout = rollout_fn(graph_recon, opt_params_trans_inv, rng=rng, carry_only=True)
    rollout_true = rollout_fn(graph_recon, true_params, rng=rng, carry_only=True)
    outputs_recon = {k: jax.tree_util.tree_map(lambda x: x[:rollout.seq[k]], v) for k, v in rollout.buffer.unfreeze().items()}
    outputs_true = {k: jax.tree_util.tree_map(lambda x: x[:rollout_true.seq[k]], v) for k, v in rollout_true.buffer.unfreeze().items()}

    # Get y
    y_recon = {k: v.y for k, v in outputs_recon.items()}
    y_true = {k: v.y for k, v in outputs_true.items()}
    y = {k: v.y[0] for k, v in outputs.items()}
    y_color_recon = {k: "red" if loss_mask[k].loss else "blue" for k in y.keys()}
    y_color = {k: "black" for k in y.keys()}
    y_color_true = {k: "green" for k in y.keys()}

    # Visualize output
    _plot_y = lambda _ax, _y, _color: _ax.plot(_y, color=_color)
    fig, y_axes = rexv2.utils.get_subplots(y_recon)
    jax.tree_util.tree_map(_plot_y, y_axes, y_recon, y_color_recon)
    jax.tree_util.tree_map(_plot_y, y_axes, y_true, y_color_true)
    jax.tree_util.tree_map(_plot_y, y_axes, y, y_color)

    # Show plots
    plt.show()