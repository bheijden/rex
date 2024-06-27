from typing import Dict, Union, Callable
import dill as pickle
import tqdm
import os
import multiprocessing
import itertools
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax

import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
from rexv2.gmm_estimator import GMMEstimator

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

DelaySim = Dict[str, Dict[str, Union[base.DelayDistribution, Dict[str, base.DelayDistribution]]]]
Delay = Dict[str, Dict[str, Union[float, Dict[str, float]]]]


def make_delay_distributions(record: Union[base.ExperimentRecord, base.EpisodeRecord],
                             num_steps: int = 100,
                             num_components: int = 2,
                             step_size: float = 0.05,
                             seed: int = 0):
    # Prepare data
    if isinstance(record, base.EpisodeRecord):
        n = next(iter(record.nodes.values()))
        if n.steps.seq.ndim == 2:
            record = jax.tree_util.tree_map(lambda x: x.flatten(), record)
            # Filter -1 from data
            record = jax.tree_util.tree_map(lambda x: x[x != -1], record)
    elif isinstance(record, base.ExperimentRecord):
        # todo: concatenate all episodes into one long sequence. I.e. no episode dimension
        raise NotImplementedError("Not implemented for ExperimentRecord")
    else:
        raise ValueError(f"Unknown record type: {type(record)}")

    # Get data
    data, info = dict(step={}, inputs={}), dict(step={}, inputs={})
    for name, n in record.nodes.items():
        data["step"][name] = n.steps.delay
        info["step"][name] = n.info
        data["inputs"][name] = {}
        info["inputs"][name] = {}
        for input_name, i in n.inputs.items():
            data["inputs"][name][input_name] = i.messages.delay
            info["inputs"][name][input_name] = (n.info.inputs[input_name], n.info)

    def init_estimator(x, i):
        name = i.name if not isinstance(i, tuple) else f"{i[0].output}->{i[1].name}"
        est = GMMEstimator(x, name)
        return est

    # Initialize estimators
    est = jax.tree_util.tree_map(lambda x, i: init_estimator(x, i), data, info)

    # Fit estimators
    jax.tree_util.tree_map(lambda e: e.fit(num_steps=num_steps, num_components=num_components, step_size=step_size, seed=seed), est)

    # Get distributions
    dist = jax.tree_util.tree_map(lambda e: e.get_dist(), est)
    return data, info, est, dist


def plot_dists(dist, data=None, info=None, est=None):
    HAS_DATA = False if data is None else True
    HAS_INFO = False if info is None else True
    HAS_EST = False if est is None else True
    if data is None:
        data = jax.tree_util.tree_map(lambda x: None, dist, is_leaf=lambda x: isinstance(x, base.DelayDistribution))
    if info is None:
        class Dummy:
            def __init__(self, name, output=None, cls=None):
                self.name = name
                self.output = output
                if cls is not None:
                    self.cls = cls

        info = {"step": {k: Dummy(k, cls=True) for k in dist["step"].keys()},
                "inputs": {k: {v: (Dummy(k, cls=True), Dummy(v, output=v)) for v in dist["inputs"][k].keys()} for k in dist["step"].keys()}}
    if est is None:
        est = jax.tree_util.tree_map(lambda x: None, dist, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    class _NoPytree:
        def __init__(self, dist):
            self.dist = dist

        def __getattr__(self, item):
            return getattr(self.dist, item)

    # First shallow copy of arguments
    dist = jax.tree_util.tree_map(lambda x: x, dist)
    data = jax.tree_util.tree_map(lambda x: x, data)
    info = jax.tree_util.tree_map(lambda x: x, info)
    est = jax.tree_util.tree_map(lambda x: x, est)

    # # Pop world from
    # [_d["inputs"]["agent"].pop("last_action", None) for _d in [data, info, est, dist]]
    # [_d["inputs"]["sensor"].pop("world", None) for _d in [data, info, est, dist]]
    # [_d["inputs"].pop("world", None) for _d in [data, info, est, dist]]
    # [_d["step"].pop("world", None) for _d in [data, info, est, dist]]

    # Split
    est_inputs, est_step = est["inputs"], est["step"]
    data_inputs, data_step = data["inputs"], data["step"]
    info_inputs, info_step = info["inputs"], info["step"]
    dist_inputs, dist_step = dist["inputs"], dist["step"]

    # Plot gmm
    from matplotlib.ticker import FormatStrFormatter
    import numpy as onp

    def plot_gmm(ax, dist, delays, i, edgecolor):
        # m = onp.max(delays) if delays is not None else dist.quantile(0.99)*1.05
        m = dist.quantile(0.99)# if delays is not None else dist.quantile(0.99)*1.05
        x = onp.linspace(0, m, 1000)
        y = dist.pdf(x)
        # if isinstance(i, tuple):
        #     output, input = i
        #     if output is None:
        #         ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
        #         ax.set_title(f"{input}")
        #     else:
        #         ax.plot(x, y, label="gmm (trun)", color=edgecolor, linestyle="--")
        #         ax.set_title(f"{input} -> {output}")
        if hasattr(i, "cls"):
            ax.plot(x, y, label="gmm", color=edgecolor, linestyle="--")
            ax.set_title(f"{i.name}")
            print(f"{i.name}: mean={dist.mean():.3f} sec, 99%={m:.3f} sec")
        else:
            input_info, node_info = i
            ax.plot(x, y, label="gmm", color=edgecolor, linestyle="--")
            ax.set_title(f"{input_info.output} -> {node_info.name}")
            print(f"{input_info.output} -> {node_info.name}: mean={dist.mean():.3f} sec, 99%={m:.3f} sec")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set_xlabel('delay (s)', fontsize=10)
        ax.set_ylabel('density', fontsize=10)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_xlim([0, m])
        ax.legend()

    # Plot distributions
    fig_step, axes_step = rutils.get_subplots(dist_step, figsize=(10, 10), sharex=False, sharey=False, major="row", is_leaf=lambda x: isinstance(x, base.DelayDistribution))
    fig_inputs, axes_inputs = rutils.get_subplots(dist_inputs, figsize=(10, 10), sharex=False, sharey=False, major="row", is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    # Plot measured delays
    from rexv2.open_colors import ecolor, fcolor

    if HAS_EST:
        jax.tree_util.tree_map(
            lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.computation, facecolor=fcolor.computation, plot_dist=False),
            axes_step, est_step)
        jax.tree_util.tree_map(
            lambda ax, e: e.plot_hist(ax=ax, edgecolor=ecolor.communication, facecolor=fcolor.communication, plot_dist=False),
            axes_inputs, est_inputs)

    # Plot gmm
    from functools import partial

    jax.tree_util.tree_map(partial(plot_gmm, edgecolor=ecolor.computation), axes_step, dist_step, data_step, info_step, is_leaf=lambda x: isinstance(x, base.DelayDistribution))
    jax.tree_util.tree_map(partial(plot_gmm, edgecolor=ecolor.communication), axes_inputs, dist_inputs, data_inputs, info_inputs, is_leaf=lambda x: isinstance(x, base.DelayDistribution))

    return fig_step, fig_inputs


if __name__ == "__main__":
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    RECORD_FILE = f"{LOG_DIR}/pendulum_data_delay_only.pkl"

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Identify delays
    data, info, est, dist = make_delay_distributions(record)

    # Plot
    fig_step, fig_inputs = plot_dists(dist, data=data, info=info, est=est)
    plt.show()

    # Save
    fig_step.savefig(f"{LOG_DIR}/step_dists.png")
    fig_inputs.savefig(f"{LOG_DIR}/inputs_dists.png")
    print(f"Saved to {LOG_DIR}/step_dists.png")
    print(f"Saved to {LOG_DIR}/inputs_dists.png")
    with open(f"{LOG_DIR}/dists.pkl", "wb") as f:
        pickle.dump(dist, f)
    print(f"Saved to {LOG_DIR}/dists.pkl")