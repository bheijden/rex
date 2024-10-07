from math import ceil
from typing import Dict, Union, Callable, Any
import dill as pickle
import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

from rex import base
from rex.open_colors import get_color_cycle
from envs.abstract.nodes import Abstract, Output, MAX_DELAY


def sysid_system(nodes: Dict[str, Abstract], rollout: base.GraphState) -> Dict[str, Abstract]:
    if rollout.step.ndim == 0:
        rollout = jax.tree_util.tree_map(lambda x: x[None], rollout)
    seq = jax.tree_util.tree_map(lambda x: onp.array(x), rollout.seq)
    base_params = {k: v[0] for k, v in rollout.params.items()}
    buffer = jax.tree_util.tree_map(lambda x: onp.array(x), rollout.buffer.unfreeze())

    # Create sysid nodes according to original nodes
    nodes_sysid = {}
    infos = {}
    for name, node in nodes.items():
        infos[name] = node.info
        nodes_sysid[name] = Abstract.from_info(infos[name], outputs=buffer[name], seq=seq[name], param_cls=node.param_cls)

    # Connect them according to original nodes
    [node.connect_from_info(infos[name].inputs, nodes_sysid) for name, node in nodes_sysid.items()]

    # Replace delay distributions with trainable ones from base_params
    for name, node in nodes_sysid.items():
        for k, i in node.inputs.items():
            if isinstance(i.delay_dist, base.StaticDist):
                node.inputs[k].delay_dist = base_params[name].delays[k]
    return nodes_sysid


def linear(rng: jax.Array, rate: int, num_nodes: int, param_cls: str, max_delay: float, std_jitter: float) -> Dict[str, Abstract]:
    assert num_nodes >= 1, "There must be at least one node."
    assert max_delay <= MAX_DELAY, f"Max delay {max_delay} exceeds MAX_DELAY: {MAX_DELAY}."
    color_cycle = get_color_cycle()
    nodes = {}
    for i in range(num_nodes):
        # comp_delay_dist = base.StaticDist.create(distrax.Deterministic(0.999 / rate))
        comp_delay_dist = base.StaticDist.create(distrax.Deterministic(0.))
        node = Abstract(name=f"{i}",  color=next(color_cycle), order=i, param_cls=param_cls, outputs=None, seq=None, rate=rate, delay=0.99/rate, delay_dist=comp_delay_dist)
        nodes[node.name] = node
        if i > 0:
            parent_index = i-1
            # print(f"Connecting {i} to {parent_index}")
            rng, rng_delay = jax.random.split(rng)
            d = float(jax.random.uniform(rng_delay, shape=(), minval=0.0, maxval=max_delay, dtype=jnp.float32))
            if std_jitter > 0:
                delay_dist = base.StaticDist.create(distrax.Normal(loc=d, scale=std_jitter))
            else:
                delay_dist = base.StaticDist.create(distrax.Deterministic(d))
            node.connect(nodes[f"{parent_index}"], delay_dist=delay_dist, delay=0.0)
    return nodes


def tree(rng: jax.Array, rate: int, num_nodes: int, param_cls: str, max_delay: float, std_jitter: float) -> Dict[str, Abstract]:
    assert num_nodes >= 1, "There must be at least one node."
    color_cycle = get_color_cycle()
    nodes = {}
    for i in range(num_nodes):
        comp_delay_dist = base.StaticDist.create(distrax.Deterministic(0.))
        node = Abstract(name=f"{i}", color=next(color_cycle), order=i, param_cls=param_cls, outputs=None, seq=None, rate=rate, delay=0.99/rate, delay_dist=comp_delay_dist)
        nodes[node.name] = node
        if i > 0:
            parent_index = (i - 1) // 2
            # print(f"Connecting {i} to {parent_index}")
            rng, rng_delay = jax.random.split(rng)
            d = float(jax.random.uniform(rng_delay, shape=(), minval=0.0, maxval=max_delay, dtype=jnp.float32))
            delay_dist = base.StaticDist.create(distrax.Normal(loc=d, scale=std_jitter) if std_jitter > 0 else distrax.Deterministic(d))
            node.connect(nodes[f"{parent_index}"], delay_dist=delay_dist, delay=0.0)
    return nodes


def sparse_helper(start: int, num_nodes: int):
    assert num_nodes >= 1, "There must be at least one node."
    assert num_nodes <= 5, "Sparse subgraph only supports up to 6 nodes."
    connections = []
    nodes = []
    for i in range(start+1, start + num_nodes+1):
        connections.append((start, i))
        nodes.append(i)
    if num_nodes >= 2:
        connections.append((1+start, 2+start))
    return nodes, connections


def sparse(rng: jax.Array, rate: int, num_nodes: int, param_cls: str, max_delay: float, std_jitter: float) -> Dict[str, Abstract]:
    assert num_nodes >= 1, "There must be at least one node."
    connections = []
    node_names = [0]
    num_nodes_left = num_nodes-1
    start = 0
    while num_nodes_left > 0:
        num_nodes_subgraph = min(num_nodes_left, 5)
        nodes_sub, connections_sub = sparse_helper(start, num_nodes_subgraph)
        node_names.extend(nodes_sub)
        connections.extend(connections_sub)
        next_start = start + num_nodes_subgraph
        num_nodes_left -= num_nodes_subgraph

        # Add connection to next subgraph
        if num_nodes_left > 0:
            connections.append((start, next_start))

        # Increment
        start = next_start

    color_cycle = get_color_cycle()
    nodes = {}
    for i in range(num_nodes):
        comp_delay_dist = base.StaticDist.create(distrax.Deterministic(0.))
        node = Abstract(name=f"{i}", color=next(color_cycle), order=i, param_cls=param_cls, outputs=None, seq=None, rate=rate, delay=0.99/rate, delay_dist=comp_delay_dist)
        nodes[node.name] = node
    for (i, j) in connections:
        # print(f"Connecting {j} to {i}")
        rng, rng_delay = jax.random.split(rng)
        d = float(jax.random.uniform(rng_delay, shape=(), minval=0.0, maxval=max_delay, dtype=jnp.float32))
        delay_dist = base.StaticDist.create(distrax.Normal(loc=d, scale=std_jitter) if std_jitter > 0 else distrax.Deterministic(d))
        nodes[f"{j}"].connect(nodes[f"{i}"], delay_dist=delay_dist, delay=0.0)
    return nodes


