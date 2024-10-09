from typing import Tuple, Dict
from math import ceil
import functools
import jax
import jax.numpy as jnp
import distrax
# from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend
# tfd = tfp.distributions

from rex.base import Timestamps, Edge, Vertex, Graph, TrainableDist, StaticDist
from rex import constants
from rex.node import BaseNode


def generate_graphs(
    nodes: Dict[str, BaseNode],
    ts_max: float,
    rng: jax.Array = None,
    num_episodes: int = 1,
) -> Graph:
    """Generate graphs based on the nodes, computation delays, and communication delays.

    All nodes are assumed to have a rate and name attribute.
    Moreover, all nodes are assumed to run and communicate asynchronously. In other words, their timestamps are independent.

    :param nodes: Dictionary of nodes.
    :param ts_max: Final time.
    :param rng: Random number generator.
    :param num_episodes: Number of graphs to generate.
    :return: Graphs for each episode.
    """
    rng = jax.random.PRNGKey(0) if rng is None else rng
    return _generate_graphs(nodes=nodes, ts_max=ts_max, rng=rng, num_episodes=num_episodes)


def augment_graphs(graphs: Graph, nodes: Dict[str, BaseNode], rng: jax.Array = None) -> Graph:
    """Augment graphs based on the nodes, computation delays, and communication delays.

    With augmenting, the graphs are expanded with additional vertices and edges based on the provided nodes.
    Nodes not in graphs.vertices are added to the graphs according to the specified delay_dist.
    Edges between vertices are added for connections not present in graphs.edges.

    :param graphs: Graphs to augment.
    :param nodes: Dictionary of nodes.
    :param rng: Random number generator.
    :return: Augmented graphs.
    """
    rng = jax.random.PRNGKey(0) if rng is None else rng

    # Expand dimension if necessary
    v = next(iter(graphs.vertices.values()))
    if len(v.seq.shape) == 1:
        graphs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), graphs)
        rm_dim = True
    elif len(v.seq.shape) == 2:
        rm_dim = False
    else:
        raise ValueError("Invalid shape for v.seq. Cannot have more than 2 dimensions.")

    # Generate graphs
    graphs_aug = _generate_graphs(nodes=nodes, rng=rng, graphs=graphs)

    # Remove dimension if necessary
    if rm_dim:
        graphs_aug = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), graphs_aug)
    return graphs_aug


def _generate_graphs(
    nodes: Dict[str, BaseNode],
    rng: jax.Array,
    num_episodes: int = None,
    ts_max: float = None,
    graphs: Graph = None,
) -> Graph:
    """Generate graphs based on the nodes, computation delays, and communication delays.

    All nodes are assumed to have a rate and name attribute.
    Moreover, all nodes are assumed to run and communicate asynchronously. In other words, their timestamps are independent.

    :param nodes: Dictionary of nodes.
    :param rng: Random number generator.
    :param num_episodes: Number of graphs to generate.  If None, the number of episodes is determined by the shape of the vertices.
    :param ts_max: Final time. If None, the final time is determined by the maximum ts_end in the graphs.
    :param graphs: Graphs to augment. If None, a new graph is generated.
    :return: A set of graphs.
    """
    assert not (ts_max is None) == (graphs is None), "Either ts_max or graphs should be provided."
    assert not (num_episodes is None) == (graphs is None), "Either num_episodes or graphs should be provided."
    assert (ts_max is None) == (num_episodes is None), "Both ts_max and num_episodes should be provided."

    # Check if graph is consistent with num_episodes
    if graphs is None:
        graphs = Graph(vertices=dict(), edges=dict())
        assert isinstance(ts_max, (float, int)), "ts_max should be provided (float, int)."
        ts_max = ts_max * jnp.ones((num_episodes,))
    else:
        # Determine num_episodes
        v = next(iter(graphs.vertices.values()))
        assert len(v.seq.shape) == 2, "Invalid shape for v.seq. Add episode dimension."
        num_episodes = v.seq.shape[0]

        # Determine ts_max
        ts_max = jnp.zeros((num_episodes,))
        for n, v in graphs.vertices.items():
            ts_max = jnp.maximum(ts_max, v.ts_end.max(axis=1))

    # Gather delays
    computation_delays, communication_delays, phase = dict(), dict(), dict()
    connections = dict()
    for n in nodes.values():
        # Convert to distrax.Distribution
        delay_dist = n.delay_dist
        if isinstance(delay_dist, TrainableDist):
            raise NotImplementedError("Cannot have trainable distribution for computation delay.")
        computation_delays[n.name] = delay_dist
        # Determine phase distribution
        phase[n.name] = StaticDist.create(distrax.Deterministic(loc=n.phase))
        for c in n.outputs.values():
            delay_dist = c.delay_dist
            if isinstance(delay_dist, TrainableDist):
                delay_dist = StaticDist.create(distrax.Deterministic(loc=delay_dist.min))  # Assume the minimal delay
            communication_delays[(c.output_node.name, c.input_node.name)] = delay_dist
            connections[(c.output_node.name, c.input_node.name)] = c
    rates = {n: nodes[n].rate for n in nodes}

    # Generate all timestamps
    def step(name: str, __ts_max: float, carry, i):
        ts_prev, rng_prev = carry
        rate = rates[name]
        comp_delay = computation_delays[name]

        # Split rng
        rng_comp, rng_next = jax.random.split(rng_prev, num=2)

        # Compute timestamps
        ts_start = ts_prev
        ts_end = ts_start + comp_delay.replace(rng=rng_comp).sample()[1]
        ts_next = jnp.max(jnp.array([ts_end, ts_prev + 1 / rate]))
        seq = jnp.where(ts_end > __ts_max, -1, i)
        vertex = Vertex(seq=seq, ts_start=ts_start, ts_end=ts_end)
        return (ts_next, rng_next), vertex

    def _scan_body_seq(skip: bool, ts_start: jax.Array, seq: int, ts_recv: float):
        def _while_cond(_seq):
            _seq_mod = _seq % ts_start.shape[0]
            is_larger = ts_start[_seq_mod] > ts_recv if skip else ts_start[_seq_mod] >= ts_recv
            is_last = ts_start.shape[0] <= _seq + 1
            # jax.debug.print("seq={_seq} | is_larger={is_larger} | is_last={is_last}", _seq=_seq, is_larger=is_larger, is_last=is_last)
            return jnp.logical_not(jnp.logical_or(is_larger, is_last))

        def _while_body(_seq):
            return _seq + 1

        # Determine the first seq that has a starting time that is larger than ts_recv
        seq = jax.lax.while_loop(_while_cond, _while_body, seq)

        # It can happen that the last seq is not larger than ts_recv, in that case, return -1
        is_larger = ts_start[seq] > ts_recv if skip else ts_start[seq] >= ts_recv
        seq_clipped = jnp.where(is_larger, seq, -1)
        return seq, seq_clipped

    def episode(rng_eps, _graphs, _ts_max):
        # Split rngs
        rngs = jax.random.split(rng_eps, num=2 * len(nodes)+len(connections))
        rngs_phase = rngs[:len(nodes)]
        rngs_episode = rngs[len(nodes):2*len(nodes)]
        rngs_comm = rngs[2*len(nodes):]

        # Determine start times
        offsets = {n: phase[n].replace(rng=_rng).sample()[1] for n, _rng in zip(nodes, rngs_phase)}
        vertices = {n: v for n, v in _graphs.vertices.items()}
        for n, _rng in zip(nodes, rngs_episode):
            if n in vertices:
                continue
            # Check if node settings are supported
            if nodes[n].advance is True:
                raise NotImplementedError("node.advance=True is not supported yet. As a workaround, you can generate graphs via AsyncGraph.")
            if nodes[n].scheduling is constants.Scheduling.PHASE:
                raise NotImplementedError("node.scheduling=PHASE is not supported yet. As a workaround, you can generate graphs via AsyncGraph.")
            # Generate timestamps
            node_step = functools.partial(step, n, _ts_max)
            ts_max_all = float(ts_max.max())  # Maxmimum ts across all episodes (needed to match static shapes).
            num_steps = ceil(ts_max_all * rates[n]) + 1
            _, vertices[n] = jax.lax.scan(node_step, (offsets[n], _rng), jnp.arange(0, num_steps), length=num_steps)

        # For every ts_recv, find the largest input_node.seq such that input_node.ts_start <= ts_recv (or < if connection.skip==True)
        edges = {(n1, n2): e for (n1, n2), e in _graphs.edges.items()}
        for ((output_name, input_name), c), _rng in zip(connections.items(), rngs_comm):
            assert output_name in vertices, f"Node {output_name} not found in vertices."
            assert input_name in vertices, f"Node {input_name} not found in vertices."
            if (output_name, input_name) in edges:
                continue  # Skip if edge already exists
            # Check if connection settings are supported
            if c.blocking is True:
                raise NotImplementedError("connection.blocking=True is not supported yet. As a workaround, you can generate graphs via AsyncGraph.")
            if c.blocking is True and isinstance(c.delay_dist, TrainableDist):
                raise NotImplementedError("connection.blocking=True and connection.delay_dist is TrainableDist is not supported yet. As a workaround, you can generate graphs via AsyncGraph.")
            if c.jitter is constants.Jitter.BUFFER:
                raise NotImplementedError("connection.jitter=BUFFER is not supported yet. As a workaround, you can generate graphs via AsyncGraph.")
            seq_out = vertices[output_name].seq
            ts_end = jnp.where(seq_out == -1, jnp.inf, vertices[output_name].ts_end)
            ts_recv = ts_end + communication_delays[(output_name, input_name)].replace(rng=_rng).sample(shape=ts_end.shape)[1]
            ts_start = vertices[input_name].ts_start
            scan_body_seq = functools.partial(_scan_body_seq, c.skip, ts_start)
            last_seq, seqs_clipped = jax.lax.scan(scan_body_seq, 0, ts_recv)

            # Replace ts_recv with -1 if ts_end == -1 (i.e., the message was never sent, so it will never be received)
            ts_recv = jnp.where(seq_out == -1, -1, ts_recv)

            # Overwrite seq_out and seq_in if ts_end is larger than ts_max
            seq_out = jnp.where(ts_end > _ts_max, -1, seq_out)
            seq_in = jnp.where(ts_end > _ts_max, -1, seqs_clipped)
            seq_in = jnp.where(seqs_clipped > vertices[input_name].seq.max(), -1, seq_in)
            edges[(output_name, input_name)] = Edge(seq_out=seq_out, seq_in=seq_in, ts_recv=ts_recv)

        graph = Graph(vertices=vertices, edges=edges)
        return graph

    # Test
    graph = jax.vmap(episode)(jax.random.split(rng, num_episodes), graphs, ts_max)
    return graph
