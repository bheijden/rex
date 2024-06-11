from typing import Tuple, Dict
from math import ceil
import functools
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend

from rexv2.base import Timestamps, Edge, Vertex, Graph

tfd = tfp.distributions

from rexv2.node import BaseNode


def generate_graphs(
    nodes: Dict[str, BaseNode],
    computation_delays: Dict[str, tfd.Distribution],
    communication_delays: Dict[Tuple[str, str], tfd.Distribution],
    t_final: float,
    rng: jax.Array = None,
    phase: Dict[str, tfd.Distribution] = None,
    num_episodes: int = 1,
) -> Graph:
    """Generate graphs based on the nodes, computation delays, and communication delays.

    All nodes are assumed to have a rate and name attribute.
    Moreover, all nodes are assumed to run and communicate asynchronously. In other words, their timestamps are independent.

    :param nodes: Dictionary of nodes.
    :param computation_delays: Dictionary of computation delays.
    :param communication_delays: Dictionary of communication delays with keys (output_node, input_node).
    :param t_final: Final time.
    :param rng: Random number generator.
    :param phase: Dictionary of phase shift distributions (start time). Default is 0.
    :param num_episodes: Number of graphs to generate.
    :return: List of graphs.
    """
    rng = jax.random.PRNGKey(0) if rng is None else rng
    phase = dict() if phase is None else phase
    connections = dict()
    for n in nodes.values():
        assert n.name in computation_delays, f"Missing computation delay for {n.name}"
        phase[n.name] = phase.get(n.name, tfd.Deterministic(loc=0.0))
        for c in n.outputs.values():
            assert (
                c.output_node.name,
                c.input_node.name,
            ) in communication_delays, f"Missing communication delay for {c.output_node.name} -> {c.input_node.name}"
            connections[(c.output_node.name, c.input_node.name)] = c
    rates = {n: nodes[n].rate for n in nodes}

    # Generate all timestamps
    def step(name: str, carry, i):
        ts_prev, rng_prev = carry
        rate = rates[name]
        comp_delay = computation_delays[name]
        comm_delays = {m: comm for (n, m), comm in communication_delays.items() if n == name}

        # Split rng
        rngs = jax.random.split(rng_prev, num=1 + len(comm_delays) + 1)
        rng_comp, rng_comm, rng_next = rngs[0], rngs[1:-1], rngs[-1]

        # Compute timestamps
        seq = i
        ts_start = ts_prev
        ts_end = ts_start + comp_delay.sample(seed=rng_comp)
        ts_next = jnp.max(jnp.array([ts_end, ts_prev + 1 / rate]))
        ts_recvs = {m: ts_end + comm_delays[m].sample(seed=rng) for m, rng in zip(comm_delays, rng_comm)}
        timestamps = Timestamps(seq=seq, ts_start=ts_start, ts_end=ts_end, ts_recv=ts_recvs)
        return (ts_next, rng_next), timestamps

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

    def episode(rng_eps):
        # Split rngs
        rngs = jax.random.split(rng_eps, num=2 * len(nodes))
        rngs_phase = rngs[: len(nodes)]
        rngs_episode = rngs[len(nodes) :]

        # Determine start times
        offsets = {n: phase[n].sample(seed=_rng) for n, _rng in zip(nodes, rngs_phase)}
        timestamps = dict()
        for n, _rng in zip(nodes, rngs_episode):
            node_step = functools.partial(step, n)
            num_steps = ceil(t_final * rates[n]) + 1
            _, timestamps[n] = jax.lax.scan(node_step, (offsets[n], _rng), jnp.arange(0, num_steps), length=num_steps)

        # For every ts_recv, find the largest input_node.seq such that input_node.ts_start <= ts_recv (or < if connection.skip==True)
        edges = dict()
        for (output_name, input_name), c in connections.items():
            ts_recv = timestamps[output_name].ts_recv[input_name]
            ts_start = timestamps[input_name].ts_start
            scan_body_seq = functools.partial(_scan_body_seq, c.skip, ts_start)
            last_seq, seqs_clipped = jax.lax.scan(scan_body_seq, 0, ts_recv)
            seq_out = timestamps[output_name].seq
            edges[(output_name, input_name)] = Edge(seq_out=seq_out, seq_in=seqs_clipped, ts_recv=ts_recv)

        # Create vertices
        vertices = dict()
        for name_node in nodes:
            seq = timestamps[name_node].seq
            ts_start = timestamps[name_node].ts_start
            ts_end = timestamps[name_node].ts_end
            vertices[name_node] = Vertex(seq=seq, ts_start=ts_start, ts_end=ts_end)
        graph = Graph(vertices=vertices, edges=edges)
        return graph

    # Test
    graph = jax.vmap(episode)(jax.random.split(rng, num_episodes))
    return graph
