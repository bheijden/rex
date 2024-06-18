from math import ceil
import functools
import networkx as nx
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict
from typing import Any, Tuple, List, TypeVar, Dict, Union, Callable

import supergraph as sg
from rexv2.partition_runner import make_run_partition_excl_supervisor, make_update_state
from rexv2 import base
from rexv2.node import BaseNode
from rexv2 import utils
from rexv2.constants import Supergraph
import rexv2.jax_utils as rjax
from supergraph.evaluate import timer


class Graph:
    def __init__(
        self,
        nodes: Dict[str, BaseNode],
        supervisor: BaseNode,
        graphs_raw: base.Graph,
        skip: List[str] = None,
        supergraph: Supergraph = Supergraph.MCS,
        S_init: nx.DiGraph = None,
        backtrack: int = 20,
        debug: bool = False,
        progress_bar: bool = True,
        buffer_sizes: Dict[str, int] = None,
        extra_padding: int = 0,
    ):
        """Compile graph with nodes, supervisor, and target computation graphs.

        This class finds a partitioning and supergraph to efficiently represent all raw computation graphs.
        It exposes a .step and .reset method that resembles the gym API. In addition, we provide a .run and .rollout method.
        We refer to the specific methods for more information.

        The supervisor node defines the boundary between partitions, and essentially dictates the timestep of every step call.

        "Raw" computation graphs are the graphs that are computation graphs that only take into account the data flow of a system,
        without considering the fact that some messages may be used in multiple step calls, when no new data is available.
        Conversely, some messages may be discarded if they fall out of the buffer size.
        In other words, we first modify the raw computation graphs to take into account the buffer sizes (i.e. window sizes)
        for every connection.

        :param nodes: Dictionary of nodes.
        :param supervisor: Supervisor node.
        :param graphs_raw: Raw computation graphs. Must be acyclic.
        :param skip: List of nodes to skip during graph execution.
        :param supergraph: Supergraph mode. Options are MCS, TOPOLOGICAL, and GENERATIONAL.
        :param S_init: Initial supergraph.
        :param backtrack: Backtrack parameter for MCS supergraph mode.
        :param debug: Debug mode. Validates the partitioning and supergraph and times various compilation steps.
        :param progress_bar: Show progress bar during supergraph generation.
        :param buffer_sizes: Dictionary of buffer sizes for each connection.
        :param extra_padding: Extra padding for buffer sizes.
        """

        self.nodes = nodes
        self.nodes[supervisor.name] = supervisor
        self.supervisor = supervisor
        self.nodes_excl_supervisor = {k: v for k, v in nodes.items() if v.name != supervisor.name}
        self._skip = skip if isinstance(skip, list) else [skip] if isinstance(skip, str) else []

        # Apply window to graphs
        v = next(iter(graphs_raw.vertices.values()))
        assert len(v.seq.shape) == 2, "Invalid shape. Expected 2 dimensions (episode, step)."
        self._graphs_raw = graphs_raw
        with timer("apply_window", verbose=debug):
            self._windowed_graphs = utils.apply_window(nodes, graphs_raw)  # Apply window to graphs
        with timer("to_graph", verbose=debug):
            self._graphs = self._windowed_graphs.to_graph()  # For visualization

        # Convert to networkx graphs
        with timer("to_networkx_graph", verbose=debug):
            self._Gs = []
            for i in range(len(self._graphs)):
                G = utils.to_networkx_graph(self._graphs[i], nodes=nodes, validate=debug)
                self._Gs.append(G)

        # Grow supergraph
        self.supergraph = supergraph
        if supergraph is Supergraph.MCS:
            S, S_init_to_S, Gs_monomorphism = sg.grow_supergraph(
                self._Gs,
                supervisor.name,
                S_init=S_init,
                combination_mode="linear",
                backtrack=backtrack,
                progress_fn=None,
                progress_bar=progress_bar,
                validate=debug,
            )
        elif supergraph is Supergraph.TOPOLOGICAL:
            from supergraph.evaluate import baselines_S

            S, _ = baselines_S(self._Gs, supervisor.name)
            S_init_to_S = {n: n for n in S.nodes()}
            Gs_monomorphism = sg.evaluate_supergraph(self._Gs, S, progress_bar=progress_bar)
        elif supergraph is Supergraph.GENERATIONAL:
            from supergraph.evaluate import baselines_S

            _, S = baselines_S(self._Gs, supervisor.name)
            S_init_to_S = {n: n for n in S.nodes()}
            Gs_monomorphism = sg.evaluate_supergraph(self._Gs, S, progress_bar=progress_bar)
        else:
            raise ValueError(f"Unknown supergraph mode '{supergraph}'.")
        self._S = S
        self._S_init_to_S = S_init_to_S
        self._Gs_monomorphism = Gs_monomorphism

        # Get timings
        with timer("to_timings", verbose=debug):
            self._timings = utils.to_timings(self._windowed_graphs, S, self._Gs, Gs_monomorphism, supervisor.name)

        # Verify that buffer_sizes are large enough (than _buffer_sizes) if provided
        with timer("get_buffer_sizes", verbose=debug):
            _buffer_sizes = self._timings.get_buffer_sizes()
        if buffer_sizes is not None:
            for name, size in buffer_sizes.items():
                size = [size] if isinstance(size, int) else size
                assert len(_buffer_sizes) == 0 or max(size) >= max(
                    _buffer_sizes[name]
                ), f"Buffer size for {name} is too small: {size} < {_buffer_sizes[name]}"
                _buffer_sizes[name] = size
        self._buffer_sizes = _buffer_sizes
        self._extra_padding = extra_padding

        # Save supervisor info
        self._supervisor_kind = supervisor.name
        self._supervisor_slot = [n for n, data in S.nodes(data=True) if data["kind"] == self.supervisor.name][0]
        self._supervisor_gen_idx = self._timings.slots[self._supervisor_slot].generation
        assert (
            self._supervisor_slot == f"s{self.supervisor.name}_0"
        ), f"Expected supervisor slot to be 's{self.supervisor.name}_0', but got '{self._supervisor_slot}'."

        # Convert timings to list of generations
        generations = self._timings.to_generation()
        assert (
            self._supervisor_gen_idx == len(generations) - 1
        ), f"Supervisor {self._supervisor_slot} must be in the last generation."
        assert (
            len(generations[self._supervisor_gen_idx]) == 1
        ), f"Supervisor {self._supervisor_slot} must be the only node in the last generation."

        # Make partition runner (excl supervisor)
        self._run_partition_excl_supervisor = make_run_partition_excl_supervisor(
            self.nodes, self._timings, self._S, self._supervisor_slot, self._skip
        )

    @property
    def S(self) -> nx.DiGraph:
        """The supergraph"""
        return self._S

    @property
    def Gs(self) -> List[nx.DiGraph]:
        """List of networkx graphs after applying windows to the raw computation graphs."""
        return self._Gs

    @property
    def graphs_raw(self) -> base.Graph:
        """Raw computation graphs."""
        return self._graphs_raw

    @property
    def graphs(self) -> base.Graph:
        """Graphs after applying windows to the raw computation graphs."""
        return self._graphs

    @property
    def timings(self) -> base.Timings:
        """Timings of the supergraph.

        Contains all predication masks to convert the supergraph to the correct partition given the current episode and step.
        """
        return self._timings

    @property
    def max_eps(self):
        """The maximum number of episodes."""
        num_eps = next(iter(self.timings.slots.values())).run.shape[-2]
        return num_eps

    @property
    def max_steps(self):
        """The maximum number of steps.

        That's usually the number of vertices of the supervisor in the raw computation graphs.
        """
        num_steps = next(iter(self.timings.slots.values())).run.shape[-1]
        return num_steps - 1

    def init(
        self,
        rng: jax.typing.ArrayLike = None,
        step_states: Dict[str, base.StepState] = None,
        starting_step: Union[int, jax.typing.ArrayLike] = 0,
        starting_eps: jax.typing.ArrayLike = 0,
        randomize_eps: bool = False,
        order: Tuple[str, ...] = None,
    ):
        """
        Initializes the graph state with optional parameters for RNG and step states.

        Nodes are initialized in a specified order, with the option to override step states.
        Step states may be partially defined, i.e. only contain the params or state,
        Useful for setting up the graph state before running the graph with .run, .rollout, or .reset.

        :param rng: Random number generator seed or state.
        :param step_states: Predefined step states for nodes.
        :param starting_step: The simulation's starting step.
        :param starting_eps: The starting episode.
        :param randomize_eps: If True, randomly selects the starting episode.
        :param order: The order in which nodes are initialized.
        :return: The initialized graph state.
        """
        # Determine initial step states
        step_states = step_states if step_states is not None else {}
        step_states = step_states.unfreeze() if isinstance(step_states, FrozenDict) else step_states
        step_states = {k: v for k, v in step_states.items()}  # Copy step states

        if rng is None:
            rng = jax.random.PRNGKey(0)

        if randomize_eps:
            rng, rng_eps = jax.random.split(rng, num=2)
            starting_eps = jax.random.choice(rng_eps, self.max_eps, shape=())

        # Determine init order. If name not in order, add it to the end
        order = tuple() if order is None else order
        order = list(order)
        for name in [self.supervisor.name] + list(self.nodes_excl_supervisor.keys()):
            if name not in order:
                order.append(name)

        # Initialize temporary graph state
        graph_state = base.GraphState(eps=jnp.int32(starting_eps), nodes=step_states)

        # Initialize step states
        rng, rng_ss = jax.random.split(rng)
        rngs = jax.random.split(rng_ss, num=len(order * 4)).reshape((len(order), 4, 2))
        rngs_inputs = {}
        for rngs_ss, name in zip(rngs, order):
            # Unpack rngs
            rng_params, rng_state, rng_step, rng_inputs = rngs_ss
            node = self.nodes[name]
            # Grab preset params and state if available
            preset_params = step_states[name].params if name in step_states else None
            preset_state = step_states[name].state if name in step_states else None
            # Params first, because the state may depend on them
            params = node.init_params(rng_params, graph_state) if preset_params is None else preset_params
            step_states[node.name] = base.StepState(
                rng=rng_step, params=params, state=None, inputs=None, eps=starting_eps, seq=onp.int32(0), ts=onp.float32(0.0)
            )
            # Then, get the state (which may depend on the params)
            state = node.init_state(rng_state, graph_state) if preset_state is None else preset_state
            # Inputs are updated once all nodes have been initialized with their params and state
            step_states[name] = base.StepState(
                rng=rng_step, params=params, state=state, inputs=None, eps=starting_eps, seq=onp.int32(0), ts=onp.float32(0.0)
            )
            rngs_inputs[name] = rng_inputs

        # Initialize inputs
        for name, rng_inputs in rngs_inputs.items():
            if name in self._skip:
                continue
            node = self.nodes[name]
            step_states[name] = step_states[name].replace(inputs=node.init_inputs(rng_inputs, graph_state))
        # NOTE: used to be eps=jp.as_int32(starting_eps) --> why?

        # New base graph state, without timing and buffer
        new_gs = base.GraphState(eps=starting_eps, nodes=FrozenDict(step_states))

        # Get buffer & episode timings (i.e. timings[eps])
        timings = self._timings
        buffer = timings.get_output_buffer(self.nodes, self._buffer_sizes, self._extra_padding, graph_state, rng=rng)

        # Create new graph state with timings and buffer
        new_cgs = base.GraphState(step=None, eps=None, nodes=new_gs.nodes, timings_eps=None, buffer=buffer)
        new_cgs = new_cgs.replace_step(timings, step=starting_step)  # (Clips step to valid value)
        new_cgs = new_cgs.replace_eps(timings, eps=new_gs.eps)  # (Clips eps to valid value & updates timings_eps)
        return new_cgs

    def run_until_supervisor(self, graph_state: base.GraphState) -> base.GraphState:
        """Runs graph until supervisor node.step is called.

        Internal use only. Use reset(), step(), run(), or rollout() instead.
        """
        # run supergraph (except supervisor)
        graph_state = self._run_partition_excl_supervisor(graph_state)
        return graph_state

    def run_supervisor(
        self, graph_state: base.GraphState, step_state: base.StepState = None, output: base.Output = None
    ) -> base.GraphState:
        """Runs supervisor node.step if step_state and output are not provided.
        Otherwise, overrides step_state and output with provided values.

        Internal use only. Use reset(), step(), run(), or rollout() instead.
        """
        assert (step_state is None) == (
            output is None
        ), "Either both step_state and output must be None or both must be not None."
        # Make update state function
        update_state = make_update_state(self._supervisor_kind)
        supervisor_slot = self._supervisor_slot
        supervisor = self.supervisor

        def _run_supervisor_step() -> base.GraphState:
            # Get next step state and output from supervisor node
            if step_state is None and output is None:  # Run supervisor node
                # ss = graph_state.nodes[supervisor_kind]
                ss = supervisor.get_step_state(graph_state)
                new_ss, new_output = supervisor.step(ss)
            else:  # Override step_state and output
                new_ss, new_output = step_state, output

            # Update graph state
            new_graph_state = graph_state.replace_step(
                self._timings, step=graph_state.step
            )  # Make sure step is clipped to max_step size
            # The step counter was already incremented in run_until_supervisor, but supervisor of the previous step was not run yet.
            # Hence, we need to grab the timings of the previous step (i.e. graph_state.step-1).
            timing = rjax.tree_take(graph_state.timings_eps.slots[supervisor_slot], i=new_graph_state.step - 1)
            new_graph_state = update_state(graph_state, timing, new_ss, new_output)
            return new_graph_state

        def _skip_supervisor_step() -> base.GraphState:
            return graph_state

        # Run supervisor node if step > 0, else skip
        graph_state = jax.lax.cond(graph_state.step == 0, _skip_supervisor_step, _run_supervisor_step)
        return graph_state

    def run(self, graph_state: base.GraphState) -> base.GraphState:
        """
        Executes one step of the graph including the supervisor node and returns the updated graph state.

        Different from the .step method, it automatically progresses the graph state post-supervisor execution,
        suitable for jax.lax.scan or jax.lax.fori_loop operations. This method is different from the gym API, as it uses the
        .step method of the supervisor node, while the reset and step methods allow the user to override the .step method.

        :param graph_state: The current graph state, or initial graph state from .init().
        :return: Updated graph state. It returns directly *after* the supervisor node's step() is run.
        """
        # Runs supergraph (except for supervisor)
        graph_state = self.run_until_supervisor(graph_state)

        # Runs supervisor node if no step_state or output is provided, otherwise uses provided step_state and output
        graph_state = self.run_supervisor(graph_state)
        return graph_state

    def reset(self, graph_state: base.GraphState) -> Tuple[base.GraphState, base.StepState]:
        """
        Prepares the graph for execution by resetting it to a state before the supervisor node's execution.

        Returns the graph and step state just before what would be the supervisor's step, mimicking the initial observation
        return of a gym environment's reset method. The step state can be considered the initial observation of a gym environment.

        :param graph_state: The graph state from .init().
        :return: Tuple of the new graph state and the supervisor node's step state *before* execution of the first step.
        """
        # Runs supergraph (except for supervisor)
        next_graph_state = self.run_until_supervisor(graph_state)
        next_step_state = self.supervisor.get_step_state(next_graph_state)  # Return supervisor node's step state
        return next_graph_state, next_step_state

    def step(
        self, graph_state: base.GraphState, step_state: base.StepState = None, output: base.Output = None
    ) -> Tuple[base.GraphState, base.StepState]:
        """
        Executes one step of the graph, optionally overriding the supervisor node's execution.

        If step_state and output are provided, they override the supervisor's step, allowing for custom step implementations.
        Otherwise, the supervisor's step() is executed as usual.

        When providing the updated step_state and output, the provided output can be viewed as the action that the agent would
        take in a gym environment, which is sent to nodes connected to the supervisor node.

        Start every episode with a call to reset() using the initial graph state from init(), then call step() repeatedly.

        :param graph_state: The current graph state.
        :param step_state: Custom step state for the supervisor node.
        :param output: Custom output for the supervisor node.
        :return: Tuple of the new graph state and the supervisor node's step state *before* execution of the next step.
        """
        # Runs supervisor node (if step_state and output are not provided, otherwise overrides step_state and output with provided values)
        new_graph_state = self.run_supervisor(graph_state, step_state, output)

        # Runs supergraph (except for supervisor)
        next_graph_state = self.run_until_supervisor(new_graph_state)
        next_step_state = self.supervisor.get_step_state(next_graph_state)  # Return supervisor node's step state
        return next_graph_state, next_step_state

    def rollout(
        self,
        graph_state: base.GraphState,
        starting_step: Union[int, jax.Array] = 0,
        eps: Union[int, jax.Array] = 0,
        max_steps: int = None,
        carry_only: bool = False,
    ) -> base.GraphState:
        """
        Executes the graph for a specified number of steps or until a condition is met, starting from a given step and episode.

        Utilizes the run method for execution, with an option to return only the final graph state or a sequence of all graph states.
        By virtue of using the run method, it does not allow for overriding the supervisor node's step method. That is,
        the supervisor node's step method is used during the rollout.

        :param graph_state: The initial graph state.
        :param starting_step: The starting step for the rollout.
        :param eps: The starting episode.
        :param max_steps: The maximum steps to execute, if None, runs until a stop condition is met.
        :param carry_only: If True, returns only the final graph state; otherwise returns all states.
        :return: The final or sequence of graph states post-execution.
        """
        # graph_state = self.init(starting_step=starting_step, starting_eps=eps, randomize_eps=False)
        graph_state = graph_state.replace_eps(self._timings, eps=eps)
        graph_state = graph_state.replace_step(self._timings, step=starting_step)
        if max_steps is None:
            max_steps = self.max_steps

        # Run graph for max_steps
        if carry_only:
            final_graph_state = jax.lax.fori_loop(0, max_steps, lambda i, gs: self.run(gs), graph_state)
            return final_graph_state
        else:
            # Use scan (USES WALRUS OPERATOR)
            final_graph_state, graph_states = jax.lax.scan(
                lambda gs, _: ((new_gs := self.run(gs)), new_gs), graph_state, jnp.arange(max_steps)
            )
            return graph_states
