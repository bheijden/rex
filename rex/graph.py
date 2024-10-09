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
from rex.partition_runner import make_run_partition_excl_supervisor, make_update_state
from rex import base
from rex.node import BaseNode
from rex import utils
from rex.constants import Supergraph, Clock, RealTimeFactor
import rex.jax_utils as rjax


class Graph:
    def __init__(
        self,
        nodes: Dict[str, BaseNode],
        supervisor: BaseNode,
        graphs_raw: base.Graph,
        skip: List[str] = None,
        supergraph: Supergraph = Supergraph.MCS,
        prune: bool = True,
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
        :param prune: Prune nodes that are no ancestors of the supervisor node.
                      Setting to False ensures that all nodes up until the time of the last supervisor node are included.
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

        # # Initialize record_settings
        # _settings = dict(params=False, rng=False, inputs=False, state=False, output=False)
        # self._record_settings = {k: _settings.copy() for k in nodes.keys()}

        # Expand dimension if necessary
        v = next(iter(graphs_raw.vertices.values()))
        if len(v.seq.shape) == 1:
            graphs_raw = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), graphs_raw)
        elif len(v.seq.shape) == 2:
            pass
        else:
            raise ValueError("Invalid shape. Expected 1 (step,) or 2 dimensions (episode, step).")
        v = next(iter(graphs_raw.vertices.values()))
        assert len(v.seq.shape) == 2, "Invalid shape. Expected 2 dimensions (episode, step)."
        # Apply window to graphs
        self._graphs_raw = graphs_raw
        self._windowed_graphs = utils.apply_window(nodes, graphs_raw)  # Apply window to graphs
        self._graphs = self._windowed_graphs.to_graph()  # For visualization

        # Convert to networkx graphs
        self._Gs = []
        for i in range(len(self._graphs)):
            G = utils.to_networkx_graph(self._graphs[i], nodes=nodes, validate=debug)
            self._Gs.append(G)

        # If prune is False, we need to include all nodes up until the last supervisor node
        if not prune:
            Gs_supergraph = [utils.to_connected_graph(G, supervisor, nodes, validate=debug) for G in self._Gs]
        else:
            Gs_supergraph = self._Gs

        # Grow supergraph
        self.supergraph = supergraph
        if supergraph is Supergraph.MCS:
            S, S_init_to_S, Gs_monomorphism = sg.grow_supergraph(
                Gs_supergraph,
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

            S, _ = baselines_S(Gs_supergraph, supervisor.name)
            S_init_to_S = {n: n for n in S.nodes()}
            Gs_monomorphism = sg.evaluate_supergraph(Gs_supergraph, S, progress_bar=progress_bar)
        elif supergraph is Supergraph.GENERATIONAL:
            from supergraph.evaluate import baselines_S

            _, S = baselines_S(Gs_supergraph, supervisor.name)
            S_init_to_S = {n: n for n in S.nodes()}
            Gs_monomorphism = sg.evaluate_supergraph(Gs_supergraph, S, progress_bar=progress_bar)
        else:
            raise ValueError(f"Unknown supergraph mode '{supergraph}'.")
        self._S = S
        self._S_init_to_S = S_init_to_S
        self._Gs_monomorphism = Gs_monomorphism

        # Get timings
        # with timer("to_timings", verbose=debug):
        self._timings = utils.to_timings(self._windowed_graphs, S, self._Gs, Gs_monomorphism, supervisor.name)

        # Verify that buffer_sizes are large enough (than _buffer_sizes) if provided
        # with timer("get_buffer_sizes", verbose=debug):
        _buffer_sizes = self._timings.get_buffer_sizes()
        if buffer_sizes is not None:
            for name, size in buffer_sizes.items():
                size = [size] if isinstance(size, int) else size
                assert name not in _buffer_sizes or len(_buffer_sizes[name]) == 0 or max(size) >= max(
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
        params: Dict[str, base.Params] = None,
        starting_step: Union[int, jax.typing.ArrayLike] = 0,
        starting_eps: jax.typing.ArrayLike = 0,
        randomize_eps: bool = False,
        order: Tuple[str, ...] = None,
    ):
        """
        Initializes the graph state with optional parameters for RNG and step states.

        Nodes are initialized in a specified order, with the option to override params.
        Useful for setting up the graph state before running the graph with .run, .rollout, or .reset.

        :param rng: Random number generator seed or state.
        :param params: Predefined params for (a subset of) the nodes.
        :param starting_step: The simulation's starting step.
        :param starting_eps: The starting episode.
        :param randomize_eps: If True, randomly selects the starting episode.
        :param order: The order in which nodes are initialized.
        :return: The initialized graph state.
        """
        # Determine init order. If name not in order, add it to the end
        order = tuple() if order is None else order
        order = list(order)
        for name in [self.supervisor.name] + list(self.nodes_excl_supervisor.keys()):
            if name not in order:
                order.append(name)

        # Prepare random number generators
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rng_eps, rng_step, rng_params, rng_state, rng_inputs = jax.random.split(rng, num=5)

        if randomize_eps:
            starting_eps = jax.random.choice(rng_eps, self.max_eps, shape=())

        # Determine preset params
        params = params if params is not None else {}
        params = params.unfreeze() if isinstance(params, FrozenDict) else params
        params = {k: v for k, v in params.items()}  # Copy params

        # Initialize graph state
        rngs_step = FrozenDict({k: _rng for k, _rng in zip(order, jax.random.split(rng_step, num=len(order)))})
        seq = FrozenDict({k: onp.int32(0) for k in order})
        ts = FrozenDict({k: onp.float32(0.0) for k in order})
        state = {}
        inputs = {}
        graph_state = base.GraphState(eps=jnp.int32(starting_eps), step=jnp.int32(starting_step),
                                      rng=rngs_step, seq=seq, ts=ts, params=params, state=state, inputs=inputs)

        # Initialize params
        rngs = jax.random.split(rng_params, num=len(order))
        for rng, name in zip(rngs, order):
            params[name] = params.get(name, self.nodes[name].init_params(rng, graph_state))

        # Initialize state
        rngs = jax.random.split(rng_state, num=len(order))
        for rng, name in zip(rngs, order):
            state[name] = self.nodes[name].init_state(rng, graph_state)

        # Initialize inputs
        rngs = jax.random.split(rng_inputs, num=len(order))
        for rng, name in zip(rngs, order):
            inputs[name] = self.nodes[name].init_inputs(rng, graph_state)

        # Replace params, state, and inputs in graph state with immutable versions
        new_gs = graph_state.replace(params=FrozenDict(params), state=FrozenDict(state), inputs=FrozenDict(inputs))

        # Get buffer & episode timings (i.e. timings[eps])
        timings = self._timings
        buffer = timings.get_output_buffer(self.nodes, self._buffer_sizes, self._extra_padding, graph_state, rng=rng)

        # Create new graph state with timings and buffer
        new_cgs = new_gs.replace(buffer=buffer)
        new_cgs = new_cgs.replace_step(timings, step=starting_step)  # (Clips step to valid value)
        new_cgs = new_cgs.replace_eps(timings, eps=new_gs.eps)  # (Clips eps to valid value & updates timings_eps)
        return new_cgs

    def init_record(self,
                    graph_state: base.GraphState,
                    params: Union[Dict[str, bool], bool] = None,
                    rng: Union[Dict[str, bool], bool] = None,
                    inputs: Union[Dict[str, bool], bool] = None,
                    state: Union[Dict[str, bool], bool] = None,
                    output: Union[Dict[str, bool], bool] = None,
                    ) -> base.GraphState:
        """Sets the record settings for the nodes in the graph.

        :param graph_state: The initial graph state from .init().
        :param params: Whether to record params for each node. Logged once.
        :param rng: Whether to record rng for each node. Logged each step.
        :param inputs: Whether to record inputs for each node. Logged each step. Can become very large.
        :param state: Whether to record state for each node. Logged each step.
        :param output: Whether to record output for each node. Logged each step.
        :return: The updated graph state with record settings.
        """
        assert "record" not in graph_state.aux, "Already initialized record in graph_state.aux."

        # Prepare record settings
        params = params if params is not None else {}
        rng = rng if rng is not None else {}
        inputs = inputs if inputs is not None else {}
        state = state if state is not None else {}
        output = output if output is not None else {}
        params = params if isinstance(params, dict) else {k: params for k in self.nodes.keys()}
        rng = rng if isinstance(rng, dict) else {k: rng for k in self.nodes.keys()}
        inputs = inputs if isinstance(inputs, dict) else {k: inputs for k in self.nodes.keys()}
        state = state if isinstance(state, dict) else {k: state for k in self.nodes.keys()}
        output = output if isinstance(output, dict) else {k: output for k in self.nodes.keys()}
        # Initialize default record_settings
        _record_settings = {k: {} for k in self.nodes.keys()}
        for name in self.nodes.keys():
            _record_settings[name]["params"] = params.get(name, False)
            _record_settings[name]["rng"] = rng.get(name, False)
            _record_settings[name]["inputs"] = inputs.get(name, False)
            _record_settings[name]["state"] = state.get(name, False)
            _record_settings[name]["output"] = output.get(name, False)

        # Initialize record
        # Determine maximum number of steps per node across episodes
        num_seqs = {}
        for sname, slot in self._timings.slots.items():
            kind = slot.kind
            num_runs = slot.run.sum(axis=-1)
            # Calculate total non-masked steps per node per episode
            num_seqs[kind] = num_seqs.get(kind, onp.zeros_like(num_runs)) + num_runs
        num_seqs = {k: onp.max(v) for k, v in num_seqs.items()} # Take the maximum number of steps over all episodes

        node_records = {}
        for name, node in self.nodes.items():
            # Initialize input record
            inputs = {}
            for _, c in node.inputs.items():
                name_out = c.output_node.name
                info = c.output_node.info
                seqs_out = num_seqs[name_out]
                dummy_arr = onp.ones((seqs_out,), dtype=int) * -1
                messages = base.MessageRecord(
                    seq_out=dummy_arr,
                    seq_in=dummy_arr,
                    ts_sent=dummy_arr.astype(float),
                    ts_recv=dummy_arr.astype(float),
                    delay=dummy_arr.astype(float),
                )
                inputs[name_out] = base.InputRecord(info=info, messages=messages)
            # Initialize step record
            rng = graph_state.rng[name] if _record_settings[name]["rng"] else None
            inputs = inputs if _record_settings[name]["inputs"] else None
            state = graph_state.state[name] if _record_settings[name]["state"] else None
            buffer = graph_state.buffer[name] if _record_settings[name]["output"] else None
            output = jax.tree_util.tree_map(lambda x: x[0], buffer) if buffer is not None else None

            seqs_in = num_seqs[name]
            step_record = base.StepRecord(
                eps=onp.array(-1, dtype=int),
                seq=onp.array(-1, dtype=int),
                ts_start=onp.array(-1, dtype=float),
                ts_end=onp.array(-1, dtype=float),
                delay=onp.array(-1, dtype=float),
                rng=rng,
                inputs=inputs,
                state=state,
                output=output,
            )
            steps = jax.tree_util.tree_map(lambda x: (jnp.ones((seqs_in,) + x.shape) * -1).astype(jax.dtypes.canonicalize_dtype(x.dtype)), step_record)
            # Initialize node record
            node_records[name] = base.NodeRecord(
                info=node.info,
                clock=Clock.COMPILED,
                real_time_factor=RealTimeFactor.FAST_AS_POSSIBLE,
                ts_start=0.0,
                params=graph_state.params[name] if _record_settings[name]["params"] else None,
                steps=steps,
                inputs=None,  # todo: use inputs instead of None (currently not logged).
            )
        new_aux = graph_state.aux.copy({"record": base.EpisodeRecord(nodes=node_records)})
        new_gs = graph_state.replace(aux=new_aux)
        return new_gs

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
        RETURN_OUTPUT = True
        assert (step_state is None) == (output is None), "Either both step_state and output must be None or both must be not None."
        # Make update state function
        update_state = make_update_state(self._supervisor_kind)
        supervisor_slot = self._supervisor_slot
        supervisor = self.supervisor
        record = graph_state.aux.get("record", None)

        if RETURN_OUTPUT:
            # Update graph state
            graph_state = graph_state.replace_step(self._timings, step=graph_state.step)  # Make sure step is clipped to max_step size
            # The step counter was already incremented in run_until_supervisor, but supervisor of the previous step was not run yet.
            # Hence, we need to grab the timings of the previous step (i.e. graph_state.step-1).
            timing = rjax.tree_take(graph_state.timings_eps.slots[supervisor_slot], i=graph_state.step - 1)
            # Define NOOP
            noop_output_record = rjax.tree_take(record.nodes[self._supervisor_kind].steps.output, timing.seq) if record is not None else None
            noop_ss = graph_state.step_state[self.supervisor.name]
            noop_output = rjax.tree_take(graph_state.buffer[self.supervisor.name], timing.seq)

            def _run_supervisor_step() -> Tuple[base.StepState, base.Output, base.Output]:
                # Get next step state and output from supervisor node
                if step_state is None and output is None:  # Run supervisor node
                    ss = noop_ss
                    _new_ss, _new_output = supervisor.step(ss)
                else:  # Override step_state and output
                    _new_ss, _new_output = step_state, output
                _new_output_record = _new_output if noop_output_record is not None else None
                return _new_ss, _new_output, _new_output_record

            def _skip_supervisor_step() -> Tuple[base.StepState, base.Output, base.Output]:
                # noop_output and noop_output_record can be different, e.g.,
                # noop_output_record is -1 everywhere, while noop_output may be a previously buffered output
                return noop_ss, noop_output, noop_output_record

            # Run supervisor node if step > 0, else skip
            new_ss, new_output, new_output_record = jax.lax.cond(graph_state.step == 0, _skip_supervisor_step, _run_supervisor_step)

            # Update graph state
            graph_state = update_state(graph_state, timing, new_ss, new_output, new_output_record)

        else:
            raise NotImplementedError("RETURN_OUTPUT=False is not implemented yet with the new record system.")
            def _run_supervisor_step() -> base.GraphState:
                # Get next step state and output from supervisor node
                if step_state is None and output is None:  # Run supervisor node
                    ss = graph_state.step_state[self.supervisor.name]
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
        next_step_state = next_graph_state.step_state[self.supervisor.name]  # Return supervisor node's step state
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
        next_step_state = next_graph_state.step_state[self.supervisor.name]  # Return supervisor node's step state
        return next_graph_state, next_step_state

    def rollout(
        self,
        graph_state: base.GraphState,
        max_steps: int = None,
        carry_only: bool = True,
    ) -> base.GraphState:
        """
        Executes the graph for a specified number of steps or until a condition is met, starting from a given step and episode.

        Utilizes the run method for execution, with an option to return only the final graph state or a sequence of all graph states.
        By virtue of using the run method, it does not allow for overriding the supervisor node's step method. That is,
        the supervisor node's step method is used during the rollout.

        Note: To record the rollout, use the init_record method on the graph_state before calling this method and
              set carry_only=True. Then, the record is available in graph_state.aux["record"].

        :param graph_state: The initial graph state.
        :param max_steps: The maximum steps to execute, if None, runs until a stop condition is met.
        :param carry_only: If True, returns only the final graph state; otherwise returns all states.
        :return: The final or sequence of graph states post-execution.
        """
        # graph_state = self.init(starting_step=starting_step, starting_eps=eps, randomize_eps=False)
        graph_state = graph_state.replace_eps(self._timings, eps=graph_state.eps)
        graph_state = graph_state.replace_step(self._timings, step=graph_state.step)
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
