import abc
from typing import Dict, Tuple, List
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import numpy as np

# import jumpy
# import jumpy.numpy as jp
from flax.core import FrozenDict

from rex.base import StepState, StepStates, GraphState, Output
from rex.proto import log_pb2
from rex.node import Node


class BaseGraph:
    def __init__(self, root: Node, nodes: Dict[str, Node], skip: List[str]):
        # Exclude the node for which this environment is a drop-in replacement (i.e. the root)
        nodes = {node.name: node for _, node in nodes.items() if node.name != root.name}
        _assert = len([n for n in nodes.values() if n.name == root.name]) == 0
        assert _assert, "The root should be provided separately, so not inside the `nodes` dict"
        self.root = root
        self.nodes = nodes
        self.nodes_and_root: Dict[str, Node] = {**nodes, root.name: root}
        self._skip = skip if isinstance(skip, list) else [skip] if isinstance(skip, str) else []

    def __getstate__(self):
        raise NotImplementedError
        # args, kwargs = (), dict(root=self.root, nodes=self.nodes)
        # return args, kwargs

    def __setstate__(self, state):
        args, kwargs = state

        # Unpickle nodes
        nodes, root = kwargs["nodes"], kwargs["root"]
        nodes_and_root = {**nodes, root.name: root}
        for node in nodes_and_root.values():
            node.unpickle(nodes_and_root)

        self.__init__(*args, **kwargs)

    def init(self, rng: ArrayLike = None, step_states: StepStates = None, starting_eps: ArrayLike = 0,
             randomize_eps: bool = False, order: Tuple[str, ...] = None):
        # Determine initial step states
        step_states = step_states if step_states is not None else {}
        step_states = step_states.unfreeze() if isinstance(step_states, FrozenDict) else step_states

        if rng is None:
            rng = jax.random.PRNGKey(0)

        if randomize_eps:
            rng, rng_eps = jax.random.split(rng, num=2)
            starting_eps = jax.random.choice(rng, self.max_eps(), shape=())

        # Determine init order. If name not in order, add it to the end
        order = tuple() if order is None else order
        order = list(order)
        for name in [self.root.name] + list(self.nodes.keys()):
            if name not in order:
                order.append(name)

        # Initialize temporary graph state
        graph_state = GraphState(eps=jnp.int32(starting_eps), nodes=step_states)

        # Initialize step states
        rngs = jax.random.split(rng, num=len(order*4)).reshape((len(order), 4, 2))
        rngs_inputs = {}
        for rngs_ss, name in zip(rngs, order):
            # Unpack rngs
            rng_params, rng_state, rng_step, rng_inputs = rngs_ss
            node = self.nodes_and_root[name]
            # Grab preset params and state if available
            preset_params = step_states[name].params if name in step_states else None
            preset_state = step_states[name].state if name in step_states else None
            # Params first, because the state may depend on them
            params = node.default_params(rng_params, graph_state) if preset_params is None else preset_params
            step_states[node.name] = StepState(rng=rng_step, params=params, state=None, inputs=None)
            # Then, get the state (which may depend on the params)
            state = node.default_state(rng_state, graph_state) if preset_state is None else preset_state
            # Inputs are updated once all nodes have been initialized with their params and state
            step_states[name] = StepState(rng=rng_step, params=params, state=state, inputs=None, eps=starting_eps, seq=np.int32(0), ts=np.float32(0.))
            rngs_inputs[name] = rng_inputs

        # Initialize inputs
        for name, rng_inputs in rngs_inputs.items():
            if name in self._skip:
                continue
            node = self.nodes_and_root[name]
            step_states[name] = step_states[name].replace(inputs=node.default_inputs(rng_inputs, graph_state))
        # NOTE: usd to be eps=jp.as_int32(starting_eps) --> why?
        return GraphState(eps=starting_eps, nodes=FrozenDict(step_states))

    def start(self, graph_state: GraphState, timeout: float = None) -> GraphState:
        return graph_state

    def stop(self, timeout: float = None):
        pass

    @abc.abstractmethod
    def run_until_root(self, graph_state: GraphState) -> GraphState:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abc.abstractmethod
    def run_root(self, graph_state: GraphState, step_state: StepState = None, output: Output = None) -> GraphState:
        """Runs root node if step_state and output are not provided. Otherwise, overrides step_state and output with provided values."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run(self, graph_state: GraphState) -> GraphState:
        """Runs graph (incl. root) for one step and returns new graph state.
        This means the graph_state *after* the root step is returned.
        """
        # todo: check if start() was called before
        # Runs supergraph (except for root)
        graph_state = self.run_until_root(graph_state)

        # Runs root node if no step_state or output is provided, otherwise uses provided step_state and output
        graph_state = self.run_root(graph_state)
        return graph_state

    def reset(self, graph_state: GraphState, timeout: float = None) -> Tuple[GraphState, StepState]:
        """Resets graph and returns before root node would run (follows gym API)."""
        # Runs supergraph (except for root)
        self.stop(timeout=timeout)
        graph_state = self.start(graph_state, timeout=timeout)
        next_graph_state = self.run_until_root(graph_state)
        next_step_state = self.root.get_step_state(next_graph_state)  # Return root node's step state
        return next_graph_state, next_step_state

    def step(self, graph_state: GraphState, step_state: StepState = None, output: Output = None) -> Tuple[GraphState, StepState]:
        """Runs graph for one step and returns before root node would run (follows gym API).
        - If step_state and output are provided, the root node's step is not run and the
        provided step_state and output are used instead.
        - Calling step() repeatedly is equivalent to calling run() repeatedly, except that
        step() returns the root node's step state *before* the root node is run, while run()
        returns the root node's step state *after* the root node is run.
        - Only calling step() after init() without reset() is possible, but note that step()
        starts by running the root node. But, because an episode should start with a run_until_root(),
        the first root step call is skipped.
        """
        # Runs root node (if step_state and output are not provided, otherwise overrides step_state and output with provided values)
        new_graph_state = self.run_root(graph_state, step_state, output)

        # Runs supergraph (except for root)
        next_graph_state = self.run_until_root(new_graph_state)
        next_step_state = self.root.get_step_state(next_graph_state)  # Return root node's step state
        return next_graph_state, next_step_state

    def get_episode_record(self) -> log_pb2.EpisodeRecord:
        raise NotImplementedError

    def max_eps(self, graph_state: GraphState = None):
        raise NotImplementedError

    def max_steps(self, graph_state: GraphState = None) -> int:
        raise NotImplementedError

    def max_runs(self, graph_state: GraphState = None) -> int:
        return self.max_steps(graph_state) + 1

    def max_starting_step(self, max_steps: int, graph_state: GraphState = None) -> int:
        raise NotImplementedError
