from typing import Any, Dict, Tuple, Callable

import jax
import jumpy
import jumpy.numpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.constants import WARN, LATEST, PHASE, FAST_AS_POSSIBLE, SIMULATED, VECTORIZED, INTERPRETED
from rex.base import StepState, GraphState, Empty
from rex.node import Node
from rex.agent import Agent as BaseAgent
from rex.env import BaseEnv
from rex.proto import log_pb2
import rex.jumpy as rjp


class Replay:
	def __init__(self, node: Node, outputs, states, params):
		super(Replay, self).__setattr__("_node", node)
		super(Replay, self).__setattr__("_outputs", outputs)
		super(Replay, self).__setattr__("_states", states)
		super(Replay, self).__setattr__("_params", params)

		# Replace references to the wrapped node with this node.
		node.output.node = self
		for i in node.output.inputs:
			i.output = self

	def __setstate__(self, state):
		"""Used for unpickling"""
		raise NotImplementedError

	def __setattr__(self, key, value):
		setattr(self._node, key, value)

	def __getattr__(self, item):
		return getattr(self._node, item)

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None):
		return self._node.default_output(rng, graph_state)

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None):
		if graph_state.step is not None:
			return rjp.tree_take(self._states, graph_state.step, axis=0)
		else:
			return self._node.default_state(rng, graph_state)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None):
		if graph_state.step is not None:
			return rjp.tree_take(self._params, graph_state.step, axis=0)
		else:
			return self._node.default_params(rng, graph_state)

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None):
		return self._node.reset(rng, graph_state)

	def step(self, step_state: StepState) -> Tuple[StepState, Any]:
		seq = step_state.seq
		output = rjp.tree_take(self._outputs, seq, axis=0)
		return step_state, output

	@property
	def unwrapped(self):
		return self._node


@struct.dataclass
class Loss:
	cum_loss: Any
	unwrapped: Any


class ReconstructionLoss:
	def __init__(self, node: Node, outputs):
		super(ReconstructionLoss, self).__setattr__("_node", node)
		super(ReconstructionLoss, self).__setattr__("_outputs", outputs)

		# Replace references to the wrapped node with this node.
		node.output.node = self
		for i in node.output.inputs:
			i.output = self

	def __setstate__(self, state):
		"""Used for unpickling"""
		raise NotImplementedError

	def __setattr__(self, key, value):
		setattr(self._node, key, value)

	def __getattr__(self, item):
		return getattr(self._node, item)

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None):
		return self._node.default_output(rng, graph_state)

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None):
		state = self._node.default_state(rng, graph_state)
		output = self._node.default_output(rng, graph_state)
		cum_loss = jax.tree_util.tree_map(lambda x: jp.zeros_like(x), output)
		return Loss(cum_loss=cum_loss, unwrapped=state)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None):
		return self._node.default_params(rng, graph_state)

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None):
		return self._node.reset(rng, graph_state)

	def step(self, step_state: StepState) -> Tuple[StepState, Any]:
		# Run step with unwrapped state
		unwrapped = step_state.replace(state=step_state.state.unwrapped)
		new_unwrapped, output = self._node.step(unwrapped)

		# Calculate squared loss of
		seq = step_state.seq
		target = rjp.tree_take(self._outputs, seq, axis=0)

		# Calculate loss
		cum_loss = step_state.state.cum_loss
		new_cum_loss = jax.tree_util.tree_map(lambda l, x, y: l + (x-y)**2, cum_loss, target, output)
		new_step_state = step_state.replace(state=Loss(cum_loss=new_cum_loss, unwrapped=new_unwrapped.state))
		return new_step_state, output

	@property
	def unwrapped(self):
		return self._node


@struct.dataclass
class EstimatorParams:
	world_states: Any


@struct.dataclass
class EstimatorState:
	starting_step: jp.int32


class Estimator(BaseAgent):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> EstimatorParams:
		if graph_state is None or graph_state.nodes.get("estimator", None) is None:
			raise NotImplementedError("Estimator must be initialized with a graph state containing an estimator node")
		return graph_state.nodes["estimator"].params

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> EstimatorState:
		return EstimatorState(starting_step=graph_state.step)

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
		return Empty()


class EstimatorEnv(BaseEnv):
	agent_cls = Estimator

	def __init__(
			self,
			nodes: Dict[str, "Node"],
			root: Estimator,
			trace: log_pb2.TraceRecord,
			loss_fn: Callable[[GraphState], Any],
			max_steps: int = 1,
			graph_type: int = VECTORIZED,
			name: str = "estimator-v0",

	):
		assert graph_type not in [INTERPRETED]

		# Exclude the node for which this environment is a drop-in replacement (i.e. the root)
		nodes = {node.name: node for _, node in nodes.items() if node.name != root.name}
		super().__init__(nodes, root, max_steps, SIMULATED, FAST_AS_POSSIBLE, graph_type, trace, name=name)

		# Required for step and reset functions
		assert "world" in nodes, "Double-pendulum environment requires a world node."
		self.loss_fn = loss_fn
		self.world = nodes["world"]
		self.agent = root
		self.estimator = root
		self.nodes = {node.name: node for _, node in nodes.items() if node.name != self.world.name}
		self.nodes_world_and_agent = self.graph.nodes_and_root

	# def sample_starting_step(self, rng: jp.ndarray):
	# 	return jumpy.random.randint(rng, shape=(), low=0, high=self._max_starting_step + 1)  # High is never selected.

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState) -> GraphState:
		"""Get the graph state."""
		# Prepare new graph state
		assert graph_state.step is not None, "Graph state must have a step index."
		assert graph_state.nodes.get("estimator", None) is not None, "Graph state must have an estimator node."
		prev_graph_state = graph_state
		starting_step = prev_graph_state.step
		outputs = prev_graph_state.outputs
		new_nodes = prev_graph_state.nodes.unfreeze()

		# For every node, prepare the initial stepstate
		rng, rng_estimator, rng_agent, rng_world = jumpy.random.split(rng, num=4)

		# Get new step_state
		def get_step_state(node: Node, _rng: jp.ndarray, _graph_state) -> StepState:
			"""Get new step_state for a node."""
			rng_params, rng_state, rng_step = jumpy.random.split(_rng, num=3)
			ss = _graph_state.nodes.get(node.name, None)
			if ss is not None and ss.params is not None:
				params = _graph_state.nodes[node.name].params
			else:
				params = node.default_params(rng_params, _graph_state)
			state = node.default_state(rng_state, _graph_state)
			return StepState(rng=rng_step, params=params, state=state)

		# Define new graph state
		graph_state = GraphState(nodes=new_nodes, step=starting_step, outputs=outputs)

		# Step_state root & world (root must be reset before world, as the world may copy some params from the root)
		new_nodes["root"] = get_step_state(self.nodes["root"], rng_agent, graph_state)
		new_nodes[self.world.name] = get_step_state(self.world, rng_world, graph_state)
		new_nodes[self.estimator.name] = get_step_state(self.estimator, rng_estimator, graph_state)  # NOTE: isinstance(self.root, Estimator)

		# Get new step_state for other nodes in arbitrary order
		rng, *rngs = jumpy.random.split(rng, num=len(self.nodes) + 1)
		for (name, n), rng_n in zip(self.nodes.items(), rngs):
			if name == "root" or name == self.estimator.name or name == self.world.name:
				continue
			# Replace step state in graph state
			new_nodes[name] = get_step_state(n, rng_n, graph_state)

		# Set initial state of world
		new_state = rjp.tree_take(new_nodes["estimator"].params.world_states, starting_step)
		new_nodes["world"] = new_nodes["world"].replace(state=new_state)

		# Reset nodes
		rng, *rngs = jumpy.random.split(rng, num=len(self.nodes_world_and_agent) + 1)
		[n.reset(rng_reset, graph_state) for (n, rng_reset) in zip(self.nodes_world_and_agent.values(), rngs)]
		return GraphState(nodes=FrozenDict(new_nodes), step=starting_step, outputs=outputs)

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> Tuple[GraphState, Any]:
		"""Reset environment."""
		new_graph_state = self._get_graph_state(rng, graph_state)

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, step_state = self.graph.reset(new_graph_state)

		# Calculate loss
		loss = self.loss_fn(graph_state)
		return graph_state, loss

	def step(self, graph_state: GraphState, action: Any) -> Tuple[GraphState, Any, float, bool, Dict]:
		"""Perform step transition in environment."""
		# Update step_state (if necessary)
		new_step_state = self.agent.get_step_state(graph_state)

		# Apply step and receive next step_state
		graph_state, step_state = self.graph.step(graph_state, new_step_state, Empty())

		# Get observation
		loss = self.loss_fn(graph_state)

		# Termination condition
		done = graph_state.step >= self._cgraph.max_steps
		info = {"TimeLimit.truncated": done}

		return graph_state, loss, 0., done, info