from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, WARN
from rex.graph import BaseGraph
from rex.base import StepState, GraphState, RexStepReturn, RexResetReturn
from rex.env import BaseEnv
from rex.node import Node
from rex.agent import Agent as BaseAgent
from rex.spaces import Box


@struct.dataclass
class Params:
	"""Pendulum root param definition"""
	max_torque: jp.float32
	max_speed: jp.float32


@struct.dataclass
class State:
	"""Pendulum root state definition"""
	pass


@struct.dataclass
class Output:
	"""Pendulum root output definition"""
	action: jp.ndarray


class Agent(BaseAgent):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Params:
		"""Default params of the root."""
		return Params(max_torque=jp.float32(2.0), max_speed=jp.float32(22.0))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
		"""Default state of the root."""
		return State()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the root."""
		return Output(action=jp.array([0.0], dtype=jp.float32))

	# def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
	# 	"""Reset the root."""
	# 	rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
	# 	params = self.default_params(rng_params, graph_state)
	# 	state = self.default_state(rng_state, graph_state)
	# 	inputs = self.default_inputs(rng_inputs, graph_state)
	# 	return StepState(rng=rng_step, params=params, state=state, inputs=inputs)


class PendulumEnv(BaseEnv):
	root_cls = Agent

	def __init__(
			self,
			graph: BaseGraph,
			max_steps: int = 100,
			name: str = "disc-pendulum-v0"
	):
		super().__init__(graph, max_steps, name=name)

		# Required for step and reset functions
		assert "world" in self.graph.nodes, "Pendulum environment requires a world node."
		self.world = self.graph.nodes["world"]
		self.agent = self.graph.root
		self.nodes = {node.name: node for _, node in self.graph.nodes.items() if node.name != self.world.name}
		self.nodes_world_and_agent = self.graph.nodes_and_root

	def observation_space(self, params: Params = None):
		"""Observation space of the environment."""
		if params is not None:
			self.log("reloading", "Current implementation does not support custom parametrized observation spaces.")
		# assert params is None, "Current implementation does not support custom parametrized observation spaces."
		params = self.agent.default_params(jumpy.random.PRNGKey(0))
		inputs = {u.input_name: u for u in self.agent.inputs}

		# Prepare
		num_state = inputs["state"].window
		num_last_action = inputs["last_action"].window if "last_action" in inputs else 0
		high = [1.0] * num_state * 2 + [params.max_speed] * num_state + [params.max_torque] * num_last_action
		high = jp.array(high, dtype=jp.float32)
		return Box(low=-high, high=high, dtype=jp.float32)

	def action_space(self, params: Params = None):
		"""Action space of the environment."""
		if params is not None:
			self.log("reloading", "Current implementation does not support custom parametrized action spaces.")
		params = self.agent.default_params(jumpy.random.PRNGKey(0))
		return Box(low=-params.max_torque, high=params.max_torque, shape=(1,), dtype=jp.float32)

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
		"""Get the graph state."""
		# Prepare new graph state
		rng, rng_eps = jumpy.random.split(rng, num=2)
		starting_step = jp.int32(0) if graph_state is None else graph_state.step
		starting_eps = jumpy.random.choice(rng, self.graph.max_eps(), shape=()) if graph_state is None else graph_state.eps
		new_nodes = dict()
		graph_state = GraphState(nodes=new_nodes)

		# For every node, prepare the initial stepstate
		rng, rng_agent, rng_world = jumpy.random.split(rng, num=3)

		# Get new step_state
		def get_step_state(node: Node, _rng: jp.ndarray, _graph_state) -> StepState:
			"""Get new step_state for a node."""
			rng_params, rng_state, rng_step = jumpy.random.split(_rng, num=3)
			params = node.default_params(rng_params, _graph_state)
			state = node.default_state(rng_state, _graph_state)
			return StepState(rng=rng_step, params=params, state=state, inputs=None)

		# Step_state root & world (root must be reset before world, as the world may copy some params from the root)
		new_nodes[self.agent.name] = get_step_state(self.agent, rng_agent, graph_state)
		new_nodes[self.world.name] = get_step_state(self.world, rng_world, graph_state)

		# Get new step_state for other nodes in arbitrary order
		rng, *rngs = jumpy.random.split(rng, num=len(self.nodes) + 1)
		for (name, n), rng_n in zip(self.nodes.items(), rngs):
			# Replace step state in graph state
			new_nodes[name] = get_step_state(n, rng_n, graph_state)

		# Reset nodes
		rng, *rngs = jumpy.random.split(rng, num=len(self.nodes_world_and_agent) + 1)
		[n.reset(rng_reset, graph_state) for (n, rng_reset) in zip(self.nodes_world_and_agent.values(), rngs)]
		return GraphState(eps=starting_eps, step=starting_step, nodes=FrozenDict(new_nodes))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> RexResetReturn:
		"""Reset environment."""
		new_graph_state = self._get_graph_state(rng, graph_state)

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, step_state = self.graph.reset(new_graph_state)

		# Get observation
		obs = self._get_obs(step_state)
		info = {}
		return graph_state, obs, info

	def step(self, graph_state: GraphState, action: jp.ndarray) -> RexStepReturn:
		"""Perform step transition in environment."""
		# Update step_state (if necessary)
		step_state = self.agent.get_step_state(graph_state)
		new_step_state = step_state

		# Prepare output action
		u = Output(action=action)
		# th = step_state.inputs["state"].data.th[0]
		# thdot = step_state.inputs["state"].data.thdot[0]
		# x = jp.array([th, thdot])
		# print(f"{self.name.ljust(14)} | x: {x} | u: {u.action[0]}")

		# Apply step and receive next step_state
		graph_state, step_state = self.graph.step(graph_state, new_step_state, u)

		# Get observation
		obs = self._get_obs(step_state)
		th = self._angle_normalize(step_state.inputs["state"].data.th[-1])  # Normalize angle
		thdot = step_state.inputs["state"].data.thdot[-1]

		# Calculate cost (penalize angle error, angular velocity and input voltage)
		# cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * action[0] ** 2
		cost = th ** 2 + 0.01 * thdot ** 2 + 0.001 * (action[0] ** 2)

		# Determine done flag
		terminated = self._is_terminal(graph_state)
		truncated = graph_state.step >= self.max_steps
		info = {"TimeLimit.truncated": graph_state.step >= self.max_steps}

		return graph_state, obs, -cost, terminated, truncated, info

	def _is_terminal(self, graph_state: GraphState) -> bool:
		return False

	def _get_obs(self, step_state: StepState) -> Any:
		"""Get observation from environment."""
		inputs = step_state.inputs
		th = inputs["state"].data.th
		thdot = inputs["state"].data.thdot
		last_action = inputs["last_action"].data.action[:, 0] if 'last_action' in inputs else jp.array([])
		obs = jp.concatenate([jp.cos(th), jp.sin(th), thdot, last_action], axis=-1)
		return obs

	def _angle_normalize(self, th: jp.array):
		th_norm = th - 2 * jp.pi * jp.floor((th + jp.pi) / (2 * jp.pi))
		return th_norm
