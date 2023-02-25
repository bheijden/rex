from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, INTERPRETED, WARN
from rex.base import StepState, GraphState
from rex.env import BaseEnv
from rex.node import Node
from rex.agent import Agent as BaseAgent
from rex.spaces import Box


@struct.dataclass
class Params:
	"""Pendulum agent param definition"""
	max_torque: jp.float32
	max_speed: jp.float32


@struct.dataclass
class State:
	"""Pendulum agent state definition"""
	pass


@struct.dataclass
class Output:
	"""Pendulum agent output definition"""
	action: jp.ndarray


class Agent(BaseAgent):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Params:
		"""Default params of the agent."""
		return Params(max_torque=jp.float32(2.0), max_speed=jp.float32(22.0))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
		"""Default state of the agent."""
		return State()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the agent."""
		return Output(action=jp.array([0.0], dtype=jp.float32))

	# def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
	# 	"""Reset the agent."""
	# 	rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
	# 	params = self.default_params(rng_params, graph_state)
	# 	state = self.default_state(rng_state, graph_state)
	# 	inputs = self.default_inputs(rng_inputs, graph_state)
	# 	return StepState(rng=rng_step, params=params, state=state, inputs=inputs)


class PendulumEnv(BaseEnv):
	agent_cls = Agent

	def __init__(
			self,
			nodes: Dict[str, "Node"],
			agent: Agent,
			max_steps: int = 100,
			trace: log_pb2.TraceRecord = None,
			clock: int = SIMULATED,
			real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
			graph: int = INTERPRETED,
			name: str = "disc-pendulum-v0"
	):
		# Exclude the node for which this environment is a drop-in replacement (i.e. the agent)
		nodes = {node.name: node for _, node in nodes.items() if node.name != agent.name}
		super().__init__(nodes, agent, max_steps, clock, real_time_factor, graph, trace, name=name)

		# Required for step and reset functions
		assert "world" in nodes, "Pendulum environment requires a world node."
		self.world = nodes["world"]
		self.agent = agent
		self.nodes = {node.name: node for _, node in nodes.items() if node.name != self.world.name}
		self.nodes_world_and_agent = self.graph.nodes_and_agent

	def observation_space(self, params: Params = None):
		"""Observation space of the environment."""
		assert params is None, "Current implementation does not support custom parametrized observation spaces."
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
		assert params is None, "Current implementation does not support custom parametrized action spaces."
		params = self.agent.default_params(jumpy.random.PRNGKey(0))
		return Box(low=-params.max_torque, high=params.max_torque, shape=(1,), dtype=jp.float32)

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
		"""Get the graph state."""
		# Prepare new graph state
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

		# Step_state agent & world (agent must be reset before world, as the world may copy some params from the agent)
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
		return GraphState(step=jp.int32(0), nodes=FrozenDict(new_nodes))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> Tuple[GraphState, Any]:
		"""Reset environment."""
		new_graph_state = self._get_graph_state(rng, graph_state)

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, step_state = self.graph.reset(new_graph_state)

		# Get observation
		obs = self._get_obs(step_state)
		return graph_state, obs

	def step(self, graph_state: GraphState, action: jp.ndarray) -> Tuple[GraphState, jp.ndarray, float, bool, Dict]:
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
		cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * action[0] ** 2

		# Determine done flag
		done = self._is_terminal(graph_state)
		info = {"TimeLimit.truncated": graph_state.step >= self.max_steps}

		return graph_state, obs, -cost, done, info

	def _is_terminal(self, graph_state: GraphState) -> bool:
		return graph_state.step >= self.max_steps

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
