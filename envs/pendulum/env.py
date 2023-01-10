from typing import Any, Dict, Tuple, Union
import jumpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, INTERPRETED, WARN
from rex.base import InputState, StepState, GraphState
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

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the agent."""
		rng_params, rng_state, rng_inputs, rng_step = jp.random_split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)


class PendulumEnv(BaseEnv):
	def __init__(
			self,
			nodes: Dict[str, "Node"],
			agent: Agent,
			max_steps: int = 100,
			trace: log_pb2.TraceRecord = None,
			sync: int = SYNC,
			clock: int = SIMULATED,
			scheduling: int = PHASE,
			real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
			graph: int = INTERPRETED,
			name: str = "pendulum-v0",
	):
		# Exclude the node for which this environment is a drop-in replacement (i.e. the agent)
		nodes = {node.name: node for _, node in nodes.items() if node.name != agent.name}
		super().__init__(nodes, agent, max_steps, sync, clock, scheduling, real_time_factor, graph, trace, name=name)

		# Required for step and reset functions
		assert "world" in nodes, "Pendulum environment requires a world node."
		self.world = nodes["world"]
		self.agent = agent
		self.nodes = {node.name: node for _, node in nodes.items() if node.name != self.world.name}

	def observation_space(self, params: Params = None):
		"""Observation space of the environment."""
		params = self.agent.default_params(jp.random_prngkey(0)) if params is None else params
		high = jp.array([1.0, 1.0, params.max_speed], dtype=jp.float32)
		return Box(low=-high, high=high, dtype=jp.float32)

	def action_space(self, params: Params = None):
		"""Action space of the environment."""
		params = self.agent.default_params(jp.random_prngkey(0)) if params is None else params
		return Box(low=-1, high=1, shape=(1,), dtype=jp.float32)

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
		"""Get the graph state."""
		# if graph_state is not None:
		# 	raise NotImplementedError("Supplying a graph state is not yet supported for Pendulum environment.")

		# Prepare new graph state
		new_nodes = dict()
		graph_state = GraphState(nodes=new_nodes)

		# For every node, prepare the initial stepstate
		rng, rng_agent, rng_world = jp.random_split(rng, num=3)

		# Reset world and agent (in that order)
		new_nodes[self.world.name] = self.world.reset(rng_world, graph_state)  # Reset world node (must be done first).
		new_nodes[self.agent.name] = self.agent.reset(rng_agent, graph_state)  # Reset agent node.

		# Reset other nodes in arbitrary order
		rngs = jp.random_split(rng, num=len(self.nodes))
		for (name, n), rng_reset in zip(self.nodes.items(), rngs):
			# Reset node and optionally provide params, state, inputs
			new_ss = n.reset(rng_reset, graph_state)  # can provide params, state, inputs here

			# Replace step state in graph state
			new_nodes[name] = new_ss

		return GraphState(step=jp.int32(0), nodes=FrozenDict(new_nodes))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> Tuple[GraphState, Any]:
		"""Reset environment."""
		new_graph_state = self._get_graph_state(rng, graph_state)

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, ts, step_state = self.graph.reset(new_graph_state)

		# Get observation
		obs = self._get_obs(step_state)
		return graph_state, obs

	def step(self, graph_state: GraphState, action: jp.ndarray) -> Tuple[GraphState, InputState, float, bool, Dict]:
		"""Perform step transition in environment."""
		# Update step_state (if necessary)
		step_state = self.agent.get_step_state(graph_state)
		new_step_state = step_state

		# Prepare output action
		u = Output(action=action*step_state.params.max_torque)
		# th = step_state.inputs["state"].data.th[0]
		# thdot = step_state.inputs["state"].data.thdot[0]
		# x = jp.array([th, thdot])
		# print(f"{self.name.ljust(14)} | x: {x} | u: {u.action[0]}")

		# Apply step and receive next step_state
		graph_state, ts, step_state = self.graph.step(graph_state, new_step_state, u)

		# Get observation
		obs = self._get_obs(step_state)
		th = self._angle_normalize(step_state.inputs["state"].data.th[0])  # Normalize angle
		thdot = step_state.inputs["state"].data.thdot[0]

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
		th = step_state.inputs["state"].data.th[0]
		thdot = step_state.inputs["state"].data.thdot[0]
		return jp.array([jp.cos(th), jp.sin(th), thdot])

	def _angle_normalize(self, th: jp.array):
		th_norm = th - 2 * jp.pi * jp.floor((th + jp.pi) / (2 * jp.pi))
		return th_norm