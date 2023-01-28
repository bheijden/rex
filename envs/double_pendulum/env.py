from typing import Any, Dict, Tuple, Union
import jumpy
import jumpy.numpy as jp
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
	max_speed: jp.float32  # todo: max_speed high enough? (also defined in ode.world.py


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
		return Params(max_torque=jp.float32(4.0), max_speed=jp.float32(22.0))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
		"""Default state of the agent."""
		return State()

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
		"""Default output of the agent."""
		return Output(action=jp.array([0.0], dtype=jp.float32))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the agent."""
		rng_params, rng_state, rng_inputs, rng_step = jumpy.random.split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)


class DoublePendulumEnv(BaseEnv):
	agent_cls = Agent

	def __init__(
			self,
			nodes: Dict[str, "Node"],
			agent: Agent,
			max_steps: int = 320,  # 4*rate
			trace: log_pb2.TraceRecord = None,
			clock: int = SIMULATED,
			real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
			graph: int = INTERPRETED,
			name: str = "double-pendulum-v0"
	):
		# Exclude the node for which this environment is a drop-in replacement (i.e. the agent)
		nodes = {node.name: node for _, node in nodes.items() if node.name != agent.name}
		super().__init__(nodes, agent, max_steps, clock, real_time_factor, graph, trace, name=name)

		# Required for step and reset functions
		assert "world" in nodes, "Double-pendulum environment requires a world node."
		self.world = nodes["world"]
		self.agent = agent
		self.nodes = {node.name: node for _, node in nodes.items() if node.name != self.world.name}

	def observation_space(self, params: Params = None):
		"""Observation space of the environment."""
		params = self.agent.default_params(jumpy.random.PRNGKey(0)) if params is None else params
		inputs = {u.input_name: u for u in self.agent.inputs}

		# Prepare
		num_state = inputs["state"].window
		num_last_action = inputs["last_action"].window if "last_action" in inputs else 0
		high = [1.0] * num_state * 4 + [params.max_speed] * num_state * 2 + [params.max_torque] * num_last_action
		high = jp.array(high, dtype=jp.float32)
		return Box(low=-high, high=high, dtype=jp.float32)

	def action_space(self, params: Params = None):
		"""Action space of the environment."""
		params = self.agent.default_params(jumpy.random.PRNGKey(0)) if params is None else params
		return Box(low=-params.max_torque, high=params.max_torque, shape=(1,), dtype=jp.float32)

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
		"""Get the graph state."""
		# if graph_state is not None:
		# 	raise NotImplementedError("Supplying a graph state is not yet supported for Pendulum environment.")

		# Prepare new graph state
		new_nodes = dict()
		graph_state = GraphState(nodes=new_nodes)

		# For every node, prepare the initial stepstate
		rng, rng_agent, rng_world = jumpy.random.split(rng, num=3)

		# Reset world and agent (in that order)
		new_nodes[self.world.name] = self.world.reset(rng_world, graph_state)  # Reset world node (must be done first).

		# Reset agent node.
		new_nodes[self.agent.name] = self.agent.reset(rng_agent, graph_state)

		# Reset other nodes in arbitrary order
		rngs = jumpy.random.split(rng, num=len(self.nodes))
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
		# todo: where is the equilibrium point? Make sure the observations are smooth there.
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
		graph_state, ts, step_state = self.graph.step(graph_state, new_step_state, u)

		# Get observation
		# todo: what is up & down?
		obs = self._get_obs(step_state)
		th = self._angle_normalize(step_state.inputs["state"].data.th[-1])  # Normalize angle
		th2 = self._angle_normalize(step_state.inputs["state"].data.th2[-1])  # Normalize angle
		thdot = step_state.inputs["state"].data.thdot[-1]
		thdot2 = step_state.inputs["state"].data.thdot2[-1]

		# Calculate cost (penalize angle error, angular velocity and input voltage)
		# cost = th2 ** 2 + 0.1 * (thdot2 / (1 + 10 * abs(th2))) ** 2 + 0.01 * action[0] ** 2
		# cost = th2 ** 2 + 0.1 * thdot2 ** 2 + 0.01 * action[0] ** 2
		# cost = (jp.pi - jp.abs(th + th2)) ** 2 + 0.1 * thdot + 0.1 * thdot2 ** 2 + 0.01 * action[0] ** 2

		u = action[0]
		vel = jp.array([thdot, thdot2])
		angle = jp.array([jp.cos(th), jp.sin(th), jp.cos(th2), jp.sin(th2)]).reshape((4, 1))
		target_vel = jp.array([-0, 0])

		pos = jp.array([angle[0][0] + jp.cos(th + th2), angle[1][0] + jp.sin(th + th2)])
		target_pos = jp.array([-2, 0])
		L2_pos = (pos - target_pos).T @ (pos - target_pos)

		cost = 0
		cost -= jp.where(L2_pos < 0.25, 200, 0)
		cost -= jp.where(L2_pos < 0.05, 600, 0)
		_tmp = jp.where(L2_pos < 0.05, 800 + 100 * jp.exp(20 * (0.3 - jp.abs(vel[1]))), 0)
		cost -= jp.where(jp.abs(vel[1]) <= 0.3, _tmp, 0)
		cost -= jp.where(jp.abs(th) > jp.pi / 2, (abs(th) - jp.pi / 2) * 40 * (jp.pi / 2 - abs(th2)), 0)
		D2 = jp.where(L2_pos < 0.25, jp.diag([1, 4]), jp.diag([0.05, 0.25]))
		D2 = jp.where(L2_pos < 0.05, jp.diag([5, 15]), D2)

		Dp = jp.diag([20, 10])
		cost += (pos - target_pos).T @ Dp @ (pos - target_pos) + (vel - target_vel).T @ D2 @ (
					vel - target_vel) + 0.001 * u ** 2

		# if ((pos - target_pos).T @ (pos - target_pos)) < 0.25:
		# 	cost -= 200
		# 	D2 = jp.diag([1, 4])
		# 	PRINT = True
		# 	if ((pos - target_pos).T @ (pos - target_pos)) < 0.05:
		# 		cost -= 600
		# 		D2 = jp.diag([5, 15])
		# 		# print("vel_cost:  ", (vel - target_vel).T @ D2 @ (vel - target_vel), vel)
		# 		if abs(vel[1]) <= 0.3:
		# 			cost -= 800
		# 			# print("vel_reward:  ", 100 * jp.exp(20 * (0.3 - abs(vel[1]))), vel)
		# 			cost -= 100 * jp.exp(20 * (0.3 - abs(vel[1])))
		# cost += (pos - target_pos).T @ Dp @ (pos - target_pos) + (vel - target_vel).T @ D2 @ (
		# 			vel - target_vel) + 0.001 * u ** 2
		# if abs(th) > jp.pi / 2:
		# 	cost -= (abs(th) - jp.pi / 2) * 40 * (jp.pi / 2 - abs(th2))

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
		th2 = inputs["state"].data.th2
		thdot = inputs["state"].data.thdot
		thdot2 = inputs["state"].data.thdot2
		last_action = inputs["last_action"].data.action[:, 0] if 'last_action' in inputs else jp.array([])
		obs = jp.concatenate([jp.cos(th), jp.sin(th), jp.cos(th2), jp.sin(th2), thdot, thdot2, last_action], axis=-1)
		return obs

	def _angle_normalize(self, th: jp.array):
		th_norm = th - 2 * jp.pi * jp.floor((th + jp.pi) / (2 * jp.pi))
		return th_norm
