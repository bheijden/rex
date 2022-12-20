from rex.base import GraphState
from rex.graph import BaseGraph
from rex.agent import Agent
from typing import Any, Tuple, Dict
import gym
import jumpy as jp
import abc


class BaseEnv:
	def __init__(self, graph: "BaseGraph", agent: Agent, max_steps: int):
		# Check that the agent is of the correct type
		assert isinstance(agent, Agent), "The agent must be an instance of Agent"
		self.agent = agent
		self.max_steps = max_steps
		self.graph = graph
		assert self.max_steps > 0 and isinstance(self.max_steps, int), "max_steps must be a positive integer"
		if hasattr(self.graph, "max_steps"):
			assert self.max_steps <= self.graph.max_steps, f"max_steps ({self.max_steps}) must be smaller than the max_steps of the graph ({self.graph.max_steps})."

	@abc.abstractmethod
	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> Tuple[GraphState, Any]:
		raise NotImplementedError

	@abc.abstractmethod
	def step(self, graph_state: GraphState, action: Any) -> Tuple[GraphState, Any, float, bool, Dict]:
		raise NotImplementedError

	def close(self):
		pass

	def stop(self):
		return self.graph.stop()

	def render(self):
		raise NotImplementedError

	def action_space(self, graph_state: GraphState) -> gym.Space:
		"""Action space of the environment."""
		raise NotImplementedError

	def observation_space(self, graph_state: GraphState) -> gym.Space:
		"""Observation space of the environment."""
		raise NotImplementedError
