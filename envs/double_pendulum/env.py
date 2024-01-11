from typing import Any, Dict, Tuple, Union
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from rex.graph import BaseGraph
from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, WARN
from rex.base import StepState, GraphState, RexStepReturn, RexResetReturn
from rex.env import BaseEnv
from rex.node import Node
from rex.spaces import Box


@struct.dataclass
class Params:
	"""Pendulum param definition"""
	max_torque: Union[float, jax.typing.ArrayLike]
	max_speed: Union[float, jax.typing.ArrayLike]
	max_speed2: Union[float, jax.typing.ArrayLike]
	length: Union[float, jax.typing.ArrayLike]
	length2: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class State:
	"""Pendulum root state definition"""
	reward: Union[float, jax.typing.ArrayLike]
	starting_step: Union[int, jax.typing.ArrayLike]


@struct.dataclass
class Output:
	"""Pendulum root output definition"""
	action: jax.typing.ArrayLike


class Agent(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> Params:
		"""Default params of the root."""
		return Params(max_torque=8.0,
		              max_speed=50.0,
		              max_speed2=50.0,
		              length=0.1,
		              length2=0.1)

	def default_state(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> State:
		"""Default state of the root."""
		return State(reward=0., starting_step=0)

	def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> Output:
		"""Default output of the root."""
		return Output(action=jnp.array([0.0], dtype=jnp.float32))

	def step(self, step_state: StepState) -> Tuple[StepState, Output]:
		return step_state, self.default_output(step_state.rng)


class DoublePendulumEnv(BaseEnv):
	root_cls = Agent

	def __init__(
			self,
			graph: BaseGraph,
			max_steps: int = 400,  # 4*rate
			name: str = "double-pendulum-v0"
	):
		super().__init__(graph, max_steps, name=name)

		# Required for step and reset functions
		assert "world" in self.graph.nodes, "Double-pendulum environment requires a world node."
		self.world = self.graph.nodes["world"]
		self.agent = self.graph.root
		self.nodes = {node.name: node for _, node in self.graph.nodes.items() if node.name != self.world.name}
		self.nodes_world_and_agent = self.graph.nodes_and_root

	def observation_space(self, params: Params = None):
		"""Observation space of the environment."""
		assert params is None, "Current implementation does not support custom parametrized observation spaces."
		params = self.agent.default_params(jax.random.PRNGKey(0))
		inputs = {u.input_name: u for u in self.agent.inputs}

		# Prepare
		num_state = inputs["state"].window
		num_last_action = inputs["last_action"].window if "last_action" in inputs else 0
		high = [1.0] * num_state * 4 + [32.0] * num_state + [32.0] * num_state + [params.max_torque] * num_last_action
		# high = [1.0] * num_state * 4 + [params.max_torque] * num_last_action
		high = jnp.array(high, dtype=jnp.float32)
		return Box(low=-high, high=high, dtype=jnp.float32)

	def action_space(self, params: Params = None):
		"""Action space of the environment."""
		assert params is None, "Current implementation does not support custom parametrized action spaces."
		params = self.agent.default_params(jax.random.PRNGKey(0))
		return Box(low=-params.max_torque, high=params.max_torque, shape=(1,), dtype=jnp.float32)

	def reset(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> RexResetReturn:
		"""Reset environment."""
		new_graph_state = self.graph.init(rng, graph_state, order=("agent", "world"))

		# Reset nodes
		rng, *rngs = jax.random.split(rng, num=len(self.nodes_world_and_agent) + 1)
		[n.reset(rng_reset, new_graph_state) for (n, rng_reset) in zip(self.nodes_world_and_agent.values(), rngs)]

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, step_state = self.graph.reset(new_graph_state)

		# Get observation
		obs = self._get_obs(step_state)
		info = {}
		return graph_state, obs, info

	def step(self, graph_state: GraphState, action: jax.typing.ArrayLike) -> RexStepReturn:
		"""Perform step transition in environment."""
		# Update step_state (if necessary)
		new_step_state = self.agent.get_step_state(graph_state)

		# Prepare output action
		u = Output(action=action)

		# Apply step and receive next step_state
		graph_state, step_state = self.graph.step(graph_state, new_step_state, u)

		# Get observation
		# goal: th2 = jnp.pi - th
		# [th, th2] = [0, 0] = th=down, th2=down
		# [th, th2] = [pi, 0] = th=up, th2=up
		# [th, th2] = [0, pi] = th=down , th2=up
		# [th, th2] = [pi, pi] = th=up, th2=down
		obs = self._get_obs(step_state)
		last_obs = step_state.inputs["state"][-1].data
		last_action = step_state.inputs["last_action"][-1].data.action
		cos_th, sin_th, cos_th2, sin_th2 = last_obs.cos_th, last_obs.sin_th, last_obs.cos_th2, last_obs.sin_th2
		th, th2 = jnp.arctan2(sin_th, cos_th), jnp.arctan2(sin_th2, cos_th2)
		thdot, thdot2 = last_obs.thdot, last_obs.thdot2

		# Calculate cost (penalize angle error, angular velocity and input voltage)
		delta_goal = (jnp.pi - jnp.abs(th + th2))
		cost = delta_goal ** 2 + 0.1*(thdot / (1 + 10 * abs(delta_goal)))**2 + 0.05*(thdot2 / (1 + 10 * abs(delta_goal)))**2 #+ 0.01 * action[0] ** 2
		cost += 0.05 * (last_action[0] - action[0]) ** 2
		cost -= 0.6*(th / (1 + 10 * abs(delta_goal)))**2

		# Termination condition
		terminated = False
		truncated = (step_state.seq - step_state.state.starting_step) >= self.max_steps
		info = {"TimeLimit.truncated": truncated}

		# update graph_state
		new_ss = step_state.replace(state=step_state.state.replace(reward=-cost))
		graph_state = graph_state.replace(nodes=graph_state.nodes.copy({self.agent.name: new_ss}))

		return graph_state, obs, -cost, terminated, truncated, info

	def _get_obs(self, step_state: StepState) -> Any:
		"""Get observation from environment."""
		inputs = step_state.inputs
		cos_th = inputs["state"].data.cos_th
		sin_th = inputs["state"].data.sin_th
		cos_th2 = inputs["state"].data.cos_th2
		sin_th2 = inputs["state"].data.sin_th2
		thdot = inputs["state"].data.thdot
		thdot2 = inputs["state"].data.thdot2
		last_action = inputs["last_action"].data.action[:, 0] if 'last_action' in inputs else jnp.array([])
		obs = jnp.concatenate([cos_th, sin_th, cos_th2, sin_th2, thdot, thdot2, last_action], axis=-1)
		# obs = jnp.concatenate([cos_th, sin_th, cos_th2, sin_th2, last_action], axis=-1)
		return obs

	def _angle_normalize(self, th: jax.typing.ArrayLike):
		th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
		return th_norm
