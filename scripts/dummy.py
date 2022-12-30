from typing import Any, Dict, Tuple, Union
import jumpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, INTERPRETED, WARN
from rex.base import InputState, StepState, GraphState
from rex.env import BaseEnv
from rex.node import Node
from rex.agent import Agent


@struct.dataclass
class DummyParams:
	"""Dummy param definition"""

	param_1: jp.float32
	param_2: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0.0, 1.0], jp.float32))


@struct.dataclass
class DummyState:
	"""Dummy state definition"""

	step: jp.int32  # Step index
	seqs_sum: jp.int32  # The sequence numbers of every input in InputState summed over the entire episode.
	dummy_1: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0.0, 1.0], jp.float32))


@struct.dataclass
class DummyOutput:
	"""Dummy output definition"""

	seqs_sum: jp.int32  # The sequence numbers of every input in InputState summed over the entire episode.
	dummy_1: jp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jp.array([0.0, 1.0], jp.float32))


class DummyNode(Node):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyParams:
		"""Default params of the node."""
		return DummyParams(jp.float32(99.0), jp.array([0.0, 1.0], dtype=jp.float32))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyState:
		"""Default state of the node."""
		return DummyState(step=jp.int32(0), seqs_sum=jp.int32(0), dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyOutput:
		"""Default output of the node."""
		if graph_state is not None:
			seqs_sum = graph_state.nodes[self.name].state.seqs_sum
		else:
			seqs_sum = jp.int32(0)
		return DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the node."""
		rng_params, rng_state, rng_inputs, rng_step = jp.random_split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)

	def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, DummyOutput]:
		"""Step the node."""
		# Unpack StepState
		rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Split rng for step call
		new_rng = rng
		# new_rng, rng_step = jp.random_split(rng, num=2)  # todo: is costly if not jitted.

		# Sum the sequence numbers of all inputs
		seqs_sum = jp.int32(0)  # state.seqs_sum if self.stateful else jp.int32(0)
		for name, i in inputs.items():
			seqs_sum += jp.sum(i.seq)

		# Update params (optional)
		new_params = params.replace(param_1=ts)

		# Update state
		new_state = state.replace(step=state.step + 1, seqs_sum=seqs_sum)

		# Prepare output
		output = DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([1.0, 2.0], jp.float32))

		# Update StepState (notice that do not replace the inputs)
		new_step_state = step_state.replace(rng=new_rng, state=new_state, params=new_params)

		# Print input info
		if not jp._in_jit() and not jp._has_jax:
			log_msg = []
			for name, input_state in inputs.items():
				# ts_msgs = [round(ts_recv, 4) for ts_recv in input_state.ts_recv]
				# info = f"{name}={ts_msgs}"
				seq_msg = [seq for seq in input_state.seq]
				info = f"{name}={seq_msg}"
				log_msg.append(info)
			log_msg = " | ".join(log_msg)
			self.log("step", f"step={state.step} | seqs_sum={seqs_sum} | {log_msg}", log_level=INFO)
			# self.log("step", f"step={state.step} | ts={ts: .3f} | {log_msg}", log_level=INFO)

		return new_step_state, output


class DummyAgent(Agent):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyParams:
		"""Default params of the agent."""
		return DummyParams(jp.float32(99.0), jp.array([0.0, 1.0], dtype=jp.float32))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyState:
		"""Default state of the agent."""
		return DummyState(step=jp.int32(0), seqs_sum=jp.int32(0), dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyOutput:
		"""Default output of the agent."""
		if graph_state is not None:
			seqs_sum = graph_state.nodes[self.name].state.seqs_sum
		else:
			seqs_sum = jp.int32(0)
		return DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
		"""Reset the agent."""
		rng_params, rng_state, rng_inputs, rng_step = jp.random_split(rng, num=4)
		params = self.default_params(rng_params, graph_state)
		state = self.default_state(rng_state, graph_state)
		inputs = self.default_inputs(rng_inputs, graph_state)
		return StepState(rng=rng_step, params=params, state=state, inputs=inputs)


class DummyEnv(BaseEnv):
	def __init__(
			self,
			nodes: Dict[str, "Node"],
			agent: DummyAgent,
			max_steps: int = 100,
			trace: log_pb2.TraceRecord = None,
			sync: int = SYNC,
			clock: int = SIMULATED,
			scheduling: int = PHASE,
			real_time_factor: Union[int, float] = FAST_AS_POSSIBLE,
			graph: int = INTERPRETED,
			name: str = "DummyEnv",
	):
		# Exclude the node for which this environment is a drop-in replacement (i.e. the agent)
		nodes = {node.name: node for _, node in nodes.items() if node.name != agent.name}

		# Required for step and reset functions
		self.agent = agent
		self.nodes = nodes
		super().__init__(nodes, agent, max_steps, sync, clock, scheduling, real_time_factor, graph, trace, name=name)

	def _is_terminal(self, graph_state: GraphState) -> bool:
		return graph_state.step >= self.max_steps

	def _get_obs(self, step_state: StepState) -> Any:
		"""Get observation from environment."""
		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***
		obs = step_state.inputs
		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***

		return obs

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
		"""Get the graph state."""
		# For every node, prepare the initial stepstate
		new_nodes = dict()

		# ***DO SOMETHING WITH graph_state TO RESET ALL NODES***
		# Reset agent node (for which this environment is a drop-in replacement)
		rng, rng_agent = jp.random_split(rng, num=2)
		new_nodes[self.agent.name] = self.agent.reset(rng_agent, graph_state)

		# Split rngs for other node resets
		rngs = jp.random_split(rng, num=len(self.nodes))

		for (name, n), rng_reset in zip(self.nodes.items(), rngs):
			# Reset node and optionally provide params, state, inputs
			new_ss = n.reset(rng_reset, graph_state)  # can provide params, state, inputs here

			# Replace step state in graph state
			new_nodes[name] = new_ss

		# ***DO SOMETHING WITH graph_state TO RESET ALL NODES***
		return GraphState(nodes=FrozenDict(new_nodes))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> Tuple[GraphState, Any]:
		"""Reset environment."""
		new_graph_state = self._get_graph_state(rng, graph_state)

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, ts, step_state = self.graph.reset(new_graph_state)

		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***
		obs = self._get_obs(step_state)
		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***

		return graph_state, obs

	def step(self, graph_state: GraphState, action: Any) -> Tuple[GraphState, InputState, float, bool, Dict]:
		"""Perform step transition in environment."""
		# ***PREPROCESS action TO GET AgentOutput***
		# Unpack StepState
		step_state = self.agent.get_step_state(graph_state)
		rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Split rng for step call
		new_rng, rng_step = jp.random_split(rng, num=2)

		# Sum the sequence numbers of all inputs
		seqs_sum = jp.int32(0)  # state.seqs_sum
		for name, i in inputs.items():
			seqs_sum += jp.sum(i.seq)

		# Update params (optional)
		new_params = params.replace(param_1=jp.float32(0.0))

		# Update state
		new_state = state.replace(step=state.step + 1, seqs_sum=seqs_sum)

		# Prepare output
		action = DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([1.0, 2.0], jp.float32))

		# Update StepState (notice that we do not replace the inputs)
		new_step_state = step_state.replace(rng=new_rng, state=new_state, params=new_params)

		# Apply step to receive next step_state
		graph_state, ts, step_state = self.graph.step(graph_state, new_step_state, action)

		# ***DO SOMETHING WITH StepState TO GET OBS/reward/done/info***
		obs = self._get_obs(step_state)
		reward = 0.
		done = self._is_terminal(graph_state)
		info = {}
		# ***DO SOMETHING WITH StepState TO GET OBS/reward/done/info***

		return graph_state, obs, reward, done, info
