from typing import Any, Dict, Tuple, Union

import networkx as nx

import jumpy
import jumpy.numpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.distributions import Gaussian
from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, LATEST, BUFFER
from rex.base import InputState, StepState, GraphState, RexResetReturn, RexStepReturn
from rex.env import BaseEnv
from rex.node import Node
from rex.agent import Agent
from rex.spaces import Box
from rex.graph import BaseGraph, Graph
from rex.compiled import CompiledGraph
from rex.tracer import get_network_record, get_timings_from_network_record


def build_dummy_compiled_env() -> Tuple["DummyEnv", "DummyEnv", Dict[str, Node]]:
	env, nodes = build_dummy_env()

	# Get spaces
	action_space = env.action_space()

	# Run environment
	done, (graph_state, obs, info) = False, env.reset(jumpy.random.PRNGKey(0))
	for _ in range(1):
		while not done:
			action = action_space.sample(jumpy.random.PRNGKey(0))
			graph_state, obs, reward, truncated, done, info = env.step(graph_state, action)
	env.stop()

	# Get episode record with timings
	record = env.graph.get_episode_record()

	# Trace computation graph
	trace_mcs, MCS, G, G_subgraphs = get_network_record(record, "agent", -1)
	timings = get_timings_from_network_record(trace_mcs, G, G_subgraphs)

	# Define compiled graph
	graph = CompiledGraph(nodes=nodes, root=nodes["agent"], MCS=MCS, default_timings=timings)

	# Create traced environment
	env_mcs = DummyEnv(graph=graph, max_steps=env.max_steps, name="dummy_env_mcs")
	return env_mcs, env, nodes


def build_dummy_env() -> Tuple["DummyEnv", Dict[str, Union[Agent, Node]]]:
	nodes = build_dummy_graph()
	agent: DummyAgent = nodes["agent"]  # type: ignore
	graph = Graph(nodes, root=agent, clock=SIMULATED, real_time_factor=FAST_AS_POSSIBLE)
	env = DummyEnv(graph=graph, max_steps=100, name="dummy_env")
	return env, nodes


def build_dummy_graph() -> Dict[str, Node]:
	# Create nodes
	world = DummyNode("world", rate=20, delay_sim=Gaussian(0.000))
	sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(0.007))
	observer = DummyNode("observer", rate=30, delay_sim=Gaussian(0.016))
	agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(0.005, 0.001), advance=True)
	actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), advance=False, stateful=True)
	nodes = [world, sensor, observer, agent, actuator]

	# Connect
	sensor.connect(world, blocking=False, delay_sim=Gaussian(0.004), skip=False, jitter=LATEST, name="testworld", window=1)
	observer.connect(sensor, blocking=False, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER)
	observer.connect(agent, blocking=False, delay_sim=Gaussian(0.003), skip=True, jitter=LATEST)
	agent.connect(observer, blocking=True, delay_sim=Gaussian(0.003), skip=False, jitter=BUFFER, window=1)
	actuator.connect(agent, blocking=False, delay_sim=Gaussian(0.003, 0.001), skip=False, jitter=BUFFER, delay=0.05, window=2)
	world.connect(actuator, blocking=False, delay_sim=Gaussian(0.004), skip=True, jitter=BUFFER)
	return {n.name: n for n in nodes}


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

	def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyParams:
		"""Default params of the node."""
		return DummyParams(jp.float32(99.0), jp.array([0.0, 1.0], dtype=jp.float32))

	def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyState:
		"""Default state of the node."""
		return DummyState(step=jp.int32(0), seqs_sum=jp.int32(0), dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyOutput:
		"""Default output of the node."""
		seqs_sum = jp.int32(0)
		return DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))

	def step(self, step_state: StepState) -> Tuple[StepState, DummyOutput]:
		"""Step the node."""
		# Unpack StepState
		rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Split rng for step call
		new_rng = rng
		# new_rng, rng_step = jumpy.random.split(rng, num=2)  # todo: is costly if not jitted.

		# Sum the sequence numbers of all inputs
		seqs_sum = jp.int32(0)  # state.seqs_sum if self.stateful else jp.int32(0)
		for name, i in inputs.items():
			seqs_sum += jp.sum(i.seq)

		# Update params (optional)
		new_params = params.replace(param_1=step_state.ts)

		# Update state
		new_state = state.replace(step=state.step + 1, seqs_sum=seqs_sum)

		# Prepare output
		output = DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([1.0, 2.0], jp.float32))

		# Update StepState (notice that do not replace the inputs)
		new_step_state = step_state.replace(rng=new_rng, state=new_state, params=new_params)

		# Print input info
		if not jumpy.core.is_jitted() and not jumpy.is_jax_installed:
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

	def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyOutput:
		"""Default output of the root."""
		# if graph_state is not None:
		# 	seqs_sum = graph_state.nodes[self.name].state.seqs_sum
		# else:
		seqs_sum = jp.int32(0)
		return DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))


class DummyEnv(BaseEnv):
	def __init__(self, graph: BaseGraph, max_steps: int = 100, name: str = "DummyEnv"):
		super().__init__(graph=graph, max_steps=max_steps, name=name)
		self.agent = self.graph.root
		self.nodes = self.graph.nodes
		self.nodes_and_root = self.graph.nodes_and_root

	def _is_terminal(self, graph_state: GraphState) -> bool:
		return graph_state.step >= self.max_steps

	def _get_obs(self, step_state: StepState) -> Any:
		"""Get observation from environment."""
		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***
		obs = list(step_state.inputs.values())[0][-1].data.seqs_sum
		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***

		return obs

	def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
		"""Get the graph state."""
		# Prepare graph_state
		graph_state = graph_state or GraphState(step=jp.int32(0), eps=jp.int32(0), nodes=None, timings=None)

		# For every node, prepare the initial stepstate
		new_nodes = dict()

		# ***DO SOMETHING WITH graph_state TO RESET ALL NODES***
		# Reset root node (for which this environment is a drop-in replacement)
		rng, rng_agent = jumpy.random.split(rng, num=2)

		# Get new step_state
		def get_step_state(node: Node, _rng: jp.ndarray, _graph_state) -> StepState:
			"""Get new step_state for a node."""
			rng_params, rng_state, rng_step = jumpy.random.split(rng, num=3)
			params = node.default_params(rng_params, _graph_state)
			state = node.default_state(rng_state, _graph_state)
			return StepState(rng=rng_step, params=params, state=state, inputs=None)

		# Get root step state first
		new_nodes[self.agent.name] = get_step_state(self.agent, rng_agent, graph_state)

		# Get new step_state for other nodes in arbitrary order
		rng, *rngs = jumpy.random.split(rng, num=len(self.nodes)+1)
		for (name, n), rng_n in zip(self.nodes.items(), rngs):
			# Replace step state in graph state
			new_nodes[name] = get_step_state(n, rng_n, graph_state)

		# Reset nodes
		rng, *rngs = jumpy.random.split(rng, num=len(self.nodes_and_root) + 1)
		[n.reset(rng_reset, graph_state) for (n, rng_reset) in zip(self.nodes_and_root.values(), rngs)]
		# ***DO SOMETHING WITH graph_state TO RESET ALL NODES***
		return graph_state.replace(nodes=FrozenDict(new_nodes))

	def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> RexResetReturn:
		"""Reset environment."""
		new_graph_state = self._get_graph_state(rng, graph_state)

		# Reset environment to get initial step_state (runs up-until the first step)
		graph_state, step_state = self.graph.reset(new_graph_state)

		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***
		obs = self._get_obs(step_state)
		info = {}
		# ***DO SOMETHING WITH StepState TO GET OBSERVATION***

		return graph_state, obs, info

	def step(self, graph_state: GraphState, action: Any) -> RexStepReturn:
		"""Perform step transition in environment."""
		# ***PREPROCESS action TO GET AgentOutput***
		# Unpack StepState
		step_state = self.agent.get_step_state(graph_state)
		rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

		# Split rng for step call
		new_rng, rng_step = jumpy.random.split(rng, num=2)

		# Sum the sequence numbers of all inputs
		seqs_sum = jp.int32(0)  # state.seqs_sum
		for name, i in inputs.items():
			seqs_sum += jp.sum(i.seq)

		# Update params (optional)
		new_params = params

		# Update state
		new_state = state  # state.replace(step=state.step + 1, seqs_sum=seqs_sum)

		# Prepare output
		action = DummyOutput(seqs_sum=seqs_sum, dummy_1=jp.array([1.0, 2.0], jp.float32))

		# Update StepState (notice that we do not replace the inputs)
		new_step_state = step_state.replace(rng=new_rng, state=new_state, params=new_params)

		# Apply step to receive next step_state
		graph_state, step_state = self.graph.step(graph_state, new_step_state, action)

		# ***DO SOMETHING WITH StepState TO GET OBS/reward/done/info***
		obs = self._get_obs(step_state)
		reward = 0.
		truncated = False
		done = self._is_terminal(graph_state)
		info = {}
		# ***DO SOMETHING WITH StepState TO GET OBS/reward/done/info***

		return graph_state, obs, reward, truncated, done, info

	def observation_space(self, params: DummyParams = None):
		"""Observation space of the environment."""
		return Box(low=-1, high=1, shape=(), dtype=jp.float32)

	def action_space(self, params: DummyParams = None):
		"""Action space of the environment."""
		return Box(low=-1, high=1, shape=(1,), dtype=jp.float32)
