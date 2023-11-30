from typing import Any, Dict, Tuple, Union

import jax
import jumpy
import jumpy.numpy as jp
from flax import struct
from flax.core import FrozenDict

from rex.distributions import Gaussian
from rex.proto import log_pb2
from rex.constants import INFO, SYNC, SIMULATED, PHASE, FAST_AS_POSSIBLE, LATEST, BUFFER
from rex.base import InputState, StepState, GraphState, RexResetReturn, RexStepReturn, Empty
from rex.env import BaseEnv
from rex.node import Node
from rex.asynchronous import Agent, AsyncGraph
from rex.spaces import Box
from rex.graph import BaseGraph
from rex.compiled import CompiledGraph
from rex.supergraph import get_network_record, get_timings_from_network_record


def build_dummy_compiled_env() -> Tuple["DummyEnv", "DummyEnv", Dict[str, Node]]:
    env, nodes = build_dummy_env()

    # Get spaces
    action_space = env.action_space()

    # Run environment
    done, (graph_state, obs, info) = False, env.reset(jumpy.random.PRNGKey(0))
    for _ in range(1):
        while not done:
            action = action_space.sample(jumpy.random.PRNGKey(0))
            graph_state, obs, reward, terminated, truncated, info = env.step(graph_state, action)
            done = terminated | truncated
    env.stop()

    # Get episode record with timings
    record = env.graph.get_episode_record()

    # Trace computation graph
    trace_mcs, S, _, Gs, Gs_monomorphism = get_network_record(record, root="agent", seq=-1, supergraph_mode="MCS")
    timings = get_timings_from_network_record(trace_mcs, Gs, Gs_monomorphism)

    # Define compiled graph
    graph = CompiledGraph(nodes=nodes, root=nodes["agent"], S=S, default_timings=timings)

    # Create traced environment
    env_mcs = DummyEnv(graph=graph, max_steps=env.max_steps, name="dummy_env_mcs")
    return env_mcs, env, nodes


def build_dummy_env() -> Tuple["DummyEnv", Dict[str, Union[Agent, Node]]]:
    nodes = build_dummy_graph()
    agent: DummyAgent = nodes["agent"]  # type: ignore
    graph = AsyncGraph(nodes, root=agent, clock=SIMULATED, real_time_factor=FAST_AS_POSSIBLE)
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
class DummyState:
    """Dummy state definition"""
    step: jp.int32
    x: jp.float32


@struct.dataclass
class DummyOutput:
    """Dummy output definition"""

    y: jp.float32


# Define the function whose root we want to find
def f(x, value):
    return x ** 2 - value

# Define the gradient of the function
df_dx = jax.grad(f)


class DummyNode(Node):

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Empty:
        """Default params of the node."""
        return Empty()

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyState:
        """Default state of the node."""
        return DummyState(step=jp.int32(0), x=jp.float32(0.0))

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyOutput:
        """Default output of the node."""
        return DummyOutput(y=jp.float32(0.0))

    def step(self, step_state: StepState) -> Tuple[StepState, DummyOutput]:
        """Step the node."""
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Split rng for step call
        new_rng = rng
        # new_rng, rng_step = jumpy.random.split(rng, num=2)  # todo: is costly if not jitted.

        new_x = state.x
        for name, i in inputs.items():
            for y in i.data.y:
                new_x += (y+1.)/(y+1.)

        # # Sum the sequence numbers of all inputs
        # seqs_sum = jp.zeros((), dtype=jp.int32)
        # for name, i in inputs.items():
        #     for value in i.seq:
        #         root = jp.sqrt(jp.abs(value))  # Initial approximation for the root
        #         for _ in range(0):  # Perform a large number of iterations (adjust as needed)
        #             root = root - f(root, value) / df_dx(root, value)  # Perform Newton-Raphson update
        #         seqs_sum += root

        # Update state
        new_state = state.replace(step=state.step + 1, x=new_x)

        # Update StepState (notice that do not replace the inputs)
        new_step_state = step_state.replace(rng=new_rng, state=new_state)

        # Prepare output
        output = DummyOutput(y=new_x)

        return new_step_state, output


class DummyAgent(Agent):

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyOutput:
        """Default output of the node."""
        return DummyOutput(y=jp.float32(0.0))

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> DummyState:
        """Default state of the node."""
        return DummyState(step=jp.int32(0), x=jp.float32(0.0))

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
        obs = list(step_state.inputs.values())[0][-1].data.y
        # ***DO SOMETHING WITH StepState TO GET OBSERVATION***

        return obs

    def _get_graph_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> GraphState:
        """Get the graph state."""
        rng, rng_eps = jumpy.random.split(rng, num=2)
        starting_step = jp.int32(0) if graph_state is None else graph_state.step
        starting_eps = jumpy.random.choice(rng, self.graph.max_eps(), shape=()) if graph_state is None else graph_state.eps

        # Prepare graph_state
        graph_state = graph_state or GraphState(step=starting_step, eps=starting_eps, nodes=None, timings=None)

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
        new_rng = rng
        # new_rng, rng_step = jumpy.random.split(rng, num=2)

        #
        new_x = state.x
        for name, i in inputs.items():
            for y in i.data.y:
                new_x += (y+1.)/(y+1.)

        # Update state
        new_state = state.replace(step=state.step + 1, x=new_x)

        # Prepare output
        action = DummyOutput(y=new_x)

        # Update StepState (notice that we do not replace the inputs)
        new_step_state = step_state.replace(rng=new_rng, state=new_state)

        # Apply step to receive next step_state
        graph_state, step_state = self.graph.step(graph_state, new_step_state, action)

        # ***DO SOMETHING WITH StepState TO GET OBS/reward/done/info***
        obs = self._get_obs(step_state)
        reward = 0.
        terminated = False
        truncated = self._is_terminal(graph_state)
        info = {}
        # ***DO SOMETHING WITH StepState TO GET OBS/reward/done/info***

        return graph_state, obs, reward, terminated, truncated, info

    def observation_space(self, params: Empty = None):
        """Observation space of the environment."""
        return Box(low=-1, high=1, shape=(), dtype=jp.float32)

    def action_space(self, params: Empty = None):
        """Action space of the environment."""
        return Box(low=-1, high=1, shape=(1,), dtype=jp.float32)
