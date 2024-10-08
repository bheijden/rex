from typing import Dict, Any, Union
import jax
import jax.numpy as jnp
import numpy as onp
import flax.struct as struct
from rex.rl import ResetReturn, StepReturn, BaseEnv, Box
from rex import base
from rex.graph import Graph
import rex.ppo as ppo
from rex.pendulum.agent import AgentParams


class SwingUpEnv(BaseEnv):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)
        self._init_params = {}

    @property
    def max_steps(self) -> Union[int, jax.typing.ArrayLike]:
        """Maximum number of steps in an evaluation episode"""
        return int(3.5 * self.graph.nodes["agent"].rate)

    def set_params(self, params: Dict[str, Any]):
        """Pre-set parameters for the environment"""
        self._init_params.update(params)

    def observation_space(self, graph_state: base.GraphState) -> Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> Box:
        params: AgentParams = graph_state.params["agent"]
        high = jnp.array([params.max_torque], dtype=jnp.float32)
        return Box(-high, high, shape=high.shape, dtype=high.dtype)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = graph_state.step_state["agent"]
        params: AgentParams = ss.params
        obs = params.get_observation(ss)
        return obs

    def reset(self, rng: jax.Array = None) -> ResetReturn:
        # Initialize the graph state
        init_gs = self.graph.init(rng=rng, params=self._init_params, order=("agent",))
        # Run the graph until the agent node
        gs, _ = self.graph.reset(init_gs)
        # Get observation
        obs = self.get_observation(gs)
        info = {}  # No info to return
        return gs, obs, info

    def step(self, graph_state: base.GraphState, action: jax.Array) -> StepReturn:
        params: AgentParams = graph_state.params["agent"]
        # Update the agent's state (i.e. action and observation history)
        new_agent = params.update_state(graph_state.step_state["agent"], action)
        # The loss_task (i.e. reward) is accumulated in the World node's step function
        # Hence, we read out the loss_task from the world node and set it to 0 before stepping
        # This is to ensure that the loss_task is only counted once
        # Note that this is not obligatory, but it's a good way to ensure that the reward is consistent in the
        # face of simulated asynchrounous effects.
        new_world = graph_state.state["world"].replace(loss_task=0.0)
        # Update the states in the graph state
        gs = graph_state.replace(state=graph_state.state.copy({"agent": new_agent,
                                                               "world": new_world}))
        # Convert action to output (i.e. the one that the Agent node outputs)
        ss = gs.step_state["agent"]
        output = params.to_output(action)
        # Step the graph (i.e. all nodes except the Agent node)
        next_gs, next_ss = self.graph.step(gs, ss, output)
        # Get observation
        obs = self.get_observation(next_gs)
        info = {}
        # Read out the loss_task from the world node's state
        reward = -graph_state.state["world"].loss_task
        # Determine if the episode is truncated
        terminated = False  # Infinite horizon task
        truncated = params.tmax <= next_ss.ts  # Truncate if the time limit is reached
        # Mitigate truncation of infinite horizon tasks by adding a final reward
        # Add the steady-state solution as if the agent had stayed in the same state for the rest of the episode
        gamma = params.gamma
        reward_final = truncated * (1 / (1 - gamma)) * reward  # Assumes that the reward is constant after truncation
        reward = reward + reward_final
        return next_gs, obs, reward, terminated, truncated, info


@struct.dataclass
class PendulumConfig(ppo.Config):
    def EVAL_METRICS_JAX_CB(self, total_steps, diagnostics: ppo.Diagnostics, eval_transitions: ppo.Transition = None) -> Dict:
        metrics = super().EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_transitions)

        # Calculate success rate
        cos_th = eval_transitions.obs[..., 0]
        thdot = eval_transitions.obs[..., 2]
        is_upright = cos_th > 0.90
        is_static = jnp.abs(thdot) < 1.0
        is_valid = jnp.logical_and(is_upright, is_static)
        success_rate = is_valid.sum() / is_valid.size
        metrics["eval/success_rate"] = success_rate
        return metrics

    def EVAL_METRICS_HOST_CB(self, metrics: Dict):
        # Standard metrics
        global_step = metrics["train/total_steps"]
        mean_approxkl = metrics["train/mean_approxkl"]
        mean_return = metrics["eval/mean_returns"]
        std_return = metrics["eval/std_returns"]
        mean_length = metrics["eval/mean_lengths"]
        std_length = metrics["eval/std_lengths"]
        total_episodes = metrics["eval/total_episodes"]

        # Extra metrics
        success_rate = metrics["eval/success_rate"]

        if self.VERBOSE:
            warn = ""
            if total_episodes == 0:
                warn = "WARNING: No eval. episodes returned | "
            print(f"{warn}train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                  f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f} | "
                  f"success_rate={success_rate:.2f}")


sweep_pmv2r1zf = PendulumConfig(
    LR=0.0003261962464827655,
    NUM_ENVS=128,
    NUM_STEPS=32,
    TOTAL_TIMESTEPS=5e6,
    UPDATE_EPOCHS=8,
    NUM_MINIBATCHES=16,
    GAMMA=0.9939508937435216,
    GAE_LAMBDA=0.9712149137900143,
    CLIP_EPS=0.16413213812946092,
    ENT_COEF=0.01,
    VF_COEF=0.8015258840683805,
    MAX_GRAD_NORM=0.9630061315073456,
    NUM_HIDDEN_LAYERS=2,
    NUM_HIDDEN_UNITS=64,
    KERNEL_INIT_TYPE='xavier_uniform',
    HIDDEN_ACTIVATION='tanh',
    STATE_INDEPENDENT_STD=True,
    SQUASH=True,
    ANNEAL_LR=False,
    NORMALIZE_ENV=True,
    FIXED_INIT=True,
    OFFSET_STEP=False,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
    VERBOSE=True,
    DEBUG=False
)