from typing import Dict, Union, Any, Tuple
import jax
import jax.numpy as jnp
import numpy as onp

from rexv2.graph import Graph
import rexv2.base as base
import rexv2.rl as rl

from envs.crazyflie.ode import MoCapOutput
from envs.crazyflie.pid import PIDParams, PIDOutput
from envs.crazyflie.agent import PPOAgentParams
from envs.crazyflie.supervisor import SupervisorParams


class Environment(rl.Environment):
    def __init__(self, graph: Graph, params: Dict[str, base.Base] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        super().__init__(graph, params, only_init, starting_eps, randomize_eps, order)

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return rl.Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        params_sup: SupervisorParams = graph_state.params.get("supervisor")
        params_agent: PPOAgentParams = graph_state.params.get("agent")
        high_mapping = dict(
            pwm_ref=params_sup.pwm_from_hover,
            phi_ref=params_sup.phi_max,
            theta_ref=params_sup.theta_max,
            psi_ref=params_sup.psi_max,
            z_ref=params_sup.z_max
        )
        high = jnp.array([high_mapping[k] for a, k in zip(range(params_agent.action_dim), params_sup.ctrl_mapping)], dtype=float)
        return rl.Box(-high, high, shape=high.shape, dtype=float)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        obs = params.get_observation(ss)
        return obs

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ts = self.get_step_state(graph_state).ts
        params: SupervisorParams = graph_state.params["supervisor"]
        return ts >= params.tmax

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        terminated = False
        return terminated  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        # Get current state
        ss = self.get_step_state(graph_state)
        # pos = last_mocap.pos
        # x, y, z = pos
        # vx, vy, vz = last_mocap.vel

        # Get transformed
        center = graph_state.params["supervisor"].center
        radius = graph_state.state["agent"].radius
        last_mocap: MoCapOutput = ss.inputs["estimator"][-1].data
        last_mocap_ia = last_mocap.in_agent_frame(center)
        radial, theta, z = last_mocap_ia.pos
        vrad, vel_on, vz = last_mocap_ia.vel
        pos_off = jnp.linalg.norm(jnp.array([radial - radius, z]))
        vel_off = jnp.linalg.norm(jnp.array([vrad, vz]))

        # Get denormalized action
        p_pid: PIDParams = graph_state.params["pid"]
        output = p_pid.to_output(action)
        prev_output = p_pid.to_output(ss.state.prev_act)
        dz_ref = output.z_ref - prev_output.z_ref
        dphi_ref = output.phi_ref - prev_output.phi_ref
        dtheta_ref = output.theta_ref - prev_output.theta_ref
        dpsi_ref = output.psi_ref - prev_output.psi_ref

        # Reward for following the path (higher when closer to the path)
        rwd_vel_on = 0.1*vel_on / jnp.clip(pos_off, 0.001, None)

        # Control cost
        # cost_dangle = jnp.sqrt(dphi_ref**2 + dtheta_ref**2 + dpsi_ref**2)
        cost_dangle = dphi_ref**2 + dtheta_ref**2 + dpsi_ref**2
        cost_decouple = jnp.abs(dphi_ref) * jnp.abs(dtheta_ref)
        cost_dz_ref = jnp.sqrt(dz_ref**2)
        cost_z_ref = jnp.abs(output.z_ref - center[2])

        # Calculate rewards
        # rwd_eps = 0.4*rwd_vel_on - 0.4*vel_off - pos_off - 10.0*cost_dangle - 0.*cost_decouple - 0.2*cost_dz_ref - 5*cost_z_ref
        rwd_eps = 0.4*rwd_vel_on - 0.4*vel_off - pos_off - 10.0*cost_dangle - 0.*cost_decouple - 0.2*cost_dz_ref - 5*cost_z_ref
        rwd_term = 0.0
        rwd_final = rwd_eps

        # Get termination conditions
        terminated = self.get_terminated(graph_state)
        truncated = self.get_truncated(graph_state)

        # Calculate final reward
        gamma = graph_state.params["supervisor"].gamma
        done = jnp.logical_or(terminated, truncated)
        rwd = rwd_eps + done * ((1 - terminated) * (1 / (1 - gamma)) + terminated) * rwd_final + done * terminated * rwd_term
        return rwd

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        # Get current state
        ss = self.get_step_state(graph_state)
        center = graph_state.params["supervisor"].center
        radius = graph_state.state["agent"].radius
        last_mocap: MoCapOutput = ss.inputs["estimator"][-1].data
        last_mocap_ia = last_mocap.in_agent_frame(center)
        radial, theta, z = last_mocap_ia.pos
        vrad, vel_on, vz = last_mocap_ia.vel

        # Cost for not following or being on the path (vel, pos)
        pos_off = jnp.linalg.norm(jnp.array([radial - radius, z]))
        vel_off = jnp.linalg.norm(jnp.array([vrad, vz]))

        # Reward for following the path (higher when closer to the path)
        rwd_vel_on = 0.01*jnp.clip(vel_on / jnp.clip(pos_off, 0.001, None), 0, None)

        return {
            "vel_on": vel_on,
            "pos_off": pos_off,
            "vel_off": vel_off,
            "rwd_vel_on": rwd_vel_on,
            "z": z,
        }

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> PIDOutput:
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        output = params.to_output(ss, action)
        return output

    def update_graph_state_pre_step(self, graph_state: base.GraphState, action: jax.Array) -> base.GraphState:
        # Update agent state
        ss = self.get_step_state(graph_state)
        params: PPOAgentParams = ss.params
        new_state = params.update_state(ss, action)
        # Update graph state
        graph_state = graph_state.replace(state=graph_state.state.copy({
            # "world": graph_state.state["world"].replace(loss_task=0.0),  # Reset task reward
            self.graph.supervisor.name: new_state
        }))
        return graph_state

    def update_graph_state_post_step(self, graph_state: base.GraphState, action: jax.Array = None) -> base.GraphState:
        """Override this method if you want to update the graph state after graph.step(...).

        Note: This method is called after the graph has been stepped (before returning from .step()),
              or before returning the initial observation from .reset().
        :param graph_state: The current graph state.
        :param action: The action taken. Is None when called from .reset().
        """
        if action is not None:
            # Update graph state
            ss = self.get_step_state(graph_state)
            graph_state = graph_state.replace(state=graph_state.state.copy({
                self.graph.supervisor.name: ss.state.replace(prev_act=action)
            }))

        return graph_state
