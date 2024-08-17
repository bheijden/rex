import functools
from typing import Callable, Union, Dict, Any, Tuple
import equinox as eqx
import jax
import numpy as onp
from flax import struct
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import rexv2.base as base
from rexv2.graph import Graph
import rexv2.evo as evo
from envs.pendulum.tasks import Task


def create_sysid_task(graph: Graph, graph_state: base.GraphState, strategy: str = "CMA_ES"):
    MIN, MAX = 0.5, 1.5

    # Create an empty filter
    base_states = {name: s for name, s in graph_state.state.items()}
    loss_filter = jax.tree_util.tree_map(lambda x: False, base_states)
    # loss_filter["estimator"] = loss_filter["estimator"].replace(loss_th=True)  # Estimator loss
    loss_filter["mocap"] = loss_filter["mocap"].replace(loss_pos=False, loss_vel=True, loss_att=True)  # Reconstruction loss

    # Create an empty skeleton of the params
    base_params = {name: p for name, p in graph_state.params.items()}  # Get base structure for params
    init_params = jax.tree_util.tree_map(lambda x: None, base_params)  # Create an empty skeleton of the params
    u_min = jax.tree_util.tree_map(lambda x: None, base_params)
    u_max = jax.tree_util.tree_map(lambda x: None, base_params)
    shared = []  # Holds the list of shared params

    # World
    world_range = init_params["world"].sysid_range()
    init_params["world"] = jax.tree_util.tree_map(lambda _x, _y: _y, world_range, base_params["world"])
    u_min["world"] = world_range[0]
    u_max["world"] = world_range[1]
    shared.append(base.Shared.init(where=lambda _p: _p["world"].actuator_delay,
                                   replace_fn=lambda _p: _p["pid"].actuator_delay))

    # Agent
    pass  # todo: optimize over fixed initial state?

    # Mocap
    init_params["mocap"] = base_params["mocap"].replace(pos_std=None, vel_std=None, att_std=None, ang_vel_std=None)
    u_min["mocap"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["mocap"])
    u_max["mocap"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["mocap"])
    u_min["mocap"] = eqx.tree_at(lambda _min: _min.sensor_delay.alpha, u_min["mocap"], 0.)
    u_max["mocap"] = eqx.tree_at(lambda _max: _max.sensor_delay.alpha, u_max["mocap"], 1.)

    # Estimator
    pass  # todo: optimize for dynamics?

    # PID
    init_params["pid"] = init_params["pid"].replace(actuator_delay=base_params["pid"].actuator_delay,
                                                    sensor_delay=base_params["pid"].sensor_delay)
    u_min["pid"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["pid"])
    u_max["pid"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["pid"])
    u_min["pid"] = eqx.tree_at(lambda _min: _min.actuator_delay.alpha, u_min["pid"], 0.)
    u_max["pid"] = eqx.tree_at(lambda _max: _max.actuator_delay.alpha, u_max["pid"], 1.)
    u_min["pid"] = eqx.tree_at(lambda _min: _min.sensor_delay.alpha, u_min["pid"], 0.)
    u_max["pid"] = eqx.tree_at(lambda _max: _max.sensor_delay.alpha, u_max["pid"], 1.)

    def rollout_fn(graph: Graph, params: Dict[str, base.Params], rng: jax.Array = None) -> base.GraphState:
        # Initialize graph state
        gs = graph.init(rng=rng, params=params, order=("agent", "pid"))

        # Rollout graph
        final_gs = graph.rollout(gs, carry_only=True)
        return final_gs

    def plot_fn(graph_states: base.GraphState, identifier: str = "") -> Any:
        # Plot best result
        figs = []

        state = graph_states.state
        inputs = graph_states.inputs
        params = graph_states.params

        center = params["agent"][0].center
        radius = params["agent"][0].fixed_radius
        mocap = inputs["estimator"]["mocap"][:, -1]
        world = inputs["mocap"]["world"][:, -1]

        from envs.crazyflie.ode import in_body_frame
        in_body_frame_jv = jax.jit(jax.vmap(in_body_frame))
        mocap_vel_ib = in_body_frame_jv(mocap.data.att, mocap.data.vel)
        world_vel_ib = in_body_frame_jv(world.data.att, world.data.vel)
        # mocap_ia = jax.vmap(mocap.data.static_in_agent_frame, in_axes=(0, None))(mocap.data, center)
        # world_ia = jax.vmap(world.data.static_in_agent_frame, in_axes=(0, None))(world.data, center)

        ts_recon = {
            "pwm_ref": inputs["world"]["pid"].ts_recv[..., -1],
            "pos": world.ts_recv,
            "vel_ib": world.ts_recv,
            "att": world.ts_recv,
            "vel": world.ts_recv,
        }
        recon = {
            "pwm_ref": inputs["world"]["pid"].data.pwm_ref[..., -1],
            "pos": world.data.pos,
            "vel_ib": world_vel_ib,
            "att": world.data.att,
            "vel": world.data.vel,
        }
        ts_output = {
            "pos": mocap.ts_sent,
            "vel_ib": mocap.ts_sent,
            "att": mocap.ts_sent,
            "vel": mocap.ts_sent,
        }
        output = {
            "pos": mocap.data.pos,
            "vel_ib": mocap_vel_ib,
            "att": mocap.data.att,
            "vel": mocap.data.vel,
        }

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        figs.append(fig)

        # First row: PWM and x-y plot
        axes[0, 0].plot(ts_recon["pwm_ref"], recon["pwm_ref"], label="pwm_ref", color="green", linestyle="-")
        axes[0, 0].legend()
        axes[0, 0].set_title("PWM")

        # Replace the scatter plot with a line plot
        xlim_all = [None, None]
        ylim_all = [None, None]
        for ts_data, data, cmap in [(ts_output, output, "viridis"), (ts_recon, recon, "inferno")]:
            points = onp.array([data["pos"][:, 0], data["pos"][:, 1]]).T.reshape(-1, 1, 2)
            segments = onp.concatenate([points[:-1], points[1:]], axis=1)
            norm = Normalize(ts_data["pos"].min(), ts_data["pos"].max())
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(ts_data["pos"])
            lc.set_linewidth(2)
            xlim = [data["pos"][:, 0].min(), data["pos"][:, 0].max()]
            xlim = [xlim[0] - 1, xlim[1] + 1] if xlim[0] == xlim[1] else xlim
            ylim = [data["pos"][:, 1].min(), data["pos"][:, 1].max()]
            ylim = [ylim[0] - 1, ylim[1] + 1] if ylim[0] == ylim[1] else ylim
            xlim_all[0] = min(xlim[0], xlim_all[0]) if xlim_all[0] is not None else xlim[0]
            xlim_all[1] = max(xlim[1], xlim_all[1]) if xlim_all[1] is not None else xlim[1]
            ylim_all[0] = min(ylim[0], ylim_all[0]) if ylim_all[0] is not None else ylim[0]
            ylim_all[1] = max(ylim[1], ylim_all[1]) if ylim_all[1] is not None else ylim[1]
            axes[0, 1].set_xlim(xlim_all)
            axes[0, 1].set_ylim(ylim_all)
            line = axes[0, 1].add_collection(lc)
            # fig.colorbar(line, ax=axes[0, 1])
        axes[0, 1].set_title("X-Y Position (color: time)")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Y")

        # Plot Z
        for ts_data, data, color in [(ts_output, output, "blue"), (ts_recon, recon, "green")]:
            axes[0, 2].plot(ts_data["pos"], data["pos"][:, 2], label="z", color=color, linestyle="-")
            # axes[0, 2].plot(ts_data["z_ref"], data["z_ref"], label="z_ref", color="green", linestyle="-")
            axes[0, 2].legend()
            axes[0, 2].set_title("Z Position")

            # Second row: Phi, Theta, Psi
            axes[1, 0].plot(ts_data["att"], data["att"][:, 0], label="phi", color=color, linestyle="-")
            # axes[1, 0].plot(ts_data["phi_ref"], data["phi_ref"], label="phi_ref", color="green", linestyle="-")
            axes[1, 0].set_ylim([-onp.pi / 5, onp.pi / 5])
            axes[1, 0].legend()
            axes[1, 0].set_title("Phi")

            axes[1, 1].plot(ts_data["att"], data["att"][:, 1], label="theta", color=color, linestyle="-")
            # axes[1, 1].plot(ts_data["theta_ref"], data["theta_ref"], label="theta_ref", color="green", linestyle="-")
            axes[1, 1].set_ylim([-onp.pi / 5, onp.pi / 5])
            axes[1, 1].legend()
            axes[1, 1].set_title("Theta")

            axes[1, 2].plot(ts_data["att"], data["att"][:, 2], label="psi", color=color, linestyle="-")
            # axes[1, 2].plot(ts_data["psi_ref"], data["psi_ref"], label="psi_ref", color="green", linestyle="-")
            axes[1, 2].set_ylim([-onp.pi / 5, onp.pi / 5])
            axes[1, 2].legend()
            axes[1, 2].set_title("Psi")

            # Plot velocities in body frame
            if "vel_ib" in data:
                axes[2, 0].plot(ts_data["vel_ib"], data["vel_ib"][:, 0], label="Vy", color=color, linestyle="-")
                axes[2, 0].legend()
                axes[2, 0].set_title("Vx (body-frame)")
                axes[2, 1].plot(ts_data["vel_ib"], data["vel_ib"][:, 1], label="Vy", color=color, linestyle="-")
                axes[2, 1].legend()
                axes[2, 1].set_title("Vy (body-frame)")
                axes[2, 2].plot(ts_data["vel_ib"], data["vel_ib"][:, 2], label="Vz", color=color, linestyle="-")
                axes[2, 2].legend()
                axes[2, 2].set_title("Vz (body-frame)")
            else:
                axes[2, 0].axis("off")  # Empty plot
                axes[2, 1].axis("off")  # Empty plot
                axes[2, 2].axis("off")  # Empty plot

            # Plot off-center position
            if "pos_ia" in data:
                pos_off = jnp.linalg.norm(jnp.concatenate([data["pos_ia"][:, [0]], data["pos_ia"][:, [2]]], axis=-1),
                                          axis=-1)
                axes[2, 0].plot(ts_data["pos_ia"], pos_off, label="pos_off", color=color, linestyle="-")
                axes[2, 0].legend()
                axes[2, 0].set_title("Radial Position")
            elif "vel_ib" in data:
                pass
            else:
                axes[2, 0].axis("off")  # Empty plot

            # Plot velocities in agent frame
            if "vel_ia" in data:
                vel_on = data["vel_ia"][:, 1]
                vel_off = jnp.linalg.norm(jnp.concatenate([data["vel_ia"][:, [0]], data["vel_ia"][:, [2]]], axis=-1),
                                          axis=-1)
                axes[2, 1].plot(ts_data["vel_ia"], vel_off, label="vel_off", color=color, linestyle="-")
                axes[2, 1].legend()
                axes[2, 1].set_title("Velocity (off-path)")
                axes[2, 2].plot(ts_data["vel_ia"], vel_on, label="vel_on", color=color, linestyle="-")
                axes[2, 2].legend()
                axes[2, 2].set_title("Velocity (on-path)")
            elif "vel_ib" in data:
                pass
            else:
                axes[2, 1].axis("off")  # Empty plot
                axes[2, 2].axis("off")  # Empty plot

        fig.suptitle(f"{identifier}")
        return figs

    # Initialize solver
    strategies = {
        # todo: optimize OpenES
        "OpenES": dict(popsize=200, use_antithetic_sampling=True, opt_name="adam",
                       lrate_init=0.125, lrate_decay=0.999, lrate_limit=0.001,
                       sigma_init=0.05, sigma_decay=0.999, sigma_limit=0.01, mean_decay=0.0),
        "CMA_ES": dict(popsize=200, elite_ratio=0.1, sigma_init=0.4, mean_decay=0.),
    }

    denorm = base.Denormalize.init(u_min, u_max)
    trans = base.Chain.init(denorm, *shared)
    solver = evo.EvoSolver.init(denorm.normalize(u_min), denorm.normalize(u_max), strategy, strategies[strategy])
    task = Task.init(graph=graph, solver=solver, max_steps=300, description="System Identification",
                     init_params=init_params, trans=trans, loss_filter=loss_filter, rollout=rollout_fn, plot=plot_fn)
    return task