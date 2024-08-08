from typing import Any, Dict, Tuple, Union, List
import jax
import jax.numpy as jnp
import numpy as onp
from math import ceil
from flax import struct
from flax.core import FrozenDict
from rexv2.base import StepState, GraphState, Empty, TrainableDist, Base
from rexv2.node import BaseNode
from rexv2.jax_utils import tree_dynamic_slice


@struct.dataclass
class SupervisorParams(Base):
    # Dynamics
    mass: Union[float, jax.typing.ArrayLike]
    pwm_constants: jax.typing.ArrayLike
    # Ctrl limits
    pwm_from_hover: Union[float, jax.typing.ArrayLike]
    pwm_range: jax.typing.ArrayLike
    phi_max: Union[float, jax.typing.ArrayLike]
    theta_max: Union[float, jax.typing.ArrayLike]
    psi_max: Union[float, jax.typing.ArrayLike]
    z_max: Union[float, jax.typing.ArrayLike]
    # Init states
    fixed_position: jax.typing.ArrayLike
    x_range: jax.typing.ArrayLike
    y_range: jax.typing.ArrayLike
    z_range: jax.typing.ArrayLike
    # Circular path
    fixed_radius: Union[float, jax.typing.ArrayLike]  # Fixed radius
    radius_range: jax.typing.ArrayLike
    center: jax.typing.ArrayLike  # Center of the platform
    # Train
    gamma: Union[float, jax.typing.ArrayLike]
    tmax: Union[float, jax.typing.ArrayLike]
    # Domain randomization
    use_noise: Union[bool, jax.typing.ArrayLike]
    use_dr: Union[bool, jax.typing.ArrayLike]
    ctrl_mapping: List[str] = struct.field(pytree_node=False)
    init_cf: str = struct.field(pytree_node=False)
    init_path: str = struct.field(pytree_node=False)


@struct.dataclass
class SupervisorState(Base):
    init_pos: jax.typing.ArrayLike
    init_vel: jax.typing.ArrayLike
    init_att: jax.typing.ArrayLike
    init_ang_vel: jax.typing.ArrayLike
    radius: Union[float, jax.typing.ArrayLike]


class Supervisor(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SupervisorParams:
        params = SupervisorParams(
            # Dynamics
            mass=0.033,  # kg
            pwm_constants=onp.array([2.130295e-11, 1.032633e-6, 5.485e-4]),
            # ctrl
            ctrl_mapping=["z_ref", "theta_ref", "phi_ref", "psi_ref"],
            pwm_from_hover=15000,
            pwm_range=jnp.array([20000, 60000]),
            phi_max=onp.pi / 6,
            theta_max=onp.pi / 6,
            psi_max=0.,  # No yaw (or onp.pi?)
            z_max=2.0,
            # init crazyflie
            init_cf="random",  # random, fixed
            fixed_position=jnp.array([0.0, 0.0, 2.0]),  # Above the platform
            x_range=jnp.array([-4.0, 4.0]),
            y_range=jnp.array([-4.0, 4.0]),
            z_range=jnp.array([0.0, 2.0]),
            # Circular path
            init_path="random",  # random, fixed
            fixed_radius=2.0,  # Fixed radius
            radius_range=jnp.array([1.0, 1.5]),
            center=jnp.array([0.0, 0.0, 1.0]),   # Center of the platform
            # Train settings
            tmax=5.0,  # Maximum time for the episode
            gamma=0.99,  # Discount factor (add other reward terms)
            # Domain randomization
            use_noise=True,  # Whether to add noise to the measurements & perform domain randomization.
            use_dr=True,  # Whether to use domain randomization.
        )
        return params

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SupervisorState:
        """Default state of the root."""
        graph_state = graph_state or GraphState()
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num=7)
        params: SupervisorParams = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        # Initial crazyflie position
        if params.init_cf == "random":
            init_x = jax.random.uniform(rngs[1], shape=(), minval=params.x_range[0], maxval=params.x_range[1])
            init_y = jax.random.uniform(rngs[2], shape=(), minval=params.y_range[0], maxval=params.y_range[1])
            init_z = jax.random.uniform(rngs[3], shape=(), minval=params.z_range[0], maxval=params.z_range[1])
            init_pos = jnp.array([init_x, init_y, init_z])
        elif params.init_cf == "fixed":
            init_pos = jnp.array([0.0, 0.0, 2.0])
        else:
            raise ValueError(f"Unknown start position method: {params.init_cf}")
        # Circular path
        if params.init_path == "random":
            radius = jax.random.uniform(rngs[5], shape=(), minval=params.radius_range[0], maxval=params.radius_range[1])
        elif params.init_path == "fixed":
            radius = params.fixed_radius
        else:
            raise ValueError(f"Unknown inclination method: {params.init_path}")

        return SupervisorState(
            init_pos=init_pos,
            init_vel=jnp.array([0.0, 0.0, 0.0]),
            init_att=jnp.array([0.0, 0.0, 0.0]),
            init_ang_vel=jnp.array([0.0, 0.0, 0.0]),
            radius=radius,  # Radius of circular path
        )

    def step(self, step_state: StepState) -> Tuple[StepState, Empty]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        output = self.init_output(step_state.rng)
        return new_step_state, output