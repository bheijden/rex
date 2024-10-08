from typing import Union, Tuple

import jax
from flax import struct
from jax import numpy as jnp
import numpy as onp
from rex import base
from rex.jax_utils import tree_dynamic_slice
from rex.base import StepState, GraphState
from rex.node import BaseNode


@struct.dataclass
class ActuatorOutput(base.Base):
    """Pendulum actuator output"""
    action: jax.typing.ArrayLike  # Torque to apply to the pendulum


@struct.dataclass
class ActuatorParams(base.Base):
    """Pendulum actuator param definition"""
    actuator_delay: Union[float, jax.typing.ArrayLike]


class Actuator(BaseNode):
    """This is a simple actuator node definition that could interface a real actuator.

    When interfacing real hardware, you would send the action to real hardware in the .step method.
    Optionally, you could also specify a startup routine that is called right before an episode starts.
    Finally, a stop routine is called after the episode is done.
    """

    def __init__(self, *args, **kwargs):
        """No special initialization needed."""
        super().__init__(*args, **kwargs)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorParams:
        """Default params of the node."""
        actuator_delay = 0.05
        return ActuatorParams(actuator_delay=actuator_delay)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        """Starts the node in the state specified by graph_state.

        This method is called right before an episode starts.
        It can be used to move (a real) robot to a starting position as specified by the graph_state.

        Not used when running in compiled mode.
        :param graph_state: The graph state.
        :param timeout: The timeout of the startup.
        :return: Whether the node has started successfully.
        """
        # Move robot to starting position specified by graph_state (e.g. graph_state.state["agent"].init_th)
        return True  # Not doing anything here

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """If we were to control a real robot, you would send the action to the robot here."""
        # Prepare output
        output = step_state.inputs["agent"][-1].data
        output = ActuatorOutput(action=output.action)

        def _apply_action(action):
            """
            Not really doing anything here, just a dummy implementation.
            Include some side-effecting code here (e.g. sending the action to a real robot).

            The .step method may be jit-compiled, it is important to wrap any side-effecting code in a host_callback.
            See the jax documentation for more information on how to do this:
            https://jax.readthedocs.io/en/latest/external-callbacks.html
            """
            # print(f"Applying action: {action}") # Apply action to the robot
            return onp.array(1.0)  # Must match dtype and shape of return_shape

        # Apply action to the robot
        return_shape = jnp.array(1.0)  # Must match dtype and shape of return_shape
        _ = jax.experimental.io_callback(_apply_action, return_shape, output)

        # Update state
        new_step_state = step_state
        return new_step_state, output

    def stop(self, timeout: float = None) -> bool:
        """Stopping routine that is called after the episode is done.

        **IMPORTANT** It may happen that stop is called *before* the final .step call of an episode returns,
        which may cause unsafe behavior when the final step undoes the work of the .stop method.
        This should be handled by the user. For example, by stopping "longer" before returning here.

        Only ran when running asynchronously.
        :param timeout: The timeout of the stop
        :return: Whether the node has stopped successfully.
        """
        # Stop the robot (e.g. set the torque to 0)
        return True


class SimActuator(BaseNode):
    """This is a simple simulated actuator node definition that can either
    1. Feedthrough the agent's action (for normal operation, e.g., training).
       Optionally, you could include some noise or other modifications to the action.
    2. Reapply the recorded actuator outputs for system identification if available.
    """

    def __init__(self, *args, outputs: ActuatorOutput = None, **kwargs):
        """Initialize Actuator for system identification.

        Here, we will reapply the recorded actuator outputs for system identification if available.

        :param outputs: Recorded actuator Outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorParams:
        """Default params of the node."""
        actuator_delay = 0.05
        return ActuatorParams(actuator_delay=actuator_delay)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        # Get action from dataset if available, else use the one provided by the agent
        if self._outputs is not None:  # Use the recorded action (for system identification)
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
        else:  # Feedthrough the agent's action (for normal operation, e.g., training)
            output = step_state.inputs["agent"][-1].data
            output = ActuatorOutput(action=output.action)
        new_step_state = step_state
        return new_step_state, output