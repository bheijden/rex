from typing import Dict, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

from rex import base
from rex.jax_utils import tree_dynamic_slice
from rex.base import GraphState, StepState
from rex.node import BaseNode


@struct.dataclass
class SensorOutput(base.Base):
    """Output message definition of the sensor node."""
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorParams(base.Base):
    """
    Other than the sensor delay, we don't have any other parameters.
    You could add more parameters here if needed, such as noise levels etc.
    """
    sensor_delay: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorState:
    """We use this state to record the reconstruction loss."""
    loss_th: Union[float, jax.typing.ArrayLike]
    loss_thdot: Union[float, jax.typing.ArrayLike]


class Sensor(BaseNode):
    """This is a simple sensor node definition that interfaces a real sensor.

    When interfacing real hardware, you would grab the sensor measurement in the .step method.
    Optionally, you could also specify a startup routine that is called right before an episode starts.
    Finally, a stop routine is called after the episode is done.
    """

    def __init__(self, *args, **kwargs):
        """No special initialization needed."""
        super().__init__(*args, **kwargs)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorParams:
        """Default params of the node."""
        sensor_delay = 0.05
        return SensorParams(sensor_delay=sensor_delay)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        th = jnp.pi
        thdot = 0.0
        return SensorOutput(th=th, thdot=thdot)

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        """Starts the node in the state specified by graph_state.

        This method is called right before an episode starts.
        It can be used to move (a real) robot to a starting position as specified by the graph_state.

        Not used when running in compiled mode.
        :param graph_state: The graph state.
        :param timeout: The timeout of the startup.
        :return: Whether the node has started successfully.
        """
        return True  # Not doing anything here

    def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
        """If we were to interface a real hardware, you would grab the sensor measurement here."""

        """
        As the .step method may be jit-compiled, it is important to wrap any side-effecting code in a host_callback.
        See the jax documentation for more information on how to do this:
        https://jax.readthedocs.io/en/latest/external-callbacks.html
        """
        world = step_state.inputs["world"][-1].data

        def _grab_measurement():
            """
            Not really doing anything here, just a dummy implementation.
            Include some side-effecting code here (e.g. grabbing measurement from sensor).

            The .step method may be jit-compiled, it is important to wrap any side-effecting code in a host_callback.
            See the jax documentation for more information on how to do this:
            https://jax.readthedocs.io/en/latest/external-callbacks.html
            """
            # print("Grabbing sensor measurement")
            sensor_msg = onp.array(1.0)  # Dummy sensor measurement (not actually used)
            return sensor_msg  # Must match dtype and shape of return_shape

        # Grab sensor measurement
        return_shape = jnp.array(1.0)  # Must match dtype and shape of return_shape
        _ = jax.experimental.io_callback(_grab_measurement, return_shape)

        # Prepare output
        output = SensorOutput(th=world.th, thdot=world.thdot)

        # Update state (NOOP)
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
        return True  # Not doing anything here


class SimSensor(BaseNode):
    """This is a simple simulated sensor node definition that can either
    1. Convert the world state into a realistic sensor measurement (for normal operation, e.g., training).
       Optionally, you could include some noise or other modifications to the sensor measurement.
    2. Calculate a reconstruction loss based on the sensor measurement and the recorded sensor outputs.

    By calculating and aggregating the reconstruction loss here, we take time-scale differences and delays into account.
    """
    def __init__(self, *args, outputs: SensorOutput = None, **kwargs):
        """Initialize a simulated sensor for system identification.

        If outputs are provided, we will calculate the reconstruction loss based on the recorded sensor outputs.

        :param outputs: Recorded sensor Outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorParams:
        """Default params of the node."""
        sensor_delay = 0.05
        return SensorParams(sensor_delay=sensor_delay)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorState:
        """Default state of the node."""
        return SensorState(loss_th=0.0, loss_thdot=0.0)  # Initialize reconstruction loss to zero at the start of the episode

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        th = jnp.pi
        thdot = 0.0
        return SensorOutput(th=th, thdot=thdot)  # Fix the initial sensor values

    def init_delays(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> Dict[str, Union[float, jax.typing.ArrayLike]]:
        """Initialize trainable communication delays.

        **Note** These only include trainable delays that were specified while connecting the nodes.

        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default output.
        :return: Trainable delays (e.g., {input_name: delay}). Can be an incomplete dictionary.
                 Entries for non-trainable delays or non-existent connections are ignored.
        """
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        delays = {"world": params.sensor_delay}
        return delays

    def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
        # Determine output
        data = step_state.inputs["world"][-1].data
        output = SensorOutput(th=data.th, thdot=data.thdot)

        # Calculate loss
        if self._outputs is not None:  # Calculate reconstruction loss and aggregate in state
            output_rec = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            th_rec, thdot_rec = output_rec.th, output_rec.thdot
            state = step_state.state
            loss_th = state.loss_th + (jnp.sin(output.th) - jnp.sin(th_rec)) ** 2 + (jnp.cos(output.th) - jnp.cos(th_rec)) ** 2
            loss_thdot = state.loss_thdot + (output.thdot - thdot_rec) ** 2
            new_state = state.replace(loss_th=loss_th, loss_thdot=loss_thdot)
        else:  # NOOP
            new_state = step_state.state

        # Update step_state
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output





