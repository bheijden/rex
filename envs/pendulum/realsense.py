from typing import Tuple, Any, Union
import atexit

import jax.experimental.host_callback as hcb
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict

from rexv2 import base
from rexv2.node import BaseNode
from rexv2.jax_utils import tree_dynamic_slice

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(f"Failed to import pyrealsense2: {e}. Please install it with `pip install pyrealsense2`.")


@struct.dataclass
class D435iParams(base.Base):
    sensor_delay: base.TrainableDist# = struct.field(default=None)


@struct.dataclass
class D435iState(base.Base):
    pass


@struct.dataclass
class Image(base.Base):
    """Pendulum image definition"""
    bgr: jax.Array
    ts: Union[float, jax.typing.ArrayLike]


class D435iBase(BaseNode):
    def __init__(self, *args, width: int = 640, height: int = 480, fps: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        # Cam
        self._fps = fps
        self._width = width
        self._height = height

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> D435iParams:
        """Default params of the node."""
        sensor_delay = base.TrainableDist.create(alpha=0.1262352466583252, min=0.0, max=1 / self.rate)
        return D435iParams(sensor_delay=sensor_delay)

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> D435iState:
        """Default state of the node."""
        return D435iState()


class D435i(D435iBase):
    def __init__(self, *args, **kwargs):
        """Initialize RealSense D435i camera.

        Sources:
        - https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
        - https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md#examples
        - https://github.com/IntelRealSense/librealsense/issues/2549#issuecomment-431863920
        - https://github.com/IntelRealSense/librealsense/issues/12185

        """
        super().__init__(*args, **kwargs)

        # Initialize realsense
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = self._config.resolve(pipeline_wrapper)
        self._device = pipeline_profile.get_device()
        self._device_product_line = str(self._device.get_info(rs.camera_info.product_line))

        # Config streams (color only)
        self._config.disable_all_streams()
        # self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        # Start streaming
        self._pipeline.start(self._config)

        # Register shutdown
        atexit.register(self._shutdown)

        # Wait until dummy output is ready
        self._dummy_output = self._get_output(None)

    def _shutdown(self):
        try:
            self._pipeline.stop()
        except RuntimeError as e:
            print(f"Failed to stop pipeline: {e}")

    def _get_output(self, _dummy) -> Image:
        # Get frame
        i = 0
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # depth_image = np.asanyarray(depth_frame.get_data())
            # if color_frame and depth_frame:
            if color_frame:
                ts = self.now()
                color_image = onp.asanyarray(color_frame.get_data())
                break

            # Retry
            i += 1
            if 1 % 100 == 0:
                print(f"Failed to get frame {i} times.")
        return Image(bgr=color_image, ts=onp.array(ts, dtype=onp.float32))

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> Image:
        """Default output of the node."""
        graph_state = graph_state or base.GraphState()
        output = jax.experimental.io_callback(self._get_output, self._dummy_output, 0.)
        # output = hcb.call(self._get_output, 0, result_shape=self._dummy_output)
        # Account for sensor delay
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        output = output.replace(ts=-1./self.rate - sensor_delay)
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, Image]:
        """Step the node."""
        output = jax.experimental.io_callback(self._get_output, self._dummy_output, 0.)

        # Update ts of step_state
        new_step_state = step_state.replace(ts=output.ts)

        # Correct for sensor delay
        delay = step_state.params.sensor_delay.mean()
        output = output.replace(ts=new_step_state.ts - delay)
        return new_step_state, output


@struct.dataclass
class SimImage:
    bgr: jax.Array
    ts: Union[float, jax.typing.ArrayLike]  # ts of the image
    world: Any


class SimD435i(D435iBase):
    def __init__(self, *args, outputs: Image = None, **kwargs):
        """Initialize RealSense D435i camera for system identification.

        Args:
        outputs: Recorded Images to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> SimImage:
        """Default output of the node."""
        eps = graph_state.eps if graph_state else 0
        seq = 0
        bgr = tree_dynamic_slice(self._outputs.bgr, jnp.array([eps, seq])) if self._outputs else None
        world = self.inputs["world"].output_node.init_output(rng, graph_state)  # Get world shape
        # Account for sensor delay
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        ts = -1. / self.rate - sensor_delay
        return SimImage(bgr=bgr, ts=ts, world=world)

    def init_inputs(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        inputs["world"] = inputs["world"].replace(delay_dist=params.sensor_delay)
        return FrozenDict(inputs)

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, SimImage]:
        """Step the node."""
        # Get recorded output if available
        bgr = tree_dynamic_slice(self._outputs.bgr, jnp.array([step_state.eps, step_state.seq])) if self._outputs else None

        # Adjust ts_start (i.e. step_state.ts) to reflect the timestamp of the world state that generated the image
        ts = step_state.ts - step_state.params.sensor_delay.mean()  # Should be equal to world_interp.ts_sent[-1]?

        # Prepare output
        output = SimImage(bgr=bgr, ts=ts, world=step_state.inputs["world"][-1].data)
        return step_state, output
