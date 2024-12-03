from typing import Tuple, Any, Union
import atexit

import jax.experimental.host_callback as hcb
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict

from rex import base
from rex.node import BaseNode
from rex.jax_utils import tree_dynamic_slice
import envs.pendulum.detector as det

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(f"Failed to import pyrealsense2: {e}. Please install it with `pip install pyrealsense2`.")


@struct.dataclass
class D435iParams(base.Base):
    sensor_delay: base.TrainableDist
    std_th: Union[float, jax.typing.ArrayLike]  # Standard deviation of angle noise


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
        sensor_delay = base.TrainableDist.create(delay=0.00315588116645813, min=0.0, max=0.05)
        std_th = 0.4375  # Standard deviation of angle noise
        return D435iParams(sensor_delay=sensor_delay, std_th=std_th)


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

    def get_intrinsics(self):
        profile = self._pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        # focal_length_x = color_intrinsics.fx
        # focal_length_y = color_intrinsics.fy
        # optical_center_x = color_intrinsics.ppx
        # optical_center_y = color_intrinsics.ppy
        # intrinsic_matrix = onp.array([
        #     [focal_length_x, 0, optical_center_x],
        #     [0, focal_length_y, optical_center_y],
        #     [0, 0, 1]
        # ])
        return color_intrinsics

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
class D435iDetectorParams(D435iParams):
    detector: det.DetectorParams


@struct.dataclass
class D435iDetectorOutput(det.DetectorOutput):
    """Pendulum image definition"""
    bgr: jax.Array


class D435iDetectorBase(D435iBase):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> D435iDetectorParams:
        """Default params of the node."""
        params = super().init_params(rng, graph_state)
        detector = det.DetectorParams(width=self._width,
                                      height=self._height,
                                      min_max_threshold=70,
                                      lower_bgr=jnp.array([0, 0, 100]),
                                      upper_bgr=jnp.array([150, 60, 255]),
                                      kernel_size=10,
                                      sigma=10 / 6,
                                      binarization_threshold=0.5,
                                      a=58.5709114074707,
                                      b=70.53766632080078,
                                      x0=211.99990844726562,
                                      y0=119.99968719482422,
                                      phi=1.0113574266433716,
                                      theta_offset=0.7549607157707214+0.9,
                                      rate=self.rate,
                                      wn=5)
        params = D435iDetectorParams(**params.__dict__, detector=detector)
        return params

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> det.DetectorState:
        """Default state of the node."""
        graph_state = graph_state or base.GraphState()
        params: D435iDetectorParams = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        state = params.detector.init_state()
        return state


class D435iDetector(D435iDetectorBase):
    def __init__(self, *args, include_image: bool = False, **kwargs):
        """Initialize RealSense D435i camera.

        Sources:
        - https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
        - https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md#examples
        - https://github.com/IntelRealSense/librealsense/issues/2549#issuecomment-431863920
        - https://github.com/IntelRealSense/librealsense/issues/12185

        """
        super().__init__(*args, **kwargs)
        # Whether to include image in the output
        self._include_image = include_image

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

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> D435iDetectorOutput:
        """Default output of the node."""
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        state = graph_state.state.get(self.name, self.init_state(rng, graph_state))

        # Get image
        image = jax.experimental.io_callback(self._get_output, self._dummy_output, 0.)
        sensor_delay = params.sensor_delay.mean()
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        image = image.replace(ts=-1. / self.rate - sensor_delay)

        # Get detector output
        detector: det.DetectorParams = params.detector
        _, output = detector.step(image.ts, image.bgr, state)
        bgr = image.bgr if self._include_image else None
        return D435iDetectorOutput(**output.__dict__, bgr=bgr)

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, D435iDetectorOutput]:
        """Step the node."""
        # Get image
        image = jax.experimental.io_callback(self._get_output, self._dummy_output, 0.)
        ss = step_state.replace(ts=image.ts)  # Update ts of step_state

        # Get detector output
        new_state, output = ss.params.detector.step(image.ts, image.bgr, ss.state)
        new_step_state = ss.replace(state=new_state)
        bgr = image.bgr if self._include_image else None
        output = D435iDetectorOutput(**output.__dict__, bgr=bgr)
        return new_step_state, output


class SimD435iDetector(D435iDetectorBase):
    # Situations:
    # 1. The node is used for system identification: outputs w/o images are available
    #   - Maps images to pixel coordinate: [3, 640, 480] -> [2]
    #   - Identify parameters of pixel identification (binarization, bgr_ranges, thresholding, etc.)
    # 2. The node is used for system identification: outputs w/ images are available
    #   - Maps pixel coordinates to theta: [2] -> [1]
    #   - Identify parameters of ellipse fitting (a, b, x0, y0, phi, theta_offset)
    # 3. The node is used to fit the low-pass filter: No outputs are available
    #   - Maps theta to world state (th, thdot): [1] -> [2]
    #   - Identify parameters of the low-pass filter (wn)
    # 4. The node is used for control: no outputs are available and the world state is used directly
    #   - Feeds through the world state to the estimator
    #   - Optionally: Add measurement noise to the world state before sending it to the estimator

    def __init__(self, *args, outputs: D435iDetectorOutput = None, **kwargs):
        """Initialize RealSense D435i camera with detector for system identification.

        Args:
        outputs: Recorded D435iDetectorOutputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> det.SimDetectorState:
        """Default state of the node."""
        state: det.DetectorState = super().init_state(rng, graph_state)
        # Remove cummin and cummax from state if not using images to save memory (cummin and cummax are not used)
        has_bgr: bool = (self._outputs is not None and self._outputs.bgr is not None)
        state = state if has_bgr else state.replace(cummin=None, cummax=None)
        sim_state = det.SimDetectorState(**state.__dict__, loss_th=0.0, loss_th_prob=0.0)
        return sim_state

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> D435iDetectorOutput:
        """Default output of the node."""
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        # Get default detector output
        output_det = params.detector.init_output()
        # Account for sensor delay
        # To avoid division by zero and reflect that the measurement is not from before the start of the episode
        sensor_delay = params.sensor_delay.mean()
        ts = -1. / self.rate - sensor_delay
        output_det = output_det.replace(ts=ts)
        output = D435iDetectorOutput(**output_det.__dict__, bgr=None)
        return output

    def init_inputs(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        inputs["world"] = inputs["world"].replace(delay_dist=params.sensor_delay)
        return FrozenDict(inputs)

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, D435iDetectorOutput]:
        """Step the node."""
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Adjust ts_start (i.e. step_state.ts) to reflect the timestamp of the world state that generated the image
        ts = step_state.ts - step_state.params.sensor_delay.mean()  # Should be equal to world_interp.ts_sent[-1]?
        th_world = step_state.inputs["world"][-1].data.th

        output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq])) if self._outputs else None
        output = jax.tree_util.tree_map(lambda _o: _o[0, 0], output) if output else None
        has_bgr: bool = (output is not None and output.bgr is not None)
        has_output: bool = output is not None
        # (1): Apply image processing pipeline if images are available
        if has_bgr:
            bgr = output.bgr
            new_state, centroid, median = params.detector.bgr_to_pixel(state, ts, bgr)
        elif has_output:
            new_state, centroid, median = state, output.centroid, output.median
        else:
            _dummy = onp.array([0, 0]).astype(int)
            new_state, centroid, median = state, _dummy, _dummy
        # (1,2): Apply ellipse fitting and low-pass filter if outputs are available
        if has_output:
            # Calculate theta from
            th = params.detector.pixel_to_th(median)
            # Calculate reconstruction loss
            loss_th = state.loss_th + (jnp.sin(th) - jnp.sin(th_world)) ** 2 + (jnp.cos(th) - jnp.cos(th_world)) ** 2
            # Calculate log probability loss
            std_th = params.std_th
            prob_th = jax.scipy.stats.norm.pdf(th, loc=th_world, scale=std_th)
            loss_th_prob = state.loss_th_prob - 0.1*prob_th
            # Update state
            new_state = new_state.replace(loss_th=loss_th, loss_th_prob=loss_th_prob)
        else:
            # Add noise
            rng, rng_noise = jax.random.split(rng)
            th = th_world + jax.random.normal(rng_noise, shape=th_world.shape) * params.std_th

        # (1,2,3,4): Get thdot from the low-pass filter
        new_state, thdot = params.detector.th_to_thdot(new_state, th, ts)

        # Update step_state
        new_step_state = step_state.replace(rng=rng, state=new_state)

        # Prepare output
        output = D435iDetectorOutput(ts=ts, centroid=centroid, median=median, th=th, thdot=thdot, bgr=None)
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
        # Not actually using bgr, but only using the shape for now. Else, strange this can happen in delay_dist interpolation.
        bgr = jnp.zeros_like(tree_dynamic_slice(self._outputs.bgr, jnp.array([eps, seq])))[0, 0] if self._outputs else None

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
        bgr = tree_dynamic_slice(self._outputs.bgr, jnp.array([step_state.eps, step_state.seq]))[0, 0] if self._outputs else None

        # Adjust ts_start (i.e. step_state.ts) to reflect the timestamp of the world state that generated the image
        ts = step_state.ts - step_state.params.sensor_delay.mean()  # Should be equal to world_interp.ts_sent[-1]?

        # Prepare output
        output = SimImage(bgr=bgr, ts=ts, world=step_state.inputs["world"][-1].data)
        return step_state, output
