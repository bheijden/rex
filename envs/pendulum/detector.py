from typing import Union, Tuple

import jax
from flax import struct
from jax import numpy as jnp
import numpy as onp

from rexv2.base import GraphState, StepState, Base
from rexv2.node import BaseNode
from rexv2.jax_utils import tree_dynamic_slice


@struct.dataclass
class DetectorParams(Base):
    # bgr
    width: int = struct.field(pytree_node=False)  # 424
    height: int = struct.field(pytree_node=False)  # 240
    # Movement detection parameters
    min_max_threshold: Union[float, jax.typing.ArrayLike]  # 90
    lower_bgr: jax.typing.ArrayLike  # [110, 150, 100]
    upper_bgr: jax.typing.ArrayLike  # [180, 220, 180]
    kernel_size: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # 10
    sigma: Union[float, jax.typing.ArrayLike]  # 10/6
    binarization_threshold: Union[float, jax.typing.ArrayLike]  # 0.5
    # Ellipse parameters
    a: Union[float, jax.typing.ArrayLike]  # Ellipse axis
    b: Union[float, jax.typing.ArrayLike]  # Ellipse axis
    x0: Union[float, jax.typing.ArrayLike]  # "width" pixel coordinate of the center of the ellipse (coincides with joint)
    y0: Union[float, jax.typing.ArrayLike]  # "height" pixel coordinate of the center of the ellipse (coincides with joint)
    phi: Union[float, jax.typing.ArrayLike]  # Ellipse angle of rotation (in radians)
    # Angle offset after transforming the ellipse to circle to match the (physical) pendulum angle
    theta_offset: Union[float, jax.typing.ArrayLike]  # 0.0
    # Low-pass filter parameters for thdot
    rate: Union[float, jax.typing.ArrayLike] = struct.field(pytree_node=False)  # Sampling frequency in Hz
    wn: Union[float, jax.typing.ArrayLike]  # Critical frequency in Hz (must be less than half the sampling frequency)

    def init_state(self) -> "DetectorState":
        cummin = 255*jnp.ones((self.height, self.width, 3))
        cummax = jnp.zeros((self.height, self.width, 3))
        b, a = self.firstorder_lowpass_coef(wn=self.wn, fs=self.rate)
        yn_1 = 0.0
        xn_1 = 0.0
        thn_1 = 0.0
        tsn_1 = 0.0 - 1 / self.rate  # to avoid potential division by zero
        return DetectorState(cummin=cummin, cummax=cummax, b=b, a=a, yn_1=yn_1, xn_1=xn_1, thn_1=thn_1, tsn_1=tsn_1)

    def init_output(self) -> "DetectorOutput":
        centroid = jnp.array([0, 0]).astype(int)
        median = centroid
        th = 0.0
        thdot = 0.0
        ts = 0.0
        return DetectorOutput(centroid=centroid, median=median, th=th, thdot=thdot, ts=ts)

    def step(self, ts: Union[float, jax.typing.ArrayLike], bgr: jax.typing.ArrayLike, state: "DetectorState") -> Tuple["DetectorState", "DetectorOutput"]:
        new_state, centroid, median = self.bgr_to_pixel(state, bgr)
        th = self.pixel_to_th(median)
        new_state, thdot = self.th_to_thdot(new_state, th, ts)
        output = DetectorOutput(centroid=centroid, median=median, th=th, thdot=thdot, ts=ts)
        return new_state, output

    def noncausal_step(self, ts: Union[float, jax.typing.ArrayLike], bgr: jax.typing.ArrayLike) -> Tuple["DetectorState", "DetectorOutput"]:
        cummin = bgr.min(axis=0)
        cummax = bgr.max(axis=0)
        init_state = self.init_state().replace(cummin=cummin, cummax=cummax)

        def _scan(state, x):
            ts, bgr = x
            next_state, output = self.step(ts, bgr, state)
            return next_state, output

        final_state, outputs = jax.lax.scan(_scan, init_state, (ts, bgr))
        return final_state, outputs

    def bgr_to_pixel(self, state: "DetectorState", bgr: jax.typing.ArrayLike) -> Tuple["DetectorState", jax.Array, jax.Array]:
        mask_mov, cummin, cummax = self._movement_mask(bgr, state.cummin, state.cummax, self.min_max_threshold)
        mask_col = self._color_mask(bgr, self.lower_bgr, self.upper_bgr)
        mask_comb = jnp.logical_and(mask_mov, mask_col)
        mask_smooth = self._gaussian_smoothing(mask_comb, self.kernel_size, self.sigma)
        mask_binarized = jnp.where(mask_smooth > self.binarization_threshold, 1.0, 0.0).astype(bool)

        # New state
        new_state = state.replace(cummin=cummin, cummax=cummax)

        # Calculate centroid and median
        centroid = self._centroid(mask_binarized)
        median = self._median(mask_binarized)
        return new_state, centroid, median

    def pixel_to_th(self, pixel: jax.typing.ArrayLike) -> jax.Array:
        th = self._point_to_theta(pixel[1], pixel[0], self.a, self.b, self.x0, self.y0, self.phi, self.theta_offset)
        return th

    def th_to_thdot(self, state: "DetectorState", th: jax.typing.ArrayLike, ts: Union[float, jax.typing.ArrayLike]) -> Tuple["DetectorState", jax.Array]:
        # Filter finite-difference estimates of the pendulum angular velocity
        dtn = ts - state.tsn_1
        thn_unwrapped, thn_1_unwrapped = jnp.unwrap(jnp.array([th, state.thn_1]))
        xn = (thn_unwrapped - thn_1_unwrapped) / dtn
        xn = jnp.nan_to_num(xn, nan=(thn_unwrapped - thn_1_unwrapped)/self.rate)
        yn = self.filter_sample(state.yn_1, xn, state.xn_1, state.b, state.a)

        # New state
        new_state = state.replace(yn_1=yn, xn_1=xn, thn_1=th, tsn_1=ts)
        return new_state, yn

    @staticmethod
    def _movement_mask(bgr, cummin, cummax, threshold):
        cummin = jnp.minimum(cummin, bgr)
        cummax = jnp.maximum(cummax, bgr)
        movement_masks = jnp.any(cummax - cummin > threshold, axis=-1)
        return movement_masks, cummin, cummax

    @staticmethod
    def _color_mask(bgr, lower, upper):
        mask = jnp.all(jnp.logical_and(lower <= bgr, bgr <= upper), axis=-1)
        return mask

    @staticmethod
    def _gaussian_smoothing(bgr, kernel_size, sigma):
        # Create Gaussian kernel
        ax = jnp.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = jnp.meshgrid(ax, ax)
        kernel = jnp.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / jnp.sum(kernel)
        # Normalize image to be between 0 and 1 if it's not already
        bgr = bgr.astype(float)
        # Apply the Gaussian kernel to each frame of the mask
        # The mask is assumed to be of shape (num_frames, height, width)
        bgr_smooth = jax.scipy.signal.convolve2d(bgr, kernel, mode='same', boundary='fill', fillvalue=0)
        return bgr_smooth

    @staticmethod
    def _centroid(mask):
        mask = mask.astype(jnp.float32)

        # Create a grid of coordinates
        rows, cols = mask.shape
        r = jnp.arange(rows)[:, None]
        c = jnp.arange(cols)

        # Compute the weighted sum of indices
        total_weight = mask.sum()
        r_centroid = (r * mask).sum() / total_weight
        c_centroid = (c * mask).sum() / total_weight

        # Replace nans with center of the image
        r_centroid = jnp.nan_to_num(r_centroid, nan=rows / 2)
        c_centroid = jnp.nan_to_num(c_centroid, nan=cols / 2)

        # Round to the nearest integer, clip to the image size, and as integer
        r_centroid = jnp.clip(jnp.round(r_centroid), 0, rows - 1).astype(jnp.int32)
        c_centroid = jnp.clip(jnp.round(c_centroid), 0, cols - 1).astype(jnp.int32)
        return jnp.array([r_centroid, c_centroid], dtype=int)

    @staticmethod
    def _median(mask):
        # Make sure the mask is a binary mask and in floating point format
        mask = mask.astype(jnp.float32)

        # Create a grid of coordinates
        rows, cols = mask.shape
        r = jnp.arange(rows)[:, None]
        c = jnp.arange(cols)

        # Compute the cumulative distribution of the weights along each axis
        cdf_r = jnp.cumsum(mask.sum(axis=1)) / mask.sum()
        cdf_c = jnp.cumsum(mask.sum(axis=0)) / mask.sum()

        # Find the median position along each axis
        r_median = jnp.argwhere(cdf_r >= 0.5, size=1)[0, 0]
        c_median = jnp.argwhere(cdf_c >= 0.5, size=1)[0, 0]

        # If r_median AND c_median are 0, set them to the center of the image
        mask = jnp.logical_and(r_median == 0, c_median == 0)
        r_median = jnp.where(mask, rows / 2, r_median)
        c_median = jnp.where(mask, cols / 2, c_median)
        return jnp.array([r_median, c_median], dtype=int)

    @staticmethod
    def _point_to_theta(x, y, a, b, x0, y0, phi, theta_offset):
        # Translate
        x_prime = x - x0
        y_prime = y - y0
        # Rotate
        x_double_prime = x_prime * jnp.cos(-phi) + y_prime * jnp.sin(-phi)
        y_double_prime = -x_prime * jnp.sin(-phi) + y_prime * jnp.cos(-phi)
        # Scale
        x_triple_prime = x_double_prime / a
        y_triple_prime = y_double_prime / b
        # Calculate the angle
        th = jnp.arctan2(y_triple_prime, x_triple_prime) + theta_offset
        return th.astype(float)

    @staticmethod
    def firstorder_lowpass_coef(wn, fs):
        """
        Compute first order Butterworth filter coefficients using the bilinear transform.

        Args:
        - wn: The cutoff frequency in Hz.
        - fs: The sampling frequency in Hz.

        Returns:
        - b: Numerator coefficients of the filter.
        - a: Denominator coefficients of the filter.
        """
        # Pre-warp the cutoff frequency
        tan_half_omega = jnp.tan(jnp.pi * wn / fs)

        # Calculate coefficients using the bilinear transform method
        b = jnp.array([tan_half_omega / (1 + tan_half_omega)])
        a = jnp.array([1, (tan_half_omega - 1) / (1 + tan_half_omega)])

        # Normalize b coefficients for digital filter (bilinear transform)
        b_normalized = jnp.array([b[0], b[0]])  # Since in a first-order Butterworth filter, the output is scaled by b[0]
        a_normalized = a  # a coefficients remain the same

        return b_normalized, a_normalized

    @staticmethod
    def filter_sample(yn_1, xn, xn_1, b, a):
        """
        Calculate the filtered signal value at time n using a first-order digital lowpass filter.

        Parameters:
        - yn_1: The filtered signal value at time n-1.
        - xn: The input signal value at time n.
        - xn_1: The input signal value at time n-1.
        - b: The numerator coefficients of the filter.
        - a: The denominator coefficients of the filter.

        Returns:
        - yn: The filtered signal value at time n.
        """
        # Assuming b = [b0, b1] and a = [a0, a1], where a0 is always assumed 1 for normalization
        yn = b[0] * xn + b[1] * xn_1 + a[1] * yn_1
        return yn

    @staticmethod
    def estimate_ellipse(yx):
        # Extract x and y coordinates
        x = yx[:, 1]
        y = yx[:, 0]

        # Construct design matrix for linear system
        # The matrix columns represent x^2, xy, y^2, x, y, 1 terms respectively
        D = jnp.vstack([x ** 2, x * y, y ** 2, x, y, jnp.ones_like(x)]).T

        # Since we are solving Ax = 0, we want to minimize the norm of Dp,
        # where p is the parameter vector [A, B, C, D, E, F]. This is a homogeneous system.
        # Use Singular Value Decomposition (SVD) to solve
        U, S, Vt = jnp.linalg.svd(D, full_matrices=False)

        # The solution is the last column of V (corresponding to the smallest singular value)
        params = Vt.T[:, -1]

        A, B, C, D, E, F = params

        # Calculate the semi-major and semi-minor axes
        numerator = -2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
        denominator = B ** 2 - 4 * A * C
        term = jnp.sqrt((A - C) ** 2 + B ** 2)
        a = jnp.sqrt(numerator / (denominator * ((A + C) + term)))
        b = jnp.sqrt(numerator / (denominator * ((A + C) - term)))

        # Calculate the ellipse center
        x0 = (2 * C * D - B * E) / denominator
        y0 = (2 * A * E - B * D) / denominator

        # Calculate the rotation angle
        phi = 0.5 * jnp.arctan2(-B, C - A)

        return a, b, x0, y0, phi

    def draw_ellipse(self, bgr: jax.typing.ArrayLike, border: float = 0.95, color=(0, 255, 0)):
        color = onp.array(color, dtype=onp.uint8)

        def draw_single_ellipse(_bgr):
            # Create a meshgrid for the image
            y, x = jnp.meshgrid(jnp.arange(_bgr.shape[0]), jnp.arange(_bgr.shape[1]), indexing='ij')

            # Translate and rotate coordinates
            x_t = x - self.x0
            y_t = y - self.y0
            x_r = -x_t * jnp.sin(self.phi) + y_t * jnp.cos(self.phi)
            y_r = -x_t * jnp.cos(self.phi) - y_t * jnp.sin(self.phi)

            # Ellipse equation
            ellipse = ((x_r / self.b) ** 2 + (y_r / self.a) ** 2 <= 1) & ((x_r / self.b) ** 2 + (y_r / self.a) ** 2 > border)

            # Create ellipse mask (green color)
            ellipse_mask = jnp.zeros_like(_bgr)
            ellipse_mask = ellipse_mask.at[ellipse, :].set(color)  # Green channel

            # Draw center point
            center_mask = ((x - self.x0) ** 2 + (y - self.y0) ** 2 <= 9)
            ellipse_mask = ellipse_mask.at[center_mask, :].set(color)  # Green channel

            # Add ellipse to _bgr
            res = jnp.maximum(_bgr, ellipse_mask)
            return res  # jnp.clip(_bgr + ellipse_mask, 0, 255)

        return jax.vmap(draw_single_ellipse)(bgr)

    @staticmethod
    def draw_centroids(bgr: jax.typing.ArrayLike, centroids: jax.typing.ArrayLike, color=(0, 255, 0), dot_radius: float = 2, stack: bool = True):
        color = onp.array(color, dtype=onp.uint8)

        def draw_single_centroid(_bgr, _centroid):
            # Create a meshgrid for the image
            y, x = jnp.meshgrid(jnp.arange(_bgr.shape[0]), jnp.arange(_bgr.shape[1]), indexing='ij')

            # Draw center point
            center_mask = ((x - _centroid[1]) ** 2 + (y - _centroid[0]) ** 2 <= dot_radius ** 2)
            center_mask = center_mask[..., None].repeat(3, axis=-1)
            center_mask = jnp.where(center_mask, color, 0)

            # Add center to _bgr
            res = jnp.maximum(_bgr, center_mask)
            return res

        if not stack:
            bgr_out = jax.vmap(draw_single_centroid)(bgr, centroids)
        else:
            def _scan(_mask, x):
                _bgr, _centroid = x
                _stacked_mask = draw_single_centroid(_mask, _centroid)
                res = jnp.maximum(_bgr, _stacked_mask)
                return _stacked_mask, res

            init_mask = jnp.zeros_like(bgr[0])
            _, bgr_out = jax.lax.scan(_scan, init_mask,(bgr, centroids))
        return bgr_out

    @staticmethod
    def play_video(bgr, fps):
        import cv2
        wait_time = int(1000 / fps)  # Time in ms between frames
        bgr_uint8 = onp.array(bgr).astype(onp.uint8)
        while True:
            for img_uint8 in bgr_uint8:
                cv2.imshow('Video', img_uint8)
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return


@struct.dataclass
class DetectorState(Base):
    # Movement detection state
    cummin: jax.typing.ArrayLike  # [num_frames, height, width, 3]
    cummax: jax.typing.ArrayLike  # [num_frames, height, width, 3]
    # Low-pass filter state for thdot
    b: jax.typing.ArrayLike  # Numerator coefficients of the filter
    a: jax.typing.ArrayLike  # Denominator coefficients of the filter
    yn_1: Union[float, jax.typing.ArrayLike]  # The filtered angular velocity at time n-1
    xn_1: Union[float, jax.typing.ArrayLike]  # The finite-difference angular velocity at time n-1
    thn_1: Union[float, jax.typing.ArrayLike]  # The angle at time n-1
    tsn_1: Union[float, jax.typing.ArrayLike]  # The timestamp at time n-1


@struct.dataclass
class DetectorOutput(Base):
    centroid: jax.typing.ArrayLike  # [height, width] (int)
    median: jax.typing.ArrayLike  # [height, width] (int)
    th: Union[float, jax.typing.ArrayLike]  # physical pendulum angle (in radians)
    thdot: Union[float, jax.typing.ArrayLike]  # (low-pass filtered finite-difference) angular velocity (in radians/second)
    ts: Union[float, jax.typing.ArrayLike]  # ts of the image


class Detector(BaseNode):
    def __init__(self, *args, width=424, height=240, **kwargs):
        self.width = width
        self.height = height
        super().__init__(*args, **kwargs)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> DetectorParams:
        """Default params of the root."""
        params = DetectorParams(width=self.width,
                                height=self.height,
                                min_max_threshold=70,
                                lower_bgr=jnp.array([0, 0, 100]),
                                upper_bgr=jnp.array([150, 60, 255]),
                                kernel_size=10,
                                sigma=10/6,
                                binarization_threshold=0.5,
                                a=58.5709114074707,
                                b=70.53766632080078,
                                x0=211.99990844726562,
                                y0=119.99968719482422,
                                phi=1.0113574266433716,
                                theta_offset=0.7549607157707214,
                                rate=self.rate,
                                wn=5,
                                )
        return params

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> DetectorState:
        """Default state of the root."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        state = params.init_state()
        return state

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> DetectorOutput:
        """Default output of the root."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        output = params.init_output()
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, DetectorOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get binary mask
        bgr = inputs["camera"].data.bgr[-1]
        ts = inputs["camera"].data.ts[-1]

        # Apply the detector
        new_state, output = params.step(ts, bgr, state)

        # Update step state
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output


@struct.dataclass
class SimDetectorState(DetectorState):
    loss_th: [float, jax.typing.ArrayLike]
    loss_th_prob: [float, jax.typing.ArrayLike]


class SimDetector(Detector):
    def __init__(self, *args, outputs: DetectorOutput = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._outputs = outputs
        self._dummy_output = self.init_output(jax.random.PRNGKey(0))

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> DetectorOutput:
        """Default output of the node."""
        graph_state = graph_state or GraphState()
        if self._outputs is not None:
            output = tree_dynamic_slice(self._outputs, jnp.array([graph_state.eps, 0]))
        else:
            output = super().init_output(rng, graph_state)
        return output

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SimDetectorState:
        """Default state of the root."""
        state = super().init_state(rng, graph_state)
        # Remove cummin and cummax from state if outputs are available (i.e., if the node is used for system identification)
        state = state if self._outputs is None else state.replace(cummin=None, cummax=None)
        state = SimDetectorState(**state.__dict__, loss_th=0.0, loss_th_prob=0.0)
        return state

    def step(self, step_state: StepState) -> Tuple[StepState, DetectorOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Grab output from state
        ts = inputs["camera"].data.ts[-1]
        bgr = inputs["camera"].data.bgr
        th_world = inputs["camera"].data.world.th[0]

        # Assert that either bgr or outputs is None or both must be None (i.e., cannot both be available)
        assert (bgr is None or self._outputs is None), "Only one of bgr or outputs should be available. Did you provide camera and detector outputs?"
        if bgr is not None:
            # Here we use the camera output, and rely on pixel detection
            # Get binary mask
            bgr = inputs["camera"].data.bgr[0]
            mask_mov, cummin, cummax = params._movement_mask(bgr, state.cummin, state.cummax, params.min_max_threshold)
            mask_col = params._color_mask(bgr, params.lower_bgr, params.upper_bgr)
            mask_comb = jnp.logical_and(mask_mov, mask_col)
            mask_smooth = params._gaussian_smoothing(mask_comb, params.kernel_size, params.sigma)
            mask_binarized = jnp.where(mask_smooth > params.binarization_threshold, 1.0, 0.0).astype(bool)

            # Calculate centroid and median
            centroid = params._centroid(mask_binarized)
            median = params._median(mask_binarized)

            # Calculate theta
            th = params._point_to_theta(median[1], median[0], params.a, params.b, params.x0, params.y0, params.phi, params.theta_offset)
        elif self._outputs is not None:
            # Here, we do not use the camera output, but the outputs provided to the node
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            median, centroid = output.median, output.centroid

            # Calculate theta
            th = params._point_to_theta(median[1], median[0], params.a, params.b, params.x0, params.y0, params.phi, params.theta_offset)

            # Repeat unchanged state
            cummin, cummax = state.cummin, state.cummax
        else:
            # Here, we do not use the camera output, or rely on pixel detection, but use the world output
            th = th_world  # Use th_world as the output

            # Repeat unchanged output
            median, centroid = self._dummy_output.median, self._dummy_output.centroid

            # Repeat unchanged state
            cummin, cummax = state.cummin, state.cummax

        # Filter finite-difference estimates of the pendulum angular velocity
        dtn = ts - state.tsn_1
        thn_unwrapped, thn_1_unwrapped = jnp.unwrap(jnp.array([th, state.thn_1]))
        xn = (thn_unwrapped - thn_1_unwrapped) / dtn
        xn = jnp.nan_to_num(xn, nan=(thn_unwrapped - thn_1_unwrapped) / self.rate)
        yn = params.filter_sample(state.yn_1, xn, state.xn_1, state.b, state.a)

        # Prepare output
        output = DetectorOutput(centroid=centroid, median=median, th=th, thdot=yn, ts=ts)

        # Calculate loss
        loss_th = state.loss_th + (jnp.sin(th) - jnp.sin(th_world)) ** 2 + (jnp.cos(th) - jnp.cos(th_world)) ** 2

        # Update state
        new_state = state.replace(cummin=cummin, cummax=cummax, yn_1=yn, xn_1=xn, thn_1=th, tsn_1=ts, loss_th=loss_th)

        # Update state
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output
