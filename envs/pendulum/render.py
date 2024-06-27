from typing import Any, Dict, Tuple, Union, List
import jax
import jax.numpy as jnp
import numpy as onp
import jax.experimental.host_callback as hcb

from flax import struct
from rexv2.constants import LogLevel
from rexv2.base import StepState, GraphState, Base
from rexv2.node import BaseNode
from rexv2.utils import log

try:
    import cv2
except ImportError as e:
    cv2 = None
    log("main", "red", LogLevel.WARN, "rendering", "Could not import cv2. Rendering will not work.")


@struct.dataclass
class Image(Base):
    """Pendulum image definition"""
    data: jax.Array


class Render(BaseNode):
    def __init__(self, *args, visual: str = "disc", mode: str = "human", encoding: str = "bgr", shape: List[int] = None, **kwargs):
        """Renders a pendulum environment.

        :param args: Args for Node base class.
        :param visual: Type of visualisation. Either "disc" or "rod".
        :param mode: Rendering mode. Options are 'human' or 'rgb_array'.
        :param encoding: Encoding of the image. Options are 'bgr' or 'rgb'.
        :param shape: Shape of the image to be rendered.
        :param kwargs: Kwargs for Node base class.
        """
        super().__init__(*args, **kwargs)
        self._visual = visual
        self._mode = mode
        self._encoding = encoding
        self._shape = shape or [400, 400, 3]
        self._dummy_image = Image(data=jnp.zeros(self._shape, dtype=jnp.uint8))

    def _render(self, last_obs) -> Image:
        """Render the image."""
        # Grab current state
        th, thdot = last_obs.th, last_obs.thdot

        # Prepare new image
        data = onp.zeros(self._shape, onp.uint8)
        if cv2 is not None:
            if self._visual == "disc":
                data = disc_pendulum_render_fn(data, th, thdot)
            else:
                raise NotImplementedError("Only disc visualisation is implemented.")

            if self._mode == "human":
                cv2.imshow(f"{self.name}", data)
            cv2.waitKey(1)

        # Prepare output
        output = Image(data=data)
        return output

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> Image:
        """Default output of the node."""
        return Image(data=jnp.zeros(self._shape, dtype=jnp.uint8))

    def step(self, step_state: StepState) -> Tuple[StepState, Image]:
        """Step the node."""
        # Unpack StepState
        inputs = step_state.inputs

        # Get image
        output = hcb.call(self._render, inputs["sensor"][-1].data, result_shape=self._dummy_image)
        return step_state, output


def disc_pendulum_render_fn(data, th, thdot):
    height, width, _ = data.shape
    side_length = min(width, height)
    state = jnp.array([th, thdot])

    data += 255
    length = 2 * side_length // 9
    sin_theta, cos_theta = jnp.sin(state[0]), jnp.cos(state[0])

    data = cv2.circle(data, (width // 2, height // 2), side_length // 3, (255, 0, 0), -1)
    data = cv2.circle(data, (width // 2, height // 2), side_length // 12, (192, 192, 192), -1)
    data = cv2.circle(
        data,
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        side_length // 9,
        (192, 192, 192),
        -1,
    )

    # Draw velocity vector
    data = cv2.arrowedLine(
        data,
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        (
            width // 2 + int(length * (sin_theta + state[1] * cos_theta / 5)),
            height // 2 + int(length * (-cos_theta + state[1] * sin_theta / 5)),
        ),
        (0, 0, 0),
        2,
    )
    return data
