from typing import Any, Dict, Tuple, Union, List
import jumpy
import jumpy.numpy as jp
import numpy as onp
from flax import struct
from rex.constants import WARN, LATEST
from rex.base import StepState, GraphState
from rex.node import Node
from rex.utils import log

try:
    import cv2
except ImportError as e:
    cv2 = None
    log("main", "red", WARN, "rendering", "Could not import cv2. Rendering will not work.")


@struct.dataclass
class Image:
    """Pendulum image definition"""
    data: jp.ndarray


class Render(Node):
    def __init__(self, *args, visual: str = "rod", mode: str = "human", encoding: str = "bgr", shape: List[int] = None, **kwargs):
        """Renders a double-pendulum environment.

        :param args: Args for Node base class.
        :param visual: Type of visualisation. Can only be "rod" at this moment.
        :param mode: Rendering mode. Options are 'human' or 'rgb_array'.
        :param encoding: Encoding of the image. Options are 'bgr' or 'rgb'.
        :param shape: Shape of the image to be rendered.
        :param kwargs: Kwargs for Node base class.
        """
        super().__init__(*args, stateful=False, **kwargs)
        self._visual = visual
        self._mode = mode
        self._encoding = encoding
        self._shape = shape or [400, 400, 3]

    def __getstate__(self):
        args, kwargs, inputs = super().__getstate__()
        kwargs.pop("stateful")  # Node is always stateless
        kwargs.update(dict(visual=self._visual, mode=self._mode, encoding=self._encoding, shape=self._shape))
        return args, kwargs, inputs

    def __setstate__(self, state):
        args, kwargs, inputs = state
        self.__init__(*args, **kwargs)
        # At this point, the inputs are not yet fully unpickled.
        self.inputs = inputs

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Image:
        """Default output of the node."""
        return Image(data=jp.zeros(self._shape, dtype=jp.uint8))

    def step(self, step_state: StepState) -> Tuple[StepState, Image]:
        """Step the node."""
        # Unpack StepState
        inputs = step_state.inputs

        # Grab current state
        last_obs = inputs["sensor"][-1].data
        cos_th, sin_th, cos_th2, sin_th2 = last_obs.cos_th, last_obs.sin_th, last_obs.cos_th2, last_obs.sin_th2
        th, th2 = jp.arctan2(sin_th, cos_th), jp.arctan2(sin_th2, cos_th2)
        thdot, thdot2 = inputs["sensor"][-1].data.thdot, inputs["sensor"][-1].data.thdot2

        # Prepare new image
        data = jp.zeros(self._shape, jp.uint8)
        if not jumpy.core.is_jitted() and cv2 is not None:
            if self._visual == "rod":
                data = double_pendulum_render_fn(data, th, th2, thdot, thdot2)
            else:
                raise NotImplementedError("Only rod visualisation is implemented.")

            if self._mode == "human":
                cv2.imshow(f"{self.name}", data)
            cv2.waitKey(1)

        # Prepare output
        output = Image(data=data)  # todo: generate image from state
        return step_state, output


def double_pendulum_render_fn(data: onp.ndarray, th: float, th2: float, thdot: float = 0., thdot2: float = 0., action=0.):
    """Render a double pendulum.

    :param data: Image data.
    :param th: Angle of first pendulum. Upright position is defined to as pi.
    :param th2: Angle of second pendulum relative to the first one.
    :param thdot: Angular velocity of first pendulum.
    :param thdot2: Angular velocity of second pendulum.
    :param action: Applied torque.
    """
    height, width, _ = data.shape
    side_length = min(width, height)

    data += 255
    length = side_length // 5
    sin_theta, cos_theta = jp.sin(th), jp.cos(th)

    # Draw title
    font = cv2.FONT_HERSHEY_SIMPLEX
    # text = "Applied Voltage " + str(action)  # todo: visualize action
    text = "Double pendulum"
    text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
    text_x = int((width - text_size[0]) / 2)
    text_y = int(text_size[1])
    data = cv2.putText(data, text, (text_x, text_y), font, 0.5, (0, 0, 0))

    # Draw pendulum
    data = cv2.line(
        data,
        (width // 2, height // 2),
        (width // 2 + int(length * sin_theta), height // 2 + int(length * cos_theta)),
        (0, 0, 255),
        max(side_length // 32, 1),
    )

    # Draw mass
    data = cv2.circle(
        data, (width // 2 + int(length * sin_theta), height // 2 + int(length * cos_theta)), side_length // 24, (0, 0, 0), -1
    )
    sin_theta2, cos_theta2 = jp.sin(th + th2), jp.cos(th + th2)
    data = cv2.line(
        data,
        (width // 2 + int(length * sin_theta), height // 2 + int(length * cos_theta)),
        (width // 2 + int(length * sin_theta) + int(length * sin_theta2),
         height // 2 + int(length * cos_theta) + int(length * cos_theta2)),
        (0, 255, 0),
        max(side_length // 64, 1),
    )
    data = cv2.circle(
        data, (width // 2 + int(length * sin_theta) + int(length * sin_theta2),
              height // 2 + int(length * cos_theta) + int(length * cos_theta2)), int(0.5 * side_length / 24),
        (0, 150, 0), -1
    )

    # Draw velocity vector  # todo: Visually looks wrong... Check.
    # data = cv2.arrowedLine(
    #     data,
    #     (width // 2 + int(length * sin_theta) + int(length * sin_theta2),
    #      height // 2 + int(length * cos_theta) + int(length * cos_theta2)),
    #     (
    #         width // 2 + int(length * (sin_theta + min(state[2], 10) * cos_theta / 10) + length * (
    #                     sin_theta2 + min(state[3], 10) * cos_theta2 / 10)),
    #         height // 2 + int(length * (cos_theta + min(state[2], 10) * sin_theta / 10) + length * (
    #                     cos_theta2 + min(state[3], 10) * sin_theta2 / 10)),
    #     ),
    #     (0, 0, 255),
    #     max(side_length // 240, 1),
    # )

    return data


if __name__ == '__main__':
    import rex.utils as utils
    import rex.multiprocessing as mp
    data = onp.zeros((1000, 500, 500, 3), onp.uint8)
    data.fill(255)
    th, th2, thdot, thdot2 = 0., 0, 0.5, 0.5
    double_pendulum_render_fn = mp.new_process(double_pendulum_render_fn, max_workers=2)

    timer = utils.timer("Render")
    with timer:
        images = []
        for i, d in enumerate(data):
            th += i/1000 * onp.pi
            # img = double_pendulum_render_fn(d, th, th2, thdot, thdot2)
            img = double_pendulum_render_fn.submit(d, th, th2, thdot, thdot2)
            images.append(img)
    print(f"[{timer.name}] Elapsed: {timer.duration}")

    timer = utils.timer("wait")
    with timer:
        images = [img.result() for img in images]
    print(f"[{timer.name}] Elapsed: {timer.duration}")

    for img in images:
        cv2.imshow(f"image", img)
        cv2.waitKey(10)

    # double_pendulum_render_fn(data, th, th2, thdot, thdot2)
    # cv2.imshow('image', data)
    # cv2.waitKey(0)