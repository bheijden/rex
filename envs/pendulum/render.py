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
	def __init__(self, *args, visual: str = "disc", mode: str = "human", encoding: str = "bgr", shape: List[int] = None, **kwargs):
		"""Renders a pendulum environment.

        :param args: Args for Node base class.
        :param visual: Type of visualisation. Either "disc" or "rod".
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
		th, thdot = inputs["sensor"][-1].data.th, inputs["sensor"][-1].data.thdot

		# Prepare new image
		data = jp.zeros(self._shape, jp.uint8)
		if not jumpy.core.is_jitted() and cv2 is not None:
			if self._visual == "disc":
				data = disc_pendulum_render_fn(data, th, thdot)
			else:
				raise NotImplementedError("Only disc visualisation is implemented.")

			if self._mode == "human":
				cv2.imshow(f"{self.name}", data)
			cv2.waitKey(1)

		# Prepare output
		output = Image(data=data)  # todo: generate image from state
		return step_state, output


def disc_pendulum_render_fn(data, th, thdot):
	height, width, _ = data.shape
	side_length = min(width, height)
	state = jp.array([th, thdot])

	data += 255
	length = 2 * side_length // 9
	sin_theta, cos_theta = jp.sin(state[0]), jp.cos(state[0])

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
