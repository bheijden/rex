import queue, threading, time
import cv2
import atexit
from typing import Any, Dict, Tuple, Union, List

import jax.experimental.host_callback as hcb
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

from rex.multiprocessing import new_process
from rex.base import StepState, GraphState, Empty
from rex.node import Node


@struct.dataclass
class Image:
    """Pendulum image definition"""
    image: jax.Array


class Camera(Node):
    def __init__(self, cam_idx: int, *args, width: int = 640, height: int = 480, **kwargs):
        super().__init__(*args, **kwargs)
        # Cam
        self._cam_idx = cam_idx
        self._width = width
        self._height = height
        self._cam = None
        self._dummy_output = self._get_output()
        self._mp_get_output = new_process(self._get_output)

    def __getstate__(self):
        args, kwargs, inputs = super().__getstate__()
        kwargs.pop("stateful")  # Node is always stateless
        kwargs.update(dict(cam_idx=self._cam_idx, width=self._width, height=self._height))
        return args, kwargs, inputs

    def __setstate__(self, state):
        args, kwargs, inputs = state
        self.__init__(*args, **kwargs)
        # At this point, the inputs are not yet fully unpickled.
        self.inputs = inputs

    def _init_cam(self):
        if self._cam is None:
            from envs.vx300s.real.aruco_detector import VideoCapture

            self._cam = VideoCapture(self._cam_idx)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            # Register shutdown
            atexit.register(self._shutdown)

            # Wait until
            iter = 0
            while True:
                time.sleep(1.0)
                ret, image_raw = self._cam.read()
                iter += 1
                if ret:
                    break
                elif iter > 100:
                    raise ValueError(f"Failed to initialize camera with index {self._cam_idx}.")

    def _shutdown(self):
        # Release camera resources
        if self._cam is not None:
            self._cam.release()
            self._cam = None

    def _get_output(self):
        self._init_cam()  # Make sure camera is initialized

        # Get image
        i = 0
        while True:
            ret, image_raw = self._cam.read()  # cv_img is in BGR format
            i += 1
            if i % 100 == 0:
                print("Waiting for image...")
            if ret:
                break
            time.sleep(0.01)

        # Display result frame (Must always be called from the same thread that created the window)
        self._cam.view(image_raw)
        return Image(image=image_raw)

    def default_output(self, rng: jax.random.KeyArray, graph_state: GraphState = None) -> Image:
        """Default output of the node."""
        output = hcb.call(self._get_output, graph_state.nodes[self.name].state.last_detection,
                          result_shape=self._dummy_output)
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, Image]:
        """Step the node."""
        # output = hcb.call(self._get_output, (), result_shape=self._dummy_output)
        output = self._get_output()
        # output = self._mp_get_output()
        # Prepare output
        return step_state, output


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, cam_idx):
        self.cap = cv2.VideoCapture(cam_idx)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        self.q_img = queue.Queue()
        t = threading.Thread(target=self._viewer)
        t.daemon = True
        t.start()

    def set(self, prop, val):
        self.cap.set(prop, val)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            time.sleep(0.03)
            ret, frame = self.cap.read()
            if not ret:
                ...
                # print("Error reading frame")
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def _viewer(self):
        while True:
            # print("get image")
            image = self.q_img.get(block=True)
            # print("view image")
            cv2.imshow("image", image)
            key = cv2.waitKey(1)

    def view(self, image):
        self.q_img.put(image)