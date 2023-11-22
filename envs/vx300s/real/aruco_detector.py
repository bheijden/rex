import typing as t
import numpy as np
import cv2
import cv2.aruco as aruco

import queue, threading, time


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, cam_idx):
        self.cap = cv2.VideoCapture(cam_idx)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def set(self, prop, val):
        self.cap.set(prop, val)

    def release(self):
        self.cap.release()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            time.sleep(0.01)
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


class ArucoPoseDetector:
    ARUCO_DICT = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
    }

    def __init__(
            self, height: int, width: int, marker_size: float, camera_matrix: t.List, dist_coeffs: t.List, aruco_type: str
    ):
        self.marker_size = marker_size
        self.camera_matrix = np.array(camera_matrix, dtype="float32")
        self.dist_coeffs = np.array(dist_coeffs, dtype="float32")
        self.h = height
        self.w = width
        self.aruco_dict = aruco.Dictionary_get(self.ARUCO_DICT[aruco_type])
        self.parameters = aruco.DetectorParameters_create()

        # Correct distortion
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (self.h, self.w), 0.0, (self.h, self.w)
        )

    def undistort(self, image_raw: np.ndarray):
        dst1 = cv2.undistort(image_raw, self.camera_matrix, self.dist_coeffs, None, self.newcameramtx)
        x, y, w1, h1 = self.roi
        image = dst1[y: y + h1, x: x + w1]
        return image

    def estimate_pose(self, image: np.ndarray, draw: bool = True):
        # Convert to gray scale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # The detectmarkers() function can detect the marker and return the ID
        # and the coordinates of the four corners of the sign board
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        # If markers detected
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            if draw:
                image = self.draw_pose(image, corners, rvec, tvec, axis_length=self.marker_size / 2)
        else:
            rvec, tvec = None, None
        return image, corners, ids, rvec, tvec

    def draw_pose(self, image: np.ndarray, corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, axis_length: float = 0.03):
        for i in range(rvec.shape[0]):
            aruco.drawDetectedMarkers(image, corners)
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[i, :, :], tvec[i, :, :], axis_length)
        return image
