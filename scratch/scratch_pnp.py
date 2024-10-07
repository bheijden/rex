import dill as pickle
import tqdm
import functools
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
import datetime


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import distrax
import equinox as eqx
import numpy as onp
from scipy.spatial.transform import Rotation as R

from brax.generalized import pipeline as gen_pipeline
from brax.spring import pipeline as spring_pipeline
from brax.positional import pipeline as pos_pipeline
from brax.base import System, State
from brax.io import mjcf

import supergraph
import rex
from rex import base, jax_utils as jutils, constants
from rex.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rex.utils import timer
import rex.utils as rutils
from rex.jax_utils import same_structure
from rex import artificial
import envs.pendulum.systems as psys
import envs.pendulum.ppo as pend_ppo
import envs.pendulum.brax as pend_brax
import rex.rl as rl

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


def find_nearest_indices(th, th_interp):
    # Ensure th is within the range [-pi, pi]
    th_norm = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
    # th_normalized = jnp.mod(th + onp.pi, 2 * onp.pi) - onp.pi

    # Compute the absolute difference between each point in th_normalized and th_interp
    diff = jnp.abs(th_norm[:, jnp.newaxis] - th_interp)

    # Find the index of the minimum difference for each point in th_interp
    indices = jnp.argmin(diff, axis=0)

    return indices


if __name__ == "__main__":
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=3, suppress=True)

    if False:
        import pyrealsense2 as rs

        # Initialize RealSense pipeline and get RGB camera intrinsics
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)
        pipeline.start(config)
        profile = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        pipeline.stop()

        # Extract intrinsics
        focal_length_x = color_intrinsics.fx
        focal_length_y = color_intrinsics.fy
        optical_center_x = color_intrinsics.ppx
        optical_center_y = color_intrinsics.ppy

        print("Focal Length (x):", focal_length_x)
        print("Focal Length (y):", focal_length_y)
        print("Optical Center (x):", optical_center_x)
        print("Optical Center (y):", optical_center_y)
        intrinsic_matrix = onp.array([
            [focal_length_x, 0, optical_center_x],
            [0, focal_length_y, optical_center_y],
            [0, 0, 1]
        ])
    else:
        # Camera intrinsic matrix.
        intrinsic_matrix = onp.array([[308.773, 0., 214.688],
                                      [0., 308.724, 118.102],
                                      [0., 0., 1.]])

    print("Intrinsic matrix:\n", intrinsic_matrix)

    # Assuming no lens distortion.
    dist_coeffs = onp.zeros(5)  # Use a length of 5 for distortion coefficients

    EPS_IDX = -1
    EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240710_141737_brax_norandomization_longerstack_v4_dark"
    # EXP_DIR = "/home/r2ci/rex/scratch/pendulum/logs/20240716_brax_pose2_sysid"
    RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"
    PARAMS_FILE = f"{EXP_DIR}/sysid_params.pkl"
    RENDER_FILE = f"{EXP_DIR}/sysid_render.html"

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Load params
    with open(PARAMS_FILE, "rb") as f:
        params = pickle.load(f)

    # World
    world = params["world"]
    length = world.offset

    # Recalculate detections
    detector = params["camera"].detector
    cam = jax.tree_util.tree_map(lambda x: x[EPS_IDX], record.nodes["camera"].steps)
    ts, bgr = cam.ts_start, cam.output.bgr
    _, detections = detector.noncausal_step(ts, bgr)
    dt = ts[-1] / len(ts)
    th = detections.th
    centroid = onp.array(detections.centroid, dtype="float32")

    # Resample
    if False:
        th_interp = jnp.linspace(-onp.pi, onp.pi, 20)
        # th_interp = jnp.sign(th_interp)*onp.pi/2 + th_interp
        th_wrapped = th - 2 * jnp.pi * jnp.floor((th + jnp.pi) / (2 * jnp.pi))
        indices = find_nearest_indices(th_wrapped, th_interp)
        th = th[indices]
        centroid = centroid[indices]

    if True:
        wxyz, camera_position = detector.estimate_camera_pose(th, centroid, world.offset)
        max_idx = -1
    else:
        # th = th
        # sys = mjcf.loads(pend_brax.CAM_DISK_PENDULUM_XML)
        # cam_pos = jnp.array([0.0, 0.0, 0.0])
        # cam_orn = jnp.array([1.0, 0.0, 0.0, 0.0])
        # sys, pipeline_state_lst = pend_brax.get_pipeline_state(th, dt=dt, cam_pos=cam_pos, cam_orn=cam_orn, sys=sys)
        # pos = jnp.stack([ps.x.pos for ps in pipeline_state_lst])
        # x = pos[:, 1, 0]
        # y = pos[:, 1, 1]
        # z = pos[:, 1, 2]

        # calculate x, y coordinate from th (pendulum upward is 0 rad, and center is joint)
        x = length * jnp.sin(-th)
        y = onp.zeros_like(-th)
        z = length * jnp.cos(-th)

        # Define 3D world points. # y, -x, z works for pose_2 with jnp.sin(th), jnp.cos(th), jnp.zeros_like(th)
        # object_points = onp.column_stack([y, x, z]).astype("float32") works with .x.pos
        object_points = onp.column_stack([x, y, z]).astype("float32")
        print("Object points:\n", object_points.shape)

        # Define 2D image points.
        image_points = centroid
        print("Image points:\n", image_points.shape)

        # Redefine image points max_idx - image_points
        u, v = image_points[:, 0], image_points[:, 1]
        image_points = onp.column_stack([v, u])

        # Solve PnP.
        import cv2
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, intrinsic_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
        )

        # Convert rotation vector to a rotation matrix.
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Convert rotation matrix to quaternion (w, x, y, z) using scipy
        rotation = R.from_matrix(rotation_matrix.T)
        quaternion = rotation.as_quat()
        wxyz = onp.roll(quaternion, shift=1)

        # Camera position in world coordinates.
        camera_position = -onp.matrix(rotation_matrix).T * onp.matrix(tvec)
        camera_position = onp.array(camera_position)[:, 0]

        print("Camera Position:\n", camera_position)
        print("tvec:\n", onp.array(tvec).squeeze())
        print("Rotation matrix:\n", rotation.as_matrix())

        max_idx = 30
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True)
        axes[0].plot(th[:max_idx], label="angle")
        axes[0].set_title("Angle")
        axes[1].plot(centroid[:max_idx], label="centroid")
        axes[1].set_title("Centroid")
        axes[2].plot(x[:max_idx], label="x")
        axes[2].plot(z[:max_idx], label="z")
        axes[2].set_title("X, Z")
        axes[2].legend()
        plt.show()

    # Calculate 3D positions
    from brax.io import image, html
    sys = mjcf.loads(pend_brax.CAM_DISK_PENDULUM_XML)
    sys, pipeline_state_lst = pend_brax.get_pipeline_state(th, dt=dt, cam_pos=camera_position, cam_orn=wxyz, sys=sys)
    data = image.render(sys, pipeline_state_lst, width=424, height=240, fmt="gif", camera="d435i_render")
    GIF_FILE = RENDER_FILE.replace(".html", ".gif")
    with open(GIF_FILE, "wb") as f:
        f.write(data)
    print("Gif to", GIF_FILE)

    # Render visual
    rollout_json = html.render(sys, pipeline_state_lst)
    pend_brax.save(RENDER_FILE, rollout_json)
    print("Rendered to", RENDER_FILE)

    # Plot
    bgr_ellipse = detector.draw_ellipse(cam.output.bgr, color=(255, 0, 0))
    bgr_centroids = detector.draw_centroids(bgr_ellipse, detections.median)
    detector.play_video(bgr_centroids[:max_idx], fps=54)


