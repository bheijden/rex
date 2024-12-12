import dill as pickle
import tqdm
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx


import supergraph
import rexv2
from rexv2 import base, jax_utils as jutils, constants
from rexv2.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rexv2.utils import timer
import rexv2.utils as rutils
from rexv2.jax_utils import same_structure
import envs.pendulum.systems as psys

# plotting
import cv2


if __name__ == "__main__":
    onp.set_printoptions(precision=3, suppress=True)
    jnp.set_printoptions(precision=5, suppress=True)
    RNG = jax.random.PRNGKey(6)
    GPU_DEVICE = jax.devices('gpu')[0]
    CPU_DEVICE = jax.devices('cpu')[0]
    CPU_DEVICES = itertools.cycle(jax.devices('cpu')[1:])
    EPS_IDX = -1
    WORLD_RATE = 100.
    ID_CAM = False  # Use images to identify camera parameters
    USE_CAM = True  # Use camera instead of sensor in estimator
    USE_BRAX = True  # Use brax for simulation  # todo: change to brax
    SUPERVISOR = "actuator"
    SUPERGRAPH = Supergraph.MCS
    LOG_DIR = "/home/r2ci/rex/scratch/pendulum/logs"
    EXP_DIR = f"{LOG_DIR}/20240710_141737_brax_norandomization_longerstack_v4_dark/20240710_183513_brax_images"
    RECORD_FILES = [
        f"{EXP_DIR}/nodelay_cam_nopred_real_data.pkl",
        f"{EXP_DIR}/nodelay_cam_real_data.pkl",
        f"{EXP_DIR}/nodelay_nocam_real_data.pkl",
        f"{EXP_DIR}/real_data.pkl",
        f"{EXP_DIR}/stacked_nodelay_nocam_real_data.pkl",
        f"{EXP_DIR}/stacked_nodelay_real_data.pkl",
        f"{EXP_DIR}/stacked_real_data.pkl",
    ]

    for RECORD_FILE in RECORD_FILES:
        NAME_FILE = RECORD_FILE.split("/")[-1].split(".")[0]
        VIDEO_FILE = f"{EXP_DIR}/video_random/{NAME_FILE}"
        print(f"Processing {RECORD_FILE}")

        # Load record
        with open(RECORD_FILE, "rb") as f:
            record: base.ExperimentRecord = pickle.load(f)

        # Get cam rate
        rate = 50  # record.episodes[0].nodes["camera"].info.rate

        # Get camera output
        bgr = []
        min_frames = 1e9
        for eps in record.episodes:
            bgr.append(eps.nodes["camera"].steps.output.bgr)
            min_frames = min(min_frames, len(bgr[-1]))
        bgr = [frames[:min_frames] for frames in bgr]  # Make sure all episodes have the same number of frames
        bgr = onp.array(bgr)  # Convert to numpy (episodes, frames, h, w, c)

        # Roll order of episodes by 5
        bgr = onp.roll(bgr, 5, axis=0)

        # Concatenate episodes
        frame_height, frame_width = bgr.shape[2], bgr.shape[3]
        bgr = bgr.reshape(-1, frame_height, frame_width, 3)  # (episodes * frames, h, w, c)

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(f"{VIDEO_FILE}.avi", fourcc, rate, (frame_width, frame_height))

        # Save video with cv2 (compressed)
        for frame in tqdm.tqdm(bgr, desc="Saving video"):
            out.write(frame)
        out.release()
        print(f"Saved video to {VIDEO_FILE}")

    # import imageio
    # imageio.mimsave(GIF_FILE, rgb, fps=60)
    # Visualize detection
    if False:
        detector = jax.tree_util.tree_map(lambda x: x[0], record.nodes["camera"].params.detector)
        bgr_ellipse = detector.draw_ellipse(bgr, color=(255, 0, 0))
        bgr_centroids = detector.draw_centroids(bgr_ellipse, cam.median)
        detector.play_video(bgr_centroids, fps=60)
