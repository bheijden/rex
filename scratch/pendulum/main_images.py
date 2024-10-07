import dill as pickle
import tqdm
from typing import Dict, Union, Callable, Any
import os
import multiprocessing
import itertools
JAX_USE_CACHE = False
# if JAX_USE_CACHE:
    # os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count() - 4
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import distrax
import equinox as eqx

# Cache settings
if JAX_USE_CACHE:
    # More info: https://github.com/google/jax/pull/22271/files?short_path=71526fb#diff-71526fb9807ead876cbde1c3c88a868e56d49888023dd561e6705d403ab026c0
    jax.config.update("jax_compilation_cache_dir", "./cache-rl")
    # -1: disable the size restriction and prevent overrides.
    # 0: Leave at default (0) to allow for overrides.
    #    The override will typically ensure that the minimum size is optimal for the file system being used for the cache.
    # > 0: the actual minimum size desired; no overrides.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # A computation will only be written to the persistent cache if the compilation time is longer than the specified value.
    # It is defaulted to 1.0 second.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_explain_cache_misses", False)  # True --> results in error

import supergraph
import rex
from rex import base, jax_utils as jutils, constants
from rex.constants import Clock, RealTimeFactor, Scheduling, LogLevel, Supergraph, Jitter
from rex.utils import timer
import rex.utils as rutils
from rex.jax_utils import same_structure
import envs.pendulum.systems as psys

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")


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
    # EXP_DIR = f"{LOG_DIR}/20240710_141737_brax_norandomization_longerstack_v4_dark"
    EXP_DIR = f"{LOG_DIR}/20240716_brax_pose2_sysid"
    RECORD_FILE = f"{EXP_DIR}/sysid_data.pkl"
    PARAMS_FILE = f"{LOG_DIR}/sysid_params.pkl"  # todo: change to brax
    FRAMES_DIR = f"{EXP_DIR}/frames_real"
    GIF_FILE = f"{FRAMES_DIR}/render.gif"
    # ORDER = ["camera", "sensor", "actuator", "controller", "estimator", "supervisor"]
    # CSCHEME = {"world": "gray", "sensor": "grape", "camera": "orange", "estimator": "violet", "controller": "lime",
    #            "actuator": "green", "supervisor": "indigo"}

    # Load record
    with open(RECORD_FILE, "rb") as f:
        record: base.EpisodeRecord = pickle.load(f)

    # Gather outputs
    outputs_sysid = {name: n.steps.output for name, n in record.nodes.items()}
    cam = jax.tree_util.tree_map(lambda x: x[EPS_IDX], outputs_sysid["camera"])

    # Save render
    import cv2

    bgr = cam.bgr
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in bgr]

    # Save every 2nd frame to
    os.makedirs(FRAMES_DIR, exist_ok=True)
    # for i, frame in enumerate(bgr):
    #     if i % 2 == 0:
    #         cv2.imwrite(f"{FRAMES_DIR}/frame_{i}.png", frame)
    # exit()

    import imageio
    imageio.mimsave(GIF_FILE, rgb, fps=60)
    exit()

    # Visualize detection
    if True:
        detector = jax.tree_util.tree_map(lambda x: x[0], record.nodes["camera"].params.detector)
        bgr_ellipse = detector.draw_ellipse(bgr, color=(255, 0, 0))
        bgr_centroids = detector.draw_centroids(bgr_ellipse, cam.median)
        detector.play_video(bgr_centroids, fps=60)
