import os
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
import jax._src.compilation_cache as src_cc
cc.initialize_cache("./cache")
import logging
logging.getLogger("jax").setLevel(logging.DEBUG)

gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]

import mujoco
from mujoco import mjx

import envs.vx300s as vx300s
PATH_VX300S = os.path.dirname(vx300s.__file__)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(f"{PATH_VX300S}/assets/vx300s_mjx.xml")
    data = mujoco.MjData(model)
    mjx_model = mjx.device_put(model)
    mjx_data = mjx.device_put(data)

    def step(data):
        data = mjx.forward(mjx_model, data)
        # data = mjx.step(mjx_model, data)
        # data = mjx.step(mjx_model, data)
        # data = mjx.step(mjx_model, data)
        return data


    from jax._src.compilation_cache import get_executable

    jit_step = jax.jit(step, device=gpu_device)
    jit_step(mjx_data)
    print(data.qpos)