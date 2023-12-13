import jax
import jumpy

import numpy as onp
import jax.numpy as jnp
import jumpy.numpy as jp
import rex.jumpy as rjp

from brax import math
from flax import struct

from envs.vx300s.env import ArmOutput


@struct.dataclass
class CostParams:
    orn: jp.float32     # Gripper bar parallel to box
    down: jp.float32    # Downward orientation --> w_ee_r (w_"ee_rotation")
    height: jp.float32  # Height of ee --> w_ee_h (w_"ee_height")
    force: jp.float32   # Force penalty --> w_c (w_"contact")
    near: jp.float32    # Ee near box --> w_t  (w_"touch")
    dist: jp.float32    # Box near goal --> w_o_p (w_"object_to_goal")
    align: jp.float32   # Align such that box is in-between goal and ee --> w_a (w_"align")
    ctrl: jp.float32    # Control penalty
    bias_height: jp.float32  # Bias [m] for ee height (accounts for ee size (from ee_link to ground ~35mm)
    bias_near: jp.float32  # Bias [m] for near penalty in xy plane (accounts for box size from center ~50mm + ~10mm for ee thickness)
    alpha: jp.float32   # Larger alpha increases penalty on near and dist once (orn, down, height, align) are satisfied
    discount: jp.float32   # Discount factor

    @classmethod
    def default(cls):
        return cls(orn=3.0,
                   down=3.0,
                   height=10.0,
                   force=1.0,  # 1.0 works
                   near=5.0,
                   dist=50.0,
                   align=2.0,
                   ctrl=0.1,
                   bias_height=0.045,
                   bias_near=0.07,
                   alpha=0.0,
                   discount=1.0)


def box_pushing_cost(cp: CostParams, boxpos, eepos, goalpos, eeorn, force=None, action=None, time_step=None):
    rot_mat = ArmOutput(jpos=None, eeorn=eeorn, eepos=None).orn_to_3x3
    ee_to_goal = goalpos - eepos[:2]
    box_to_ee = (eepos - boxpos)[:2]
    box_to_goal = (goalpos - boxpos[:2])

    # if dot(ee_yaxis (in global), ee_to_goal (in global))==0 --> ee_yaxis = perpendicular to box
    # ee_yaxis is parallel to gripper_bar axis
    norm_ee_to_goal = ee_to_goal / math.safe_norm(ee_to_goal)
    cost_orn = jnp.abs(jnp.dot(rot_mat[:2, 1], norm_ee_to_goal))

    # ee_xaxis points in -z if ee is oriented downward
    # norm_box_to_goal = box_to_goal / math.safe_norm(box_to_goal)
    # target_ee_xaxis = jnp.concatenate([norm_box_to_goal, jnp.array([-5.0])])  # making this more negative forces the ee to be pointing downward
    # norm_target_ee_xaxis = target_ee_xaxis / math.safe_norm(target_ee_xaxis)
    # cost_down = (1-jnp.dot(rot_mat[:3, 0], norm_target_ee_xaxis))  # Here, the dot is 1 if ee_xaxis == target_ee_axis
    cost_down = jnp.abs(rot_mat[:2, 0]).sum()

    # Distances in xy-plane
    dist_box_to_ee = math.safe_norm(box_to_ee)
    dist_box_to_goal = math.safe_norm(box_to_goal)
    cost_align = (jnp.sum(box_to_ee * box_to_goal) / (dist_box_to_ee*dist_box_to_goal) + 1)

    # Force cost
    # cost_force = (pipeline_state.qf_constraint[2]-0.46) ** 2
    cost_force = jnp.abs(force).max() ** 2 if force is not None else 0.0
    cost_height = jnp.abs(eepos[2] - cp.bias_height)
    cost_near = jp.abs(math.safe_norm((boxpos - eepos)[:2]) - cp.bias_near)
    cost_dist = math.safe_norm(boxpos[:2] - goalpos)
    cost_ctrl = math.safe_norm(action) if action is not None else 0.0

    # Store centimeter distance
    cm = cost_dist*100

    # Weight all costs
    cost_orn = cp.orn * cost_orn
    cost_down = cp.down * cost_down
    cost_align = cp.align * cost_align
    cost_height = cp.height * cost_height
    alpha = 1 / (1 + cp.alpha * jnp.abs(cost_orn + cost_down + cost_align + cost_height))
    cost_force = cp.force * cost_force
    cost_near = cp.near * cost_near * alpha
    cost_dist = cp.dist * cost_dist * alpha
    cost_ctrl = cp.ctrl * cost_ctrl

    total_cost = cost_ctrl + cost_height + cost_near + cost_dist + cost_orn + cost_down + cost_align + cost_force
    discounted_cost = total_cost * cp.discount ** time_step if time_step is not None else total_cost
    info = {"cm": cm, "cost": discounted_cost, "cost_orn": cost_orn, "cost_force": cost_force, "cost_down": cost_down, "cost_align": cost_align, "cost_height": cost_height, "cost_near": cost_near, "cost_dist": cost_dist, "cost_ctrl": cost_ctrl, "alpha": alpha}
    return discounted_cost, info