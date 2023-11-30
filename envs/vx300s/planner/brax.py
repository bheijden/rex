from functools import partial
import jax
import jax.numpy as jnp
from jax.debug import print as jprint
import jax.experimental.host_callback as hcb

import jumpy.random
import rex.jumpy as rjp
import jumpy.numpy as jp
import jax.numpy as np
from math import ceil

import brax
from brax.io import mjcf
from brax.generalized import pipeline as gen_pipeline
from brax.positional import pipeline as pos_pipeline
from brax.spring import pipeline as spr_pipeline
from brax import base, math
from flax import struct
from flax.core import FrozenDict

from rex.base import StepState, GraphState
from rex.node import Node
from rex.utils import deprecation_warning

from envs.vx300s.planner.cost import CostParams, box_pushing_cost
from envs.vx300s.planner.cem import CEMParams, cem_planner
from envs.vx300s.env import PlannerOutput, get_next_jpos

PIPELINES = {"generalized": gen_pipeline,
             "positional": pos_pipeline,
             "spring": spr_pipeline}


@struct.dataclass
class State:
    pipeline_state: base.State
    q_des: jnp.ndarray


@struct.dataclass
class BraxCEMParams:
    """See https://arxiv.org/pdf/1907.03613.pdf for details on CEM"""
    dt_substeps: jp.float32
    substeps: jp.int32
    sys: base.System
    cem_params: CEMParams
    cost_params: CostParams


@struct.dataclass
class BraxCEMState:
    goalpos: jp.ndarray
    goalyaw: jp.ndarray
    last_plan: PlannerOutput


class BraxCEMPlanner(Node):
    def __init__(self, *args,
                 mj_path: str,
                 pipeline: str = "generalized",
                 u_max: float = 0.35 * 3.14,
                 dt: float = 0.15,
                 dt_substeps: float = 0.015,
                 horizon: int = 4,
                 num_samples: int = 300,
                 max_iter: int = 1,
                 sampling_smoothing: float = 0.,
                 evolution_smoothing: float = 0.1,
                 elite_portion: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._pipeline = pipeline
        self._horizon = horizon
        self._u_max = u_max
        self._dt = dt
        self._dt_substeps = dt_substeps
        self._num_samples = num_samples
        self._max_iter = max_iter
        self._sampling_smoothing = sampling_smoothing
        self._evolution_smoothing = evolution_smoothing
        self._elite_portion = elite_portion
        self._mj_path = mj_path
        self._sys = mjcf.load(mj_path)
        self._gpu_device = jax.devices('gpu')[0] if len(jax.devices('gpu')) > 0 else None
        self._cpu_device = jax.devices('cpu')[0]
        self._jit_run_cem = jax.jit(self.run_cem, device=self._gpu_device if self._gpu_device else self._cpu_device)
        self._jit_get_init_jpos = jax.jit(self.get_init_plan)

    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> BraxCEMParams:
        if "planner" in graph_state.nodes:
            return graph_state.nodes["planner"].params
        else:
            dt, dt_substeps, substeps = get_stepsizes(dt=self._dt, dt_substeps=self._dt_substeps)
            # params_sup = graph_state.nodes["supervisor"].params
            cost_params = CostParams(orn=3.0,
                                     down=3.0,
                                     height=1.0,
                                     force=1.0,  # 1.0 works
                                     near=5.0,
                                     dist=50.0,
                                     align=2.0,
                                     ctrl=0.1,
                                     bias_height=0.035,
                                     bias_near=0.07,
                                     alpha=0.0,
                                     discount=0.98)
            cem_params = CEMParams(u_min=-self._u_max * jp.ones((6,), dtype=jp.float32),
                                   u_max=self._u_max * jp.ones((6,), dtype=jp.float32),
                                   sampling_smoothing=self._sampling_smoothing,
                                   evolution_smoothing=self._evolution_smoothing)
            params = BraxCEMParams(dt_substeps=dt_substeps,
                                   substeps=substeps,
                                   sys=self._sys,
                                   cem_params=cem_params,
                                   cost_params=cost_params)
        return params

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> BraxCEMState:
        goalpos = graph_state.nodes["supervisor"].state.goalpos
        goalyaw = graph_state.nodes["supervisor"].state.goalyaw
        return BraxCEMState(last_plan=self.default_output(rng, graph_state), goalpos=goalpos, goalyaw=goalyaw)

    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> PlannerOutput:
        p_sup = graph_state.nodes["supervisor"].params
        p_plan = graph_state.nodes["planner"].params
        jpos = p_sup.home_jpos
        jvel = jp.zeros((self._horizon, jpos.shape[0]), dtype=jp.float32)
        dt = p_plan.dt_substeps * p_plan.substeps
        timestamps = dt * jp.arange(0, self._horizon + 1, dtype=jp.float32)
        return PlannerOutput(jpos=jpos, jvel=jvel, timestamps=timestamps)

    def step(self, step_state: StepState):
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Determine timestamps at t+t_offset --> time of applying the action.
        ts_now = step_state.ts
        ts_offset = self.output.phase  # todo: Add communication delay?
        ts_future = ts_now + ts_offset
        dt = params.dt_substeps * params.substeps
        timestamps = dt * jp.arange(0, self._horizon + 1, dtype=jp.float32) + ts_future

        # Determine where arm.jpos will be at t+t_offset
        rng, rng_cem = jumpy.random.split(rng, num=2)
        goalpos = state.goalpos
        eepos_now = step_state.inputs["armsensor"][-1].data.eepos
        eeorn_now = step_state.inputs["armsensor"][-1].data.eeorn
        jpos_now = step_state.inputs["armsensor"][-1].data.jpos
        # jpos_future = get_next_jpos(state.last_plan, timestamps[0])
        boxpos_now = step_state.inputs["boxsensor"][-1].data.boxpos
        boxyaw_now = jp.array([step_state.inputs["boxsensor"][-1].data.wrapped_yaw])

        # Run CEM todo: make sure box and ee are not in collision.
        new_plan = self.run_cem(rng_cem, params, state.last_plan, timestamps, jpos_now, boxpos_now, boxyaw_now, goalpos)
        # new_plan = self._jit_run_cem(rng_cem, params, state.last_plan, timestamps, jpos_now, boxpos_now, goalpos)

        # print(f"new_plan.jpos: {new_plan.jpos} | jpos_now: {jpos_now} | diff: {new_plan.jpos - jpos_now}")
        # jpos_diff = new_plan.jpos - jpos_now
        # print(f"jpos diff | max={jpos_diff.max()} | min={jpos_diff.min()} | mean={jpos_diff.mean()}")

        # new_plan = PlannerOutput(jpos=jpos_now, jvel=mean, timestamps=timestamps)
        new_state = state.replace(last_plan=new_plan)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        return new_step_state, new_plan

    def get_init_plan(self, last_plan: PlannerOutput, timestamps: jp.ndarray) -> PlannerOutput:
        get_next_jpos_vmap = jax.vmap(get_next_jpos, in_axes=(None, 0))
        jpos_timestamps = get_next_jpos_vmap(last_plan, timestamps)
        jvel_timestamps = jpos_timestamps[1:] - jpos_timestamps[:-1]
        return PlannerOutput(jpos=jpos_timestamps[0], jvel=jvel_timestamps, timestamps=timestamps)

    def run_cem(self, rng, params: BraxCEMParams, last_plan: PlannerOutput, timestamps: jnp.ndarray, jpos: jnp.ndarray, boxpos: jnp.ndarray, boxyaw: jnp.ndarray, goalpos: jnp.ndarray) -> PlannerOutput:
        # Update system with correct dt from parameters
        params = params.replace(sys=params.sys.replace(dt=params.dt_substeps))
        cem_params = params.cem_params

        # Determine initial plan from last_plan
        init_plan = self.get_init_plan(last_plan, timestamps)

        # Define cost function
        def cost(_params: BraxCEMParams, state: State, action, time_step: int):
            pipeline_state = state.pipeline_state
            sys = params.sys

            # Get indices
            ee_arm_idx = sys.link_names.index("ee_link")
            box_idx = sys.link_names.index("box")
            goal_idx = sys.link_names.index("goal")

            x_i = pipeline_state.x.vmap().do(
                base.Transform.create(pos=sys.link.inertia.transform.pos)
            )

            boxpos = x_i.pos[box_idx]
            eepos = x_i.pos[ee_arm_idx]
            eeorn = self._convert_wxyz_to_xyzw(x_i.rot[ee_arm_idx])  # require xyzw convention
            goalpos = x_i.pos[goal_idx][:2]
            force = pipeline_state.qf_constraint

            total_cost, info = box_pushing_cost(_params.cost_params, boxpos, eepos, goalpos, eeorn, force, action, time_step)
            return total_cost

        # Define dynamics function
        def dynamics(_params: BraxCEMParams, state: State, action, time_step):
            def loop_cond(args):
                i, _ = args
                return i < _params.substeps

            def loop_body(args):
                i, state = args
                q_des = state.q_des + action * params.sys.dt  # todo: clip to max angles?
                pipeline_state = PIPELINES[self._pipeline].step(params.sys, state.pipeline_state, q_des)
                return i + 1, State(pipeline_state=pipeline_state, q_des=q_des)

            i, state = jax.lax.while_loop(loop_cond, loop_body, (0, state))

            return state

        # Get initial pose
        qpos = jnp.concatenate([boxpos, boxyaw, goalpos, jpos, jnp.array([0])])  # added [0] for additional joint.

        # Initialize pipeline state
        pipeline_state = PIPELINES[self._pipeline].init(params.sys, qpos, jnp.zeros(params.sys.qd_size()))
        # q_des = pipeline_state.q[params.sys.actuator.q_id]
        q_des = init_plan.jpos
        init_state = State(pipeline_state=pipeline_state, q_des=q_des)

        # Run CEM
        cem_hyperparams = dict(sampling_smoothing=cem_params.sampling_smoothing, evolution_smoothing=cem_params.evolution_smoothing,
                               elite_portion=self._elite_portion, num_samples=self._num_samples, max_iter=self._max_iter)
        _, mean, obj = cem_planner(cost, dynamics, params, init_state, init_plan.jvel, cem_params.u_min, cem_params.u_max, rng,
                                   hyperparams=cem_hyperparams)

        # Return new plan
        new_plan = init_plan.replace(jvel=mean)

        return new_plan

    def print_sys_info(self):
        print_sys_info(self._sys, self._pipeline)

    def _convert_wxyz_to_xyzw(self, quat: jp.ndarray):
        """Convert quaternion (w,x,y,z) -> (x,y,z,w)"""
        return jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype="float32")


def print_sys_info(sys: base.System = None, pipeline: str = "generalized"):

    print("\nCOLLISIONS")
    from brax.geometry.contact import _geom_pairs
    for (geom_i, geom_j) in _geom_pairs(sys):
        # print(geom_i.link_idx, geom_j.link_idx)
        name_i = sys.link_names[geom_i.link_idx[0]] if geom_i.link_idx is not None else "world"
        name_j = sys.link_names[geom_j.link_idx[0]] if geom_j.link_idx is not None else "world"
        print(f"collision pair: {name_i} --> {name_j}")

    # Actuators
    print("\nACTUATOR SIZE")
    print(f"actuator size: {sys.act_size()}")

    # Print relevant parameters
    parameters_dict = {
        pos_pipeline: [
            'dt',
            'joint_scale_pos',
            'joint_scale_ang',
            'collide_scale',
            'ang_damping',  # shared with `brax.physics.spring`
            'vel_damping',  # shared with `brax.physics.spring`
            'baumgarte_erp',  # shared with `brax.physics.spring`
            'spring_mass_scale',  # shared with `brax.physics.spring`
            'spring_inertia_scale',  # shared with `brax.physics.spring`
            'constraint_ang_damping',  # shared with `brax.physics.spring`
            'elasticity',  # shared with `brax.physics.spring`
        ],
        spr_pipeline: [
            'dt',
            'constraint_stiffness',
            'constraint_limit_stiffness',
            'constraint_vel_damping',
            'ang_damping',  # shared with `brax.physics.positional`
            'vel_damping',  # shared with `brax.physics.positional`
            'baumgarte_erp',  # shared with `brax.physics.positional`
            'spring_mass_scale',  # shared with `brax.physics.positional`
            'spring_inertia_scale',  # shared with `brax.physics.positional`
            'constraint_ang_damping',  # shared with `brax.physics.positional`
            'elasticity',  # shared with `brax.physics.positional`
        ],
        gen_pipeline: [
            'dt',
            'matrix_inv_iterations',
            'solver_iterations',
            'solver_maxls',
        ]
        # The 'convex' parameter is not included due to its unknown usage.
    }

    print(f"\nPARAMETERS: {pipeline}")
    for p in parameters_dict[PIPELINES[pipeline]]:
        try:
            print(f"{p}: {sys.__getattribute__(p)}")
            continue
        except AttributeError:
            pass
        try:
            print(f"{p}: {sys.link.__getattribute__(p)}")
            continue
        except AttributeError:
            pass
        try:
            print(f"{p}: {sys.geoms[0].__getattribute__(p)}")
            continue
        except AttributeError:
            pass


def get_stepsizes(dt: float = None, dt_substeps: float = None, substeps: float = None):
    if dt is None:
        dt = dt_substeps * substeps
    elif dt_substeps is None:
        dt_substeps = dt / substeps
    elif substeps is None:
        substeps = ceil(dt/dt_substeps)
        dt_substeps = dt / substeps
    else:
        raise ValueError("Only 2 out of 3 arguments may be specified.")
    return dt, dt_substeps, substeps