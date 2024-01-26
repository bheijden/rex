from functools import partial
import hashlib
import os
import dill as pickle
from math import ceil
from typing import Union, Tuple
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import brax
from brax.io import mjcf
from brax.base import Transform, System
from brax.generalized import pipeline as gen_pipeline
from brax.positional import pipeline as pos_pipeline
from brax.spring import pipeline as spr_pipeline

from rex.proto import log_pb2
from rex.jax_utils import tree_extend
import sysid.utils as sid
import envs.vx300s as vx300s


@struct.dataclass
class EEPose:
	eepos: jax.typing.ArrayLike
	eeorn: jax.typing.ArrayLike


@struct.dataclass
class Params:
	box_armature: jax.typing.ArrayLike  # sys.dof.armature[:4] = (4,) (xyz, yaw)
	joint_armature: jax.typing.ArrayLike  # sys.dof.armature[6:12] = (6,)
	box_damping: jax.typing.ArrayLike  # sys.dof.damping[:4] = (4,) (xyz, yaw)
	joint_damping: jax.typing.ArrayLike  # sys.dof.damping[6:12] = (6,)
	kp: jax.typing.ArrayLike  # sys.actuator.gain[:6] = (6,)
	kv: jax.typing.ArrayLike  # sys.actuator.gain[6:] = (6,)
	box_mass: jax.typing.ArrayLike  # sys.link.inertia.mass[1] = ()
	box_diaginertia: jax.typing.ArrayLike  # sys.link.inertia.i[1] = (3,)  --> diaginertia
	link_mass: jax.typing.ArrayLike  # sys.link.inertia.mass[3:9] = (6,)
	link_diaginertia: jax.typing.ArrayLike  # sys.link.inertia.i[3:9] = (6, 3)  --> diaginertia
	sys: Union[gen_pipeline.System]
	dt_sysid: Union[float, jax.typing.ArrayLike]
	substeps: Union[int, jax.typing.ArrayLike] = struct.field(pytree_node=False, default=None)

	@classmethod
	def default(cls, sys: gen_pipeline.System = None, dt_sysid: Union[float, jax.typing.ArrayLike] = None, substeps: int = None):
		box_armature = jnp.array([0.04, 0.04, 0.04, 0.04])
		joint_armature = jnp.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
		box_damping = jnp.array([0.5, 0.5, 0.5, 0.5])
		joint_damping = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
		kp = jnp.array([20., 50., 50., 20., 20., 20.])
		kv = jnp.array([2., 10., 10., 2., 2., 2.])
		box_mass = jnp.array(0.047)  # box mass (kg)
		l = 0.1  # box length (m)
		box_diaginertia = 1 / 12 * box_mass * (l ** 2 + l ** 2) * jnp.ones(
			(3,))  # Calculate diagonal inertia w.r.t center of mass
		link_mass = jnp.array([0.09976057, 0.3411557, 0.09205945, 0.10984317, 0.03510773, 0.18321589])  # link mass (kg)
		link_diaginertia = jnp.array([[1.07625368e-04, 1.02358516e-04, 7.43850591e-05],
		                              [3.42201520e-03, 3.33783616e-03, 2.67244238e-04],
		                              [3.47116150e-04, 3.09179014e-04, 4.92400310e-05],
		                              [1.27300846e-04, 1.18447661e-04, 3.78955757e-05],
		                              [1.94190028e-05, 1.49090589e-05, 1.34243552e-05],
		                              [2.95034184e-04, 2.35780308e-04, 1.42303556e-04]])
		return cls(box_armature=box_armature,
		           joint_armature=joint_armature,
		           box_damping=box_damping,
		           joint_damping=joint_damping,
		           kp=kp,
		           kv=kv,
		           box_mass=box_mass,
		           box_diaginertia=box_diaginertia,
		           link_mass=link_mass,
		           link_diaginertia=link_diaginertia,
		           sys=sys, dt_sysid=dt_sysid, substeps=substeps)


@struct.dataclass
class State:
	init_boxpos: jax.typing.ArrayLike
	init_boxyaw: Union[float, jax.typing.ArrayLike]
	init_goalpos: jax.typing.ArrayLike
	init_jpos: jax.typing.ArrayLike
	init_jvel: jax.typing.ArrayLike
	pipeline_state: Union[gen_pipeline.State]


@struct.dataclass
class Action:
	jpos: jax.typing.ArrayLike
	jvel: jax.typing.ArrayLike


@struct.dataclass
class Output:
	arm_output: vx300s.env.ArmOutput
	box_output: vx300s.env.BoxOutput


class BraxBackend(sid.Backend):
	name = "brax"
	xml_path = "/home/r2ci/rex/envs/vx300s/assets/vx300s_brax.xml"

	def init_backend(self, dt_sysid: float, dt: float = None, sys: gen_pipeline.System = None) -> Params:
		sys = mjcf.load(self.xml_path) if sys is None else sys
		sys = sys.replace(dt=dt) if dt is not None else sys

		print(f"degrees of freedom: {sys.qd_size()}")

		# Determine collision pairs
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
		q_id = sys.actuator.q_id[:1]

		# DOFS
		print("\nDEGREES OF FREEDOM SIZE")
		print(f"degrees of freedom: {sys.qd_size()}")

		substeps = ceil(dt_sysid / sys.dt)
		dt_sysid = substeps * sys.dt
		print(f"\nTIME")
		print(f"dt: {sys.dt}, dt_sysid: {dt_sysid}, substeps: {substeps}")
		return Params.default(sys=sys, dt_sysid=dt_sysid, substeps=substeps)

	def init_sys(self, pre_params: Params) -> Params:
		sys = pre_params.sys
		armature = sys.dof.armature.at[:4].set(pre_params.box_armature)
		armature = armature.at[6:12].set(pre_params.joint_armature)
		damping = sys.dof.damping.at[:4].set(pre_params.box_damping)
		damping = damping.at[6:12].set(pre_params.joint_damping)
		dof = sys.dof.replace(armature=armature, damping=damping)
		gain = sys.actuator.gain.at[:6].set(pre_params.kp)
		gain = gain.at[6:].set(pre_params.kv)
		actuator = sys.actuator.replace(gain=gain)
		mass = sys.link.inertia.mass.at[1].set(pre_params.box_mass)
		mass = mass.at[3:9].set(pre_params.link_mass)
		i = sys.link.inertia.i.at[1].set(jnp.diag(pre_params.box_diaginertia))
		i = i.at[3:9].set(jax.vmap(jnp.diag)(pre_params.link_diaginertia))
		inertia = sys.link.inertia.replace(mass=mass, i=i)
		link = sys.link.replace(inertia=inertia)
		sys = sys.replace(dof=dof, actuator=actuator, link=link)
		return pre_params.replace(sys=sys)

	def init_pipeline(self, params: Params, pre_state: State) -> State:
		boxpos = pre_state.init_boxpos
		boxyaw = pre_state.init_boxyaw
		goalpos = pre_state.init_goalpos
		jpos = pre_state.init_jpos
		jvel = pre_state.init_jvel
		sys = params.sys
		qpos_box_goal = jnp.concatenate([boxpos, jnp.array([boxyaw]), goalpos])
		ndof_box_goal = qpos_box_goal.shape[0]
		qpos = sys.init_q.at[0:ndof_box_goal].set(qpos_box_goal)
		qpos = qpos.at[ndof_box_goal:-1].set(jpos)
		qvel = jnp.zeros(sys.qd_size())
		qvel = qvel.at[ndof_box_goal:-1].set(jvel)
		pipeline_state = gen_pipeline.init(sys, qpos, qvel)
		pipeline_state.x.vmap()
		post_state = pre_state.replace(pipeline_state=pipeline_state)
		return post_state

	def get_output(self, params: Params, state: State) -> Output:
		pipeline_state = state.pipeline_state
		sys = params.sys
		# _joint_idx = sys.actuator.q_id[:6].tolist()
		_joint_slice = slice(6, 12)  # todo: this is not general
		_ee_arm_idx = sys.link_names.index("ee_link")
		_box_idx = sys.link_names.index("box")
		# _goal_idx = sys.link_names.index("goal")
		x_i = pipeline_state.x.vmap().do(
			Transform.create(pos=sys.link.inertia.transform.pos)
		)
		jpos = pipeline_state.q[_joint_slice]
		eepos = x_i.pos[_ee_arm_idx]
		eeorn = sid.convert_wxyz_to_xyzw(x_i.rot[_ee_arm_idx])  # quaternion (w,x,y,z) -> (x,y,z,w)
		boxpos = x_i.pos[_box_idx]
		boxorn = sid.convert_wxyz_to_xyzw(x_i.rot[_box_idx])  # quaternion (w,x,y,z) -> (x,y,z,w)
		arm_output = vx300s.env.ArmOutput(jpos=jpos, eepos=eepos, eeorn=eeorn)
		box_output = vx300s.env.BoxOutput(boxpos=boxpos, boxorn=boxorn)
		y = Output(arm_output=arm_output, box_output=box_output)
		return y

	def step(self, params: Params, state: State, action: Action) -> Tuple[State, Output]:

		def loop_body(_, s: State) -> State:
			jpos_ref, jvel_ref = action.jpos, action.jvel
			a = jnp.concatenate([jpos_ref, jvel_ref])
			# new_ps = gen_pipeline.step(params.sys, s.pipeline_state, a)
			new_ps = s.pipeline_state
			new_s = s.replace(pipeline_state=new_ps)
			return new_s

		new_state = jax.lax.fori_loop(0, onp.array(params.substeps), loop_body, state)
		y = self.get_output(params, new_state)
		return new_state, y


def get_ee_pose(sys: System, jpos: jax.typing.ArrayLike) -> EEPose:
	# Set
	qpos = jnp.concatenate([sys.init_q[:6], jpos, jnp.array([0])])
	pipeline_state = spr_pipeline.init(sys, qpos, jnp.zeros_like(sys.init_q))
	x_i = pipeline_state.x.vmap().do(
		Transform.create(pos=sys.link.inertia.transform.pos)
	)

	# Get position
	ee_arm_idx = sys.link_names.index("ee_link")
	eepos = x_i.pos[ee_arm_idx]

	# Get orientation
	quat = x_i.rot[ee_arm_idx]
	eeorn = jnp.array([quat[1], quat[2], quat[3], quat[0]])
	return EEPose(eepos, eeorn)


def load_or_gen_data(cache_dir: str, log_dir: str, dt_sysid: float, xml_path: str):
	hash_object = hashlib.md5(str({"log_dir": log_dir, "dt_sysid": dt_sysid, "xml_path": xml_path}).encode())
	hash = hash_object.hexdigest()

	hash_file = f"{cache_dir}/{hash}.pkl"
	if os.path.exists(hash_file):
		print(f"loading data from {hash_file}")
		DATA_INTERP = pickle.load(open(hash_file, "rb"))
	else:
		print(f"generating data and saving to {hash_file}")
		# Prepare data
		m_brax = mjcf.load(xml_path)
		LOG_DIR = log_dir
		record = log_pb2.ExperimentRecord()
		with open(f"{LOG_DIR}/record_pre.pb", "rb") as f:
			record.ParseFromString(f.read())

		import experiments as exp
		helper = exp.RecordHelper(record, method="truncated")
		DATA = helper._data_stacked
		TIMESTAMPS = helper._timestamps_stacked
		ts = min(TIMESTAMPS["armsensor"]["ts_output"].max(), TIMESTAMPS["boxsensor"]["ts_output"].max())
		timestamps_interp = onp.arange(0, ts, dt_sysid)

		# Get jit functions
		jit_vmap_interp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
		jit_vmap_get_ee_pose = jax.jit(jax.vmap(get_ee_pose, in_axes=(None, 0)))
		jit_vmap_cost_fn = jax.jit(jax.vmap(vx300s.planner.cost.box_pushing_cost, in_axes=(None, 0, 0, None, 0)))

		timestamps = TIMESTAMPS
		num_eps = timestamps["armsensor"]["ts_output"].shape[0]

		# Store all of the below in a dict
		eps_data = []
		for eps_idx in range(num_eps):
			jpos_target = jit_vmap_interp(timestamps_interp, timestamps["armactuator"]["ts_output"][eps_idx],
			                              DATA["armactuator"]["outputs"].jpos[eps_idx])
			armsensor = jax.tree_util.tree_map(
				lambda x: jit_vmap_interp(timestamps_interp, timestamps["armsensor"]["ts_output"][eps_idx], x[eps_idx]),
				DATA["armsensor"]["outputs"])
			jpos = jit_vmap_interp(timestamps_interp, timestamps["armsensor"]["ts_output"][eps_idx],
			                       DATA["armsensor"]["outputs"].jpos[eps_idx])
			boxsensor = jax.tree_util.tree_map(
				lambda x: jit_vmap_interp(timestamps_interp, timestamps["boxsensor"]["ts_output"][eps_idx], x[eps_idx]),
				DATA["boxsensor"]["outputs"])
			boxpos = jit_vmap_interp(timestamps_interp, timestamps["boxsensor"]["ts_output"][eps_idx],
			                         DATA["boxsensor"]["outputs"].boxpos[eps_idx])
			ee_pose = jit_vmap_get_ee_pose(m_brax, jpos)
			ee_pose_target = jit_vmap_get_ee_pose(m_brax, jpos_target)
			cost_params = jax.tree_util.tree_map(lambda x: x[eps_idx, 0], DATA["planner"]["step_states"].params.cost_params)
			goalpos = jax.tree_util.tree_map(lambda x: x[eps_idx, 0], DATA["planner"]["step_states"].state.goalpos)

			# Get cost
			_, cost_info = jit_vmap_cost_fn(cost_params, boxpos, ee_pose.eepos, goalpos, ee_pose.eeorn)
			cost = cost_info.pop("cost")
			cm = cost_info.pop("cm")
			_ = cost_info.pop("alpha")

			# Get error
			jpos_err_abs = jnp.abs(jpos_target - jpos)
			ee_error = (ee_pose_target.eepos - ee_pose.eepos) * 100
			ee_error_abs = jnp.abs(ee_error)
			ee_error_norm = jnp.linalg.norm(ee_error, axis=-1)

			# Store all of the above in eps_data
			eps_data.append({
				"timestamps": timestamps_interp,
				"jpos_target": jpos_target,
				"jpos": jpos,
				"boxpos": boxpos,
				"armsensor": armsensor,
				"boxsensor": boxsensor,
				"ee_pose": (ee_pose.eepos, ee_pose.eeorn),
				"ee_pose_target": (ee_pose_target.eepos, ee_pose_target.eeorn),
				"cost_params": cost_params,
				"goalpos": goalpos,
				"cost": cost,
				"cm": cm,
				"jpos_err_abs": jpos_err_abs,
				"ee_error": ee_error,
				"ee_error_abs": ee_error_abs,
				"ee_error_norm": ee_error_norm,
			})

		# stack eps_data
		DATA_INTERP = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *eps_data)

		# Pickle
		with open(hash_file, "wb") as f:
			pickle.dump(DATA_INTERP, f)
	return DATA_INTERP


def residual(backend: sid.Backend, opt_params: Params, args) -> Union[Output, vx300s.env.ArmOutput, vx300s.env.BoxOutput]:
	"""Residual function for least squares.

	:param opt_params: pytree of optimizable parameters (None for non-optimizable params that get replaced by base_params)
	:param args: Passed through optx least_squares (non-optimzed args)
	"""
	# todo: apply exp?
	base_params, pre_s, actions, init_y_ys = args

	# Replace base_params with optimizable params and re-initialize system with new params
	opt_params = opt_params.replace(substeps=base_params.substeps)  # NOTE: ugly fix for substeps --> trees don't deal with static fields very well
	opt_params = tree_extend(base_params, opt_params)  # Extend opt_params to match base_params pytree structure
	pre_params = jax.tree_util.tree_map(lambda base_x, opt_x: base_x if opt_x is None else opt_x, base_params, opt_params)
	params = backend.init_sys(pre_params)  # Updates parameters in params.sys

	# Get initial state (incl. pipeline state)
	init_s = jax.vmap(backend.init_pipeline, in_axes=(None, 0))(params, pre_s)

	# Rollout with params
	pred_final_s, pred_init_y_ys = jax.vmap(backend.rollout, in_axes=(None, 0, 0))(params, init_s, actions)

	# Get residual (label - pred)
	pre_res = jax.tree_util.tree_map(lambda label, pred: label - pred, init_y_ys, pred_init_y_ys)

	# Remove orientations, because they require some special handling (e.g. wrap around, only yaw is relevant, etc.)
	res_arm = pre_res.arm_output.replace(eeorn=None)
	res_box = pre_res.box_output.replace(boxorn=None)
	res = pre_res.replace(arm_output=res_arm, box_output=res_box)
	return res
