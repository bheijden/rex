import dill
import pickle
import jax
import jumpy.numpy as jp
import jax.tree_util as tree_util
import flax.struct as struct

from dill_dataclasses import Params, State, StepState


# @struct.dataclass
# class Params:
# 	# p: jp.float32
# 	# t: jp.float32
# 	q: jp.float32
#
#
# @struct.dataclass
# class State:
# 	th: jp.float32
# 	thdot: jp.float32
#
#
# @struct.dataclass
# class StepState:
# 	state: State
# 	params: Params


# Load state from file with dill
with open("state.dill", "rb") as f:
	state = dill.load(f, ignore=False)
	if isinstance(state, tuple):
		# set state of pytreedef with the classes of the pytree
		#

		pytree_flat, pytree_cls_flat, pytreedef, pytreedef_state_bytes = state
		pytree = jax.tree_util.tree_map(lambda cls, val: cls(val), pytreedef, pytree_flat)
		jax
		pytree = jax.tree_util.tree_unflatten(pytreedef, pytree_flat)
	else:
		pytree = state
	print(f"{pytree.__class__} | {pytree}")
	print(f"isinstance(state, Params) = {isinstance(pytree.params, Params)}")

# Load state from file with pickle
# with open("state.pkl", "rb") as f:
# 	state = pickle.load(f)
