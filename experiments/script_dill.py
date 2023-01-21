# todo: add versioning to nodes.  This will allow us to detect changes when unpickling.
# todo: use argh for command line parsing?
# todo: Optionally log pickled nodes in proto
# todo: Optionally log pickled outputs of each node with message pack in proto (write to list with max_queue_size)
# todo: Optionally log pickled step_state of each node with message pack in proto (write to list with  max_queue_size)
# todo: Allow environments to be pickled and unpickled.
# todo: DILL: https://stackoverflow.com/questions/25613543/how-to-use-dill-to-serialize-a-class-definition
import dill
import pickle
import jax
import jumpy.numpy as jp
import jumpy.random as random
import flax.serialization as ser
import flax.struct as struct
import jax.tree_util as tree_util

from dill_dataclasses import Params, State, StepState


@struct.dataclass
class NewParams:
    t: jp.float32
    a: str = struct.field(pytree_node=False, default_factory=lambda: "a")

pytree = {"test": StepState(state=State(th=[67, 1], thdot=68), params=Params(p=77, t=78))}

# Flatten tree
pytree_flat, pytreedef = jax.tree_util.tree_flatten(pytree)
struct.dataclass(NewParams)

pytreedef_state = pytreedef.__getstate__()  # --> pickle classes
pytreedef_state_bytes = tree_util.tree_map(lambda cls: dill.dumps(cls), pytreedef_state)
pytree_node_cls = [dill.dumps(n[3]) for n in pytreedef_state]

# Prepare for serialization
# state = pytree
state = (pytree_flat, pytree_node_cls, pytreedef, pytreedef_state_bytes)

# Dump the state to a file with dill
with open("state.dill", "wb") as f:
    byref = None
    dill.dump(state, f, byref=byref)
    print(f"dill | bytes = {len(dill.dumps(state, byref=byref))}")

# # Dump the state to a file with pickle
# with open("state.pkl", "wb") as f:
#     print(f"dill | bytes = {len(pickle.dumps(state))}")
#     pickle.dump(state, f)

# # Serialize
# data = ser.to_bytes(state)
# print('num bytes:', len(data))
#
# corrupted_state = jax.tree_map(lambda x: 0 * x, state)
# print(ser.to_state_dict(corrupted_state))
#
# # Restore the state using the state dict using the serialized state dict stored in data
# restored_state = ser.from_bytes(corrupted_state, data)
# print(ser.to_state_dict(restored_state))

# Convert to state_dict
# state_dict = ser.to_state_dict(state)
