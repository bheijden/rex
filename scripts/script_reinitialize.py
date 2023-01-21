from scripts.dummy import DummyNode, DummyAgent
from rex.distributions import Distribution, Gaussian, GMM
from rex.constants import LATEST, BUFFER, WARN, DEBUG, ERROR
import rex.utils as utils
import dill
from flax import serialization
from rex.proto import log_pb2


# Load episode record
log_dir = "/home/r2ci/rex/logs"
exp_record = log_pb2.ExperimentRecord()
with open(f"{log_dir}/all_logged_sac_pendulum.pb", "rb") as f:
	exp_record.ParseFromString(f.read())

# reload nodes
eps = []
for e in exp_record.episode:
	node = {n.info.name: dict(obj=None, outputs=None, rngs=None, states=None, params=None, step_states=None) for n in e.node}
	eps.append(node)

	# Reinitialize nodes
	for n in e.node:
		obj = dill.loads(n.info.state)
		node[obj.name]["obj"] = obj

	# Finalize unpickling
	objs = {name: n["obj"] for name, n in node.items()}
	for n in node.values():
		n["obj"].unpickle(objs)

	for n in e.node:
		# Reinitialize outputs
		target = dill.loads(n.outputs.target)
		encoded_bytes = n.outputs.encoded_bytes
		outputs = [serialization.from_bytes(target, b) for b in encoded_bytes]
		node[n.info.name]["outputs"] = outputs

		# Reinitialize rngs
		target = dill.loads(n.rngs.target)
		encoded_bytes = n.rngs.encoded_bytes
		rngs = [serialization.from_bytes(target, b) for b in encoded_bytes]
		node[n.info.name]["rngs"] = rngs

		# Reinitialize states
		target = dill.loads(n.states.target)
		encoded_bytes = n.states.encoded_bytes
		states = [serialization.from_bytes(target, b) for b in encoded_bytes]
		node[n.info.name]["states"] = states

		# Reinitialize params
		target = dill.loads(n.params.target)
		encoded_bytes = n.params.encoded_bytes
		params = [serialization.from_bytes(target, b) for b in encoded_bytes]
		node[n.info.name]["params"] = params

		# Reinitialize step states
		target = dill.loads(n.step_states.target)
		encoded_bytes = n.step_states.encoded_bytes
		step_states = [serialization.from_bytes(target, b) for b in encoded_bytes]
		node[n.info.name]["step_states"] = step_states

