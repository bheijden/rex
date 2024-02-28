from typing import Dict, List, Tuple, Union, Callable, Any, Type, Optional
import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as onp

import flax.serialization as serialization
import dill as pickle
from pickle import UnpicklingError

from rex.node import Node
from rex.proto import log_pb2


class _NoValue: pass


class _HasValue:
    def __init__(self, tree):
        self.tree = tree


def stack_padding(it):

    def resize(row, size):
        new = onp.array(row)
        new.resize(size, refcheck=False)
        return new

    # find longest row length
    row_length = max(it, key=len).shape
    mat = onp.array([resize(row, row_length) for row in it])

    return mat


def _padded_stack(*data: jax.typing.ArrayLike) -> Optional[jax.Array]:
    empties = [isinstance(d, _NoValue) for d in data]
    has_empty = any(empties)
    if has_empty:
        if all(empties):
            return None
        else:
            raise ValueError("Cannot stack partially empty data")
    elif all([isinstance(d, _HasValue) for d in data]):
        return tree_map(_padded_stack, *[d.tree for d in data])
    assert all([not isinstance(d, _HasValue) for d in data])

    # Stack data
    data_stacked = stack_padding(data)
    return data_stacked


def _truncated_stack(*data: jax.typing.ArrayLike) -> Optional[jax.Array]:
    has_empty = any([isinstance(d, _NoValue) for d in data])
    if has_empty:
        return None
    elif all([isinstance(d, _HasValue) for d in data]):
        return tree_map(_truncated_stack, *[d.tree for d in data])
    assert all([not isinstance(d, _HasValue) for d in data])

    # Determine min_length
    min_length = min([x.shape[0] for x in data if x.ndim > 0])

    # Truncate data to min_length
    data = tree_map(lambda d: d[:min_length], data)

    # Stack data
    data_stacked = tree_map(lambda *d: onp.stack(d), *data)
    return data_stacked


def unpickle_data(record: log_pb2.Serialization):
    if len(record.encoded_bytes) == 0:
        return None
    encoded_bytes = record.encoded_bytes
    try:
        target = pickle.loads(record.target)
        data = [serialization.from_bytes(target, b) for b in encoded_bytes]
    except UnpicklingError as e:
        print(f"Failed to load target. Unpickling to state_dict instead: {e}")
        data = [serialization.msgpack_restore(b) for b in encoded_bytes]
    return tree_map(lambda *x: jnp.stack(x), *data)


class RecordHelper:
    def __init__(self, record: Union[log_pb2.ExperimentRecord, log_pb2.EpisodeRecord], method: str = "padded"):
        self.record = record

        # Convert to experiment record
        if isinstance(record, log_pb2.EpisodeRecord):
            self._record = log_pb2.ExperimentRecord(episode=[record])
        else:
            self._record = record
        assert isinstance(self._record, log_pb2.ExperimentRecord), "Record must be an ExperimentRecord or EpisodeRecord"

        # Store preprocessed data in convenient format
        self._delays: List[Dict[str, Dict[str, Union[jax.typing.ArrayLike, Dict[str, jax.typing.ArrayLike]]]]] = None
        self._delays_stacked: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Dict[str, jax.typing.ArrayLike]]]] = None
        self._data: List[Dict[str, Dict[str, Any]]] = None
        self._data_stacked: Dict[str, Dict[str, Any]] = None
        self._timestamps: List[Dict[str, Dict[str, Any]]] = None
        self._timestamps_stacked: Dict[str, Dict[str, Any]] = None
        self._nodes: List[Dict[str, Union[str, Node]]] = []

        # Pre-process record data
        self._preprocess_data()

        # Validate record
        self._validate_data()

        # Stack data
        assert method in ["truncated", "padded"], "Stacking method must be either 'truncated' or 'padded'"
        self._stack_data(method)

    @property
    def data(self):
        return self._data_stacked

    @property
    def outputs(self):
        outputs = {}
        for name, d in self._data_stacked.items():
            outputs[name] = d["outputs"]
        return outputs

    @property
    def states(self):
        states = {}
        for name, d in self._data_stacked.items():
            states[name] = d["states"]
        return states

    @property
    def params(self):
        params = {}
        for name, d in self._data_stacked.items():
            params[name] = d["params"]
        return params

    @property
    def step_states(self):
        step_states = {}
        for name, d in self._data_stacked.items():
            step_states[name] = d["step_states"]
        return step_states

    @property
    def rngs(self):
        rngs = {}
        for name, d in self._data_stacked.items():
            rngs[name] = d["rngs"]
        return rngs

    @property
    def delays(self):
        node_delays = {}
        for name, delays in self._delays_stacked["step"].items():
            node_delays[name] = dict(step=delays, inputs={})
            for input_name, input_delays in self._delays_stacked["inputs"][name].items():
                node_delays[name]["inputs"][input_name] = input_delays
        return node_delays

    @property
    def timestamps(self):
        return self._timestamps_stacked

    @property
    def ts_outputs(self):
        ts_outputs = {}
        for name, ts in self._timestamps_stacked.items():
            ts_outputs[name] = ts["ts_output"]
        return ts_outputs

    @property
    def ts_step(self):
        ts_step = {}
        for name, ts in self._timestamps_stacked.items():
            ts_step[name] = ts["ts_step"]
        return ts_step

    def get_nodes(self, episode: int = -1) -> Dict[str, Node]:
        nodes = {}
        for name, node_bytes in self._nodes[episode].items():
            assert len(node_bytes) > 0, "Node state must be non-empty."
            nodes[name] = pickle.loads(node_bytes)

        # Fully restore node by unpickling (re-connects to other nodes, execute custom unpickling routines if any)
        for n in nodes.values():
            n.unpickle(nodes)
        return nodes

    def _preprocess_data(self):
        # Get delays
        self._delays, _ = get_delay_data(self._record, concatenate=False)

        # Get timestamps
        self._timestamps = get_timestamps(self._record)

        # Get data
        self._data = []
        self._nodes = []
        for i, e in enumerate(self._record.episode):
            # Store nodes
            eps_nodes = {}
            self._nodes.append(eps_nodes)
            for n in e.node:
                eps_nodes[n.info.name] = n.info.state

            # Store data
            eps_data = {n.info.name: dict(outputs=None, rngs=None, states=None, params=None, step_states=None) for n in e.node}
            self._data.append(eps_data)
            for n in e.node:
                # Store outputs
                eps_data[n.info.name]["outputs"] = unpickle_data(n.outputs)
                eps_data[n.info.name]["rngs"] = unpickle_data(n.rngs)
                eps_data[n.info.name]["states"] = unpickle_data(n.states)
                eps_data[n.info.name]["params"] = unpickle_data(n.params)
                eps_data[n.info.name]["step_states"] = unpickle_data(n.step_states)
            for n in e.node:
                # Convert empty data to _NoValue and _HasValue
                eps_data[n.info.name]["outputs"] = _HasValue(eps_data[n.info.name]["outputs"]) if eps_data[n.info.name]["outputs"] is not None else _NoValue()
                eps_data[n.info.name]["rngs"] = _HasValue(eps_data[n.info.name]["rngs"]) if eps_data[n.info.name]["rngs"] is not None else _NoValue()
                eps_data[n.info.name]["states"] = _HasValue(eps_data[n.info.name]["states"]) if eps_data[n.info.name]["states"] is not None else _NoValue()
                eps_data[n.info.name]["params"] = _HasValue(eps_data[n.info.name]["params"]) if eps_data[n.info.name]["params"] is not None else _NoValue()
                eps_data[n.info.name]["step_states"] = _HasValue(eps_data[n.info.name]["step_states"]) if eps_data[n.info.name]["step_states"] is not None else _NoValue()

    def _validate_data(self):
        # todo: Do not raise errors, but rather set flags. Check flags in stack and truncate.
        # todo: check if all data is present
        # todo: Check if data is of same length?
        # todo: Check if computation graph is the same
        # todo: Check if all nodes are present
        # todo: Check if step_states correspond to logged inputs
        # Stack
        # self._lengths = tree_map(lambda *l: list(l), *self._lengths)
        # self._max_lengths = tree_map(lambda *l: max(l), *self._lengths
        pass

    def _stack_data(self, method: str):
        # Stack data
        if method == "truncated":
            self._data_stacked = tree_map(_truncated_stack, *self._data)
            self._delays_stacked = tree_map(_truncated_stack, *self._delays)
            self._timestamps_stacked = tree_map(_truncated_stack, *self._timestamps)
        elif method == "padded":
            self._data_stacked = tree_map(_padded_stack, *self._data)
            self._delays_stacked = tree_map(_padded_stack, *self._delays)
            self._timestamps_stacked = tree_map(_padded_stack, *self._timestamps)
        else:
            raise NotImplementedError


def get_delay_data(record: log_pb2.ExperimentRecord, concatenate: bool = True):
    get_step_delay = lambda s: s.delay  # todo: use comp_delay?
    get_input_delay = lambda m: m.delay  # todo: use comm_delay?

    exp_data, exp_info = [], []
    for e in record.episode:
        data, info = dict(inputs=dict(), step=dict()), dict(inputs=dict(), step=dict())
        exp_data.append(data), exp_info.append(info)
        for n in e.node:
            node_name = n.info.name
            # Fill info tree
            info["inputs"][node_name] = dict()
            info["step"][node_name] = n.info
            for i in n.inputs:
                input_name = i.info.name
                info["inputs"][node_name][input_name] = (n.info, i.info)

            # Fill data tree
            delays = [get_step_delay(s) for s in n.steps]
            data["step"][node_name] = onp.array(delays)
            data["inputs"][node_name] = dict()
            for i in n.inputs:
                input_name = i.info.name
                delays = [get_input_delay(m) for g in i.grouped for m in g.messages]
                data["inputs"][node_name][input_name] = onp.array(delays)

    data = jax.tree_map(lambda *x: onp.concatenate(x, axis=0), *exp_data) if concatenate else exp_data
    info = jax.tree_map(lambda *x: x[0], *exp_info) if concatenate else exp_info
    return data, info


def get_timestamps(record: log_pb2.ExperimentRecord):
    # Get timestamps
    ts_data = []
    for e in record.episode:
        ts = dict()
        ts_data.append(ts)
        for n in e.node:
            node_name = n.info.name
            ts_node = dict()
            ts[node_name] = ts_node

            ts_node["ts_step"] = onp.array([s.ts_output for s in n.steps])
            ts_node["ts_output"] = onp.array([s.ts_output for s in n.steps])

    return ts_data
