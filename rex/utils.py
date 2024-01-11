import inspect
from typing import TYPE_CHECKING, List, Deque, Callable
import pickle
from functools import wraps
import importlib
from time import time
from termcolor import colored
from os import getpid
from threading import current_thread
import numpy as onp
from warnings import warn
import jax

from rex.proto import log_pb2
from rex.constants import WARN, INFO, SIMULATED
from rex.base import StepState

if TYPE_CHECKING:
    from rex.node import BaseNode


# Global log levels
LOG_LEVEL = WARN
NODE_LOG_LEVEL = {}
NODE_COLOR = {}

# def synchronized(lock):
#     """A decorator for synchronizing access to a given function."""
#
#     def wrapper(fn):
#         def inner(*args, **kwargs):
#             with lock:
#                 return fn(*args, **kwargs)
#         return inner
#     return wrapper


def deprecation_warning(msg: str, stacklevel: int = 2):
    # Get the current stack frame
    stack = inspect.stack()
    frame = stack[stacklevel]
    # Extract information
    module = inspect.getmodule(frame[0]).__name__ if inspect.getmodule(frame[0]) is not None else "unknown"
    filename = frame.filename
    lineno = frame.lineno
    function_name = frame.function
    # Print log message
    msg = f"{msg}: (fun={module}.{function_name} at line {lineno})."
    log("rex", "yellow", WARN, "rex", msg)
    # warn(msg, DeprecationWarning, stacklevel=stacklevel)


def load(attribute: str):
    """Loads an attribute from a module.

    :param attribute: The attribute to load. Has the form "module/attribute".
    :return: The attribute.
    """
    module, attribute = attribute.split("/")
    module = importlib.import_module(module)
    attribute = getattr(module, attribute)
    return attribute


def log(
    name: str,
    color: str,
    log_level: int,
    id: str,
    msg=None,
):
    if log_level >= LOG_LEVEL:
        # Add process ID, thread ID, name
        log_msg = f"[{str(getpid())[:5].ljust(5)}][{current_thread().name.ljust(25)}][{name.ljust(20)}][{id.ljust(20)}]"
        if msg is not None:
            log_msg += f" {msg}"
        print(colored(log_msg, color))


def set_log_level(log_level: int, node: "BaseNode" = None, color: str = None):
    if node is not None:
        NODE_LOG_LEVEL[node] = log_level
        if color is not None:
            NODE_COLOR[node] = color
    else:
        global LOG_LEVEL
        LOG_LEVEL = log_level
        assert color is None, "Cannot set color without node"


# def timing(num: int = 1):
#     """Use as decorator @timing(number of repetitions)"""
#     def _timing(f):
#         @wraps(f)
#         def wrap(*args, **kw):
#             ts = time()
#             for _i in range(num):
#                 _ = f(*args, **kw)
#             te = time()
#             print(f"func:{f.__name__} args:[{args}, {kw}] took: {(te-ts)/num: 2.4f} sec")
#             return _
#         return wrap
#     return _timing


class timer:
    def __init__(self, name: str = None, log_level: int = INFO):
        self.name = name or "timer"
        self.log_level = log_level
        self.duration = None

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        self.duration = time() - self.tstart
        if self.log_level >= LOG_LEVEL:
            log(name="tracer", color="white", log_level=self.log_level, id=f"{self.name}", msg=f"Elapsed: {self.duration}")


# def analyse_deadlock(nodes: List["Node"], log_level: int = INFO):
#     # Get list of all threads
#     threads = []
#     for n in nodes:
#         threads += list(n._executor._threads)
#         for i in n.inputs:
#             threads += list(i._executor._threads)
#     for n in nodes:
#         # Check node executor
#         t = list(n._executor._threads)[0]
#         for f, fn, _args, _kwargs in n._q_task:
#             if f.done():
#                 continue
#             elif f.running():
#                 frame = sys._current_frames().get(t.ident, None)
#                 if frame.f_code.co_name == "inner":
#                     frame_lock = frame.f_locals["lock"]
#                     frame_ident = int(re.search('owner=(.*) count', frame_lock.__repr__()).group(1))
#                     frame_owner = [t.getName() for t in threads if t.ident == frame_ident][0]
#                     frame_fn = frame.f_back.f_code.co_name
#                     frame_lineno = frame.f_back.f_lineno
#                 else:
#                     frame_lock = None
#                     frame_owner = None
#                     frame_fn = frame.f_code.co_name
#                     frame_lineno = frame.f_code.co_firstlineno
#                 msg = f"fn={fn.__name__} | stuck={frame_fn}({frame_lineno}) | owner={frame_owner}"
#                 n.log(id="RUNNING", value=msg, log_level=log_level)
#             elif not f.running():
#                 msg = f"fn={fn.__name__}"
#                 n.log(id="WAITING", value=msg, log_level=log_level)
#
#         # Check input executor
#         for i in n.inputs:
#             t = list(i._executor._threads)[0]
#             for f, fn, _args, _kwargs in i._q_task:
#                 if f.done():
#                     continue
#                 elif f.running():
#                     frame = sys._current_frames().get(t.ident, None)
#                     if frame.f_code.co_name == "inner":
#                         frame_lock = frame.f_locals["lock"]
#                         frame_ident = int(re.search('owner=(.*) count', frame_lock.__repr__()).group(1))
#                         frame_owner = [t.getName() for t in threads if t.ident == frame_ident][0]
#                         frame_fn = frame.f_back.f_code.co_name
#                         # frame_lineno = frame.f_back.f_code.co_firstlineno
#                         frame_lineno = frame.f_back.f_lineno
#                     else:
#                         frame_lock = None
#                         frame_owner = None
#                         frame_fn = frame.f_code.co_name
#                         frame_lineno = frame.f_code.co_firstlineno
#                     msg = f"fn={fn.__name__} | stuck={frame_fn}({frame_lineno}) | owner={frame_owner}"
#                     i.log(id="RUNNING", value=msg, log_level=log_level)
#                 elif not f.running():
#                     f"fn={fn.__name__}"
#                     i.log(id="WAITING", value=fn.__name__, log_level=log_level)
#     return


# def batch_split_rng(rng: jnp.ndarray, fn: Callable, queue: Deque[jnp.ndarray], num: int = 20):
#     assert num > 0, "Must sample a positive number"
#     if len(queue) == 0:
#         rngs = fn(rng, num+1)
#         rng = rngs[0]
#         queue.extend((rngs[1:]))
#
#     # Get key
#     split_rng = queue.popleft()
#     return rng, split_rng


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


def make_put_output_on_device(wrapped_fn, device):
    def put_output_on_device(step_state: StepState):
        new_step_state, output = wrapped_fn(step_state)
        # todo: use jax.device_get(x) (transfers "x" to host) instead of jax.device_put(x, device)?
        output_on_device = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), output)
        return new_step_state, output_on_device

    return put_output_on_device
