from typing import TYPE_CHECKING, List, Deque, Callable
import re
from functools import wraps
from time import time
import sys
from termcolor import colored
from os import getpid
from threading import current_thread
import jax.numpy as jnp  # todo: replace with from brax import jumpy as jp.ndarray?

from rex.constants import WARN, INFO, SIMULATED

if TYPE_CHECKING:
    from rex.node import Node


# Global log level
LOG_LEVEL = WARN


# def synchronized(lock):
#     """A decorator for synchronizing access to a given function."""
#
#     def wrapper(fn):
#         def inner(*args, **kwargs):
#             with lock:
#                 return fn(*args, **kwargs)
#         return inner
#     return wrapper


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


def set_log_level(log_level: int):
    import rex.utils as utils
    utils.LOG_LEVEL = log_level


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
    def __init__(self, name: str, log_level: int = INFO):
        self.name = name
        self.log_level = log_level

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        if self.log_level >= LOG_LEVEL:
            print(f"[{self.name}] Elapsed: {time() - self.tstart}")


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


