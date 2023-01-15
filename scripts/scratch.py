import functools
import time
import os
from concurrent.futures.process import _sendback_result, _ExceptionWithTraceback
from concurrent.futures import _base, Future, CancelledError, ProcessPoolExecutor
import traceback
import jumpy
import jumpy.numpy as jp

from rex.constants import WARN, ERROR
from rex.utils import log
from rex.base import StepState, Output
from scripts.dummy import build_dummy_env

# Built dummy environment and extract graph_state
env, nodes = build_dummy_env()
graph_state, obs = env.reset(jumpy.random.PRNGKey(0))
env.stop()

sensor = nodes["sensor"]
step_state = graph_state.nodes[sensor.name]

log(name="mainprocess", color="red", log_level=WARN, id="main_thread", msg=f"START")


def _process_worker(fn, call_queue, result_queue, initializer, initargs):
	"""Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        fn: The function to evaluate. With fn.__self__, a reference to that class instance can be obtained.
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None. The arguments of the initializer start with fn, followed by initargs.
        initargs: A tuple of args for the initializer
    """
	if initializer is not None:
		try:
			initializer(fn, *initargs)
		except BaseException as e:
			error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
			log(initializer.__qualname__, "red", ERROR, "worker", error_msg)
			_base.LOGGER.critical('Exception in initializer:', exc_info=True)
			# The parent will notice that the process stopped and
			# mark the pool broken
			return
	while True:
		call_item = call_queue.get(block=True)
		if call_item is None:
			# Wake up queue management thread
			result_queue.put(os.getpid())
			return
		try:
			r = fn(*call_item.args, **call_item.kwargs)
		except BaseException as e:
			exc = _ExceptionWithTraceback(e, e.__traceback__)
			_sendback_result(result_queue, call_item.work_id, exception=exc)
		else:
			_sendback_result(result_queue, call_item.work_id, result=r)
			del r

		# Liberate the resource as soon as possible, to avoid holding onto
		# open files or shared memory that is not needed anymore
		del call_item


class Proxy:
	# todo: BaseNode.push_step must call submit, while CompiledGraph must call __call__.
	# todo: define initializer method for nodes.
	# todo: wraps a step/reset function?
	# Design patterns in 10 min: https://www.youtube.com/watch?v=tv-_1er1mWI
	def __init__(self, fn, max_workers=1, initializer=None, initargs=()):
		self._unwrapped_fn = fn
		executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer, initargs=initargs)
		self._executor = executor

		def _adjust_process_count():
			"""This monkey patches the _adjust_process_count method of the executor.

			We need to do this, because with original method requires fn to be pickleable.
			This is not the case for our functions, because nodes cannot be pickled.
			 """
			for _ in range(len(executor._processes), executor._max_workers):
				p = executor._mp_context.Process(
					target=_process_worker,
					args=(fn,
					      executor._call_queue,
					      executor._result_queue,
					      executor._initializer,
					      executor._initargs))
				p.start()
				executor._processes[p.pid] = p

		# Monkey patch the _adjust_process_count method
		executor._adjust_process_count = _adjust_process_count

	def __call__(self, *args, **kwargs):
		"""Per default, the wrapped function is synchronously called in the caller's process."""
		return self._unwrapped_fn(*args, **kwargs)

	def _f_callback(self, f: Future):
		"""Callback function that is called when the future is done."""
		e = f.exception()
		if e is not None and e is not CancelledError:
			error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
			log(self._unwrapped_fn.__qualname__, "red", ERROR, "proxy", error_msg)

	@staticmethod
	def _dummy_fn(*args, **kwargs):
		"""Dummy function that pickled (but never used)."""
		pass

	def submit(self, *args, **kwargs) -> Future:
		"""Calls the wrapped function asynchronously in a separate process."""
		# Schedule work item
		f = self._executor.submit(self._dummy_fn, *args, **kwargs)
		f.add_done_callback(self._f_callback)

		return f


def new_process(max_workers=1, initializer=None, initargs=()):
	def decorator(fn):
		return Proxy(fn, max_workers=max_workers, initializer=initializer, initargs=initargs)
	return decorator


def initializer(fn, *args):
	print("INITIALIZER", fn, args)
	# fn.__self__.pid = os.getpid()


class Node:
	def __init__(self):
		self.pid = os.getpid()

	@new_process(max_workers=1, initializer=initializer, initargs=("TEST_ARG"))
	def step(self, ts, step_state):
		time.sleep(1)
		log("worker", "blue", WARN, "worker", f"pid={self.pid} | step {ts}")
		return step_state, "OUTPUT"


n = Node()

# n.step = Proxy(sensor.step, max_workers=5, initializer=initializer, initargs=("SOME_ARG",))

# Sync step
sync_ss, sync_output = sensor.step(0.5, step_state)

# Async step
fs = []
for i in range(100000):
	f = n.step.submit(i, step_state)
	fs.append(f)
print("Submitted all")
[f.result() for f in fs]
print("DONE")
