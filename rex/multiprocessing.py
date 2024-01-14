from typing import Callable, Tuple
import os
import multiprocessing as mp
from concurrent.futures.process import _sendback_result, _ExceptionWithTraceback
from concurrent.futures import _base, Future, CancelledError, ProcessPoolExecutor
import traceback

from rex.constants import ERROR
from rex.utils import log


# We skip this function in the coverage report because it is not accurately reported by pytest-cov
# We do actually fully cover this function in test_multiprocessing.py.
def _process_worker(fn, call_queue, result_queue, initializer, initargs):  # pragma: no cover
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
        except BaseException as _e:
            # error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            # log(initializer.__qualname__, "red", ERROR, "worker", error_msg)
            _base.LOGGER.critical("Exception in initializer:", exc_info=True)
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


class _NewProcess:
    def __init__(
        self, fn: Callable, max_workers: int = 1, initializer: Callable = None, initargs: Tuple = (), method: str = "fork"
    ):
        self._unwrapped_fn = fn

        # In general we fork, so that we inherit all global variables of the parent process.
        # Fork is not threadsafe, so we may need to use 'forkserver' as the start method.
        # More info: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        # As a consequence of inheriting all global variables, re-initializing pubs/subs in worker process seem to fail when
        # the main process was already initialized as a ros node.
        mp_context = mp.get_context(method)

        executor = ProcessPoolExecutor(
            max_workers=max_workers, initializer=initializer, initargs=initargs, mp_context=mp_context
        )
        self._executor = executor

        def _adjust_process_count():
            """This monkey patches the _adjust_process_count method of the executor.

            We need to do this, because with original method requires fn to be pickleable.
            This is in general not the case for most functions (e.g. step, reset), because nodes cannot be pickled.
            """
            for _ in range(len(executor._processes), executor._max_workers):
                p = executor._mp_context.Process(
                    target=_process_worker,
                    args=(fn, executor._call_queue, executor._result_queue, executor._initializer, executor._initargs),
                )
                p.start()
                executor._processes[p.pid] = p

        # Monkey patch the _adjust_process_count method
        executor._adjust_process_count = _adjust_process_count

    def __call__(self, *args, **kwargs):
        """Call the wrapped function synchronously in a separate process.

        When we are tracing, we call the wrapped function directly,
        because tracing must happen in the same thread.
        """
        f = self.submit(*args, **kwargs)
        return f.result()

    def _done_callback(self, f: Future):
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
        f.add_done_callback(self._done_callback)
        return f

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)


def new_process(fn: Callable, max_workers: int = 1, initializer: Callable = None, initargs: Tuple = (), method: str = "spawn") -> _NewProcess:
    return _NewProcess(fn, max_workers=max_workers, initializer=initializer, initargs=initargs, method=method)


# This is a decorator (does not work with class methods...)
# def new_process(max_workers=1, initializer=None, initargs=()):
# 	def decorator(fn):
# 		return _NewProcess(fn, max_workers=max_workers, initializer=initializer, initargs=initargs)
# 	return decorator
