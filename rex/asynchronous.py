from typing import Tuple, Deque, Dict, Union, List, Callable, Any, Optional
import time
from collections import deque
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from collections import deque
import traceback
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as onp

from rex.constants import Clock, RealTimeFactor, Async, LogLevel, Scheduling, Jitter
from rex import base
from rex.node import BaseNode, Connection
from rex import utils


class _AsyncNodeWrapper:
    def __init__(self, node: BaseNode):
        self.node = node
        self.outputs: Dict[str, _AsyncConnectionWrapper] = {}  # Outgoing edges. Keys are the actual node names incident to the edge.
        self.inputs: Dict[str, _AsyncConnectionWrapper] = {}  # Incoming edges. Keys are the input_names of the nodes from which the edge originates. May be different from the actual node names.

        # Record settings
        self._record_setting = dict(params=False, rng=False, inputs=False, state=False, output=False)
        self._max_records = 20000

        # Output related
        self._num_buffer = 50
        self._jit_reset = None
        self._jit_sample = None

        # State and episode counter
        self._has_warmed_up = False
        self._eps = -1  # Is incremented in ._reset() before the episode starts (hence, -1)
        self._state = Async.STOPPED

        # Executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=node.name)
        self._q_task: Deque[Tuple[Future, Callable, Any, Any]] = deque(maxlen=10)
        self._lock = RLock()

        # Reset every run
        self._tick = None
        self._record: base.NodeRecord = None
        self._record_steps: List[base.AsyncStepRecord] = None
        self._phase_scheduled = None
        self._phase = None
        self._sync = None
        self._clock = None
        self._real_time_factor = 1.0
        self._phase_output = None  # output related
        self._dist_state: base.DelayDistribution = None  # output related

        # Log
        self._discarded = 0

        # Set starting ts
        self._ts_start = Future()
        self._set_ts_start(0.0)

        self.q_tick: Deque[int] = None
        self.q_ts_scheduled: Deque[Tuple[int, float]] = None
        self.q_ts_end_prev: Deque[float] = None
        self.q_ts_start: Deque[Tuple[int, float, float, base.AsyncStepRecord]] = None
        self.q_rng_step: Deque[jax.Array] = None
        self.q_sample = None  # output related

        # Only used if no step and reset fn are provided
        self._i = 0

        if not 1 / self.node.rate > self.node.delay:
            self.log(
                "WARNING",
                f"The sampling time ({1/node.rate=:.3f} s) is smaller than"
                f" the output phase ({self.node.delay=:.3f} s)."
                " This may lead to large (accumulating) delays.",
                LogLevel.WARN,
            )

    @property
    def max_records(self) -> int:
        # assert self.node.max_records >= 0, "max_records must be non-negative."
        return self._max_records

    @property
    def record_setting(self) -> Dict[str, bool]:
        return self._record_setting

    @property
    def eps(self) -> int:
        return self._eps

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def phase_output(self) -> float:
        return self._phase_output

    def wrap_connections(self, nodes: Dict[str, "_AsyncNodeWrapper"]):
        for c in self.node.outputs.values():
            assert c.output_node.name == self.node.name, f"Output node name {c.output_node.name} does not match node name {self.node.name}"
            output_node = nodes[c.output_node.name]  # should be equal to self
            input_node = nodes[c.input_node.name]
            connection = _AsyncConnectionWrapper(c, output_node, input_node)
            self.outputs[c.input_node.name] = connection
            input_node.inputs[c.input_name] = connection

    def log(self, id: Union[str, Async], value: Optional[Any] = None, log_level: Optional[int] = None):
        self.node.log(id=id, value=value, log_level=log_level)

    def _submit(self, fn, *args, stopping: bool = False, **kwargs):
        with self._lock:
            if self._state in [Async.READY, Async.STARTING, Async.READY_TO_START, Async.RUNNING] or stopping:
                f = self._executor.submit(fn, *args, **kwargs)
                self._q_task.append((f, fn, args, kwargs))
                f.add_done_callback(self._done_callback)
            else:
                self.log("SKIPPED", fn.__name__, log_level=LogLevel.DEBUG)
                f = Future()
                f.cancel()
        return f

    def _done_callback(self, f: Future):
        e = f.exception()
        if e is not None and e is not CancelledError:
            error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            utils.log(self.node.name, "red", LogLevel.ERROR, "ERROR", error_msg)

    def _set_ts_start(self, ts_start: float):
        assert isinstance(self._ts_start, Future)
        self._ts_start.set_result(ts_start)
        self._ts_start = ts_start
        # Set _async_now for the wrapped node
        self.node._async_now = self.now

    def now(self) -> float:
        """Get the passed time since start of episode according to the simulated and wall clock"""
        # Determine starting timestamp
        ts_start = self._ts_start
        ts_start = ts_start.result() if isinstance(ts_start, Future) else ts_start

        # Determine passed time
        wc = time.time()
        wc_passed = wc - ts_start
        sc = wc_passed if self._real_time_factor == 0 else wc_passed * self._real_time_factor
        return sc

    def throttle(self, ts: float):
        if self._real_time_factor > 0:
            # Determine starting timestamp
            ts_start = self._ts_start
            ts_start = ts_start.result() if isinstance(ts_start, Future) else ts_start

            wc_passed_target = ts / self._real_time_factor
            wc_passed = time.time() - ts_start
            wc_sleep = max(0.0, wc_passed_target - wc_passed)
            time.sleep(wc_sleep)

    def get_record(self, params: bool = True, rng: bool = True, inputs: bool = True, state: bool = True, output: bool = True) -> base.NodeRecord:
        params = self.record_setting["params"] and params
        rng = self.record_setting["rng"] and rng
        inputs = self.record_setting["inputs"] and inputs
        state = self.record_setting["state"] and state
        output = self.record_setting["output"] and output

        # If records were discarded, warn the user that the record is incomplete.
        if self._discarded > 0:
            self.log(
                "WARNING", f"Discarded {self._discarded} records. Incomplete records may lead to errors when tracing.", LogLevel.WARN
            )

        # If the record is incomplete, warn the user that the record is incomplete.
        if self._record is None:
            raise RuntimeError("No record has been created yet.")

        # Add the steps to the record
        if self._record.steps is None:
            to_array = lambda *x: onp.array(x[:-1]) if (len(x) > 0 and x[-1] is None) else onp.array(x)
            # to_array = lambda *x: onp.array([_x for _x in x if _x is not None]) if (len(x) > 0 and x[-1] is None) else onp.array(x)
            steps = jax.tree_map(to_array, *self._record_steps)
            self._record = self._record.replace(steps=steps)

        # Add the inputs to the record
        if self._record.inputs is None:
            last_seq_in = self._record.steps.seq[-1] if len(self._record.steps.seq) > 0 else -1
            inputs = {c.connection.output_node.name: c.get_record(last_seq_in) for i, c in self.inputs.items()}
            self._record = self._record.replace(inputs=inputs)

        # Filter any fields that are not requested
        steps = self._record.steps
        steps = steps.replace(rng=steps.rng if rng else None)
        steps = steps.replace(inputs=steps.inputs if inputs else None)
        steps = steps.replace(state=steps.state if state else None)
        steps = steps.replace(output=steps.output if output else None)
        record = self._record.replace(
            steps=steps,
            params=self._record.params if params else None
        )
        return record

    def set_record_settings(self, params: bool = None, rng: bool = None, inputs: bool = None, state: bool = None, output: bool = None, max_records: int = None):
        """ Set the record settings for the node.

        :param params: Whether to record params for each node. Logged once.
        :param rng: Whether to record rng for each node. Logged each step.
        :param inputs: Whether to record inputs for each node. Logged each step. Can become very large.
        :param state: Whether to record state for each node. Logged each step.
        :param output: Whether to record output for each node. Logged each step.
        :param max_records: The maximum number of records to keep in memory. If exceeded, the oldest records are discarded.
        :return:
        """
        self._record_setting["params"] = params if params is not None else self._record_setting["params"]
        self._record_setting["rng"] = rng if rng is not None else self._record_setting["rng"]
        self._record_setting["inputs"] = inputs if inputs is not None else self._record_setting["inputs"]
        self._record_setting["state"] = state if state is not None else self._record_setting["state"]
        self._record_setting["output"] = output if output is not None else self._record_setting["output"]
        self._max_records = max_records if max_records is not None else self._max_records

    def warmup(self, graph_state: base.GraphState, device_step: jax.Device = None, device_dist: jax.Device = None, jit_step: bool = True, profile: bool = False):
        """ Warmup the node by running it once to compile the step function and sample the delay distribution.

        :param graph_state: The graph state.
        :param device_step: The device on which the step function should be compiled.
        :param device_dist: The device on which the delay distribution should be compiled.
        :param jit_step: Whether to jit the step function.
        :param profile: Whether to profile the step function.
        :return:
        """

        device_step = device_step if device_step is not None else jax.devices('cpu')[0]
        device_dist = device_dist if device_dist is not None else jax.devices('cpu')[0]

        # Jit step function
        ss = graph_state.step_state[self.node.name]
        if jit_step:
            self.async_step = jax.jit(self.async_step, device=device_step)

            # AOT compilation
            with utils.timer(f"{self.node.name}.step | pre-compile ", log_level=self.node.log_level):
                self.async_step = self.async_step.lower(ss).compile()

        # Time profile the pre-compiled function
        if profile:
            with utils.timer(f"{self.node.name}.step | time-profile", log_level=self.node.log_level, repeat=10):
                for _ in range(10):
                    ss, o = self.async_step(ss)

        # Warms-up jitted functions in the output (i.e. pre-compiles)
        self._jit_reset = jax.jit(self.node.delay_dist.reset, device=device_dist)
        self._jit_sample = jax.jit(self.node.delay_dist.sample_pure, static_argnums=1, device=device_dist)
        dist_state = self._jit_reset(rnd.PRNGKey(0))
        new_dist_state, samples = self._jit_sample(dist_state, shape=self._num_buffer)

        # Warms-up jitted functions in the inputs (i.e. pre-compiles)
        [i.warmup(graph_state, device_step=device_step, device_dist=device_dist) for i in self.inputs.values()]

        # Warmup random number generators
        _ = [r for r in rnd.split(graph_state.rng[self.node.name], num=len(self.node.inputs))]

        # Warmup phase
        _ = float(self.node.phase)

        # Wait for the results to be ready
        samples.block_until_ready()  # Only to trigger jit compilation
        self._has_warmed_up = True

    def _reset(self, graph_state: base.GraphState, clock: Clock, real_time_factor: float):
        assert self._state in [Async.STOPPED, Async.READY], f"{self.node.name} must first be stopped, before it can be reset"
        assert (real_time_factor > 0 or clock == Clock.SIMULATED), "Real time factor must be greater than zero if clock is not simulated"

        # Determine whether to run synchronously or asynchronously
        # self._sync = SYNC if clock == SIMULATED else ASYNC
        # assert not (
        #     clock in [WALL_CLOCK] and self._sync in [SYNC]
        # ), "You can only simulate synchronously, if the clock=`SIMULATED`."

        # Get blocking inputs
        num_blocking_inputs = len([i for i in self.node.inputs.values() if i.blocking])
        if clock == Clock.SIMULATED and self.node.advance and num_blocking_inputs == 0:
            # A node without inputs cannot run with advance=True in SIMULATED mode with zero simulated computation delay,
            # because it would mean this node would run infinitely fast, which deadlocks downstream nodes that depend
            # on it asynchronously).
            if not self.node.delay_dist.mean() > 0.0:
                raise ValueError(f"Node `{self.node.name}` cannot run with advance=True in SIMULATED mode with zero simulated computation delay and zero blocking connections. "
                                 f"This would mean that this node would run infinitely fast, which deadlocks connected nodes with blocking=False.")
            else:
                # We could overwrite advance to False, and run the node asynchronously according to the simulated computation delay?
                # For now, we just raise an error and ask the user to set advance to False.
                raise NotImplementedError(f"Node `{self.node.name}` is running with advance=True in SIMULATED mode with non-zero simulated computation delay and zero blocking connections. "
                                          f"This is not yet supported. Please set advance to False.")

        # Warmup the node
        if not self._has_warmed_up:
            raise RuntimeError(f"Node `{self.node.name}` must first be warmed up, before it can be reset. "
                               f"Call graph.warmup() first.")

        # Save run configuration
        self._clock = clock  #: Simulate timesteps
        self._real_time_factor = real_time_factor  #: Scaling of simulation speed w.r.t wall clock

        # Up the episode counter (must happen before resetting outputs & inputs)
        self._eps += 1

        # Reset every run
        self._tick = 0
        self._phase_scheduled = 0.0  #: Structural phase shift that the step scheduler takes into account
        self._phase = float(self.node.phase)
        self._record = None
        self._record_steps = None
        self._step_state = graph_state.step_state[self.node.name]

        # Log
        self._discarded = 0  # Number of discarded records after reaching self.max_records

        # Set starting ts
        self._ts_start = Future()  #: The starting timestamp of the episode.

        # Initialize empty queues
        self.q_tick = deque()
        self.q_ts_scheduled = deque()
        self.q_ts_end_prev = deque()
        self.q_ts_start = deque()
        self.q_rng_step = deque()

        # Get rng for delay sampling
        rng = self._step_state.rng
        rng = jnp.array(rng) if isinstance(rng, onp.ndarray) else rng  # Keys will differ for jax vs numpy

        # Reset output
        # NOTE: This is hacky because we reuse the seed.
        # However, changing the seed of the step_state would break the reproducibility between graphs (compiled, async).
        self._phase_output = self.node.phase_output
        self._dist_state = self._jit_reset(rng)
        self.q_sample = deque()

        # Reset all inputs and output
        rngs_in = rnd.split(rng, num=len(self.inputs))
        [i.reset(r, self._step_state.inputs[i.connection.input_name]) for r, i in zip(rngs_in, self.inputs.values())]

        # Set running state
        self._state = Async.READY
        self.log(self._state, log_level=LogLevel.DEBUG)

    def _startup(self, graph_state: base.GraphState, timeout: float = None) -> Future:
        assert self._state in [Async.READY], f"{self.node.name} must first be reset, before it can start running."

        def _starting() -> Union[bool, jax.Array]:
            res = self.node.startup(graph_state, timeout=timeout)

            # Set running state
            self._state = Async.READY_TO_START
            self.log(self._state, log_level=LogLevel.DEBUG)
            return res

        with self._lock:
            # Then, flip running state so that no more tasks can be scheduled
            # This means that
            self._state = Async.STARTING
            self.log(self._state, log_level=LogLevel.DEBUG)

            # First, submit _stopping task
            f = self._submit(_starting)
        return f

    def _stop(self, timeout: Optional[float] = None) -> Future:
        # Pass here, if we are not running
        if self._state not in [Async.RUNNING]:
            self.log("", f"{self.node.name} is not running, so it cannot be stopped.", log_level=LogLevel.DEBUG)
            f = Future()
            f.set_result(None)
            return f
        assert self._state in [Async.RUNNING], f"Cannot stop, because {self.node.name} is currently not running."

        def _stopping():
            # Stop producing messages and communicate total number of sent messages
            # self.output.stop()

            # Stop all channels to receive all sent messages from their connected outputs
            [i.stop().result(timeout=timeout) for i in self.inputs.values()]

            # Run the stop function of the node
            success = self.node.stop()
            if not success:
                self.log("WARNING", f"Node {self.node.name} failed to stop.", log_level=LogLevel.WARN)

            # Record last step_state
            self._step_state = None  # Reset step_state

            # Unset _async_now for the wrapped node
            self.node._async_now = None

            # Set running state
            self._state = Async.STOPPED
            self.log(self._state, log_level=LogLevel.DEBUG)

        with self._lock:
            # Then, flip running state so that no more tasks can be scheduled
            # This means that
            self._state = Async.STOPPING
            self.log(self._state, log_level=LogLevel.DEBUG)

            # First, submit _stopping task
            f = self._submit(_stopping, stopping=True)
        return f

    def _start(self, start: float):
        assert self._state in [Async.READY_TO_START], f"{self.node.name} must first be ready to start (i.e. call ._startup), before it can start running."
        assert self._has_warmed_up, f"{self.node.name} must first be warmed up, before it can start running."

        # Set running state
        self._state = Async.RUNNING
        self.log(self._state, log_level=LogLevel.DEBUG)

        # Create logging record
        self._set_ts_start(start)
        self._record = base.NodeRecord(
            info=self.node.info,
            clock=self._clock,
            real_time_factor=self._real_time_factor,
            ts_start=start,
            # rng_dist=self._dist_state.rng,
            params=self._step_state.params if self.record_setting["params"] else None,
            inputs=None,  # added at the end
            steps=None,  # added at the end
        )
        self._record_steps = []  # Not a deque, because we do not want to overwrite the first steps when we overflow

        # Start all inputs and output
        [i.start() for i in self.inputs.values()]

        # Set first last_output_ts equal to phase (as if we just finished our previous output).
        self.q_ts_end_prev.append(0.0)

        # NOTE: Deadlocks may occur when num_tokens is chosen too low for cyclical graphs, where a low rate node
        #       depends (blocking) on a high rate node, while the high rate node depends (skipped, non-blocking)
        #       on the low rate node. In that case, the num_token of the high-rate node must be at least
        #       (probably more) the rate multiple + 1. May be larger if there are delays, etc...
        # Queue first two ticks (so that output_ts runs ahead of message)
        # The number of tokens > 1 determines "how far" into the future the
        # output timestamps are simulated when clock=simulated.
        num_tokens = 10  # todo: find non-heuristic solution. Add tokens adaptively based on requests from downstream nodes?
        self.q_tick.extend((True,) * num_tokens)

        # Push scheduled ts
        # todo: CONTINUE HERE
        _f = self._submit(self.push_scheduled_ts)
        return _f

    def _async_step(self, step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
        """This function is internally called, and should not be jited.
        If a node is the supervisor, this function is overwritten by the _Synchronizer._step

        """
        # with jax.log_compiles():
        #     same_structure(self._step_state, step_state, tag=self.name)
        #     new_step_state, output = self.step(step_state)  # Synchronizer or Node
        # Run the step function of the node (async_step may be jitted)
        new_step_state, output = self.async_step(step_state)

        # Block until output is ready
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else True, output)
        return self.async_step(step_state)

    def async_step(self, step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
        """Async step function that is called when running asynchronously.
        This function can be overridden/wrapped (e.g. jit) without affecting node.step() directly.

        This can be beneficial when settings (Clock.SIMULATED, Clock.WALL_CLOCK) require different compilation settings.
        For example, compiling for CPU in real-time, and for GPU while simulating.
        """
        # Run the step function of the node
        new_step_state, output = self.node.step(step_state)

        # Update step_state (increment sequence number)
        if new_step_state is not None:
            new_step_state = new_step_state.replace(seq=new_step_state.seq + 1)
        return new_step_state, output

    def push_scheduled_ts(self):
        # Only run if there are elements in q_tick
        has_tick = len(self.q_tick) > 0
        if has_tick:
            # Remove token from tick queue (not used)
            _ = self.q_tick.popleft()

            # Determine tick and increment
            tick = self._tick
            self._tick += 1

            # Calculate scheduled ts
            # Is unaffected by scheduling delays, i.e. assumes the zero-delay situation.
            scheduled_ts = round(tick / self.node.rate + self.phase, 6)

            # Log
            self.log("push_scheduled_ts", f"tick={tick} | scheduled_ts={scheduled_ts: .2f}", log_level=LogLevel.DEBUG)

            # Queue expected next step ts and wait for blocking delays to be determined
            self.q_ts_scheduled.append((tick, scheduled_ts))
            self.push_phase_shift()

            # Push next step ts event to blocking connections (does not throttle)
            for i in self.inputs.values():
                if not i.connection.blocking:
                    continue
                i.q_ts_next_step.append((tick, scheduled_ts))

                # Push expect (must be called from input thread)
                i._submit(i.push_expected_blocking)

    def push_phase_shift(self):
        # If all blocking delays are known, and we know the expected next step timestamp
        has_all_ts_max = all([len(i.q_ts_max) > 0 for i in self.inputs.values() if i.connection.blocking])
        has_scheduled_ts = len(self.q_ts_scheduled) > 0
        has_last_output_ts = len(self.q_ts_end_prev) > 0
        if has_scheduled_ts and has_last_output_ts and has_all_ts_max:
            self.log("push_phase_shift", log_level=LogLevel.DEBUG)

            # Grab blocking delays from queues and calculate max delay
            ts_max = [i.q_ts_max.popleft() for i in self.inputs.values() if i.connection.blocking]
            ts_max = max(ts_max) if len(ts_max) > 0 else 0.0

            # Grab next scheduled step ts (without considering phase_scheduling shift)
            tick, ts_scheduled = self.q_ts_scheduled.popleft()

            # Grab previous output ts
            ts_end_prev = self.q_ts_end_prev.popleft()

            # Calculate sources of phase shift
            only_blocking = self.node.advance and all(i.connection.blocking for i in self.inputs.values())
            phase_inputs = ts_max - ts_scheduled
            phase_last = ts_end_prev - ts_scheduled
            phase_scheduled = self._phase_scheduled

            # Calculate phase shift
            # If only blocking connections, phase is not determined by phase_scheduled
            phase = max(phase_inputs, phase_last) if only_blocking else max(phase_inputs, phase_last, phase_scheduled)

            # Update structural scheduling phase shift
            if self.node.scheduling in [Scheduling.FREQUENCY]:
                self._phase_scheduled += max(0, phase_last - phase_scheduled)
            else:  # self.scheduling in [PHASE]
                self._phase_scheduled = 0.0

            # Calculate starting timestamp for the step call
            ts_start = ts_scheduled + phase

            # Sample delay if we simulate the clock
            delay = None  # Overwritten in the next block if we simulate the clock
            if self._clock in [Clock.SIMULATED]:
                if len(self.q_sample) == 0:  # Generate samples batch-wise
                    self._dist_state, samples = self._jit_sample(self._dist_state, shape=self._num_buffer)
                    self.q_sample.extend(tuple(samples.tolist()))

                # Sample delay
                delay = float(self.q_sample.popleft())

            # Create step record
            record_step = base.AsyncStepRecord(
                eps=self._eps,
                seq=tick,
                ts_scheduled=ts_scheduled,
                ts_max=ts_max,
                ts_start=ts_start,  # Overwritten in .push_step()
                ts_end_prev=ts_end_prev,
                ts_end=None,  # Filled in .push_step()
                phase=phase,
                phase_scheduled=phase_scheduled,
                phase_inputs=phase_inputs,
                phase_last=phase_last,
                sent=None,  # Filled in .push_step()
                delay=None,  # Filled in .push_step()
                phase_overwrite=0.,  # May be overwritten in _step --> see push_step
                rng=None,  # Filled in .push_step()
                inputs=None,  # Filled in .push_step()
                state=None,  # Filled in .push_step()
                output=None,  # Filled in .push_step()
            )
            self.q_ts_start.append((tick, ts_start, delay, record_step))

            # Predetermine output timestamp when we simulate the clock
            if self._clock in [Clock.SIMULATED]:
                # Determine output timestamp
                ts_output = ts_start + delay
                header = base.Header(eps=self._eps, seq=tick, ts=ts_output)
                if self._state in [Async.RUNNING]:
                    # Push message to inputs
                    self.log("push_ts_output", ts_output, log_level=LogLevel.DEBUG)
                    [i._submit(i.push_ts_input, ts_output, header) for i in self.outputs.values()]  # todo: check!

                # todo: Somehow, num_tokens can be lowered if we would sleep here (because push_ts_output runs before push_scheduled_ts).
                #  time.sleep(1.0)

                # Add previous output timestamp to queue
                self.q_ts_end_prev.append(ts_output)

                # Simulate output timestamps into the future
                # If we use the wall-clock, ts_end_prev is queued after the step in push_step
                _f = self._submit(self.push_scheduled_ts)

            # Only throttle if we have non-blocking connections
            if any(not i.connection.blocking for i in self.inputs.values()) or not self.node.advance:
                self.throttle(ts_start)  # todo: This also throttles when running synced. Correct?

            # Push for step (will never trigger here if there are non-blocking connections).
            self.push_step()

            # Push next step timestamp to non-blocking connections
            for i in self.inputs.values():
                if i.connection.blocking:
                    continue
                i.q_ts_next_step.append((tick, ts_start))

                # Push expect (must be called from input thread)
                i._submit(i.push_expected_nonblocking)

    def push_step(self):
        has_grouped = all([len(i.q_grouped) > 0 for i in self.inputs.values()])
        has_ts_step = len(self.q_ts_start) > 0
        if has_ts_step and has_grouped:
            self.log("push_step", log_level=LogLevel.DEBUG)

            # Grab next expected step ts and step record
            tick, ts_start_sc, delay_sc, record_step = self.q_ts_start.popleft()

            # Grab grouped msgs
            inputs = {}
            for input_name, i in self.inputs.items():
                input_state = self._step_state.inputs[input_name]
                grouped = i.q_grouped.popleft()
                for seq, ts_sent, ts_recv, msg in grouped:
                    # todo: The args to _jit_update_input_state may reside on different devices. This may cause issues.
                    input_state = i._jit_update_input_state(input_state, seq, ts_sent, ts_recv, msg)
                inputs[input_name] = input_state

            # Update StepState with grouped messages
            # todo: have a single buffer for step_state used for both in and out
            tick_promoted = onp.array(tick).astype(self._step_state.seq.dtype)
            ts_start_sc_promoted = onp.array(ts_start_sc).astype(self._step_state.ts.dtype)
            step_state = self._step_state.replace(seq=tick_promoted, ts=ts_start_sc_promoted, inputs=FrozenDict(inputs))

            # Record before running step
            record_step = record_step.replace(
                rng=step_state.rng if self.record_setting["rng"] else None,
                inputs=inputs if self.record_setting["inputs"] else None,
                state=step_state.state if self.record_setting["state"] else None,
            )

            # Run step and get new state and output
            new_step_state, output = self._async_step(step_state)

            # Get new ts_start_sc_promoted
            new_ts_start_sc_promoted = new_step_state.ts if new_step_state is not None else ts_start_sc_promoted

            # Log output
            if isinstance(output, _SkippedSteps):
                record_step = record_step.replace(
                    output=None,  # Log None if we are stopping/resetting with the supervisor
                )
            else:
                record_step = record_step.replace(
                    output=output if self.record_setting["output"] else None,
                )

            # Update step_state (sequence number is incremented in ._step())
            if new_step_state is not None:
                self._step_state = new_step_state

            # Determine output timestamp
            if self._clock in [Clock.SIMULATED]:
                assert delay_sc is not None
                ts_end_sc = ts_start_sc + delay_sc
                phase_overwrite = 0.0
                ts_start_sc = ts_start_sc
            else:
                assert delay_sc is None
                ts_end_sc = self.now()
                # ts_step_sc (i.e. step_state.ts) may be overwritten in the step function (i.e. to adjust to later time when sensor data was taken).
                # Therefore, we use the potentially overwritten step_state.ts to calculate the delay.
                new_start_ts_sc = ts_start_sc if ts_start_sc_promoted == new_ts_start_sc_promoted else float(new_ts_start_sc_promoted)
                if new_start_ts_sc < ts_start_sc:
                    msg = ("Did you overwrite `step_state.ts` in the step function? Make sure it's not smaller than the original `step_state.ts`. "
                           "Are the clocks producing the adjusted step_state.ts and the original step_state.ts consistent?")
                    self.log("timestamps", msg, log_level=LogLevel.ERROR)
                    raise ValueError(msg)
                delay_sc = ts_end_sc - new_start_ts_sc
                if delay_sc <= 0:
                    msg = ("Did you overwrite `step_state.ts` in the step function? Make sure it does not exceed the current time (i.e. `self.now()`)"
                        "Are the clocks producing the adjusted step_state.ts and the original step_state.ts consistent?")
                    self.log("timestamps", msg, log_level=LogLevel.ERROR)
                    raise ValueError(msg)

                # Re-calculate phase overwrite and ts_start_sc
                phase_overwrite = new_start_ts_sc - ts_start_sc
                ts_start_sc = new_start_ts_sc

                # Add previous output timestamp to queue
                # If we simulate the clock, ts_end_prev is already queued in push_phase_shift
                self.q_ts_end_prev.append(ts_end_sc)

            # Throttle to timestamp
            self.throttle(ts_end_sc)

            # Create header with timing information on output
            header = base.Header(eps=self._eps, seq=tick, ts=ts_end_sc)

            # Log sent times
            record_step = record_step.replace(
                ts_start=ts_start_sc,
                ts_end=ts_end_sc,
                sent=header,
                delay=delay_sc,
                phase_overwrite=phase_overwrite,
            )

            # Push output
            if not isinstance(output, _SkippedSteps) and self._state in [Async.RUNNING]:  # Agent returns _SkippedSteps when we are stopping/resetting.
                [i._submit(i.push_input, output, header) for i in self.outputs.values()]

            # Add step record
            if len(self._record_steps) < self.max_records:
                # The supervisor produces a _SkippedSteps in-place of the output when we reset, resulting in a _SkippedSteps in the output.
                # To ensure that the record has a consistent shape, we replace the _SkippedSteps with a None tree.
                if len(self._record_steps) > 0 and (self._record_steps[-1].output is None) != (record_step.output is None):
                    assert isinstance(output, _SkippedSteps), "Output should be _SkippedSteps when we are stopping/resetting."
                    if output.skipped_steps == 1:  # Only append the final step we are stopping/resetting, not the ones that follow.
                        record_step = record_step.replace(
                            output=jax.tree_util.tree_map(lambda x: None, self._record_steps[0].output)
                        )  # Should only happen for the last step of the supervisor.
                        self._record_steps.append(record_step)
                else:
                    self._record_steps.append(record_step)
            elif self._discarded == 0:
                self.log(
                    "recording",
                    "Reached max number of records (timings, outputs, step_state). So no longer recording.",
                    log_level=LogLevel.WARN
                )
                self._discarded += 1
            else:
                self._discarded += 1

            # Only schedule next step if we are running
            if self._state in [Async.RUNNING]:
                # Add token to tick queue (ticks are incremented in push_scheduled_ts function)
                self.q_tick.append(True)

                # Schedule next step (does not consider scheduling shifts)
                _f = self._submit(self.push_scheduled_ts)


class _AsyncConnectionWrapper:
    def __init__(self, connection: Connection, output_node: "_AsyncNodeWrapper", input_node: "_AsyncNodeWrapper"):
        self.connection = connection
        self.output_node = output_node
        self.input_node = input_node

        self._state = Async.STOPPED

        # Jit function (call self.warmup() to pre-compile)
        self._num_buffer = 50
        self._jit_update_input_state = None
        self._jit_reset = None
        self._jit_sample = None
        self._has_warmed_up = False

        # Executor
        node_name = self.connection.input_node if isinstance(self.connection.input_node, str) else self.connection.input_node.name  # todo: str only if pickled (can be removed if not pickling)
        output_name = self.connection.output_node if isinstance(self.connection.output_node, str) else self.connection.output_node.name  # todo: str only if pickled (can be removed if not pickling)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{node_name}/{output_name}")
        self._q_task: Deque[Tuple[Future, Callable, Any, Any]] = deque(maxlen=10)
        self._lock = RLock()

        # Reset every time
        self._tick = None
        self._input_state = None
        self._record: base.InputRecord = None
        self._record_messages: List[base.MessageRecord] = None  # Can be more than max_records if the output is high
        self._phase = None
        self._phase_dist = None
        self._prev_recv_sc = None
        self._dist_state: base.DelayDistribution = None
        self.q_msgs: Deque[Tuple[base.MessageRecord, Any]] = None
        self.q_ts_input: Deque[Tuple[int, float]] = None
        self.q_ts_max: Deque[float] = None
        self.q_zip_delay: Deque[float] = None
        self.q_zip_msgs: Deque[Tuple[Any, base.Header]] = None
        self.q_expected_select: Deque[Tuple[float, int]] = None
        self.q_expected_ts_max: Deque[int] = None
        self.q_grouped: Deque[Tuple[int, float, float, Any]] = None
        self.q_ts_next_step: Deque[Tuple[int, float]] = None
        self.q_sample: Deque = None

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def log_level(self):
        return self.connection.input_node.log_level

    def log(self, id: Union[str, Async], value: Optional[Any] = None, log_level: Optional[int] = None):
        if not utils.NODE_LOGGING_ENABLED:
            return
        log_level = self.connection.input_node.log_level if log_level is None else log_level
        color = self.connection.input_node.log_color
        utils.log(f"{self.connection.output_node.name}/{self.connection.input_node.name}", color, min(log_level, self.log_level), id, value)

    def _submit(self, fn, *args, stopping: bool = False, **kwargs):
        with self._lock:
            if self._state in [Async.READY, Async.RUNNING] or stopping:
                f = self._executor.submit(fn, *args, **kwargs)
                self._q_task.append((f, fn, args, kwargs))
                f.add_done_callback(self._done_callback)
            else:
                self.log("SKIPPED", fn.__name__, log_level=LogLevel.DEBUG)
                f = Future()
                f.cancel()
        return f

    def _done_callback(self, f: Future):
        e = f.exception()
        if e is not None and e is not CancelledError:
            error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            utils.log(f"{self.connection.output_node.name}/{self.connection.input_node.name}", "red", LogLevel.ERROR, "ERROR", error_msg)

    def get_record(self, last_seq_in: int) -> base.InputRecord:
        # If the record is incomplete, warn the user that the record is incomplete.
        if self._record is None:
            raise RuntimeError(f"No record has been created yet for the input {self.connection.output_node.name} ({self.connection.input_name}) of {self.connection.input_node.name}.")

        # Add the steps to the record
        if self._record.messages is None:
            # Filter received messages meant for steps that were not yet run.
            messages = list(filter(lambda x: x.seq_in <= last_seq_in, self._record_messages))
            # Convert to numpy array
            messages = jax.tree_map(lambda *x: onp.array(x), *messages)
            self._record = self._record.replace(messages=messages)
        return self._record

    def warmup(self, graph_state: base.GraphState, device_step, device_dist):
        # Warmup input update
        self._jit_update_input_state = jax.jit(update_input_state, device=device_step)
        i = graph_state.inputs[self.connection.input_node.name][self.connection.input_name]
        new_i = self._jit_update_input_state(i, 0, 0., 0., i[0].data)

        # Warms-up jitted functions in the output (i.e. pre-compiles)
        self._jit_reset = jax.jit(i.delay_dist.reset, device=device_dist)
        self._jit_sample = jax.jit(i.delay_dist.sample_pure, static_argnums=1, device=device_dist)
        dist_state = self._jit_reset(rnd.PRNGKey(0))
        new_dist_state, samples = self._jit_sample(dist_state, shape=self._num_buffer)

        # Warmup phase
        _ = float(self.connection.phase)

        # Wait for the results to be ready
        samples.block_until_ready()  # Only to trigger jit compilation
        if isinstance(new_i.seq, jax.Array):
            new_i.seq.block_until_ready()

        self._has_warmed_up = True

    def reset(self, rng: jnp.ndarray, input_state: base.InputState):
        assert self._state in [Async.STOPPED, Async.READY], f"Input `{self.connection.output_node.name}` (`{self.connection.input_name}`) of node `{self.connection.input_node.name}` must first be stopped, before it can be reset."

        # Empty queues
        self._tick = 0
        self._input_state = input_state
        self._phase = float(self.connection.phase)
        self._record = None
        self._record_messages = None  # Can be more than max_records if the output is high
        self._prev_recv_sc = 0.0  # Ensures the FIFO property for incoming messages.
        self._dist_state = self._jit_reset(rng)
        self.q_msgs = deque()
        self.q_ts_input = deque()
        self.q_zip_delay = deque()
        self.q_zip_msgs = deque()
        self.q_ts_max = deque()
        self.q_expected_select = deque()
        self.q_expected_ts_max = deque()
        self.q_grouped = deque()
        self.q_ts_next_step = deque()
        self.q_sample = deque()

        # Set running state
        self._state = Async.READY
        self.log(self._state, log_level=LogLevel.DEBUG)

    def stop(self) -> Future:
        assert self._state in [Async.RUNNING], f"Input `{self.connection.output_node.name}` (`{self.connection.input_name}`) of node `{self.connection.input_node.name}` must be running in order to stop."

        def _stopping():
            # Set running state
            self._state = Async.STOPPED
            self.log(self._state, log_level=LogLevel.DEBUG)

        with self._lock:
            # Then, flip running state so that no more tasks can be scheduled
            self._state = Async.STOPPING
            self.log(self._state, log_level=LogLevel.DEBUG)

            # First, submit _stopping task
            f = self._submit(_stopping, stopping=True)
        return f

    def start(self):
        assert self._state in [Async.READY], f"Input `{self.connection.output_node.name}` (`{self.connection.input_name}`) of node `{self.connection.input_node.name}` must first be reset, before it can start running."
        assert self._has_warmed_up, f"Input `{self.connection.output_node.name}` (`{self.connection.input_name}`) of node `{self.connection.input_node.name}` must first be warmed up, before it can start."

        # Set running state
        self._state = Async.RUNNING
        self.log(self._state, log_level=LogLevel.DEBUG)

        # Store running configuration
        self._record = base.InputRecord(
            info=self.connection.info,
            # rng_dist=self._dist_state.rng,
            messages=None,  # added at the end
        )
        self._record_messages = []  # Can be more than max_records if the output is high

    def push_expected_nonblocking(self):
        assert not self.connection.blocking, "This function should only be called for non-blocking inputs."
        has_ts_next_step = len(self.q_ts_next_step) > 0
        has_ts_inputs = self.input_node._clock in [Clock.WALL_CLOCK] or len(self.q_ts_input) > 0
        if has_ts_next_step and has_ts_inputs:
            tick, ts_step = self.q_ts_next_step[0]
            has_ts_in_future = self.input_node._clock in [Clock.WALL_CLOCK] or any(ts > ts_step for seq, ts in self.q_ts_input)
            if has_ts_in_future:
                # Pop elements from queues
                # blocking connections:= scheduled_ts  (ignores any phase shifts, i.e. "original" schedule).
                # non-blocking:= ts_step (includes phase shifts due to blocking, scheduling shifts, comp. delays)
                tick, ts_step = self.q_ts_next_step.popleft()

                # Determine number of entries where ts > ts_step
                num_msgs = 0
                if self.connection.jitter in [Jitter.BUFFER]:
                    # Uses input phase and sequence number to determine expected timestamp instead of the actual timestamp.
                    phase = self.phase
                    for seq, ts_recv in self.q_ts_input:
                        ts_expected = seq / self.connection.output_node.rate + phase
                        if ts_expected > ts_step:
                            break
                        if ts_recv > ts_step:
                            break
                        num_msgs += 1
                else:  # self.jitter in [LATEST]:
                    # Simply uses the latest messages (and clears entire buffer until ts_step).
                    for seq, ts in self.q_ts_input:
                        if ts > ts_step or (self.connection.skip and ts == ts_step):
                            break
                        num_msgs += 1

                # Clear q_ts_input until ts_inputs >= ts_step
                [self.q_ts_input.popleft() for _ in range(num_msgs)]

                # Log
                self.log("push_exp_nonblocking", f"ts_step={ts_step: .2f} | num_msgs={num_msgs}", log_level=LogLevel.DEBUG)

                # Push selection
                self.q_expected_select.append((ts_step, num_msgs))
                self.push_selection()

    def push_expected_blocking(self):
        assert self.connection.blocking, "This function should only be called for blocking inputs."
        has_ts_next_step = len(self.q_ts_next_step) > 0
        if has_ts_next_step:
            # Pop elements from queues
            # blocking connections:= ts_next_step == scheduled_ts  (ignores any phase shifts, i.e. "original" schedule).
            # non-blocking:= ts_next_step == ts_step (includes phase shifts due to blocking, scheduling shifts, comp. delays)
            N_node, scheduled_ts = self.q_ts_next_step.popleft()

            skip = self.connection.skip
            phase_node, phase_in = round(self.connection.input_node.phase, 6), round(self.connection.output_node.phase, 6)
            rate_node, rate_in = self.connection.input_node.rate, self.connection.output_node.rate
            dt_node, dt_in = 1 / rate_node, 1 / rate_in
            t_high = dt_node * N_node + phase_node
            t_low = dt_node * (N_node - 1) + phase_node
            t_high = round(t_high, 6)
            t_low = round(t_low, 6)

            # Determine starting t_in
            # todo: find numerically stable (and fast) implementation.
            i = int((t_low - phase_in) // dt_in) if N_node > 0 else 0

            text_t = []
            t = round(i / rate_in + phase_in, 6)
            while not t > t_high:
                flag = 0
                if not t < phase_in:
                    if N_node == 0:
                        if t <= t_low and not skip:
                            text_t.append(str(t))
                            flag += 1
                        elif t < t_low and skip:
                            text_t.append(str(t))
                            flag += 1
                    if t_low < t <= t_high and not skip:
                        text_t.append(str(t))
                        flag += 1
                    elif t_low <= t < t_high and skip:
                        text_t.append(str(t))
                        flag += 1
                assert flag < 2
                i += 1
                t = round(i / rate_in + phase_in, 6)

            num_msgs = len(text_t)

            # Log
            self.log("push_exp_blocking", f"scheduled_ts={scheduled_ts: .2f} | num_msgs={num_msgs}", log_level=LogLevel.DEBUG)

            # Push ts max
            self.q_expected_ts_max.append(num_msgs)
            self.push_ts_max()

            # Push selection
            self.q_expected_select.append((scheduled_ts, num_msgs))
            self.push_selection()

    def push_ts_max(self):
        # Only called by blocking connections
        has_msgs = len(self.q_expected_ts_max) > 0 and self.q_expected_ts_max[0] <= len(self.q_ts_input)
        if has_msgs:
            num_msgs = self.q_expected_ts_max.popleft()

            # Determine max timestamp of grouped message for blocking connection
            input_ts = [self.q_ts_input.popleft()[1] for _i in range(num_msgs)]
            ts_max = max([0.0] + input_ts)
            self.q_ts_max.append(ts_max)

            # Push push_phase_shift (must be called from node thread)
            self.input_node._submit(self.input_node.push_phase_shift)

    def push_ts_input(self, msg, header: base.Header):
        # WALL_CLOCK: called by input.push_input --> msg: actual message
        # SIMULATED: called by output.push_ts_output --> msg: ts_output
        # Skip if we are not running
        if self._state not in [Async.READY, Async.RUNNING]:
            self.log("push_ts_input (NOT RUNNING)", log_level=LogLevel.DEBUG)
            return
        # Skip if from a previous episode
        elif header.eps != self.input_node.eps:
            self.log("push_ts_input (PREV EPS)", log_level=LogLevel.DEBUG)
            return
        # Else, continue
        else:
            self.log("push_ts_input", log_level=LogLevel.DEBUG)

        # Determine sent timestamp
        seq, sent_sc = header.seq, header.ts

        # Determine input timestamp
        if self.input_node._clock in [Clock.SIMULATED]:
            # Sample delay
            if len(self.q_sample) == 0:  # Generate samples batch-wise if queue is empty
                self._dist_state, samples = self._jit_sample(self._dist_state, shape=self._num_buffer)
                self.q_sample.extend(tuple(samples.tolist()))
            delay = float(self.q_sample.popleft())  # Sample delay from queue
            # Enforce FIFO property
            recv_sc = round(max(sent_sc + delay, self._prev_recv_sc), 6)  # todo: 1e-9 required here?
            self._prev_recv_sc = recv_sc
        else:
            # This only happens when push_ts_input is called by push_input
            recv_sc = self.input_node.now()

        # Communication delay
        # IMPORTANT! delay_wc measures communication delay of output_ts instead of message.
        # Value of delay_wc is overwritten in push_input() when clock=wall-clock.
        delay_sc = recv_sc - sent_sc
        self.q_zip_delay.append(delay_sc)

        # Push zip to buffer messages
        self.push_zip()

        # Add phase to queue
        self.q_ts_input.append((seq, recv_sc))

        # Push event
        if self.connection.blocking:
            self.push_ts_max()
        else:
            self.push_expected_nonblocking()

    def push_input(self, msg: Any, header_sent: base.Header):
        # Skip if we are not running
        if self._state not in [Async.READY, Async.RUNNING]:
            self.log("push_input (NOT RUNNING)", log_level=LogLevel.DEBUG)
            return
        # Skip if from a previous episode
        elif header_sent.eps != self.input_node.eps:
            self.log("push_input (PREV EPS)", log_level=LogLevel.DEBUG)
            return
        # Else, continue
        else:
            self.log("push_input", log_level=LogLevel.DEBUG)

        # todo: add transform here
        # todo: add to input_state here?

        # Push ts_input when the clock is not simulated
        if self.input_node._clock in [Clock.WALL_CLOCK]:
            # This will queue delay (and call push_zip)
            self.push_ts_input(msg, header_sent)

        # Queue msg
        self.q_zip_msgs.append((msg, header_sent))

        # Push zip to buffer messages
        self.push_zip()

    def push_zip(self):
        has_msg = len(self.q_zip_msgs) > 0
        has_delay = len(self.q_zip_delay) > 0
        if has_msg and has_delay:
            msg, header_sent = self.q_zip_msgs.popleft()

            # Determine sent timestamp
            sent_sc = header_sent.ts

            # Determine the ts of the input message
            # If clock=wall-clock, call push_ts_input with header_sent, but overwrite recv_wc if clock=simulated
            if self.input_node._clock in [Clock.SIMULATED]:
                delay_sc = self.q_zip_delay.popleft()
                recv_sc = round(sent_sc + delay_sc, 6)
            else:
                # This will queue the delay
                delay_sc = self.q_zip_delay.popleft()
                recv_sc = sent_sc + delay_sc

            # Throttle to timestamp
            self.input_node.throttle(recv_sc)

            # Create message record
            record_msg = base.MessageRecord(
                seq_out=header_sent.seq,
                seq_in=None,  # Filled in .push_selection()
                ts_sent=sent_sc,
                ts_recv=recv_sc,
                delay=delay_sc,
            )

            # Add message to queue
            self.q_msgs.append((record_msg, msg))

            # See if we can prepare tuple for next step
            self.push_selection()

    def push_selection(self):
        has_expected = len(self.q_expected_select) > 0
        if has_expected:
            has_recv_all_expected = (len(self.q_msgs) >= self.q_expected_select[0][1])
            if has_recv_all_expected:
                ts_next_step, num_msgs = self.q_expected_select.popleft()
                log_msg = f"blocking={self.connection.blocking} | step_ts={ts_next_step: .2f} | num_msgs={num_msgs}"
                self.log("push_selection", log_msg, log_level=LogLevel.DEBUG)

                # Create record
                # todo: calculate probability of selection using modeled distribution.
                #  1. Assume scheduling delay to be constant, or....
                #  2. Assume zero scheduling delay --> probably easier.
                #  3. Integrate each delay distribution over past and future sampling times.

                # Determine tick and increment
                tick = self._tick  # Serves as seq_in for the grouped messages
                self._tick += 1

                # Group messages
                grouped: List[Tuple[int, float, float, Any]] = []
                for i in range(num_msgs):
                    record_msg, msg = self.q_msgs.popleft()
                    record_msg = record_msg.replace(seq_in=tick)  # Set seq_in

                    # Add to record
                    self._record_messages.append(record_msg)

                    # Push message to input_state
                    seq = record_msg.seq_out
                    ts_sent = record_msg.ts_sent
                    ts_recv = record_msg.ts_recv
                    grouped.append((seq, ts_sent, ts_recv, msg))

                # Add grouped message to queue
                self.q_grouped.append(grouped[-self.connection.window:])

                # Push step (must be called from node thread)
                self.input_node._submit(self.input_node.push_step)


def update_input_state(input_state: base.InputState, seq: int, ts_sent: float, ts_recv: float, data: Any) -> base.InputState:
    new_input_state = input_state.push(seq, ts_sent, ts_recv, data)
    return new_input_state


class _SkippedSteps:
    def __init__(self):
        self._skipped_steps = 0

    def increment(self):
        self._skipped_steps += 1

    @property
    def skipped_steps(self):
        return self._skipped_steps

    def __repr__(self):
        return f"<SkippedSteps: {self._skipped_steps}>"


class _Synchronizer:
    def __init__(self, supervisor: _AsyncNodeWrapper):
        self._supervisor = supervisor
        self._supervisor._async_step = self._async_step  # Redirect supervisor's _async_step to this function
        self._must_reset: bool
        self._f_act: Future
        self._f_obs: Future
        self._q_act: Deque[Future] = deque()
        self._q_obs: Deque[Future]

    @property
    def action(self) -> Deque[Future]:
        return self._q_act

    @property
    def observation(self) -> Deque[Future]:
        return self._q_obs

    def reset(self):
        self._skipped = _SkippedSteps()
        self._must_reset = False
        self._q_act: Deque[Future] = deque()
        self._q_obs: Deque[Future] = deque()
        self._f_obs = Future()
        self._q_obs.append(self._f_obs)

    def _async_step(self, step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
        """Should not be jitted due to side-effects."""
        self._f_act = Future()
        self._q_act.append(self._f_act)

        # Prepare new obs future
        _new_f_obs = Future()
        self._q_obs.append(_new_f_obs)

        # Set observations as future result
        # print(f"[SET] _step: seq={step_state.seq}, ts={step_state.ts:.2f}")
        self._f_obs.set_result(step_state)
        self._f_obs = _new_f_obs

        # Wait for action future's result to be set with action
        if not self._must_reset:
            try:
                step_state, output = self._f_act.result()
                # print(f"[GET] _step: seq={step_state.seq}, ts={step_state.ts:.2f}")
                self._q_act.popleft()
                return step_state, output
            except CancelledError:  # If cancelled is None, we are going to reset
                self._q_act.popleft()
                self._must_reset = True
        self._skipped.increment()  # Increment skipped steps
        return None, self._skipped  # Do not return anything if we must reset


class AsyncGraph:
    def __init__(self,
                 nodes: Dict[str, BaseNode],
                 supervisor: BaseNode,
                 clock: Clock = Clock.WALL_CLOCK,
                 real_time_factor: Union[float, int] = RealTimeFactor.REAL_TIME,
                 ):
        self.nodes = nodes
        self.nodes[supervisor.name] = supervisor
        self.nodes_excl_supervisor = {k: v for k, v in nodes.items() if v.name != supervisor.name}
        self.supervisor = supervisor
        self.clock = clock
        self.real_time_factor = real_time_factor

        # Wrap nodes and connections
        self._async_nodes: Dict[str, _AsyncNodeWrapper] = {k: _AsyncNodeWrapper(v) for k, v in nodes.items()}
        for node in self._async_nodes.values():
            node.wrap_connections(self._async_nodes)
        self._synchronizer = _Synchronizer(self._async_nodes[supervisor.name])
        self._initial_step = True

    @property
    def max_eps(self):
        """The maximum number of episodes."""
        return 1

    @property
    def max_steps(self):
        """The maximum number of steps."""
        return jnp.inf

    def init(
        self,
        rng: jax.typing.ArrayLike = None,
        params: Dict[str, base.Params] = None,
        order: Tuple[str, ...] = None,
    ):
        """
        Initializes the graph state with optional parameters for RNG and step states.

        Nodes are initialized in a specified order, with the option to override params.
        Useful for setting up the graph state before running the graph with .run, .rollout, or .reset.

        :param rng: Random number generator seed or state.
        :param params: Predefined params for (a subset of) the nodes.
        :param order: The order in which nodes are initialized.
        :return: The initialized graph state.
        """
        # Determine init order. If name not in order, add it to the end
        order = tuple() if order is None else order
        order = list(order)
        for name in [self.supervisor.name] + list(self.nodes_excl_supervisor.keys()):
            if name not in order:
                order.append(name)

        # Prepare random number generators
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rng_step, rng_params, rng_state, rng_inputs = jax.random.split(rng, num=4)

        # Determine preset params
        params = params if params is not None else {}
        params = params.unfreeze() if isinstance(params, FrozenDict) else params
        params = {k: v for k, v in params.items()}  # Copy params

        # Initialize graph state
        rngs_step = FrozenDict({k: _rng for k, _rng in zip(order, jax.random.split(rng_step, num=len(order)))})
        seq = FrozenDict({k: onp.int32(0) for k in order})
        ts = FrozenDict({k: onp.float32(0.0) for k in order})
        state = {}
        inputs = {}
        graph_state = base.GraphState(eps=onp.int32(0), rng=rngs_step, seq=seq, ts=ts, params=params, state=state, inputs=inputs)

        # Initialize params
        rngs = jax.random.split(rng_params, num=len(order))
        for rng, name in zip(rngs, order):
            params[name] = params.get(name, self.nodes[name].init_params(rng, graph_state))

        # Initialize state
        rngs = jax.random.split(rng_state, num=len(order))
        for rng, name in zip(rngs, order):
            state[name] = self.nodes[name].init_state(rng, graph_state)

        # Initialize inputs
        rngs = jax.random.split(rng_inputs, num=len(order))
        for rng, name in zip(rngs, order):
            inputs[name] = self.nodes[name].init_inputs(rng, graph_state)

        # Replace params, state, and inputs in graph state with immutable versions
        new_gs = graph_state.replace(params=FrozenDict(params), state=FrozenDict(state), inputs=FrozenDict(inputs))
        return new_gs

    def set_record_settings(self,
                            params: Union[Dict[str, bool], bool] = None,
                            rng: Union[Dict[str, bool], bool] = None,
                            inputs: Union[Dict[str, bool], bool] = None,
                            state: Union[Dict[str, bool], bool] = None,
                            output: Union[Dict[str, bool], bool] = None,
                            max_records: Union[Dict[str, int], int] = None,
                        ):
        """Sets the record settings for the nodes in the graph."""
        params = params if params is not None else {}
        rng = rng if rng is not None else {}
        inputs = inputs if inputs is not None else {}
        state = state if state is not None else {}
        output = output if output is not None else {}
        max_records = max_records if max_records is not None else {}
        params = params if isinstance(params, dict) else {k: params for k in self._async_nodes.keys()}
        rng = rng if isinstance(rng, dict) else {k: rng for k in self._async_nodes.keys()}
        inputs = inputs if isinstance(inputs, dict) else {k: inputs for k in self._async_nodes.keys()}
        state = state if isinstance(state, dict) else {k: state for k in self._async_nodes.keys()}
        output = output if isinstance(output, dict) else {k: output for k in self._async_nodes.keys()}
        max_records = max_records if isinstance(max_records, dict) else {k: max_records for k in self._async_nodes.keys()}
        for name, node in self._async_nodes.items():
            node.set_record_settings(
                params=params.get(name, None),
                rng=rng.get(name, None),
                inputs=inputs.get(name, None),
                state=state.get(name, None),
                output=output.get(name, None),
                max_records=max_records.get(name, None),
            )

    def warmup(self, graph_state: base.GraphState,
               device_step: Union[Dict[str, jax.Device], jax.Device] = None,
               device_dist: Union[Dict[str, jax.Device], jax.Device] = None,
               jit_step: Union[Dict[str, bool], bool] = True,
               profile: Union[Dict[str, bool], bool] = False,
               ):
        """Ahead-of-time compilation of step and I/O functions to avoid latency at runtime.

        :param graph_state: The graph state that is expected to be used during runtime.
        :param device_step: The device to compile the step functions on. It's also the device used to prepare the input states.
                            If None, the default device is used.
        :param device_dist: The device to compile the sampling of the delay distribution functions on. If None, the default device is used.
                             Only relevant when using a simulated clock.
        :param jit_step: Whether to compile the step functions with JIT. If True, the step functions are compiled with JIT.
                         Step functions with jit are faster, but may not have side-effects by default.
                         Either wrap the side-effecting code in a jax callback wrapper, or set jit=False for those nodes.
                         See https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html for more info.
        :param profile: Whether to compile the step functions with time profiling. If True, the step functions are compiled with time profiling.
                        *IMPORTANT*: This will test-run the step functions, which may lead to unexpected side-effects.
        :param
        """
        device_step = device_step if device_step is not None else {}
        device_dist = device_dist if device_dist is not None else {}
        jit_step = jit_step if isinstance(jit_step, dict) else {k: jit_step for k in self._async_nodes.keys()}
        profile = profile if isinstance(profile, dict) else {k: profile for k in self._async_nodes.keys()}
        device_step = device_step if isinstance(device_step, dict) else {k: device_step for k in self._async_nodes.keys()}
        device_dist = device_dist if isinstance(device_dist, dict) else {k: device_dist for k in self._async_nodes.keys()}
        for k, n in self._async_nodes.items():
            n.warmup(graph_state,
                     device_step=device_step.get(k, None),
                     device_dist=device_dist.get(k, None),
                     jit_step=jit_step.get(k, True),
                     profile=profile.get(k, False))

    def start(self, graph_state: base.GraphState, timeout: float = None) -> base.GraphState:
        """Starts the graph and all its nodes."""
        # Stop first, if we were previously running.
        self.stop(timeout=timeout)

        # An additional reset is required when running async (futures, etc..)
        self._synchronizer.reset()

        # Prepare inputs
        no_inputs = {k: _inputs is not None for k, _inputs in graph_state.inputs.items()}
        assert all(no_inputs.values()), "No inputs provided to all entries in graph_state. Use graph.init()."

        # Reset async backend of every node
        for node in self._async_nodes.values():
            node._reset(graph_state, clock=self.clock, real_time_factor=self.real_time_factor)

        # Check that all nodes have the same episode counter
        assert len({n.eps for n in self._async_nodes.values()}) == 1, "All nodes must have the same episode counter."

        # Async startup
        fs = [n._startup(graph_state, timeout=timeout) for n in self._async_nodes.values()]
        res = [f.result() for f in fs]  # Wait for all nodes to finish startup
        assert all(res), "Not all nodes were able to start up."

        # Start nodes (provide same starting timestamp to every node)
        start = time.time()
        for node in self._async_nodes.values():
            node._start(start=start)
        return graph_state

    def stop(self, timeout: float = None):
        """Stops the graph and all its nodes."""

        # # Initiate stop (this unblocks the root's step, that is waiting for an action).
        # if len(self._synchronizer.action) > 0:
        #     self._synchronizer.action[-1].cancel()

        # Stop all nodes
        fs = [n._stop(timeout=timeout) for n in self._async_nodes.values()]

        # Initiate stop (this unblocks the root's step, that is waiting for an action).
        if len(self._synchronizer.action) > 0:
            self._synchronizer.action[-1].cancel()

        # Wait for all nodes to stop
        [f.result() for f in fs]  # Wait for all nodes to stop

        # Toggle
        self._initial_step = True

    def run_until_supervisor(self, graph_state: base.GraphState) -> base.GraphState:
        """Runs graph until supervisor node.step is called.

        Internal use only. Use reset(), step(), run(), or rollout() instead.
        """
        # Retrieve obs (waits for graph until supervisor to finish)
        next_step_state = self._synchronizer.observation.popleft().result()
        # print(f"[GET] run_until_root: seq={next_step_state.seq}, ts={next_step_state.ts:.2f}")
        self._initial_step = False
        step_states = {name: node._step_state for name, node in self._async_nodes.items()}
        step_states[self.supervisor.name] = next_step_state
        next_graph_state = base.GraphState(eps=next_step_state.eps).replace_step_states(step_states=step_states)
        return next_graph_state

    def run_supervisor(
        self, graph_state: base.GraphState, step_state: base.StepState = None, output: base.Output = None
    ) -> base.GraphState:
        """Runs supervisor node.step if step_state and output are not provided.
        Otherwise, overrides step_state and output with provided values.

        Internal use only. Use reset(), step(), run(), or rollout() instead.
        """
        assert (step_state is None) == (output is None), "Either both step_state and output must be None or both must be not None."
        # If run_root is run before run_until_root, we skip.
        if self._initial_step:
            return graph_state

        # Get next step state and output from root node
        if step_state is None and output is None:  # Run root node
            ss = graph_state.step_state[self.supervisor.name]
            next_step_state, new_output = self._async_nodes[self.supervisor.name].async_step(ss)
        else:  # Override step_state and output
            next_step_state, new_output = step_state, output
            # Update step_state (increment sequence number)
            next_step_state = next_step_state.replace(seq=next_step_state.seq + 1)

        # Set the result to be the step_state and output (action)  of the root.
        # print(f"[SET] run_root: seq={new_ss.seq}, ts={new_ss.ts:.2f}")
        self._synchronizer.action[-1].set_result((next_step_state, new_output))

        # Get graph_state
        step_states = {name: node._step_state for name, node in self._async_nodes.items()}
        step_states[self.supervisor.name] = next_step_state
        next_graph_state = base.GraphState(eps=next_step_state.eps).replace_step_states(step_states=step_states)
        return next_graph_state

    def run(self, graph_state: base.GraphState, timeout: float = None) -> base.GraphState:
        """
        Executes one step of the graph including the supervisor node and returns the updated graph state.

        Different from the .step method, it automatically progresses the graph state post-supervisor execution.
        This method is different from the gym API, as it uses the .step method of the supervisor node,
        while the reset and step methods allow the user to override the .step method.

        :param graph_state: The current graph state, or initial graph state from .init().
        :param timeout: The maximum time to wait for the graph to complete a step.
        :return: Updated graph state. It returns directly *after* the supervisor node's step() is run.
        """
        # Check if start() is called before run() and if not, call start() before run().
        if self._initial_step:
            self.start(graph_state, timeout=timeout)

        # Runs supergraph (except for supervisor)
        graph_state = self.run_until_supervisor(graph_state)

        # Runs supervisor node if no step_state or output is provided, otherwise uses provided step_state and output
        graph_state = self.run_supervisor(graph_state)
        return graph_state

    def reset(self, graph_state: base.GraphState, timeout: float = None) -> Tuple[base.GraphState, base.StepState]:
        """
        Prepares the graph for execution by resetting it to a state before the supervisor node's execution.

        Returns the graph and step state just before what would be the supervisor's step, mimicking the initial observation
        return of a gym environment's reset method. The step state can be considered the initial observation of a gym environment.

        :param graph_state: The graph state from .init().
        :return: Tuple of the new graph state and the supervisor node's step state *before* execution of the first step.
        """
        # Stop and start graph
        graph_state = self.start(graph_state, timeout=timeout)
        # Runs supergraph (except for supervisor)
        next_graph_state = self.run_until_supervisor(graph_state)
        next_step_state = next_graph_state.step_state[self.supervisor.name]  # Return supervisor node's step state
        return next_graph_state, next_step_state

    def step(
        self, graph_state: base.GraphState, step_state: base.StepState = None, output: base.Output = None
    ) -> Tuple[base.GraphState, base.StepState]:
        """
        Executes one step of the graph, optionally overriding the supervisor node's execution.

        If step_state and output are provided, they override the supervisor's step, allowing for custom step implementations.
        Otherwise, the supervisor's step() is executed as usual.

        When providing the updated step_state and output, the provided output can be viewed as the action that the agent would
        take in a gym environment, which is sent to nodes connected to the supervisor node.

        Start every episode with a call to reset() using the initial graph state from init(), then call step() repeatedly.

        :param graph_state: The current graph state.
        :param step_state: Custom step state for the supervisor node.
        :param output: Custom output for the supervisor node.
        :return: Tuple of the new graph state and the supervisor node's step state *before* execution of the next step.
        """
        # Runs supervisor node (if step_state and output are not provided, otherwise overrides step_state and output with provided values)
        new_graph_state = self.run_supervisor(graph_state, step_state, output)

        # Runs supergraph (except for supervisor)
        next_graph_state = self.run_until_supervisor(new_graph_state)
        next_step_state = next_graph_state.step_state[self.supervisor.name]  # Return supervisor node's step state
        return next_graph_state, next_step_state

    def get_record(self) -> base.EpisodeRecord:
        """Returns the episode record for all nodes in the graph."""
        records = {}
        for name, node in self._async_nodes.items():
            records[name] = node.get_record()
        return base.EpisodeRecord(nodes=records)

