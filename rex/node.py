import abc
import time
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from typing import Callable, List, Tuple, Optional, Any, Union, Deque, Dict
from collections import deque
import traceback
import jumpy as jp
import jax.numpy as jnp  # todo: replace with jumpy as jp.ndarray?
import jax.random as rnd
import numpy as onp
from flax.core import FrozenDict
from jax import jit

from rex.base import GraphState, StepState, InputState, State,  Output, Params
from rex.constants import READY, RUNNING, STOPPING, STOPPED, RUNNING_STATES, PHASE, FREQUENCY, SIMULATED, \
    FAST_AS_POSSIBLE, SYNC, BUFFER, DEBUG, INFO, WARN, ERROR, WALL_CLOCK, LATEST
from rex.input import Input
from rex.output import Output
from rex.utils import log
from rex.distributions import Distribution, Gaussian, GMM
import rex.proto.log_pb2 as log_pb2


class BaseNode:
    def __init__(self, name: str, rate: float, delay_sim: Distribution, delay: float = None, advance: bool = True,
                 stateful: bool = True, log_level: int = WARN, color: str = "green"):
        self.name = name
        self.rate = rate
        self.log_level = log_level
        self.color = color
        self.advance = advance
        self.stateful = stateful
        self.inputs: List[Input] = []
        self.output = Output(self, self.log_level, self.color, delay, delay_sim)

        # State and episode counter
        self._eps = 0
        self._state = STOPPED

        # Executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._q_task: Deque[Tuple[Future, Callable, Any, Any]] = deque(maxlen=10)
        self._lock = RLock()

        # Reset every run
        self._tick = None
        self._record: log_pb2.NodeRecord = None
        self._phase_scheduled = None
        self._phase = None
        self._phase_dist = None
        self._sync = None
        self._clock = None
        self._scheduling = None
        self._real_time_factor = 1.

        # Set starting ts
        self._ts_start = Future()
        self._set_ts_start(0.)

        self.q_tick: Deque[int] = None
        self.q_ts_scheduled: Deque[Tuple[int, float]] = None
        self.q_ts_output_prev: Deque[float] = None
        self.q_ts_step: Deque[Tuple[int, float, float, log_pb2.StepRecord]] = None
        self.q_rng_step: Deque[jnp.ndarray] = None

        # Only used if no step and reset fn are provided
        self._i = 0

        if not 1/rate > self.output.phase:
            self.log("WARNING", f"The sampling time ({1/rate=:.6f} s) is smaller than"
                                f" the output phase ({self.output.phase=:.6f} s)."
                                " This may lead to large (accumulating) delays.", WARN)

    def warmup(self):
        # Warms-up the jitted functions (i.e. pre-compiles)
        # batch_split_rng(rnd.PRNGKey(0), self._jit_split, deque(), num=self._num_buffer)  # Only to trigger jit compilation

        # Warms-up jitted functions in the output (i.e. pre-compiles)
        self.output.warmup()

        # Warms-up jitted functions in the inputs (i.e. pre-compiles)
        [i.warmup() for i in self.inputs]

    @property
    def record(self) -> log_pb2.NodeRecord:
        if self._record is None:
            record = log_pb2.NodeRecord(info=self.info)
            record.inputs.extend([log_pb2.InputRecord(info=i.info) for i in self.inputs])
            return record
        else:
            return self._record

    @property
    def eps(self) -> int:
        return self._eps

    @property
    def phase(self) -> float:
        """Phase shift of the node: max phase over all incoming blocking & non-skipped connections."""
        # Recalculate phase once per episode.
        if self._phase is None:
            try:
                return max([0.] + [i.phase for i in self.inputs if i.blocking and not i.skip])
            except RecursionError as e:
                msg = "The constructed graph is not DAG. To break an algebraic loop, " \
                      "either skip a connection or make the connection non-blocking."
                log(self.name, "red", ERROR, "ERROR", msg)
                # exit()
                raise e
        else:
            return self._phase

    @property
    def phase_dist(self) -> Distribution:
        if self._phase_dist is None:
            return Gaussian(self.phase)
        else:
            return self._phase_dist

    @property
    def info(self) -> log_pb2.NodeInfo:
        info = log_pb2.NodeInfo(name=self.name, rate=self.rate, stateful=self.stateful, advance=self.advance, phase=self.phase,
                                delay_sim=self.output.delay_sim.info, delay=self.output.delay)
        info.inputs.extend([i.info for i in self.inputs])
        return info

    @classmethod
    def from_info(cls, info: log_pb2.NodeInfo, log_level: int = WARN, color: str = "green", **kwargs):
        # Initializes a node from a NodeInfo proto log
        node = cls(name=info.name, rate=info.rate, delay_sim=GMM.from_info(info.delay_sim), delay=info.delay, advance=info.advance,
                   stateful=info.stateful, log_level=log_level, color=color, **kwargs)
        return node

    def connect_from_info(self, info: log_pb2.InputInfo, node: "Node", log_level: Optional[int] = None, color: Optional[str] = None):
        # Connects a node to another node from an InputInfo proto log
        self.connect(node,
                     blocking=info.blocking,
                     skip=info.skip,
                     delay_sim=GMM.from_info(info.delay_sim),
                     delay=info.delay,
                     jitter=info.jitter,
                     name=info.name,
                     color=color,
                     log_level=log_level)

    def log(self, id: str, value: Optional[Any] = None, log_level: Optional[int] = None):
        log_level = log_level if isinstance(log_level, int) else self.log_level
        log(self.name, self.color, min(log_level, self.log_level), id, value)

    def _set_ts_start(self, ts_start: float):
        assert isinstance(self._ts_start, Future)
        self._ts_start.set_result(ts_start)
        self._ts_start = ts_start

    def _submit(self, fn, *args, stopping: bool = False, **kwargs):
        with self._lock:
            if self._state in [READY, RUNNING] or stopping:
                f = self._executor.submit(fn, *args, **kwargs)
                self._q_task.append((f, fn, args, kwargs))
                f.add_done_callback(self._f_callback)
            else:
                self.log("SKIPPED", fn.__name__, log_level=DEBUG)
                f = Future()
                f.cancel()
        return f

    def _f_callback(self, f: Future):
        e = f.exception()
        if e is not None and e is not CancelledError:
            error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            log(self.name, "red", ERROR, "ERROR", error_msg)

    def now(self) -> Tuple[float, float]:
        """Get the passed time since according to the simulated and wall clock"""
        # Determine starting timestamp
        ts_start = self._ts_start
        ts_start = ts_start.result() if isinstance(ts_start, Future) else ts_start

        # Determine passed time
        wc = time.time()
        wc_passed = wc - ts_start
        sc = wc_passed if self._real_time_factor == 0 else wc_passed * self._real_time_factor
        return sc, wc_passed

    def throttle(self, ts: float):
        if self._real_time_factor not in [FAST_AS_POSSIBLE]:
            # Determine starting timestamp
            ts_start = self._ts_start
            ts_start = ts_start.result() if isinstance(ts_start, Future) else ts_start

            wc_passed_target = ts / self._real_time_factor
            wc_passed = time.time() - ts_start
            wc_sleep = max(0., wc_passed_target-wc_passed)
            time.sleep(wc_sleep)

    def connect(self, node: "Node", blocking: bool, delay_sim: Distribution, delay: float = None, window: int = 1, skip: bool = False,
                jitter: int = LATEST, name: Optional[str] = None, log_level: Optional[int] = None, color: Optional[str] = None):
        # Create new input
        assert node.name not in [i.output.node.name for i in self.inputs], "Cannot use the same output source for more than one input."
        log_level = log_level if isinstance(log_level, int) else self.log_level
        color = color if isinstance(color, str) else self.color
        name = name if isinstance(name, str) else node.output.name
        i = Input(self, node.output, window, blocking, skip, jitter, delay, delay_sim, log_level, color, name)
        self.inputs.append(i)

        # Register the input with the output of the specified node
        node.output.connect(i)

    @abc.abstractmethod
    def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Output]:
        raise NotImplementedError

    def _reset(self, graph_state: GraphState, sync: int = SYNC, clock: int = SIMULATED, scheduling: int = PHASE, real_time_factor: Union[int, float] = FAST_AS_POSSIBLE):
        assert self._state in [STOPPED, READY], f"{self.name} must first be stopped, before it can be reset"
        assert not (clock in [WALL_CLOCK] and sync in [SYNC]), "You can only simulate synchronously, if the clock=`SIMULATED`."

        # Save run configuration
        self._sync = sync                           #: True if we must run synchronized
        self._clock = clock                         #: Simulate timesteps
        self._scheduling = scheduling               #: Synchronization mode for step scheduling
        self._real_time_factor = real_time_factor   #: Scaling of simulation speed w.r.t wall clock

        # Up the episode counter (must happen before resetting outputs & inputs)
        self._eps += 1

        # Reset every run
        self._tick = 0
        self._phase_scheduled = 0.     #: Structural phase shift that the step scheduler takes into account
        self._phase, self._phase_dist = None, None
        self._phase = self.phase
        self._phase_dist = self.phase_dist
        self._record = None
        self._step_state = graph_state.nodes[self.name]

        # Set starting ts
        self._ts_start = Future()  #: The starting timestamp of the episode.

        # Initialize empty queues
        self.q_tick = deque()
        self.q_ts_scheduled = deque()
        self.q_ts_output_prev = deque()
        self.q_ts_step = deque()
        self.q_rng_step = deque()

        # Get rng for delay sampling
        # This is hacky because we reuse the seed.
        # However, changing the seed of the step_state would break the reproducibility between graphs (compiled, async).
        rng = self._step_state.rng
        rng = rnd.PRNGKey(rng[0]) if isinstance(rng, onp.ndarray) else rng

        # Reset output
        rng_out, rng = rnd.split(rng, num=2)
        self.output.reset(rng_out)

        # Reset all inputs and output
        rngs_in = rnd.split(rng, num=len(self.inputs))
        [i.reset(r, self._step_state.inputs[i.input_name]) for r, i in zip(rngs_in, self.inputs)]

        # Set running state
        self._state = READY
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

    def _start(self, start: float):
        assert self._state in [READY], f"{self.name} must first be reset, before it can start running."

        # Set running state
        self._state = RUNNING
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

        # Create logging record
        self._set_ts_start(start)
        self._record = log_pb2.NodeRecord(info=self.info, sync=self._sync, clock=self._clock, scheduling=self._scheduling,
                                          real_time_factor=self._real_time_factor, ts_start=start)

        # Start all inputs and output
        [i.start(record=self._record.inputs.add()) for i in self.inputs]
        self.output.start()

        # Set first last_output_ts equal to phase (as if we just finished our previous output).
        self.q_ts_output_prev.append(0.)

        # Queue first two ticks (so that output_ts runs ahead of message)
        # The number of tokens > 1 determines "how far" into the future the
        # output timestamps are simulated when clock=simulated.
        self.q_tick.extend((True, True))

        # Push scheduled ts
        _f = self._submit(self.push_scheduled_ts)

    def _stop(self, timeout: Optional[float] = None) -> Future:
        # Pass here, if we are not running
        if self._state not in [RUNNING]:
            self.log(f"{self.name} is not running, so it cannot be stopped.", log_level=DEBUG)
            f = Future()
            f.set_result(None)
            return f
        assert self._state in [RUNNING], f"Cannot stop, because {self.name} is currently not running."

        def _stopping():
            # Stop producing messages and communicate total number of sent messages
            self.output.stop()

            # Stop all channels to receive all sent messages from their connected outputs
            [i.stop().result(timeout=timeout) for i in self.inputs]

            # Set running state
            self._state = STOPPED
            self.log(RUNNING_STATES[self._state], log_level=DEBUG)

        with self._lock:
            # Then, flip running state so that no more tasks can be scheduled
            # This means that
            self._state = STOPPING
            self.log(RUNNING_STATES[self._state], log_level=DEBUG)

            # First, submit _stopping task
            f = self._submit(_stopping, stopping=True)
        return f

    # @synchronized(RLock())
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
            scheduled_ts = round(tick / self.rate + self.phase, 6)

            # Log
            self.log("push_scheduled_ts", f"tick={tick} | scheduled_ts={scheduled_ts: .2f}", log_level=DEBUG)

            # Queue expected next step ts and wait for blocking delays to be determined
            self.q_ts_scheduled.append((tick, scheduled_ts))
            self.push_phase_shift()

            # Push next step ts event to blocking connections (does not throttle)
            for i in self.inputs:
                if not i.blocking:
                    continue
                i.q_ts_next_step.append((tick, scheduled_ts))

                # Push expect (must be called from input thread)
                i._submit(i.push_expected_blocking)

    # @synchronized(RLock())
    def push_phase_shift(self):
        # If all blocking delays are known, and we know the expected next step timestamp
        has_all_ts_max = all([len(i.q_ts_max) > 0 for i in self.inputs if i.blocking])
        has_scheduled_ts = len(self.q_ts_scheduled) > 0
        has_last_output_ts = len(self.q_ts_output_prev) > 0
        if has_scheduled_ts and has_last_output_ts and has_all_ts_max:
            self.log("push_phase_shift", log_level=DEBUG)

            # Grab blocking delays from queues and calculate max delay
            ts_max = [i.q_ts_max.popleft() for i in self.inputs if i.blocking]
            ts_max = max(ts_max) if len(ts_max) > 0 else 0.

            # Grab next scheduled step ts (without considering phase_scheduling shift)
            tick, ts_scheduled = self.q_ts_scheduled.popleft()

            # Grab previous output ts
            ts_output_prev = self.q_ts_output_prev.popleft()

            # Calculate sources of phase shift
            only_blocking = self.advance and all(i.blocking for i in self.inputs)
            phase_inputs = ts_max - ts_scheduled
            phase_last = ts_output_prev - ts_scheduled
            phase_scheduled = self._phase_scheduled

            # Calculate phase shift
            # If only blocking connections, phase is not determined by phase_scheduled
            phase = max(phase_inputs, phase_last) if only_blocking else max(phase_inputs, phase_last, phase_scheduled)

            # Update structural scheduling phase shift
            if self._scheduling in [FREQUENCY]:
                self._phase_scheduled += max(0, phase_last-phase_scheduled)
            else:  # self._scheduling in [PHASE]
                self._phase_scheduled = 0.

            # Calculate starting timestamp for the step call
            ts_step = ts_scheduled + phase

            # Sample delay if we simulate the clock
            delay = self.output.sample_delay() if self._clock in [SIMULATED] else None

            # Create step record
            record_step = log_pb2.StepRecord(tick=tick, ts_scheduled=ts_scheduled, ts_max=ts_max, ts_output_prev=ts_output_prev,
                                             ts_step=ts_step, phase=phase, phase_scheduled=phase_scheduled,
                                             phase_inputs=phase_inputs, phase_last=phase_last)
            self.q_ts_step.append((tick, ts_step, delay, record_step))

            # Predetermine output timestamp when we simulate the clock
            if self._clock in [SIMULATED]:
                # Determine output timestamp
                ts_output = ts_step+delay
                _, ts_output_wc = self.now()
                header = log_pb2.Header(eps=self._eps, seq=tick, ts=log_pb2.Time(sc=ts_output, wc=ts_output_wc))
                self.output.push_ts_output(ts_output, header)

                # Add previous output timestamp to queue
                self.q_ts_output_prev.append(ts_output)

                # Simulate output timestamps into the future
                # If we use the wall-clock, ts_output_prev is queued after the step in push_step
                _f = self._submit(self.push_scheduled_ts)

            # Only throttle if we have non-blocking connections
            if any(not i.blocking for i in self.inputs) or not self.advance:
                # todo: This also throttles when running synced. Correct?
                self.throttle(ts_step)

            # Push for step (will never trigger here if there are non-blocking connections).
            self.push_step()

            # Push next step timestamp to non-blocking connections
            for i in self.inputs:
                if i.blocking:
                    continue
                i.q_ts_next_step.append((tick, ts_step))

                # Push expect (must be called from input thread)
                i._submit(i.push_expected_nonblocking)

    # @synchronized(RLock())
    def push_step(self):
        has_grouped = all([len(i.q_grouped) > 0 for i in self.inputs])
        has_ts_step = len(self.q_ts_step) > 0
        if has_ts_step and has_grouped:
            self.log("push_step", log_level=DEBUG)

            # Grab next expected step ts and step record
            tick, ts_step_sc, delay_sc, record_step = self.q_ts_step.popleft()

            # Actual step start ts
            # todo: ts_step_wc should also be inferred in push_phase_shift when running ASYNC (using wall clock).
            _, ts_step_wc = self.now()

            # Grab grouped msgs
            inputs = FrozenDict({i.input_name: i.q_grouped.popleft() for i in self.inputs})

            # Update StepState with grouped messages
            step_state = self._step_state.replace(inputs=inputs)

            # Run step and get msg
            new_step_state, output = self.step(jp.float32(ts_step_sc), step_state)

            # Update step_state
            self._step_state = new_step_state

            # Determine output timestamp
            if self._clock in [SIMULATED]:
                assert delay_sc is not None
                ts_output_sc = ts_step_sc + delay_sc
                _, ts_output_wc = self.now()
            else:
                assert delay_sc is None
                ts_output_sc, ts_output_wc = self.now()
                delay_sc = ts_output_sc - ts_step_sc
                assert delay_sc >= 0, "delay cannot be negative"

                # Add previous output timestamp to queue
                # If we simulate the clock, ts_output_prev is already queued in push_phase_shift
                self.q_ts_output_prev.append(ts_output_sc)

            # Throttle to timestamp
            self.throttle(ts_output_sc)

            # Create header with timing information on output
            header = log_pb2.Header(eps=self._eps, seq=tick, ts=log_pb2.Time(sc=ts_output_sc, wc=ts_output_wc))

            # Log sent times
            record_step.sent.CopyFrom(header)
            record_step.delay = delay_sc
            record_step.ts_output = ts_output_sc
            record_step.comp_delay.CopyFrom(log_pb2.Time(sc=ts_output_sc-ts_step_sc, wc=ts_output_wc-ts_step_wc))

            # Push output
            if output is not None:  # Agent returns None when we are stopping/resetting.
                self.output.push_output(output, header)

            # Add step record
            self._record.steps.append(record_step)

            # Only schedule next step if we are running
            if self._state in [RUNNING]:

                # Add token to tick queue (ticks are incremented in push_scheduled_ts function)
                self.q_tick.append(True)

                # Schedule next step (does not consider scheduling shifts)
                _f = self._submit(self.push_scheduled_ts)


class Node(BaseNode):
    def default_params(self, rng: jp.ndarray, graph_state: GraphState = None) -> Params:
        """Default params of the node."""
        raise NotImplementedError

    def default_state(self, rng: jp.ndarray, graph_state: GraphState = None) -> State:
        """Default state of the node."""
        raise NotImplementedError

    def default_inputs(self, rng: jp.ndarray, graph_state: GraphState = None) -> FrozenDict[str, InputState]: #Dict[str, InputState]:
        """Default inputs of the node."""
        rngs = jp.random_split(rng, num=len(self.inputs))
        inputs = dict()
        for i, rng_output in zip(self.inputs, rngs):
            window = i.window
            seq = jp.arange(-window, 0, dtype=jp.int32)
            ts_sent = 0 * jp.arange(-window, 0, dtype=jp.float32)
            ts_recv = 0 * jp.arange(-window, 0, dtype=jp.float32)
            outputs = [i.output.node.default_output(rng_output, graph_state) for _ in range(window)]
            inputs[i.input_name] = InputState.from_outputs(seq, ts_sent, ts_recv, outputs)
        return FrozenDict(inputs)

    @abc.abstractmethod
    def default_output(self, rng: jp.ndarray, graph_state: GraphState = None) -> Output:
        """Default output of the node."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, rng: jp.ndarray, graph_state: GraphState = None) -> StepState:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, ts: jp.float32, step_state: StepState) -> Tuple[StepState, Output]:
        raise NotImplementedError