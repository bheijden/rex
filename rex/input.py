import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from threading import RLock
from typing import Optional, Any, Deque, TYPE_CHECKING, Tuple, Callable
import jumpy as jp
from jax import numpy as jnp
from jax import jit
import jax.random as rnd

from rex.base import InputState
from rex.constants import READY, RUNNING, STOPPING, STOPPED, RUNNING_STATES, DEBUG, WARN, ERROR, SIMULATED, WALL_CLOCK, ASYNC, SYNC, FAST_AS_POSSIBLE, ASYNC, LATEST, BUFFER
from rex.distributions import Distribution
from rex.proto import log_pb2 as log_pb2
from rex.utils import log

if TYPE_CHECKING:
    from rex.node import Node
    from rex.output import Output


class Input:
    def __init__(self, node: "Node", output: "Output", window: int, blocking: bool, skip: bool, jitter: int, delay: float, delay_sim: Distribution, log_level: int, color: str, name: str):
        # todo: add this constraint?
        # assert not (skip and not blocking), "You can only skip blocking connections."
        self.node = node
        self.output = output
        self.input_name = name
        self.window = window
        self.blocking = blocking    #: Connection type
        self.delay_sim = delay_sim  #: Communication delay
        self.skip = skip            #: Skip first dependency
        self.jitter = jitter        #: Jitter mode
        self.log_level = log_level
        self.color = color
        self.delay = delay if delay is not None else delay_sim.high
        assert self.delay >= 0, "Phase should be non-negative."
        self._state = STOPPED

        # Jit function (call self.warmup() to pre-compile)
        self._num_buffer = 50
        self._jit_sample = jit(self.delay_sim.sample, static_argnums=1)
        self._jit_split = jit(rnd.split, static_argnums=1)

        # Executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.node.name}/{self.output.name}")
        self._q_task: Deque[Tuple[Future, Callable, Any, Any]] = deque(maxlen=10)
        self._lock = RLock()

        # Reset every time
        self._input_state = None
        self._record: log_pb2.InputRecord = None
        self._phase = None
        self._phase_dist = None
        self._prev_recv_sc = None
        self._rng = None
        self.q_msgs: Deque[Any] = None
        self.q_ts_input: Deque[Tuple[int, float]] = None
        self.q_ts_max: Deque[float] = None
        self.q_zip_delay: Deque[Tuple[float, float]] = None
        self.q_zip_msgs: Deque[Tuple[Any, log_pb2.Header]] = None
        self.q_expected_select: Deque[Tuple[float, int]] = None
        self.q_expected_ts_max: Deque[int] = None
        self.q_grouped: Deque[InputState] = None
        self.q_ts_next_step: Deque[Tuple[int, float]] = None
        self.q_sample: Deque = None

    @property
    def name(self) -> str:
        return self.output.name

    @property
    def phase(self) -> float:
        """Phase shift of the input: phase shift of the incoming output + the expected communication delay."""
        if self._phase is None:
            return self.output.phase + self.delay
        else:
            return self._phase

    @property
    def phase_dist(self) -> Distribution:
        """Phase shift of the input: phase shift of the incoming output + the expected communication delay."""
        if self._phase_dist is None:
            return self.output.phase_dist + self.delay_sim
        else:
            return self._phase_dist

    def warmup(self):
        self._jit_sample(rnd.PRNGKey(0), shape=self._num_buffer).block_until_ready()  # Only to trigger jit compilation
        self._jit_split(rnd.PRNGKey(0), num=2).block_until_ready()  # Only to trigger jit compilation

    def sample_delay(self) -> float:
        # Generate samples batch-wise
        if len(self.q_sample) == 0:
            self._rng, sample_rng = self._jit_split(self._rng, num=2)
            samples = tuple(self._jit_sample(sample_rng, shape=self._num_buffer).tolist())
            self.q_sample.extend(samples)

        # Sample delay
        sample = self.q_sample.popleft()
        return sample

    @property
    def info(self) -> log_pb2.InputInfo:
        return log_pb2.InputInfo(name=self.input_name, output=self.output.name, rate=self.output.rate, window=self.window, blocking=self.blocking,
                                 skip=self.skip, jitter=self.jitter, phase=self.phase, phase_dist=self.phase_dist.info,
                                 delay_sim=self.delay_sim.info, delay=self.delay)

    def log(self, id: str, value: Optional[Any] = None, log_level: Optional[int] = None):
        log_level = log_level if isinstance(log_level, int) else self.log_level
        log(f"{self.node.name}/{self.name}", self.color, min(log_level, self.log_level), id, value)

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
            log(f"{self.node.name}/{self.name}", "red", ERROR, "ERROR", error_msg)

    def reset(self, rng: jnp.ndarray, input_state: InputState):
        assert self._state in [STOPPED, READY], f"Input {self.name} of {self.node.name} must first be stopped, before it can be reset."

        # Empty queues
        self._input_state = input_state
        self._phase, self._phase_dist = None, None
        self._phase_dist = self.phase_dist
        self._phase = self.phase
        self._record = None
        self._prev_recv_sc = 0.  # Ensures the FIFO property for incoming messages.
        self._rng = rng
        self.q_msgs = deque()
        self.q_ts_input = deque()
        self.q_zip_delay = deque()
        self.q_zip_msgs = deque()
        self.q_ts_max = deque()
        self.q_expected_select = deque()
        self.q_expected_ts_max = deque()
        self.q_grouped = deque()
        self.q_ts_next_step = deque()
        self.q_sample = deque() #if self.q_sample is None else self.q_sample

        # Set running state
        self._state = READY
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

    def start(self, record: log_pb2.InputRecord):
        assert self._state in [READY], f"Input {self.name} of {self.node.name} must first be reset, before it can start running."

        # Set running state
        self._state = RUNNING
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

        # Store running configuration
        self._record = record
        self._record.info.CopyFrom(self.info)

    def stop(self) -> Future:
        assert self._state in [RUNNING], f"Input {self.name} of {self.node.name} must be running in order to stop."

        def _stopping():
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

    def push_expected_nonblocking(self):
        assert not self.blocking, "This function should only be called for non-blocking inputs."
        has_ts_next_step = len(self.q_ts_next_step) > 0
        has_ts_inputs = self.node._sync in [ASYNC] or len(self.q_ts_input) > 0
        if has_ts_next_step and has_ts_inputs:
            tick, ts_step = self.q_ts_next_step[0]
            has_ts_in_future = self.node._sync in [ASYNC] or any(ts > ts_step for seq, ts in self.q_ts_input)
            if has_ts_in_future:
                # Pop elements from queues
                # blocking connections:= scheduled_ts  (ignores any phase shifts, i.e. "original" schedule).
                # non-blocking:= ts_step (includes phase shifts due to blocking, scheduling shifts, comp. delays)
                tick, ts_step = self.q_ts_next_step.popleft()

                # Determine number of entries where ts > ts_step
                num_msgs = 0
                if self.jitter in [BUFFER]:
                    # Uses input phase and sequence number to determine expected timestamp instead of the actual timestamp.
                    phase = self.phase
                    for seq, ts_recv in self.q_ts_input:
                        ts_expected = seq / self.output.rate + phase
                        if ts_expected > ts_step:
                            break
                        num_msgs += 1
                else:  # self.jitter in [LATEST]:
                    # Simply uses the latest messages (and clears entire buffer until ts_step).
                    for seq, ts in self.q_ts_input:
                        if ts > ts_step or (self.skip and ts == ts_step):
                            break
                        num_msgs += 1

                # Clear q_ts_input until ts_inputs >= ts_step
                [self.q_ts_input.popleft() for _ in range(num_msgs)]

                # Log
                self.log("push_exp_nonblocking", f"ts_step={ts_step: .2f} | num_msgs={num_msgs}", log_level=DEBUG)

                # Push selection
                self.q_expected_select.append((ts_step, num_msgs))
                self.push_selection()

    def push_expected_blocking(self):
        assert self.blocking, "This function should only be called for blocking inputs."
        has_ts_next_step = len(self.q_ts_next_step) > 0
        if has_ts_next_step:
            # Pop elements from queues
            # blocking connections:= ts_next_step == scheduled_ts  (ignores any phase shifts, i.e. "original" schedule).
            # non-blocking:= ts_next_step == ts_step (includes phase shifts due to blocking, scheduling shifts, comp. delays)
            N_node, scheduled_ts = self.q_ts_next_step.popleft()

            skip = self.skip
            phase_node, phase_in = round(self.node.phase, 6), round(self.output.phase, 6)
            rate_node, rate_in = self.node.rate, self.output.rate
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
            self.log("push_exp_blocking", f"scheduled_ts={scheduled_ts: .2f} | num_msgs={num_msgs}", log_level=DEBUG)

            # Push ts max
            self.q_expected_ts_max.append(num_msgs)
            self.push_ts_max()

            # Push selection
            self.q_expected_select.append((scheduled_ts, num_msgs))
            self.push_selection()

    # @synchronized(RLock())
    def push_ts_max(self):
        # Only called by blocking connections
        has_msgs = len(self.q_expected_ts_max) > 0 and self.q_expected_ts_max[0] <= len(self.q_ts_input)
        if has_msgs:
            num_msgs = self.q_expected_ts_max.popleft()

            # Determine max timestamp of grouped message for blocking connection
            input_ts = [self.q_ts_input.popleft()[1] for _i in range(num_msgs)]
            ts_max = max([0.] + input_ts)
            self.q_ts_max.append(ts_max)

            # Push push_phase_shift (must be called from node thread)
            self.node._submit(self.node.push_phase_shift)

    # @synchronized(RLock())
    def push_selection(self):
        has_expected = len(self.q_expected_select) > 0
        if has_expected:
            has_recv_all_expected = len(self.q_msgs) >= self.q_expected_select[0][1]  # self.q_expected[0]=(ts_next_step, exp_num_msgs)
            if has_recv_all_expected:
                ts_next_step, num_msgs = self.q_expected_select.popleft()
                log_msg = f"blocking={self.blocking} | step_ts={ts_next_step: .2f} | num_msgs={num_msgs}"
                self.log("push_selection", log_msg, log_level=DEBUG)

                # Create record
                # todo: calculate probability of selection using modeled distribution.
                #  1. Assume scheduling delay to be constant, or....
                #  2. Assume zero scheduling delay --> probably easier.
                #  3. Integrate each delay distribution over past and future sampling times.
                record_grouped = self._record.grouped.add()
                record_grouped.num_msgs = num_msgs

                # Group messages
                grouped = []
                ts_max = 0
                for i in range(num_msgs):
                    record_msg, msg = self.q_msgs.popleft()

                    # Determine max timestamp
                    seq = record_msg.sent.seq
                    ts_sent = record_msg.sent.ts.sc
                    ts_recv = record_msg.received.ts.sc

                    ts_max = max(ts_max, ts_recv)

                    # Add to record
                    record_grouped.messages.append(record_msg)

                    # Push message to input_state
                    grouped.append((seq, ts_sent, ts_recv, msg))

                # Only add messages that will not get pushed out immediately.
                for (seq, ts_sent, ts_recv, msg) in grouped[-self.window:]:
                    # self._input_state.push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=msg)
                    self._input_state = self._input_state.push(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, data=msg)

                # Add grouped message to queue
                self.q_grouped.append(self._input_state)

                # Push step (must be called from node thread)
                self.node._submit(self.node.push_step)

    def push_ts_input(self, msg, header):
        # Skip if we are not running
        if self._state not in [READY, RUNNING]:
            self.log("push_ts_input (NOT RUNNING)", log_level=DEBUG)
            return
        # Skip if from a previous episode
        elif header.eps != self.node.eps:
            self.log("push_ts_input (PREV EPS)", log_level=DEBUG)
            return
        # Else, continue
        else:
            self.log("push_ts_input", log_level=DEBUG)

        # Determine sent timestamp
        seq, sent_sc, sent_wc = header.seq, header.ts.sc, header.ts.wc

        # Determine input timestamp
        if self.node._clock in [SIMULATED]:
            # Sample delay
            delay = self.sample_delay()
            # Enforce FIFO property
            recv_sc = round(max(sent_sc + delay, self._prev_recv_sc), 6)  # todo: 1e-9 required here?
            self._prev_recv_sc = recv_sc
            _, recv_wc = self.node.now()
        else:
            # This only happens when push_ts_input is called by push_input
            recv_sc, recv_wc = self.node.now()

        # Communication delay
        # IMPORTANT! delay_wc measures communication delay of output_ts instead of message.
        # Value of delay_wc is overwritten in push_input() when clock=wall-clock.
        delay_sc = recv_sc - sent_sc
        delay_wc = recv_wc - sent_wc
        self.q_zip_delay.append((delay_sc, delay_wc))

        # Push zip to buffer messages
        self.push_zip()

        # Add phase to queue
        self.q_ts_input.append((seq, recv_sc))

        # Push event
        if self.blocking:
            self.push_ts_max()
        else:
            self.push_expected_nonblocking()

    def push_input(self, msg: Any, header_sent: log_pb2.Header):
        # Skip if we are not running
        if self._state not in [READY, RUNNING]:
            self.log("push_input (NOT RUNNING)", log_level=DEBUG)
            return
        # Skip if from a previous episode
        elif header_sent.eps != self.node.eps:
            self.log("push_input (PREV EPS)", log_level=DEBUG)
            return
        # Else, continue
        else:
            self.log("push_input", log_level=DEBUG)

        # todo: add transform here
        # todo: add to input_state here?

        # Push ts_input when the clock is not simulated
        if self.node._clock in [WALL_CLOCK]:
            # This will queue delay (and call push_zip)
            self.push_ts_input(msg, header_sent)

        # Queue msg
        self.q_zip_msgs.append((msg, header_sent))

        # Push zip to buffer messages
        self.push_zip()

    # @synchronized(RLock())
    def push_zip(self):
        has_msg = len(self.q_zip_msgs) > 0
        has_delay = len(self.q_zip_delay) > 0
        if has_msg and has_delay:
            msg, header_sent = self.q_zip_msgs.popleft()

            # Determine sent timestamp
            sent_sc, sent_wc = header_sent.ts.sc, header_sent.ts.wc

            # Determine the ts of the input message
            # If clock=wall-clock, call push_ts_input with header_sent, but overwrite recv_wc if clock=simulated
            if self.node._clock in [SIMULATED]:
                delay_sc, _ = self.q_zip_delay.popleft()
                recv_sc = round(sent_sc + delay_sc, 6)

                # Recalculate delay_wc, because it reflects the communication delay of the output_ts message
                # instead of the actual message.
                _, recv_wc = self.node.now()
                delay_wc = recv_wc - sent_wc
            else:
                # This will queue the delay
                delay_sc, delay_wc = self.q_zip_delay.popleft()
                recv_sc = sent_sc + delay_sc
                recv_wc = sent_wc + delay_wc

            # Throttle to timestamp
            # todo: Throttle on separate thread? Else, wc communication delay could be wrong when clock=simulated.
            # todo: Perhaps, not too big of a problem because we assume the FIFO property?
            self.node.throttle(recv_sc)

            # Create header with timing information on received message
            header_recv = log_pb2.Header(eps=self.node._eps, seq=header_sent.seq, ts=log_pb2.Time(sc=recv_sc, wc=recv_wc))

            # Create message record
            record_msg = log_pb2.MessageRecord()
            record_msg.sent.CopyFrom(header_sent)
            record_msg.received.CopyFrom(header_recv)
            record_msg.delay = delay_sc
            record_msg.comm_delay.CopyFrom(log_pb2.Time(sc=delay_sc, wc=delay_wc))

            # Add message to queue
            self.q_msgs.append((record_msg, msg))

            # See if we can prepare tuple for next step
            self.push_selection()

