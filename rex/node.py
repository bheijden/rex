# import codecs
# import pickle
import dill as pickle
import abc
import time
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from typing import Callable, List, Tuple, Optional, Any, Union, Deque, Dict
from collections import deque
import traceback
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax.typing import ArrayLike
import numpy as onp
from flax.core import FrozenDict
from flax import serialization

from rex.base import GraphState, StepState, InputState, State, Output as BaseOutput, Params, Empty
from rex.constants import (
    READY,
    STARTING,
    READY_TO_START,
    RUNNING,
    STOPPING,
    STOPPED,
    RUNNING_STATES,
    PHASE,
    FREQUENCY,
    SIMULATED,
    FAST_AS_POSSIBLE,
    SYNC,
    ASYNC,
    BUFFER,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    WALL_CLOCK,
    LATEST,
)
from rex.input import Input
from rex.output import Output
from rex.utils import log, NODE_COLOR, NODE_LOG_LEVEL
from rex.distributions import Distribution, Gaussian, GMM
import rex.proto.log_pb2 as log_pb2


class BaseNode:
    def __init__(
        self,
        name: str,
        rate: float,
        delay_sim: Distribution = None,
        delay: float = None,
        advance: bool = True,
        stateful: bool = True,
        scheduling: int = PHASE,
    ):
        self.name = name
        self.rate = rate
        self.advance = advance
        self.stateful = stateful
        self.scheduling = scheduling
        self.inputs: List[Input] = []

        # Initialize output
        delay_sim = delay_sim if delay_sim is not None else Gaussian(0.0, 0.0)
        self.output = Output(self, delay, delay_sim)

        # The following attributes are used to keep track of the state of the node
        self._unpickled = True  # Used to determine whether this node is fully unpickled (ignores if all(inputs.unpickled)).
        self._from_info = False  # Used to determine whether this node is created from an InputInfo proto log.

        # State and episode counter
        self._has_warmed_up = False
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
        self._real_time_factor = 1.0

        # Log
        self._discarded = 0
        self.max_records = 20000  # todo: make this configurable with a assert > 1.
        self._record_step_states: Deque[StepState] = None
        self._record_outputs: Deque[Output] = None
        self.record_setting = dict(node=False, outputs=False, rngs=False, states=False, params=False, step_states=False)

        # Set starting ts
        self._ts_start = Future()
        self._set_ts_start(0.0)

        self.q_tick: Deque[int] = None
        self.q_ts_scheduled: Deque[Tuple[int, float]] = None
        self.q_ts_output_prev: Deque[float] = None
        self.q_ts_step: Deque[Tuple[int, float, float, log_pb2.StepRecord]] = None
        self.q_rng_step: Deque[jax.Array] = None

        # Only used if no step and reset fn are provided
        self._i = 0

        if not 1 / rate > self.output.phase:
            self.log(
                "WARNING",
                f"The sampling time ({1/rate=:.6f} s) is smaller than"
                f" the output phase ({self.output.phase=:.6f} s)."
                " This may lead to large (accumulating) delays.",
                WARN,
            )

    def __getstate__(self):
        """Used for pickling

        Does not pickle the connections, because it is not pickleable.
        """
        args = ()
        kwargs = dict(
            name=self.name,
            rate=self.rate,
            delay_sim=self.output.delay_sim,
            delay=self.output.delay,
            advance=self.advance,
            stateful=self.stateful,
            scheduling=self.scheduling,
        )
        inputs = self.inputs
        return args, kwargs, inputs

    def __setstate__(self, state):
        """Used for unpickling"""
        args, kwargs, inputs = state
        self.__init__(*args, **kwargs)
        # At this point, the inputs are not yet fully unpickled.
        self.inputs = inputs
        # If the node is not fully unpickled after setting the state, consider setting self._unpickled to `False`.
        # For example, some attributes may need to be replaced with references to objects, initialized somewhere else.
        self._unpickled = True

    def warmup(self, graph_state: GraphState, device=None):
        if device is None:
            # gpu_devices = jax.devices('gpu')
            cpu_device = jax.devices('cpu')[0]
            device = cpu_device
        # device = None

        # Warms-up jitted functions in the output (i.e. pre-compiles)
        self.output.warmup(graph_state, device=device)

        # Warms-up jitted functions in the inputs (i.e. pre-compiles)
        [i.warmup(graph_state, device=device) for i in self.inputs]

        # Warm-up phase_dist
        _ = self.phase_dist

        # Warmup random number generators
        _ = [r for r in rnd.split(graph_state.nodes[self.name].rng, num=len(self.inputs))]

        # Warms-up jitted functions in the node
        self._has_warmed_up = True

    @property
    def unpickled(self):
        return self._unpickled

    @property
    def log_level(self):
        return NODE_LOG_LEVEL.get(self, WARN)

    @property
    def color(self):
        return NODE_COLOR.get(self, "green")

    def record(
        self,
        node: Union[bool, None] = None,
        outputs: Union[bool, None] = None,
        rngs: Union[bool, None] = None,
        states: Union[bool, None] = None,
        params: Union[bool, None] = None,
        step_states: Union[bool, None] = None,
    ) -> log_pb2.NodeRecord:
        """Returns a NodeRecord proto log.

        Recording all the data can be very memory intensive. It is recommended to only record the data you need. In particular,
        recording the step states can be very memory intensive, because it contains inputs that are duplicates of the
        outputs of other nodes.

        If you want to record all data with minimal duplication,
        set `node=True, outputs=True, rngs=True, states=True, params=True, step_states=False`.

        You can pre-set the default record settings by setting the `record_setting` attribute of the node. For example:
        ```
        node.record_setting["node"] = True
        node.record_setting["outputs"] = True
        ```

        The maximum number of records is set to 20000. If the number of records exceeds this number, the oldest records are
        discarded. This is to prevent memory overflow. If you want to change this number, set the `max_records` attribute
        of the node.

        :param node: Whether to record the (pickled) node state (overrides the default self.record_setting).
        :param outputs: Whether to record the (pickled) outputs (overrides the default self.record_setting).
        :param rngs: Whether to record the random number generator state (overrides the default self.record_setting).
        :param states: Whether to record the (pickled) state (overrides the default self.record_setting).
        :param params: Whether to record the (pickled) params (overrides the default self.record_setting).
        :param step_states: Whether to record the (pickled) step states (overrides the default self.record_setting).
        :return: A NodeRecord proto log.
        """
        assert self._state not in [RUNNING], "Cannot extract the record while running. Stop first (by calling .stop())."

        # Adjust record settings
        node = node if node is not None else self.record_setting.get("node", False)
        outputs = outputs if outputs is not None else self.record_setting.get("outputs", False)
        rngs = rngs if rngs is not None else self.record_setting.get("rngs", False)
        states = states if states is not None else self.record_setting.get("states", False)
        params = params if params is not None else self.record_setting.get("params", False)
        step_states = step_states if step_states is not None else self.record_setting.get("step_states", False)

        # If records were discarded, warn the user that the record is incomplete.
        if self._discarded > 0:
            self.log(
                "WARNING", f"Discarded {self._discarded} records. Incomplete records may lead to errors when tracing.", WARN
            )

        # Get timing record
        if self._record is None:
            # Create new record if not yet created.
            record = log_pb2.NodeRecord(info=self.info)
            record.inputs.extend([log_pb2.InputRecord(info=i.info) for i in self.inputs])
        else:
            # Copy record from self._record.
            record = log_pb2.NodeRecord()
            record.CopyFrom(self._record)

        # Store (pickled) node state
        if node:
            record.info.state = pickle.dumps(self)

        # Store output trajectory
        if outputs and self._record_outputs and len(self._record_outputs) > 0:
            record.outputs.target = pickle.dumps(self._record_outputs[0])
            # data = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *self._record_outputs)
            # record.outputs.encoded_bytes.extend([serialization.to_bytes(data)])
            record.outputs.encoded_bytes.extend([serialization.to_bytes(o) for o in self._record_outputs])

        # Store random number generator state trajectory
        if rngs and self._record_step_states and len(self._record_step_states) > 0:
            record.rngs.target = pickle.dumps(self._record_step_states[0].rng)
            # data = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *self._record_step_states).rng
            # record.rngs.encoded_bytes.extend([serialization.to_bytes(data)])
            record.rngs.encoded_bytes.extend([serialization.to_bytes(s.rng) for s in self._record_step_states])

        # Store state trajectory
        if states and self._record_step_states and len(self._record_step_states) > 0:
            record.states.target = pickle.dumps(self._record_step_states[0].state)
            # data = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *self._record_step_states).state
            # record.states.encoded_bytes.extend([serialization.to_bytes(data)])
            record.states.encoded_bytes.extend([serialization.to_bytes(s.state) for s in self._record_step_states])

        # Store params trajectory
        if params and self._record_step_states and len(self._record_step_states) > 0:
            record.params.target = pickle.dumps(self._record_step_states[0].params)
            # data = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *self._record_step_states).params
            # record.params.encoded_bytes.extend([serialization.to_bytes(data)])
            record.params.encoded_bytes.extend([serialization.to_bytes(s.params) for s in self._record_step_states])

        # Store step_state trajectory
        if step_states and self._record_step_states and len(self._record_step_states) > 0:
            record.step_states.target = pickle.dumps(self._record_step_states[0])
            # data = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *self._record_step_states)
            # record.step_states.encoded_bytes.extend([serialization.to_bytes(data)])
            record.step_states.encoded_bytes.extend([serialization.to_bytes(ss) for ss in self._record_step_states])

        return record

    @property
    def eps(self) -> int:
        return self._eps

    @property
    def phase(self) -> float:
        """Phase shift of the node: max phase over all incoming blocking & non-skipped connections."""
        # Recalculate phase once per episode.
        if self._phase is None:
            try:
                return max([0.0] + [i.phase * 1.002 for i in self.inputs if not i.skip])
                # return max([0.] + [i.phase for i in self.inputs if i.blocking and not i.skip])
            except RecursionError as e:
                msg = (
                    "The constructed graph is not DAG. To break an algebraic loop, "
                    "either skip a connection or make the connection non-blocking."
                )
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
        cls_str = self.__class__.__module__ + "/" + self.__class__.__qualname__

        info = log_pb2.NodeInfo(
            name=self.name,
            cls=cls_str,
            rate=self.rate,
            stateful=self.stateful,
            advance=self.advance,
            phase=self.phase,
            delay_sim=self.output.delay_sim.info,
            delay=self.output.delay,
            scheduling=self.scheduling,
        )
        info.inputs.extend([i.info for i in self.inputs])
        return info

    @property
    def inputs_dict(self) -> Dict[str, Input]:
        return {i.input_name: i for i in self.inputs}

    @staticmethod
    def from_info(info: log_pb2.NodeInfo) -> "BaseNode":
        """Creates a BaseNode object from a node info object.

        Make sure to call connect_from_info() on the resulting BaseNode object to connect it to the rest of the graph.

        A Basenode object is instantiated instead of the subclass. This means that subclass information is ignored and
        lost silently. Some notes:
        - This is useful for reconstructing a node if you want to use the node for inferring basic properties (e.g. phase).
        - This is not useful for simulation (and will fail).
        - Use pickling and unpickling if you want to use the node for simulation (e.g. step, reset, etc...).
        """
        # Initializes a node from a NodeInfo proto log
        node = BaseNode(
            name=info.name,
            rate=info.rate,
            delay_sim=GMM.from_info(info.delay_sim),
            delay=info.delay,
            advance=info.advance,
            stateful=info.stateful,
        )
        return node

    def unpickle(self, nodes: Dict[str, "Node"]):
        """Unpickles the connections of the node.

        This is done after all nodes are created, so that the connections can be set to the correct nodes.

        If the node is not fully unpickled after __setstate__, set self._unpickled to `False` in __setstate__
        and overwrite this unpickle method (where you can set self._unpickled to `True` after this method is called).
        This may happen in situations where:
         - Some attributes need to be replaced with references to objects
         - Some attributes need to be re-initialized (e.g. sockets, publishers, subscribers, etc.)
        """
        for i, input in enumerate(self.inputs):
            input.unpickle(nodes)  # Returns directly if already unpickled

    def connect_from_info(self, infos: List[log_pb2.InputInfo], nodes: Dict[str, "BaseNode"]):
        # Convert to list
        if isinstance(infos, log_pb2.InputInfo):
            infos = [infos]

        for info in infos:
            output_name = info.output
            output_node = nodes[output_name]

            # Connects a node to another node from an InputInfo proto log
            self.connect(
                output_node,
                blocking=info.blocking,
                skip=info.skip,
                delay_sim=GMM.from_info(info.delay_sim),
                delay=info.delay,
                jitter=info.jitter,
                name=info.name,
            )

    def log(self, id: str, value: Optional[Any] = None, log_level: Optional[int] = None):
        log_level = log_level if isinstance(log_level, int) else self.log_level
        log(self.name, self.color, min(log_level, self.log_level), id, value)

    def _set_ts_start(self, ts_start: float):
        assert isinstance(self._ts_start, Future)
        self._ts_start.set_result(ts_start)
        self._ts_start = ts_start

    def _submit(self, fn, *args, stopping: bool = False, **kwargs):
        with self._lock:
            if self._state in [READY, STARTING, READY_TO_START, RUNNING] or stopping:
                f = self._executor.submit(fn, *args, **kwargs)
                self._q_task.append((f, fn, args, kwargs))
                f.add_done_callback(self._done_callback)
            else:
                self.log("SKIPPED", fn.__name__, log_level=DEBUG)
                f = Future()
                f.cancel()
        return f

    def _done_callback(self, f: Future):
        e = f.exception()
        if e is not None and e is not CancelledError:
            error_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            log(self.name, "red", ERROR, "ERROR", error_msg)

    def get_step_state(self, graph_state: Union[GraphState, None], name: str = None) -> Union[StepState, None]:
        """Returns the step state of itself, or another node based on the name.

        If the node is not found, because the graph state is None or the node is not in the graph state,
        None is returned.
        """
        name = name if isinstance(name, str) else self.name
        if graph_state is None:
            return None
        else:
            return graph_state.nodes.get(name, None)

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
            wc_sleep = max(0.0, wc_passed_target - wc_passed)
            time.sleep(wc_sleep)

    def connect(
        self,
        node: "Node",
        blocking: bool,
        delay_sim: Distribution = None,
        delay: float = None,
        window: int = 1,
        skip: bool = False,
        jitter: int = LATEST,
        name: Optional[str] = None,
    ):
        # Use zero deterministic delay if no simulated delay distribution is specified
        delay_sim = delay_sim if delay_sim is not None else Gaussian(0.0, 0.0)

        # Create new input
        name = name if isinstance(name, str) else node.output.name
        assert name not in [i.input_name for i in self.inputs], "Cannot use the same input name for more than one input."
        assert node.name not in [
            i.output.node.name for i in self.inputs
        ], "Cannot use the same output source for more than one input."
        i = Input(self, node.output, window, blocking, skip, jitter, delay, delay_sim, name)
        self.inputs.append(i)

        # Register the input with the output of the specified node
        node.output.connect(i)

    def get_connected_output(self, input_name: str) -> Output:
        """Get the output channel of a node that is connected to the input of this node based on the input name.

        :param input_name: The name of the input corresponding to the connected output channel. This name may be different
                           from the name of the connected node corresponding to the output channel.
        :return: The connected output channel.
        """
        output = None
        for i in self.inputs:
            if i.input_name == input_name:
                output = i.output
                break
        assert output is not None, f"No input named `{input_name}` found!"
        return output

    def get_connected_input(self, name: str) -> Input:
        """Get the input channel of a node that is connected to the output of this node.

        :param name: The name of the node corresponding to the connected input channel.
        :return: The connected input channel.
        """
        _input = None
        for i in self.output.inputs:
            if i.node.name == name:
                _input = i
                break
        assert _input is not None, f"No connected input `{name}` found!"
        return _input

    @abc.abstractmethod
    def step(self, step_state: StepState) -> Tuple[StepState, Output]:
        raise NotImplementedError

    @abc.abstractmethod
    def startup(self, graph_state: GraphState = None, timeout: float = None) -> Union[bool, jax.Array]:
        """Starts the node in the state specified by graph_state."""
        return True

    def _step(self, step_state: StepState) -> Tuple[StepState, Output]:
        """Internal step function that is called in push_step().
        This function can be overridden/wrapped without affecting step() directly.
        Hence, it does not change the effect of the step() function.
        """
        new_step_state, output = self.step(step_state)

        # Update step_state (increment sequence number)
        if new_step_state is not None:
            new_step_state = new_step_state.replace(seq=new_step_state.seq + 1)

        # Block until output is ready
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else True, output)

        return new_step_state, output

    def _reset(self, graph_state: GraphState, clock: int = SIMULATED, real_time_factor: Union[int, float] = FAST_AS_POSSIBLE):
        assert self.unpickled, (
            "Node must be fully unpickled before it can be reset. "
            "This may mean that the node has some additional unpickling routines to do."
            "For example, some node attributes may need to be set after the node has been unpickled."
        )
        assert self._state in [STOPPED, READY], f"{self.name} must first be stopped, before it can be reset"
        assert (
            real_time_factor > 0 or clock == SIMULATED
        ), "Real time factor must be greater than zero if clock is not simulated"

        # Determine whether to run synchronously or asynchronously
        self._sync = SYNC if clock == SIMULATED else ASYNC
        assert not (
            clock in [WALL_CLOCK] and self._sync in [SYNC]
        ), "You can only simulate synchronously, if the clock=`SIMULATED`."

        # Warmup the node
        if not self._has_warmed_up:
            self.warmup(graph_state)

        # Save run configuration
        self._clock = clock  #: Simulate timesteps
        self._real_time_factor = real_time_factor  #: Scaling of simulation speed w.r.t wall clock

        # Up the episode counter (must happen before resetting outputs & inputs)
        self._eps += 1

        # Reset every run
        self._tick = 0
        self._phase_scheduled = 0.0  #: Structural phase shift that the step scheduler takes into account
        self._phase, self._phase_dist = None, None
        self._phase = self.phase
        self._phase_dist = self.phase_dist
        self._record = None
        self._step_state = graph_state.nodes[self.name]

        # Log
        self._discarded = 0  # Number of discarded records after reaching self.max_records
        self._record_step_states: List[Union[StepState, bytes]] = []
        self._record_outputs: List[Union[Output, bytes]] = []

        # Set starting ts
        self._ts_start = Future()  #: The starting timestamp of the episode.

        # Initialize empty queues
        self.q_tick = deque()
        self.q_ts_scheduled = deque()
        self.q_ts_output_prev = deque()
        self.q_ts_step = deque()
        self.q_rng_step = deque()

        # Get rng for delay sampling
        rng = self._step_state.rng
        rng = jnp.array(rng) if isinstance(rng, onp.ndarray) else rng  # Keys will differ for jax vs numpy

        # Reset output
        # NOTE: This is hacky because we reuse the seed.
        # However, changing the seed of the step_state would break the reproducibility between graphs (compiled, async).
        self.output.reset(rng)

        # Reset all inputs and output
        rngs_in = rnd.split(rng, num=len(self.inputs))
        [i.reset(r, self._step_state.inputs[i.input_name]) for r, i in zip(rngs_in, self.inputs)]

        # Set running state
        self._state = READY
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

    def _startup(self, graph_state: GraphState = None, timeout: float = None) -> Future:
        assert self._state in [READY], f"{self.name} must first be reset, before it can start running."

        def _starting() -> Union[bool, jax.Array]:
            res = self.startup(graph_state, timeout=timeout)

            # Set running state
            self._state = READY_TO_START
            self.log(RUNNING_STATES[self._state], log_level=DEBUG)
            return res

        with self._lock:
            # Then, flip running state so that no more tasks can be scheduled
            # This means that
            self._state = STARTING
            self.log(RUNNING_STATES[self._state], log_level=DEBUG)

            # First, submit _stopping task
            f = self._submit(_starting)
        return f

    def _start(self, start: float):
        assert self._state in [READY_TO_START], f"{self.name} must first be ready to start (i.e. call ._startup), before it can start running."
        assert self._has_warmed_up, f"{self.name} must first be warmed up, before it can start running."

        # Set running state
        self._state = RUNNING
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

        # Create logging record
        self._set_ts_start(start)
        self._record = log_pb2.NodeRecord(
            info=self.info,
            sync=self._sync,
            clock=self._clock,
            real_time_factor=self._real_time_factor,
            ts_start=start,
            rng=self.output._dist_state.rng.tolist(),
        )

        # Start all inputs and output
        [i.start(record=self._record.inputs.add()) for i in self.inputs]
        self.output.start()

        # Set first last_output_ts equal to phase (as if we just finished our previous output).
        self.q_ts_output_prev.append(0.0)

        # NOTE: Deadlocks may occur when num_tokens is chosen too low for cyclical graphs, where a low rate node
        #       depends (blocking) on a high rate node, while the high rate node depends (skipped, non-blocking)
        #       on the low rate node. In that case, the num_token of the high-rate node must be at least
        #       (probably more) the rate multiple + 1. May be larger if there are delays, etc...
        # Queue first two ticks (so that output_ts runs ahead of message)
        # The number of tokens > 1 determines "how far" into the future the
        # output timestamps are simulated when clock=simulated.
        num_tokens = 10  # todo: find non-heuristic solution. Add tokens adaptively based on requests from downstream nodes?
        # if self.name in ["observer"]:
        #     num_tokens = 4
        # else:
        #     num_tokens = 2
        self.q_tick.extend((True,) * num_tokens)

        # Push scheduled ts
        _f = self._submit(self.push_scheduled_ts)
        return _f

    def _stop(self, timeout: Optional[float] = None) -> Future:
        # Pass here, if we are not running
        if self._state not in [RUNNING]:
            self.log("", f"{self.name} is not running, so it cannot be stopped.", log_level=DEBUG)
            f = Future()
            f.set_result(None)
            return f
        assert self._state in [RUNNING], f"Cannot stop, because {self.name} is currently not running."

        def _stopping():
            # Stop producing messages and communicate total number of sent messages
            self.output.stop()

            # Stop all channels to receive all sent messages from their connected outputs
            [i.stop().result(timeout=timeout) for i in self.inputs]

            # Record last step_state
            if self._step_state is not None and len(self._record_step_states) < self.max_records + 1:
                self._record_step_states.append(self._step_state)
                self._step_state = None

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

    def push_phase_shift(self):
        # If all blocking delays are known, and we know the expected next step timestamp
        has_all_ts_max = all([len(i.q_ts_max) > 0 for i in self.inputs if i.blocking])
        has_scheduled_ts = len(self.q_ts_scheduled) > 0
        has_last_output_ts = len(self.q_ts_output_prev) > 0
        if has_scheduled_ts and has_last_output_ts and has_all_ts_max:
            self.log("push_phase_shift", log_level=DEBUG)

            # Grab blocking delays from queues and calculate max delay
            ts_max = [i.q_ts_max.popleft() for i in self.inputs if i.blocking]
            ts_max = max(ts_max) if len(ts_max) > 0 else 0.0

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
            if self.scheduling in [FREQUENCY]:
                self._phase_scheduled += max(0, phase_last - phase_scheduled)
            else:  # self.scheduling in [PHASE]
                self._phase_scheduled = 0.0

            # Calculate starting timestamp for the step call
            ts_step = ts_scheduled + phase

            # Sample delay if we simulate the clock
            delay = self.output.sample_delay() if self._clock in [SIMULATED] else None

            # Create step record
            record_step = log_pb2.StepRecord(
                tick=tick,
                ts_scheduled=ts_scheduled,
                ts_max=ts_max,
                ts_output_prev=ts_output_prev,
                ts_step=ts_step,  # May be overwritten in _step --> see push_step
                phase=phase,
                phase_scheduled=phase_scheduled,
                phase_inputs=phase_inputs,
                phase_last=phase_last,
                phase_overwrite=0.,  # May be overwritten in _step --> see push_step
            )
            self.q_ts_step.append((tick, ts_step, delay, record_step))

            # Predetermine output timestamp when we simulate the clock
            if self._clock in [SIMULATED]:
                # Determine output timestamp
                ts_output = ts_step + delay
                _, ts_output_wc = self.now()
                header = log_pb2.Header(eps=self._eps, seq=tick, ts=log_pb2.Time(sc=ts_output, wc=ts_output_wc))
                self.output.push_ts_output(ts_output, header)

                # todo: Somehow, num_tokens can be lowered if we would sleep here (because push_ts_output runs before push_scheduled_ts).
                #  time.sleep(1.0)

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
            inputs = {}
            for i in self.inputs:
                input_state = self._step_state.inputs[i.input_name]
                grouped = i.q_grouped.popleft()
                for seq, ts_sent, ts_recv, msg in grouped:
                    input_state = i._jit_update_input_state(input_state, seq, ts_sent, ts_recv, msg)
                inputs[i.input_name] = input_state
            # inputs = FrozenDict({i.input_name: i.q_grouped.popleft() for i in self.inputs})

            # Update StepState with grouped messages
            # todo: have a single buffer for step_state used for both in and out
            step_state = self._step_state.replace(seq=tick, ts=ts_step_sc, inputs=inputs)

            # Log step_state
            if len(self._record_step_states) < self.max_records:
                self._record_step_states.append(step_state)

            # Run step and get msg
            new_step_state, output = self._step(step_state)

            # Log output
            if (
                output is not None and len(self._record_outputs) < self.max_records
            ):  # Agent returns None when we are stopping/resetting.
                self._record_outputs.append(output)

            # Update step_state (sequence number is incremented in ._step())
            if new_step_state is not None:
                self._step_state = new_step_state

            # Determine output timestamp
            if self._clock in [SIMULATED]:
                assert delay_sc is not None
                ts_output_sc = ts_step_sc + delay_sc
                _, ts_output_wc = self.now()
                phase_overwrite = 0.0
                ts_step_sc = ts_step_sc
            else:
                assert delay_sc is None
                ts_output_sc, ts_output_wc = self.now()
                # ts_step_sc (i.e. step_state.ts) may be overwritten in the step function (i.e. to adjust to later time when sensor data was taken).
                # Therefore, we use the potentially overwritten step_state.ts to calculate the delay.
                new_step_ts_sc = float(step_state.ts)
                if new_step_ts_sc < ts_step_sc:
                    msg = ("Did you overwrite `step_state.ts` in the step function? Make sure it's not smaller than the original `step_state.ts`. "
                           "Are the clocks producing the adjusted step_state.ts and the original step_state.ts consistent?")
                    self.log(
                        "timestamps",
                        msg,
                        log_level=ERROR,
                    )
                    raise ValueError(msg)
                delay_sc = ts_output_sc - new_step_ts_sc
                if delay_sc <= 0:
                    msg = (
                        "Did you overwrite `step_state.ts` in the step function? Make sure it does not exceed the current time (i.e. `self.now()[0]`)"
                        "Are the clocks producing the adjusted step_state.ts and the original step_state.ts consistent?")
                    self.log(
                        "timestamps",
                        msg,
                        log_level=ERROR,
                    )
                    raise ValueError(msg)

                # Re-calculate phase overwrite and ts_step_sc
                phase_overwrite = new_step_ts_sc - ts_step_sc
                ts_step_sc = new_step_ts_sc

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
            record_step.ts_step = ts_step_sc
            record_step.phase_overwrite = phase_overwrite
            record_step.ts_output = ts_output_sc
            record_step.comp_delay.CopyFrom(log_pb2.Time(sc=ts_output_sc - ts_step_sc, wc=ts_output_wc - ts_step_wc))

            # Push output
            if output is not None:  # Agent returns None when we are stopping/resetting.
                self.output.push_output(output, header)

            # Add step record
            if len(self._record.steps) < self.max_records:
                self._record.steps.append(record_step)
            elif self._discarded == 0:
                self.log(
                    "recording",
                    "Reached max number of records (timings, outputs, step_state). So no longer recording.",
                    log_level=WARN,
                )
                self._discarded += 1
            else:
                self._discarded += 1

            # Only schedule next step if we are running
            if self._state in [RUNNING]:
                # Add token to tick queue (ticks are incremented in push_scheduled_ts function)
                self.q_tick.append(True)

                # Schedule next step (does not consider scheduling shifts)
                _f = self._submit(self.push_scheduled_ts)


class Node(BaseNode):
    def default_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> Params:
        """Default params of the node."""
        return Empty()

    def default_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> State:
        """Default state of the node."""
        return Empty()

    def default_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, InputState]:  # Dict[str, InputState]:
        """Default inputs of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num=len(self.inputs))
        inputs = dict()
        for i, rng_output in zip(self.inputs, rngs):
            window = i.window
            seq = onp.arange(-window, 0, dtype=onp.int32)
            ts_sent = 0 * onp.arange(-window, 0, dtype=onp.float32)
            ts_recv = 0 * onp.arange(-window, 0, dtype=onp.float32)
            outputs = [i.output.node.default_output(rng_output, graph_state) for _ in range(window)]
            inputs[i.input_name] = InputState.from_outputs(seq, ts_sent, ts_recv, outputs)
        return FrozenDict(inputs)

    @abc.abstractmethod
    def default_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> BaseOutput:
        """Default output of the node.
        NOTE: This is also used to determine the shape of every leaf in the output tree.
              Therefore, it should be able to return a valid output even if no graph_state is provided.
        """
        return Empty()

    def default_step_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> StepState:
        """Default step state of the node.

        Note: It can happen that, in order to get the default step_state, we need to have the step_states of other nodes.
        :param rng: The random number generator.
        :param graph_state: The graph state that contains the step states of all nodes required to get the default step state.
        """
        # Get default rng
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rng_params, rng_state, rng_step, rng_inputs = jax.random.split(rng, num=4)

        # Get default graph state
        graph_state = graph_state if graph_state is not None else GraphState(eps=onp.int32(0), nodes=FrozenDict({}))

        # Get step states
        step_states = graph_state.nodes
        step_states = step_states.unfreeze() if isinstance(step_states, FrozenDict) else step_states
        graph_state = graph_state.replace(nodes=step_states)

        # Grab preset params and state if available
        preset_eps = graph_state.eps
        preset_seq = graph_state.nodes[self.name].seq if self.name in graph_state.nodes else onp.int32(0)
        preset_ts = graph_state.nodes[self.name].ts if self.name in graph_state.nodes else onp.float32(0.)
        preset_params = graph_state.nodes[self.name].params if self.name in graph_state.nodes else None
        preset_state = graph_state.nodes[self.name].state if self.name in graph_state.nodes else None
        preset_inputs = graph_state.nodes[self.name].inputs if self.name in graph_state.nodes else None
        # Params first, because the state may depend on them
        params = self.default_params(rng_params, graph_state) if preset_params is None else preset_params
        step_states[self.name] = StepState(rng=rng_step, params=params, state=None, inputs=None, eps=preset_eps, seq=preset_seq, ts=preset_ts)
        # Then, get the state (which may depend on the params)
        state = self.default_state(rng_state, graph_state) if preset_state is None else preset_state
        step_states[self.name] = StepState(rng=rng_step, params=params, state=state, inputs=None, eps=preset_eps, seq=preset_seq, ts=preset_ts)
        # Finally, get the inputs
        inputs = self.default_inputs(rng_inputs, graph_state) if preset_inputs is None else preset_inputs
        # Prepare step state
        step_state = StepState(rng=rng_step, params=params, state=state, inputs=inputs, eps=preset_eps, seq=preset_seq, ts=preset_ts)
        return step_state

    @abc.abstractmethod
    def startup(self, graph_state: GraphState = None, timeout: float = None) -> Union[bool, jax.Array]:
        """Starts the node in the state specified by graph_state."""
        return True

    @abc.abstractmethod
    def step(self, step_state: StepState) -> Tuple[StepState, Output]:
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self
