from typing import Any, Tuple, List, TypeVar, Dict, Union, Callable, Optional
import time
from concurrent.futures import Future
import jax
import jax.numpy as jnp
import numpy as onp
from flax.core import FrozenDict
from rexv2 import base
from rexv2.constants import Scheduling, Jitter, LogLevel, Async
from rexv2 import utils
from rexv2 import jax_utils as jutil
import supergraph.open_colors as oc
import distrax


class Connection:
    def __init__(
        self, input_node: "BaseNode", output_node: "BaseNode", blocking: bool,
            delay: float = None, delay_dist: base.DelayDistribution = None,
            window: int = 1, skip: bool = False, jitter: Jitter = Jitter.LATEST,
            input_name: str = None
    ):
        self.input_node = input_node
        self.output_node = output_node
        self.blocking = blocking
        self.delay_dist = delay_dist if delay_dist is not None else base.StaticDist.create(distrax.Normal(loc=0., scale=0.))
        self.delay = delay if delay is not None else float(self.delay_dist.quantile(0.99))
        assert self.delay >= 0, "Delay should be non-negative."
        self.window = window
        self.skip = skip
        self.jitter = jitter
        self.input_name = input_name if isinstance(input_name, str) else output_node.name

    @property
    def info(self) -> base.InputInfo:
        return base.InputInfo(
            rate=self.output_node.rate,
            window=self.window,
            blocking=self.blocking,
            skip=self.skip,
            jitter=self.jitter,
            phase=self.phase,
            delay_dist=self.delay_dist,  # todo: does not neatly pickle
            delay=self.delay,
            name=self.input_name,
            output=self.output_node.name,
        )

    @property
    def phase(self) -> float:
        return self.output_node.phase_output + self.delay


class BaseNode:
    def __init__(self, name: str, rate: float, delay: float = None, delay_dist: base.DelayDistribution = None,
                 advance: bool = False, scheduling: Scheduling = Scheduling.FREQUENCY,
                 color: str = None, order: int = None):
        """Base node class. All nodes should inherit from this class.

        :param name: The name of the node (unique).
        :param rate: The rate of the node (Hz).
        :param delay: The expected computation delay of the node (s).
        :param delay_dist: The computation delay distribution of the node for simulation.
        :param advance: Whether the node's step triggers when all inputs are ready, or throttles until the scheduled time.
        :param scheduling: The scheduling of the node. If FREQUENCY, the node is scheduled at a fixed rate, while ignoring
                           any phase shift w.r.t the clock. If PHASE, the node steps are scheduled at a fixed rate and phase
                           w.r.t the clock.
        :param color: The color of the node (for visualization).
        :param order: The order of the node (for visualization).
        """
        self.name = name
        self.rate = rate
        self.delay_dist = delay_dist if delay_dist is not None else base.StaticDist.create(distrax.Normal(loc=0., scale=0.))
        if isinstance(self.delay_dist, base.TrainableDist):
            raise NotImplementedError("Cannot have trainable distribution for computation delay.")
        self.delay = delay if delay is not None else float(self.delay_dist.quantile(0.99))  # take 99th percentile of delay
        assert self.delay >= 0, "Delay should be non-negative."
        self.advance = advance
        self.scheduling = scheduling
        self.color = color
        self.order = order
        self.outputs: Dict[str, Connection] = {}  # Outgoing edges. Keys are the actual node names incident to the edge.
        self.inputs: Dict[str, Connection] = {}  # Incoming edges. Keys are the input_names of the nodes from which the edge originates. May be different from the actual node names.

        # Record settings
        self.record_setting = dict(params=False, rng=False, inputs=False, state=False, output=False)
        self.max_records = 20000

        # Async
        self._async_now: Optional[Callable[[], float]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        identifier = cls.__name__
        if 'init_output' in cls.__dict__:
            cls.init_output = jutil.no_weaktype(identifier=f"{identifier}.init_output")(cls.init_output)
        if 'init_params' in cls.__dict__:
            cls.init_params = jutil.no_weaktype(identifier=f"{identifier}.init_params")(cls.init_params)
        if 'init_state' in cls.__dict__:
            cls.init_state = jutil.no_weaktype(identifier=f"{identifier}.init_state")(cls.init_state)
        if 'init_inputs' in cls.__dict__:
            cls.init_inputs = jutil.no_weaktype(identifier=f"{identifier}.init_inputs")(cls.init_inputs)
        if 'init_step_state' in cls.__dict__:
            cls.init_step_state = jutil.no_weaktype(identifier=f"{identifier}.init_step_state")(cls.init_step_state)
        if 'step' in cls.__dict__:
            cls.step = jutil.no_weaktype(identifier=f"{identifier}.step")(cls.step)

    @classmethod
    def from_info(cls, info: base.NodeInfo, *args, **kwargs):
        """Creates a subclass object from a node info object.

        Make sure to call connect_from_info() on the resulting subclass object to connect it to the rest of the graph.

        A subclass object is instantiated instead of the BaseNode. This means that subclass information is preserved,
        but must also be provided in the *args and **kwargs.

        Moreover, the signature of the subclass must be the same as the BaseNode, except for the additional *args and **kwargs.
        """
        return cls(
            *args,  # Pass additional arguments to the subclass
            name=info.name, rate=info.rate, delay_dist=info.delay_dist, delay=info.delay, advance=info.advance,
            scheduling=info.scheduling, color=info.color, order=info.order,
            **kwargs  # Pass additional keyword arguments to the subclass
        )

    def connect_from_info(self, infos: [str, base.InputInfo], nodes: Dict[str, "BaseNode"]):
        """Connects the node to other nodes based on the input infos.

        :param infos: The input infos. It is a dictionary of input names to input infos.
        :param nodes: The dictionary of node names to node objects.
        """
        for input_name, info in infos.items():
            output_node = nodes[info.output]
            self.connect(
                output_node, blocking=info.blocking, delay=info.delay, delay_dist=info.delay_dist,
                window=info.window, skip=info.skip, jitter=info.jitter, name=input_name
            )

    @property
    def info(self) -> base.NodeInfo:
        return base.NodeInfo(
            rate=self.rate,
            advance=self.advance,
            scheduling=self.scheduling,
            phase=self.phase,
            delay_dist=self.delay_dist,  # todo: does not neatly pickle
            delay=self.delay,
            inputs={c.output_node.name: c.info for i, c in self.inputs.items()},  # Use name in context of graph, instead of shadow name
            name=self.name,
            cls=self.__class__.__module__ + "/" + self.__class__.__qualname__,
            color=self.color,
            order=self.order,
        )

    @property
    def log_level(self):
        return utils.NODE_LOG_LEVEL.get(self, LogLevel.WARN)

    @property
    def log_color(self):
        return utils.NODE_COLOR.get(self, "green")

    @property
    def fcolor(self):
        """Get the face color of the node."""
        color = self.color if isinstance(self.color, str) else "gray"
        ecolors, fcolors = oc.cscheme_fn({self.name: color})
        return fcolors[self.name]

    @property
    def ecolor(self):
        """Get the edge color of the node."""
        color = self.color if isinstance(self.color, str) else "gray"
        ecolors, fcolors = oc.cscheme_fn({self.name: color})
        return ecolors[self.name]

    @property
    def phase(self) -> float:
        """Phase shift of the node: max phase over all incoming blocking & non-skipped connections."""
        # Recalculate phase once per episode.
        try:
            return max([0.0] + [i.phase * 1.002 for i in self.inputs.values() if not i.skip])
            # return max([0.] + [i.phase for i in self.inputs if i.blocking and not i.skip])
        except RecursionError as e:
            msg = (
                "The constructed graph is not DAG. To break an algebraic loop, "
                "either skip a connection or make the connection non-blocking."
            )
            utils.log(self.name, "red", LogLevel.ERROR.value, "ERROR", msg)
            raise e

    @property
    def phase_output(self) -> float:
        return self.phase + self.delay

    def log(self, id: Union[str, Async], value: Any = None, log_level: int = None):
        if not utils.NODE_LOGGING_ENABLED:
            return
        log_level = self.log_level if log_level is None else log_level
        color = self.log_color
        utils.log(f"{self.name}", color, min(log_level, self.log_level), id, value)

    def set_record_settings(self, params: bool = None, rng: bool = None, inputs: bool = None, state: bool = None, output: bool = None):
        self.record_setting["params"] = params if params is not None else self.record_setting["params"]
        self.record_setting["rng"] = rng if rng is not None else self.record_setting["rng"]
        self.record_setting["inputs"] = inputs if inputs is not None else self.record_setting["inputs"]
        self.record_setting["state"] = state if state is not None else self.record_setting["state"]
        self.record_setting["output"] = output if output is not None else self.record_setting["output"]

    def now(self) -> float:
        """Get the passed time since start of episode according to the simulated and wall clock.

        Only returns > 0 timestamps if running asynchronously.
        """
        if self._async_now is not None:
            return self._async_now()
        else:
            return 0.  # Return 0 if not running asynchronously

    def connect(self, output_node: "BaseNode", blocking: bool = False,
                delay: float = None, delay_dist: base.DelayDistribution = None,
                window: int = 1, skip: bool = False, jitter: Jitter = Jitter.LATEST,
                name: str = None):
        """Connects the node to another node.

        :param output_node: The node to connect to.
        :param blocking: Whether the connection is blocking.
        :param delay: The expected communication delay of the connection.
        :param delay_dist: The communication delay distribution of the connection for simulation.
        :param window: The window size of the connection. It determines how many output messages are used as input to
                       the .step() function.
        :param skip: Whether to skip the connection. It resolves acyclic dependencies, by skipping the output if it arrives
                     at the same time as the start of the .step() function (i.e. step_state.ts).
        :param jitter: How to deal with jitter of the connection. If LATEST, the latest messages are used. If BUFFER, the
                       messages are buffered and used in accordance with the expected delay.
        :param name: A shadow name for the connected node. If None, the name of the output node is used.
        """
        name = name if isinstance(name, str) else output_node.name
        connection = Connection(self, output_node, blocking, delay, delay_dist, window, skip, jitter, input_name=name)
        self.inputs[name] = connection
        output_node.outputs[self.name] = connection

    def get_step_state(self, graph_state: Union[base.GraphState, None], name: str = None) -> Union[base.StepState, None]:
        """Grab the step state of a node.

        :param graph_state: The graph state.
        :param name: The name of the node to get the step state of.
                     If not specified, the step state of the node itself is returned (if it exists).
        :return: The step state of the node if it exists, else None.
        """
        raise NotImplementedError("Refactor to use graph_state.attribute.get(self.name, ...)")
        name = name if isinstance(name, str) else self.name
        if graph_state is None:
            return None
        else:
            return graph_state.step_state.get(name, None)

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.Params:
        """Init params of the node.

        The params of the node are usually considered to be static during an episode (e.g. dynamic params, network weights).

        At this point, the graph state may contain the params of other nodes required to get the default params.
        The order of node initialization can be specified in Graph.init(... order=[node1, node2, ...]).

        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default params.
        :return: The default params of the node.
        """
        return base.Empty()

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.State:
        """Init state of the node.

        The state of the node is usually considered to be dynamic during an episode (e.g. position, velocity).

        At this point, the params of all nodes are already initialized and present in the graph state (if specified).
        Moreover, the state of other nodes required to get the default state may also be present in the graph state.
        The order of node initialization can be specified in Graph.init(... order=[node1, node2, ...]).

        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default state.
        :return: The default state of the node.
        """
        return base.Empty()

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.Output:
        """Default output of the node.

        It is common for nodes not to share their full state with other nodes.
        Hence, the output of the node is usually a subset of the state that is shared with other nodes.

        These outputs are used to initialize the inputs of other nodes. In the case where a node may not have received
        any messages from a connected node, the default outputs are used to fill the input buffers.

        Usually, the params and state of every node are already initialized and present in the graph state.
        However, it's usually preferred to define the output without relying on the graph state.

        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default output.
        :return: The default output of the node.
        """
        return base.Empty()

    def init_inputs(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> FrozenDict[str, base.InputState]:
        """Default inputs of the node.

        Fills the input buffers of the node with the default outputs of the connected nodes.
        These input buffers are usually only used during the first few steps of the simulation when the node has not yet
        received enough messages from the connected nodes to fill the buffers.

        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default inputs.
        :return: The default inputs of the node.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num=len(self.inputs))
        inputs = dict()
        for (input_name, c), rng_output in zip(self.inputs.items(), rngs):
            window = c.window
            seq = onp.arange(-window, 0, dtype=onp.int32)
            ts_sent = 0 * onp.arange(-window, 0, dtype=onp.float32)
            ts_recv = 0 * onp.arange(-window, 0, dtype=onp.float32)
            outputs = [c.output_node.init_output(rng_output, graph_state) for _ in range(window)]
            delay_dist = c.delay_dist  # NOTE!: Set trainable parameters from the graph state, if needed
            inputs[input_name] = base.InputState.from_outputs(seq, ts_sent, ts_recv, outputs, delay_dist)
        return FrozenDict(inputs)

    def init_step_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.StepState:
        """Initializes the step state of the node.

        The step state is a dataclass that contains all data to run the seq'th step of the node at time ts,
        It contains the params, state, inputs[some_name], eps, seq, and ts.

        Note that this function is **NOT** called in graph.init(...). It mostly serves as a helper function to get a
        representative step state for the node. This is useful for debugging, testing, and pre-compiling the .step method in isolation.

        The order of how init_params, init_state, and init_inputs are called in this function is similar to
        how they are called in the Graph.init(...) function.

        Note that this may fail when this node's initialization depends on the params, state, or inputs of other nodes.
        In such cases, the user can provide a graph_state with the necessary information to get the default step state.

        In some cases, the default step state may depend on the step states of other nodes. In such cases, the graph state
        must be provided to get the default step state.
        :param rng: Random number generator.
        :param graph_state: The graph state that may be used to get the default step state.
        :return: The default step state of the node.
        """
        # Prepare random number generators
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rng_step, rng_params, rng_state, rng_inputs = jax.random.split(rng, num=4)

        # Gather pre-set graph state
        graph_state = graph_state or base.GraphState()
        rng = graph_state.rng.unfreeze() if isinstance(graph_state.rng, FrozenDict) else graph_state.rng
        seq = graph_state.seq.unfreeze() if isinstance(graph_state.seq, FrozenDict) else graph_state.seq
        ts = graph_state.ts.unfreeze() if isinstance(graph_state.ts, FrozenDict) else graph_state.ts
        params = graph_state.params.unfreeze() if isinstance(graph_state.params, FrozenDict) else graph_state.params
        state = graph_state.state.unfreeze() if isinstance(graph_state.state, FrozenDict) else graph_state.state
        inputs = graph_state.inputs.unfreeze() if isinstance(graph_state.inputs, FrozenDict) else graph_state.inputs
        graph_state = graph_state.replace(rng=rng, seq=seq, ts=ts, params=params, state=state, inputs=inputs)

        # Get default step state
        rng[self.name] = graph_state.rng.get(self.name, rng_step)
        seq[self.name] = graph_state.seq.get(self.name, onp.int32(0))
        ts[self.name] = graph_state.ts.get(self.name, onp.float32(0.0))
        params[self.name] = graph_state.params.get(self.name, self.init_params(rng_params, graph_state))
        state[self.name] = graph_state.state.get(self.name, self.init_state(rng_state, graph_state))
        inputs[self.name] = graph_state.inputs.get(self.name, self.init_inputs(rng_inputs, graph_state))
        return graph_state.step_state[self.name]

    def startup(self, graph_state: base.GraphState, timeout: float = None):
        """Starts the node in the state specified by graph_state.

        Only ran when running asynchronously.
        :param graph_state: The graph state.
        :param timeout: The timeout of the startup.
        """
        return True

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
        """Step the node for the seq'th time step at time ts.

        This step function is the main function that is called to update the state of the node and produce an output, that
        is sent to the connected nodes. The step function is called at the rate of the node.

        The step_state is a dataclass that contains all data to run the seq'th step at time ts, during episode `eps`
        Specifically, it contains the params, state, inputs[connected_node_name], eps, seq, and ts.

        The inputs are interfaced as inputs[some_name][window_index].data. A window_index of -1 leads to the most recent message.
        Auxiliary information such as the sequence number, and the time sent and received are also stored in the InputState.
        This information can be accessed as inputs[some_name][window_index].seq, inputs[some_name][window_index].ts_sent, and
        inputs[some_name][window_index].ts_recv, respectively, where ts_sent and ts_recv are the time the message was sent and
        received, respectively.

        Note that the user is expected to update the state (and rng if used), but not the seq and ts, as they are
        automatically updated.

        :param step_state: The step state of the node.
        :return: The updated step state and the output of the node.
        """
        raise NotImplementedError

    def async_step(self, step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
        """Async step function that is called when running asynchronously.
        This function can be overridden/wrapped (e.g. jit) without affecting node.step() directly.

        This can be beneficial when settings (Clock.SIMULATED, Clock.WALL_CLOCK) require different compilation settings.
        For example, compiling for CPU in real-time, and for GPU in simulation time.
        """
        # with jax.log_compiles():
        #     same_structure(self._step_state, step_state, tag=self.name)
        #     new_step_state, output = self.step(step_state)  # Synchronizer or Node
        new_step_state, output = self.step(step_state)

        # Update step_state (increment sequence number)
        if new_step_state is not None:
            new_step_state = new_step_state.replace(seq=new_step_state.seq + 1)

        # Block until output is ready
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else True, output)
        return new_step_state, output