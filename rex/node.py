from typing import Any, Callable, Dict, Optional, Tuple, Union

import distrax
import jax
import numpy as onp
import supergraph.open_colors as oc
from flax.core import FrozenDict

from rex import base, jax_utils as jutil, utils
from rex.constants import Async, Jitter, LogLevel, Scheduling


class Connection:
    def __init__(
        self,
        input_node: "BaseNode",
        output_node: "BaseNode",
        blocking: bool,
        delay: float = None,
        delay_dist: Union[base.DelayDistribution, distrax.Distribution] = None,
        window: int = 1,
        skip: bool = False,
        jitter: Jitter = Jitter.LATEST,
        input_name: str = None,
    ):
        """Connection class that represents an edge in the graph.

        Args:
            input_node: The input node.
            output_node: The output node.
            blocking: Whether the connection is blocking.
            delay: The expected communication delay of the connection.
            delay_dist: The communication delay distribution of the connection for simulation.
            window: The window size of the connection. It determines how many output messages are used as input to
            jitter: How to deal with jitter of the connection. If `LATEST`, the latest messages are used. If `BUFFER`, the
            skip: Whether to skip the connection. It resolves cyclic dependencies, by skipping the output if it arrives
                  at the same time as the start of the `.step()` function (i.e. `step_state.ts`).
            input_name: A shadow name for the connected node. If `None`, the name of the output node is used.
        """
        self.input_node = input_node
        self.output_node = output_node
        self.blocking = blocking
        self.delay_dist = delay_dist if delay_dist is not None else base.StaticDist.create(distrax.Normal(loc=0.0, scale=0.0))
        self.delay_dist = (
            base.StaticDist.create(self.delay_dist) if isinstance(self.delay_dist, distrax.Distribution) else self.delay_dist
        )
        assert isinstance(
            self.delay_dist, base.DelayDistribution
        ), "Delay distribution should be a subclass of DelayDistribution."
        self.delay = delay if delay is not None else float(self.delay_dist.quantile(0.99))
        assert self.delay >= 0, "Delay should be non-negative."
        self.window = window
        self.skip = skip
        self.jitter = jitter
        self.input_name = input_name if isinstance(input_name, str) else output_node.name

    def set_delay(self, delay_dist: Union[base.DelayDistribution, distrax.Distribution] = None, delay: float = None):
        """Set the delay distribution and delay for the communication delay of the connection.

        Args:
            delay_dist: The delay distribution to simulate.
            delay: The delay to take into account for the phase shift.
        """
        self.delay_dist = self.delay_dist if delay_dist is not None else self.delay_dist
        self.delay_dist = (
            base.StaticDist.create(self.delay_dist) if isinstance(self.delay_dist, distrax.Distribution) else self.delay_dist
        )
        assert isinstance(
            self.delay_dist, base.DelayDistribution
        ), "Delay distribution should be a subclass of DelayDistribution."
        self.delay = delay if delay is not None else self.delay

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

    def disconnect(self):
        raise NotImplementedError("Clean disconnect is not properly implemented yet.")
        # del self.input_node.inputs[self.input_name]
        # del self.output_node.outputs[self.input_node.name]


class BaseNode:
    def __init__(
        self,
        name: str,
        rate: float,
        delay: float = None,
        delay_dist: Union[base.DelayDistribution, distrax.Distribution] = None,
        advance: bool = False,
        scheduling: Scheduling = Scheduling.FREQUENCY,
        color: str = None,
        order: int = None,
    ):
        """Base node class. All nodes should inherit from this class.

        Example: Basic template for a node class:
            ```python
            class MyNode(BaseNode):
                def __init__(self, *args, extra_arg, **kwargs):  # Optional
                    super().__init__(*args, **kwargs)
                    self.extra_arg = extra_arg

                def init_params(self, rng=None, graph_state=None):  # Optional
                    return MyParams(param1=1.0, param2=2.0)

                def init_state(self, rng=None, graph_state=None): # Optional
                    return MyState(state1=1.0, state2=2.0)

                def init_output(self, rng=None, graph_state=None):  # Required
                    return MyOutput(output1=1.0, output2=2.0)

                def init_delays(self, rng=None, graph_state=None):  # Optional
                    # Set trainable delays to values from params
                    params = graph_state.params[self.name]
                    return {"some_node": params.param1}  # Connected node name

                def startup(self, graph_state, timeout=None):  # Optional
                    # Move the robot to a starting position
                    return True

                def step(self, step_state):  # Required
                    # Unpack step state
                    params = step_state.params
                    state = step_state.state
                    inputs = step_state.inputs
                    # Calculate output
                    output = MyOutput(...)
                    # Update state
                    new_state = MyState(...)
                    return step_state.replace(state=new_state), output

                def stop(self, timeout=None):  # Optional
                    # Safely the robot at the end of the episode
                    return True
            ```

        Args:
            name: The name of the node (unique).
            rate: The rate of the node (Hz).
            delay: The expected computation delay of the node (s). Used to calculate the phase shift.
            delay_dist: The computation delay distribution of the node for simulation.
            advance: Whether the node's step triggers when all inputs are ready, or throttles until the scheduled time.
            scheduling: The scheduling of the node. If `FREQUENCY`, the node is scheduled at a fixed rate, while ignoring
                        any phase shift w.r.t the clock. If `PHASE`, the node steps are scheduled at a fixed rate and phase
                        w.r.t the clock.
            color: The color of the node (for visualization).
            order: The order of the node (for visualization).
        """
        self.name = name
        self.rate = rate
        self.delay_dist = delay_dist if delay_dist is not None else base.StaticDist.create(distrax.Normal(loc=0.0, scale=0.0))
        self.delay_dist = (
            base.StaticDist.create(self.delay_dist) if isinstance(self.delay_dist, distrax.Distribution) else self.delay_dist
        )
        if isinstance(self.delay_dist, base.TrainableDist):
            raise NotImplementedError("Cannot have trainable distribution for computation delay.")
        assert isinstance(
            self.delay_dist, base.DelayDistribution
        ), "Delay distribution should be a subclass of DelayDistribution."
        self.delay = delay if delay is not None else float(self.delay_dist.quantile(0.99))  # take 99th percentile of delay
        assert self.delay >= 0, "Delay should be non-negative."
        self.advance = advance
        self.scheduling = scheduling
        self.color = color
        self.order = order
        self.outputs: Dict[str, Connection] = {}  # Outgoing edges. Keys are the actual node names incident to the edge.
        self.inputs: Dict[
            str, Connection
        ] = {}  # Incoming edges. Keys are the input_names of the nodes from which the edge originates. May be different from the actual node names.

        # Async
        self._async_now: Optional[Callable[[], float]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        identifier = cls.__name__
        if "init_output" in cls.__dict__:
            cls.init_output = jutil.no_weaktype(identifier=f"{identifier}.init_output")(cls.init_output)
        if "init_params" in cls.__dict__:
            cls.init_params = jutil.no_weaktype(identifier=f"{identifier}.init_params")(cls.init_params)
        if "init_state" in cls.__dict__:
            cls.init_state = jutil.no_weaktype(identifier=f"{identifier}.init_state")(cls.init_state)
        if "init_inputs" in cls.__dict__:
            cls.init_inputs = jutil.no_weaktype(identifier=f"{identifier}.init_inputs")(cls.init_inputs)
        if "init_step_state" in cls.__dict__:
            cls.init_step_state = jutil.no_weaktype(identifier=f"{identifier}.init_step_state")(cls.init_step_state)
        if "step" in cls.__dict__:
            cls.step = jutil.no_weaktype(identifier=f"{identifier}.step")(cls.step)

    def set_delay(self, delay_dist: Union[base.DelayDistribution, distrax.Distribution] = None, delay: float = None):
        """Set the delay distribution and delay for the computation delay of the node.

        Args:
            delay_dist: The delay distribution to simulate.
            delay: The delay to take into account for the phase shift.
        """
        self.delay_dist = self.delay_dist if delay_dist is not None else self.delay_dist
        self.delay_dist = (
            base.StaticDist.create(self.delay_dist) if isinstance(self.delay_dist, distrax.Distribution) else self.delay_dist
        )
        if isinstance(self.delay_dist, base.TrainableDist):
            raise NotImplementedError("Cannot have trainable distribution for computation delay.")
        assert isinstance(
            self.delay_dist, base.DelayDistribution
        ), "Delay distribution should be a subclass of DelayDistribution."
        self.delay = delay if delay is not None else self.delay

    @classmethod
    def from_info(cls, info: base.NodeInfo, **kwargs: Dict[str, Any]):
        """Re-instantiates a Node from a NodeInfo object.

        Note: Don't forget to call `connect_from_info()`.
            Make sure to call connect_from_info() on the resulting subclass object to restore the connections.

        Note:
            This method attempts to restore the subclass object from the BaseNode object.
            Hence, it requires any additional arguments to be passed as keyword arguments.
            Moreover, the signature of the subclass must be the same as the BaseNode, except for the additional *args and **kwargs.

        Args:
            info: Node info object.
            **kwargs: Additional keyword arguments for the subclass.
        """
        reserved_kwargs = ["name", "rate", "delay_dist", "delay", "advance", "scheduling", "color", "order"]
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in reserved_kwargs}
        return cls(
            name=kwargs.get("name", info.name),
            rate=kwargs.get("rate", info.rate),
            delay_dist=kwargs.get("delay_dist", info.delay_dist),
            delay=kwargs.get("delay", info.delay),
            advance=kwargs.get("advance", info.advance),
            scheduling=kwargs.get("scheduling", info.scheduling),
            color=kwargs.get("color", info.color),
            order=kwargs.get("order", info.order),
            **extra_kwargs,  # Pass additional keyword arguments to the subclass
        )

    def connect_from_info(self, infos: [str, base.InputInfo], nodes: Dict[str, "BaseNode"]):
        """Connects the node to other nodes based on the input infos.

        Args:
            infos: A dictionary of input names to input infos.
            nodes: A dictionary of node names to node objects.
        """
        for input_name, info in infos.items():
            output_node = nodes[info.output]
            self.connect(
                output_node,
                blocking=info.blocking,
                delay=info.delay,
                delay_dist=info.delay_dist,
                window=info.window,
                skip=info.skip,
                jitter=info.jitter,
                name=input_name,
            )

    @property
    def info(self) -> base.NodeInfo:
        """Get the node info."""
        return base.NodeInfo(
            rate=self.rate,
            advance=self.advance,
            scheduling=self.scheduling,
            phase=self.phase,
            delay_dist=self.delay_dist,
            delay=self.delay,
            inputs={
                c.output_node.name: c.info for i, c in self.inputs.items()
            },  # Use name in context of graph, instead of shadow name
            name=self.name,
            cls=self.__class__.__module__ + "/" + self.__class__.__qualname__,
            color=self.color if self.color is not None else "gray",
            order=self.order,
        )

    @property
    def log_level(self):
        """Get the log level of the node."""
        return utils.NODE_LOG_LEVEL.get(self, LogLevel.WARN)

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
            return max([0.0] + [i.phase * 1.00 for i in self.inputs.values() if not i.skip])
            # return max([0.] + [i.phase for i in self.inputs if i.blocking and not i.skip])
        except RecursionError as e:
            msg_info = (
                "Algebraic loop detected. The connection form a cycle in the graph."
                "To break the loop, either skip a connection or make the connection non-blocking."
            )
            str_e = str(e)
            if msg_info not in str_e:
                msg = msg_info + f"\nLoop: {self.name}"
            else:
                msg = str_e + f"->{self.name}"
            raise RecursionError(msg)

    @property
    def phase_output(self) -> float:
        """Phase shift including the node's computation delay."""
        return self.phase + self.delay

    def log(self, id: Union[str, Async], value: Any = None, log_level: int = None):
        """
        Logs a message with the specified log level.

        Args:
            id: Identifier for the log message.
            value: The value to log.
            log_level: The log level to use. If None, the node's log level is used.
        """
        if not utils.NODE_LOGGING_ENABLED:
            return
        log_level = self.log_level if log_level is None else log_level
        utils.log(f"{self.name}", min(log_level, self.log_level), id, value)

    def now(self) -> float:
        """Get the passed time since start of episode according to the simulated and wall clock.

        Returns:
            Time since start of episode. Only returns > 0 timestamps if running asynchronously.
        """
        if self._async_now is not None:
            return self._async_now()
        else:
            return 0.0  # Return 0 if not running asynchronously

    def connect(
        self,
        output_node: "BaseNode",
        blocking: bool = False,
        delay: float = None,
        delay_dist: Union[base.DelayDistribution, distrax.Distribution] = None,
        window: int = 1,
        skip: bool = False,
        jitter: Jitter = Jitter.LATEST,
        name: str = None,
    ):
        """Connects the node to another node.

        Args:
            output_node: The node to connect to.
            blocking: Whether the connection is blocking.
            delay: The expected communication delay of the connection.
            delay_dist: The communication delay distribution of the connection for simulation.
            window: The window size of the connection. It determines how many output messages are used as input to
                    the `.step()` function.
            skip: Whether to skip the connection. It resolves cyclic dependencies, by skipping the output if it arrives
                  at the same time as the start of the `.step()` function (i.e. `step_state.ts`).
            jitter: How to deal with jitter of the connection. If `LATEST`, the latest messages are used. If `BUFFER`, the
                    messages are buffered and used in accordance with the expected delay.
            name: A shadow name for the connected node. If `None`, the name of the output node is used.
        """
        name = name if isinstance(name, str) else output_node.name
        connection = Connection(self, output_node, blocking, delay, delay_dist, window, skip, jitter, input_name=name)
        self.inputs[name] = connection
        output_node.outputs[self.name] = connection

    def disconnect(self, output_node: Union[str, "BaseNode"]):
        """Disconnects the node from another node.

        TODO: Implement clean disconnect.
            This is a placeholder for a clean disconnect.
            When nodes are already used in a graph, it may not properly disconnect.
            As a workaround, users are advised to create new nodes for each graph, instead of reusing nodes.

        Args:
            output_node: The node to disconnect from. Can be the name of the node or the node object.
        """
        raise NotImplementedError(
            "Clean disconnect is not properly implemented. "
            "Users are advised to create new nodes for each graph, instead of reusing nodes."
        )
        # name = output_node if isinstance(output_node, str) else output_node.name
        # self.inputs[name].disconnect()

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.Params:
        """Init params of the node.

        The params are composed of values that remain constant during an episode (e.g. network weights).

        At this point, the graph state may contain the params of other nodes required to get the default params.
        The order of node initialization can be specified in Graph.init(... order=[node1, node2, ...]).

        Args:
            rng: Random number generator.
            graph_state: The graph state that may be used to get the default params.

        Returns:
            The default params of the node.
        """
        return base.Empty()

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.State:
        """Init state of the node.

        The state is composed of values that are updated during the episode in the `.step()` function (e.g. position, velocity).

        At this point, the params of all nodes are already initialized and present in the graph state (if specified).
        Moreover, the state of other nodes required to get the default state may also be present in the graph state.
        The order of node initialization can be specified in `Graph.init(... order=[node1, node2, ...])`.

        Args:
            rng: Random number generator.
            graph_state: The graph state that may be used to get the default state.

        Returns:
            The default state of the node.
        """
        return base.Empty()

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.Output:
        """Default output of the node. Used to fill the input buffers of connected nodes.

        Tip:
            It's recommended to define the output without relying on the `graph_state` to avoid dependency on other nodes.

        Args:
            rng: Random number generator.
            graph_state: The graph state that may be used to get the default output.

        Returns:
            The default output of the node.
        """
        return base.Empty()

    def init_delays(
        self, rng: jax.Array = None, graph_state: base.GraphState = None
    ) -> Dict[str, Union[float, jax.typing.ArrayLike]]:
        """Initialize trainable communication delays.

        Note:
            These delays include only trainable connections.
            To make a delay trainable, replace the parameters in the delay distribution with trainable parameters.

        Example: A rough template for the init_delays function is as follows:
            ```python
            def init_delays(self, rng=None, graph_state=None):
                # Assumes graph_state contains the params of the node
                params = graph_state.params[self.name]
                trainable_delays = {"world": params.delay_param}
                return trainable_delays
            ```

        Args:
            rng: Random number generator.
            graph_state: The graph state that may be used to get the default output.

        Returns:
            Trainable delays. Can be an incomplete dictionary.
            Entries for non-trainable delays or non-existent connections are ignored.
        """
        trainable_delays = dict()
        for input_name, c in self.inputs.items():
            delay_dist = c.delay_dist
            if isinstance(delay_dist, base.TrainableDist):
                trainable_delays[input_name] = delay_dist.mean()
        return trainable_delays

    def init_inputs(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> FrozenDict[str, base.InputState]:
        """Initialize default inputs for the node.

        Fills input buffers with default outputs from connected nodes.
        Used during the initial steps of an episode when input buffers are not yet filled.

        Args:
            rng: Random number generator.
            graph_state: The graph state that may be used to get the default inputs.

        Returns:
            The default inputs of the node.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rng, rng_delays = jax.random.split(rng)
        delays = self.init_delays(rng_delays, graph_state)
        rngs = jax.random.split(rng, num=len(self.inputs))
        inputs = dict()
        for (input_name, c), rng_output in zip(self.inputs.items(), rngs):
            window = c.window
            seq = onp.arange(-window, 0, dtype=onp.int32)
            ts_sent = 0 * onp.arange(-window, 0, dtype=onp.float32)
            ts_recv = 0 * onp.arange(-window, 0, dtype=onp.float32)
            outputs = [c.output_node.init_output(rng_output, graph_state) for _ in range(window)]
            delay_dist = c.delay_dist  # NOTE!: Set trainable parameters from the graph state, if needed
            if isinstance(delay_dist, base.TrainableDist) and input_name in delays:
                # Set alpha according to the specified delay
                alpha = delay_dist.get_alpha(delays[input_name])
                delay_dist = delay_dist.replace(alpha=alpha)
            inputs[input_name] = base.InputState.from_outputs(seq, ts_sent, ts_recv, outputs, delay_dist)
        return FrozenDict(inputs)

    def init_step_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> base.StepState:
        """
        Initializes the step state of the node, which is used to run the `seq`'th step of the node at time `ts`.

        - `BaseNode.init_params`
        - `BaseNode.init_state`
        - `BaseNode.init_inputs` using `BaseNode.init_output` of connected nodes (to fill the input buffers)

        Note:
            If a node's initialization depends on the params, state, or inputs of other nodes this may fail.
             In such cases, the user can provide a graph state with the necessary information to get the default step state.

        Args:
            rng: Random number generator.
            graph_state: The graph state that may be used to get the default step state.

        Returns:
            The default step state of the node.
        """

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

    def startup(self, graph_state: base.GraphState, timeout: float = None) -> bool:
        """
        Initializes the node to the state specified by `graph_state`.
        This method is called just before an episode starts.
        It can be used to move a real robot to a starting position as specified by the `graph_state`.

        Note:
            Only called when running asynchronously.

        Args:
            graph_state: The graph state.
            timeout: The timeout of the startup.

        Returns:
            Whether the node has started successfully.
        """
        return True

    def stop(self, timeout: float = None) -> bool:
        """Stopping routine that is called after the episode is done.

        Note:
            Only called when running asynchronously.

        Warning:
            It may happen that stop is already called *before* the final `.step` call of an episode returns,
            which may cause unsafe behavior when the final step undoes the work of the .stop method.
            This should be handled by the user. For example, by stopping "longer" before returning here.

        Args:
            timeout: The timeout of the stop.

        Returns:
            Whether the node has stopped successfully.
        """
        return True

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
        """
        Execute the node for the `seq`-th time step at time `ts`.
        This function updates the node's state and generates an output, which is sent to connected nodes. It is called at the node's rate.
         Users are expected to update the state (and rng if used), but not the seq and ts, as they are automatically updated.

        Tip: Wrapping side-effecting code
            Side-effecting code should be wrapped to ensure execution on the host machine when using `jax.jit`.
            See [here](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html) for more info.

        Example: A rough template for the step function is as follows:
            ```python
            def step(step_state: base.StepState) -> Tuple[base.StepState, base.Output]:
                # Per input with `input_name`, the following information is available:
                step_state.inputs[input_name][window_index].data # A window_index of -1 leads to the most recent message.
                step_state.inputs[input_name][window_index].seq # The sequence number of the message.
                step_state.inputs[input_name][window_index].ts_sent # The time the message was sent.
                step_state.inputs[input_name][window_index].ts_recv # The time the message was received.

                # The following information is available for the node:
                step_state.params # The parameters of the node.
                step_state.state # The state of the node.
                step_state.eps # The episode number.
                step_state.seq # The sequence number.
                step_state.ts # The time of the step within the episode.
                step_state.rng # The random number generator.

                # Calculate output and updated state
                new_rng, rng_step = jax.random.split(step_state.rng)
                output = ...
                new_state = ...

                # Update the state of the node
                new_ss = step_state.replace(rng=new_rng, state=new_state)  #
                return new_ss, output
            ```

        Args:
            step_state: The step state of the node.

        Returns:
            The updated step state and the output of the node.
        """
        raise NotImplementedError


class BaseWorld(BaseNode):
    def __init__(self, name: str, rate: float, color: str = None, order: int = None, **kwargs):
        """Base node class for world (i.e. simulator) nodes.

        A convenience class that pre-sets parameters for nodes that simulate real-world processes. That is, nodes that
        simulate continuous processes in a discrete manner.

        - The delay distribution is set to the time step of the node (~1/rate). It's currently set slightly below the time
            step to ensure numerical stability, as else we may unavoidably introduce more delay.
        - The advance is set to False, as the world node should adhere to the rate of the node.
        - The scheduling is set to FREQUENCY, as the world node should adhere to the rate of the node.

        Args:
            name: The name of the node (unique).
            rate: The rate of the node (Hz).
            color: The color of the node (for visualization).
            order: The order of the node (for visualization).
        """
        delay_dist = distrax.Normal(loc=0.999 / rate, scale=0.0)
        delay = float(delay_dist.mean())
        super().__init__(
            name=name,
            rate=rate,
            delay=delay,
            delay_dist=delay_dist,
            advance=False,
            scheduling=Scheduling.FREQUENCY,
            color=color,
            order=order,
        )
