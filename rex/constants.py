from enum import Enum, auto


class Async(Enum):
    """
    Represents the various states of asynchronous operations.

    Members:
        READY: The system is ready to start.
        STARTING: The system is in the process of starting.
        READY_TO_START: The system is prepared to initiate running.
        RUNNING: The system is actively running.
        STOPPING: The system is in the process of stopping.
        STOPPED: The system has been stopped.
    """
    READY = auto()
    STARTING = auto()
    READY_TO_START = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()

    def description(self):
        return {
            Async.READY: "READY",
            Async.STARTING: "STARTING",
            Async.READY_TO_START: "READY_TO_START",
            Async.RUNNING: "RUNNING",
            Async.STOPPING: "STOPPING",
            Async.STOPPED: "STOPPED",
        }[self]

    def __str__(self):
        return self.description()


class Scheduling(Enum):
    """
    Represents the scheduling modes for node operations.

    This determines how node.step(..) calls are scheduled, particularly in the presence of delays.

    In PHASE mode, the node schedules its step calls based on both the desired frequency and its phase offset relative
    to a global clock. If the node is delayed, it tries to "catch up" to stay synchronized with its intended phase,
    potentially leading to a burst of activity as it compensates for lost time.

    In FREQUENCY mode, the node is scheduled solely based on its frequency. Each step call is made 1/frequency seconds
    after the previous one, regardless of any delays. This ensures steady spacing between steps but can result in drift
    from the desired phase if delays occur.

    Members:
        PHASE: Scheduling is based on both frequency and phase relative to the clock.
        FREQUENCY: Scheduling is based strictly on frequency, ignoring phase shifts.
    """
    PHASE = auto()
    FREQUENCY = auto()

    def description(self):
        return {
            Scheduling.PHASE: "phase",
            Scheduling.FREQUENCY: "frequency",
        }[self]

    def __str__(self):
        return self.description()


class Clock(Enum):
    """
    Defines the different types of clocks used for simulation.

    When using the WALL_CLOCK, the real-time factor should be set to REAL_TIME.

    Members:
        SIMULATED: A simulated clock for time-based operations.
        WALL_CLOCK: The actual wall clock for real-time operations.
        COMPILED: A clock compiled into a deterministic schedule.
    """
    SIMULATED = auto()
    WALL_CLOCK = auto()
    COMPILED = auto()

    def description(self):
        return {
            Clock.SIMULATED: "simulated-clock",
            Clock.WALL_CLOCK: "wall-clock",
        }[self]

    def __str__(self):
        return self.description()


class RealTimeFactor:
    """
    Defines constants related to real-time simulation.

    Any other positive value can be used to scale the simulation time.

    Only REAL_TIME should be used with Clock.WALL_CLOCK.

    Attributes:
        FAST_AS_POSSIBLE: Simulates the system as fast as possible (factor 0).
        REAL_TIME: Runs the simulation at real-time speed (factor 1.0).
    """
    FAST_AS_POSSIBLE = 0
    REAL_TIME = 1.


class Jitter(Enum):
    """
    Defines how to handle jitter in node connections.

    Jitter refers to irregular intervals between the arrival of messages from other nodes. This can disrupt nodes
    that expect a steady flow of messages. To address this, two modes of handling jitter are provided:

    - BUFFER mode: This mode buffers incoming messages, smoothing them based on expected delays and the node's
    frequency. It allows the node to process messages as if they arrived in a regular stream, reducing the impact
    of message bursts and helping to maintain more predictable behavior.

    - LATEST mode: In this mode, the node uses the most recent message received, disregarding any earlier messages
    that might still be in the queue. While this reduces latency and ensures the node is working with the most current
    data, it can introduce variability, as the node may behave differently depending on when messages arrive.

    BUFFER mode increases determinism and regularity, while LATEST mode prioritizes responsiveness at the cost of
    more stochastic behavior.

    Members:
        LATEST: Uses the most recent message in the presence of jitter.
        BUFFER: Buffers messages to account for expected delays, smoothing their arrival.
    """
    LATEST = auto()
    BUFFER = auto()

    def description(self):
        return {
            Jitter.LATEST: "latest",
            Jitter.BUFFER: "buffer",
        }[self]

    def __str__(self):
        return self.description()


class LogLevel:
    """
    Defines the log levels used for logging node activity.

    # Setting the log level for all nodes
    rex.utils.set_log_level(LogLevel.DEBUG)

    # Setting the log level for a specific node
    rex.utils.set_log_level(LogLevel.DEBUG, node=my_node, color="red")

    Available text colors:
    black, red, green, yellow, blue, magenta, cyan, white,
    light_grey, dark_grey, light_red, light_green, light_yellow, light_blue,
    light_magenta, light_cyan.

    Attributes:
        SILENT: No logging.
        DEBUG: Log detailed debugging information.
        INFO: Log general information.
        WARN: Log warnings about potential issues.
        ERROR: Log errors in the system.
        FATAL: Log critical errors leading to a system failure.
    """
    SILENT = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50


class Supergraph(Enum):
    """
    Represents different supergraph modes for efficiently compiling computation graphs.

    Members:
        MCS: Minimum Common Supergraph generally results in the most efficient solution. However, it can increase compilation time.
        GENERATIONAL: Less efficient than MCS but sometimes faster to compile.
        TOPOLOGICAL: Generally less efficient than GENERATIONAL.
    """
    MCS = auto()
    GENERATIONAL = auto()
    TOPOLOGICAL = auto()

    def description(self):
        return {
            Supergraph.MCS: "minimum common supergraph",
            Supergraph.GENERATIONAL: "generational",
            Supergraph.TOPOLOGICAL: "topological",
        }[self]

    def __str__(self):
        return self.description()
