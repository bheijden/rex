from enum import Enum, auto


class Async(Enum):
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
    FAST_AS_POSSIBLE = 0
    REAL_TIME = 1.


class Jitter(Enum):
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
    SILENT = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50


class Supergraph(Enum):
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
