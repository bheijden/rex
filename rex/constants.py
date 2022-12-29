# Running state
READY = 0
RUNNING = 1
STOPPING = 2
STOPPED = 3
RUNNING_STATES = {READY: "READY", RUNNING: "RUNNING", STOPPING: "STOPPING", STOPPED: "STOPPED"}

# Scheduling modes
PHASE = 0
FREQUENCY = 1
SCHEDULING_MODES = {PHASE: "phase", FREQUENCY: "frequency"}

# Clock modes
SIMULATED = 0
WALL_CLOCK = 1
CLOCK_MODES = {PHASE: "simulated-clock", FREQUENCY: "wall-clock"}

# Real-time factor modes
FAST_AS_POSSIBLE = 0
REAL_TIME = 1.
RTF_MODES = {FAST_AS_POSSIBLE: "fast-as-possible", REAL_TIME: "real-time"}

# Synchronization modes
SYNC = 0
ASYNC = 1
SYNC_MODES = {SYNC: "synchronous", ASYNC: "asynchronous"}

# Jitter modes
LATEST = 0
BUFFER = 1
JITTER_MODES = {LATEST: "synchronous", BUFFER: "asynchronous"}

# Graph modes
INTERPRETED = 0
VECTORIZED = 1
SEQUENTIAL = 2
BATCHED = 3
GRAPH_MODES = {INTERPRETED: "interpreted", VECTORIZED: "vectorized", SEQUENTIAL: "sequential", BATCHED: "batched"}


# Log levels
SILENT = 0
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
FATAL = 50
LOG_LEVELS = {SILENT: "silent", DEBUG: "debug", INFO: "info", WARN: "warning", ERROR: "error", FATAL: "fatal"}

