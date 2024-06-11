import functools
import jax
import tqdm
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend
tfd = tfp.distributions

import supergraph
import rexv2 as compiler
from supergraph.evaluate import timer

if __name__ == "__main__":
    # todo: steps
    #   - Make test setup with pendulum
    #   - Add __init__ args (delay, delay_sim, advance, scheduling)
    #   - Add connect args (delay, delay_sim, jitter, blocking)
    #   - Add methods (startup, now) to BaseNode
    #   - Define Async Node wrapper & Async Graph
    #       - Redefine Input, Output
    #   - Record graph, delays, data in new format (no protobuf)
    #   - Augment recorded graphs with simulation nodes (take sysid windows into account)
    #   -
    #   -
    #   - Add Pendulum (ode, real)
    #   - Add Crazyflie (ode, real)
    # todo: refactor steps
    #   -  BaseNode
    #       - advance, scheduling, delay, delay_sim
    #       -
    #       - SysIDDelay?
    #       - Logging
    #       - Recording (async & compiled)
    #       - .unwrapped, .warmup, .log_level
    #       - State Machine
    #       - .startup, .now()
    #       - ._reset, ._startup, ._start, .stop, .warmup, .now, .throttle
    #       -
    #   - AsyncNode
    #       - clock: SIMULATED, WALL_CLOCK
    #       - real_time_factor: REAL_TIME, FAST_AS_POSSIBLE, ...
    #   - Graph
    #       -
    #   -
    #   -
    #   -
    #   -

    ...