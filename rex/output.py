from typing import Optional, Any, TYPE_CHECKING
from collections import deque
from jax import jit
from jax import numpy as jnp
import jax.random as rnd
from rex.constants import READY, RUNNING, STOPPING, STOPPED, RUNNING_STATES, DEBUG
from rex.distributions import Gaussian, Distribution
from rex.input import Input
from rex.proto import log_pb2 as log_pb2
from rex.utils import log

if TYPE_CHECKING:
    from rex.node import Node


class Output:
    def __init__(self, node: "Node", log_level: int, color: str, delay: float, delay_sim: Distribution):
        self.node = node
        self.log_level = log_level
        self.color = color
        self.inputs = []
        self.delay_sim = delay_sim
        self._state = STOPPED
        self.delay = delay if delay is not None else delay_sim.high
        assert self.delay >= 0, "Phase should be non-negative."

        # Jit function (call self.warmup() to pre-compile)
        self._num_buffer = 50
        self._jit_sample = jit(self.delay_sim.sample, static_argnums=1)
        self._jit_split = jit(rnd.split, static_argnums=1)

        # Reset every run
        self._phase_dist = None
        self._phase = None
        self.q_sample = None
        self._rng = None

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def rate(self) -> float:
        return self.node.rate

    @property
    def phase(self) -> float:
        if self._phase is None:
            return self.node.phase + self.delay
        else:
            return self._phase

    @property
    def phase_dist(self) -> Distribution:
        """Distribution of the output phase shift: phase shift of the node (deterministic) + the computation delay dist."""
        if self._phase_dist is None:
            return Gaussian(self.node.phase) + self.delay_sim
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

    def log(self, id: str, value: Optional[Any] = None, log_level: Optional[int] = None):
        log_level = log_level if isinstance(log_level, int) else self.log_level
        log(f"{self.name}/output", self.color, min(log_level, self.log_level), id, value)

    def connect(self, i: "Input"):
        self.inputs.append(i)

    def reset(self, rng: jnp.ndarray):
        assert self._state in [STOPPED, READY], f"Output of {self.name} must first be stopped, before it can be reset."
        self._phase, self._phase_dist = None, None
        self._phase_dist = self.phase_dist
        self._phase = self.phase
        self.q_sample = deque() #if self.q_sample is None else self.q_sample
        self._rng = rng

        # Set running state
        self._state = READY
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

    def start(self):
        assert self._state in [READY], f"The output of {self.name} must first be reset, before it can start running."

        # Set running state
        self._state = RUNNING
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

    def stop(self):
        assert self._state in [RUNNING], f"The output of {self.name} should only push the reset total_ticks once."

        # Initiate stopping routine
        self._state = STOPPING
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

        # Set running state
        self._state = STOPPED
        self.log(RUNNING_STATES[self._state], log_level=DEBUG)

    def push_output(self, msg, header: log_pb2.Header):
        self.log("push_output", log_level=DEBUG)

        # Only send output if we are running
        if self._state in [RUNNING]:

            # Push message to inputs
            [i._submit(i.push_input, msg, header) for i in self.inputs]

    def push_ts_output(self, ts_output: float, header: log_pb2.Header):
        # Only send output if we are running
        if self._state in [RUNNING]:

            # Push message to inputs
            [i._submit(i.push_ts_input, ts_output, header) for i in self.inputs]
