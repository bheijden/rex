import jax
import jax.numpy as jnp
from jax import make_jaxpr
import numpy as onp
import time


class timer:
    def __init__(self, name: str = None, repeat: int = 1):
        self.name = name or "timer"
        self.repeat = repeat
        self.duration = None
        self.msg = "No message."

    def __enter__(self):
        self.tstart = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.duration = time.perf_counter() - self.tstart
        if self.repeat == 1:
            self.msg = f"Elapsed: {self.duration:.4f} sec"
        else:
            self.msg = f"Elapsed: {self.duration / self.repeat:.4f} sec (x{self.repeat} repeats = {self.duration:.4f} sec)"
        print(f"{self.name}: {self.msg}")


if __name__ == "__main__":
    # Seed numpy generator
    onp.random.seed(0)

    # Parameters
    num_vmap = 2000  # Number of parallel vmap operations
    num_runs = 100   # Number of consecutive runs (some may be NOOP depending on predicate)
    num_buffer = 2 ** 13  # Buffer size --> Larger buffer size results in slower performance but shouldn't if we use in-place operations
    num_output = 8  # Arbitrary output dimension
    assert num_buffer >= num_runs, "Buffer size must be larger than number of runs."
    init_buffer = onp.random.randint(0, 10, size=(num_vmap, num_buffer, num_output))  # Initialize random buffer
    init_index_out = onp.random.randint(0, 100, size=(num_vmap,))  # Determines where to start writing in buffer
    indices_in = onp.random.randint(0, num_buffer, size=(num_vmap, num_runs,))  # Determines where to read from buffer
    predicates = onp.random.randint(0, 2, size=(num_vmap, num_runs,), dtype=bool)  # Determines whether to write to buffer

    def write_output_to_buffer(_buffer, _index, _output):
        """Write output to buffer."""
        mod_index = _index % _buffer.shape[0]  # Ensure mod_index is within buffer size
        res = _buffer.at[mod_index].set(_output)  # Write output to buffer --> Force to be in-place!
        return res

    def get_output_from_buffer(_buffer, _index):
        """Get output from buffer."""
        mod_index = _index % _buffer.shape[0]  # Ensure mod_index is within buffer size
        return _buffer[mod_index]

    def run(_index_in, _index_out, _buffer):
        """"""
        # Get input
        i = get_output_from_buffer(_buffer, _index_in)
        # Get state and output
        _index_out = _index_out + 1  # Increment index
        output = i + 1  # Arbitrary operation
        return _index_out, output

    def conditional_run_or_NOOP(carry, x):
        # Unpack
        _index_out, _buffer = carry
        _index_in, _predicate = x

        # Prepare NOOP branch
        noop_index_out = _index_out
        noop_output = get_output_from_buffer(_buffer, _index_out)
        no_op = lambda *args: (noop_index_out, noop_output)

        # Run step (conditioned on predicate)
        next_index_out, output = jax.lax.cond(_predicate, run, no_op, _index_in, _index_out, _buffer)

        # Update buffer (either noop_output or output)
        # This should always be in-place, but I can't enforce it in JAX
        next_buffer = write_output_to_buffer(_buffer, _index_out, output)
        return (next_index_out, next_buffer), output

    def _scan_all(_init_index_out, _init_buffer, _indices_in, _predicates):
        return jax.lax.scan(conditional_run_or_NOOP, (_init_index_out, _init_buffer), (_indices_in, _predicates))

    # Run
    jv_scan_all = jax.jit(jax.vmap(_scan_all), donate_argnums=(1,))
    print(make_jaxpr(_scan_all)(init_index_out[0], init_buffer[0], indices_in[0], predicates[0]))

    # jit compile
    with timer("jit"):
        (final_seq, final_buffer), _ = jv_scan_all(init_index_out, init_buffer, indices_in, predicates)
        final_buffer.block_until_ready()

    # Time
    repeat = 10
    with timer("run", repeat=repeat):
        for _ in range(repeat):
            (final_seq, final_buffer), _ = jv_scan_all(init_index_out, init_buffer, indices_in, predicates)
            final_buffer.block_until_ready()
