import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax import make_jaxpr
import numpy as onp
import rex.jax_utils as rjax
from rex.utils import timer


if __name__ == "__main__":
    # Seed numpy generator
    onp.random.seed(0)
    num_mapped = 2000
    buffer_size = 2 ** 12
    num_runs = 100
    output_dim = 8
    assert buffer_size >= num_runs
    # output = onp.zeros((output_dim,), dtype=onp.float32)
    init_state = onp.zeros((num_mapped, output_dim,), dtype=onp.float32)
    init_buffer = onp.zeros((num_mapped, buffer_size, output_dim), dtype=onp.float32)
    init_seq = onp.random.randint(0, 100, size=(num_mapped,), dtype=onp.int32)
    seqs_in = onp.random.randint(0, buffer_size, size=(num_mapped, num_runs,), dtype=onp.int32)
    runs = onp.random.randint(0, 2, size=(num_mapped, num_runs,), dtype=bool)
    RETURN_OUTPUT = True
    PALLAS_INPLACE = False
    PALLAS_COPY_MAT_REF = False

    def _test_fn(_arr, i):
        return _arr.at[0].set(i)
    print(make_jaxpr(_test_fn)(jnp.array([0, 0]), 1))

    def inplace_kernel(mat_ref, vec_ref, idx_ref, o_ref):
        _vec = vec_ref[...]
        # Is it possible to do this without copying _mat into o_ref?
        if PALLAS_COPY_MAT_REF:
            o_ref[:] = mat_ref[...]  # Uncommenting returns the correct result, but doesn't this copy _mat into o_ref?
        pl.store(o_ref, (idx_ref[...],), _vec)

    def inplace(_mat, _vec, _idx):
        return pl.pallas_call(
            inplace_kernel,
            out_shape=jax.ShapeDtypeStruct(_mat.shape, _mat.dtype),
        )(_mat, _vec, _idx)

    num = 4  # Only works for powers of 2 --> why is this a problem?
    mat = jnp.arange(num*num, dtype=jnp.int32).reshape((num, num))
    vec = jnp.ones(num, dtype=jnp.int32)
    idx = jnp.array(2 % num, dtype=jnp.int32)
    print(make_jaxpr(inplace)(mat, vec, idx))
    print(jax.jit(inplace)(mat, vec, idx))


    def update_buffer(_buffer, _seq, _output):
        size = _buffer.shape[0]
        mod_seq = _seq % size
        if PALLAS_INPLACE:
            res = inplace(_buffer, _output, mod_seq)
        else:
            res = _buffer.at[mod_seq].set(_output)
        return res

    def take_output(_buffer, _seq):
        size = _buffer.shape[0]
        mod_seq = _seq % size
        return _buffer[mod_seq]
        # return rjax.tree_take(_buffer, mod_seq)

    def _step(seq_in, seq, _buffer, state):
        # Get input
        i = take_output(_buffer, seq_in)
        # Get state and output
        new_seq = seq + 1
        new_state = state + i
        output = state + 1
        # Update buffer
        # new_buffer = output if RETURN_OUTPUT else update_buffer(_buffer, seq, output)
        new_buffer = output  # if RETURN_OUTPUT else update_buffer(_buffer, seq, output)
        return new_seq, new_buffer, new_state

    def _run_generation(carry, x):
        # Unpack
        seq, _buffer, state = carry
        run, seq_in = x

        # Prepare NOOP
        old_seq = seq
        old_output = take_output(_buffer, seq)
        old_state = state
        no_op = lambda *args: (old_seq, old_output, old_state)

        # Run step
        next_seq, output, next_state = jax.lax.cond(run, _step, no_op, seq_in, seq, _buffer, state)

        # next_buffer = update_buffer(_buffer, seq, next_buffer) if RETURN_OUTPUT else next_buffer
        next_buffer = update_buffer(_buffer, seq, output)
        return (next_seq, next_buffer, next_state), next_state

    def _run_scan(_runs, _seq, _seqs_in, _state, _buffer):

        return jax.lax.scan(_run_generation, (_seq, _buffer, _state), (_runs, _seqs_in))


    def _run(_runs, _seq, _seqs_in, _state, _buffer):
        carry, out = None, None
        for i in range(num_runs):
            carry, out = _run_generation((_seq, _buffer, _state), (_runs[i], _seqs_in[i]))
            _seq, _buffer, _state = carry
        return carry, out

    # Run
    jv_run = jax.jit(jax.vmap(_run))
    print(make_jaxpr(_run_scan)(runs[0], init_seq[0], seqs_in[0], init_state[0], init_buffer[0]))

    # jit compile
    with timer("jit"):
        (final_seq, final_buffer, final_state), _ = jv_run(runs, init_seq, seqs_in, init_state, init_buffer)
        final_state.block_until_ready()

    # Time
    repeat = 10
    with timer("run", repeat=repeat):
        for _ in range(repeat):
            (final_seq, final_buffer, final_state), _ = jv_run(runs, init_seq, seqs_in, init_state, init_buffer)
            final_state.block_until_ready()
