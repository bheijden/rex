import jax
import jumpy


def g(b):
    return b*3


def f(a):
    c = g(a)
    if jumpy.core.is_jitted():
        print("JITTED", jax._src.core.thread_local_state.trace_state.trace_stack)
    else:
        # print("NOT JITTED", jax._src.core.thread_local_state.trace_state.trace_stack)
        print("NOT JITTED", jax._src.core.thread_local_state.trace_state.trace_stack.stack)
        # print("NOT JITTED", len(jax._src.core.thread_local_state.trace_state.trace_stack.stack))
        jax._src.core.thread_local_state.trace_state.trace_stack
    return c*2

f(1)
f_jit = jax.jit(f)
f_jit(1)
f_jit(2)
