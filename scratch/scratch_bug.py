import jax
import jax.numpy as jnp


def fn(x):
    R1 = jnp.array([[x[0], 0, 0],
                    [0, 1, 0],
                    [0, 0, x[0]]])  # Changing x[0] to x[2] here resolves the issue...

    # Another matrix
    R2 = jnp.array([[x[0], 0, 0],
                   [0, x[1], 0],
                   [0, 0, x[2]]])
    # R2 = jnp.diag(x)  # Using jnp.diag resolves the issue...
    H = jnp.eye(4)
    H = H.at[:3, :3].set(R2.T)  # Removing .T resolves the issue
    pos = H @ jnp.concatenate([x, jnp.array([1.0])])
    # pos = H[:3, :3] @ x  # Using this line resolves the issue...
    return pos, R1  # Only returning either pos, or R resolves the issue...


def is_close(old, new):
    res = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new))
    return all(res)

gpu = jax.devices("gpu")[0]
cpu = jax.devices("cpu")[0]

N = 5
x_v = jnp.zeros((N, 3))
fn_v = jax.vmap(fn)
fn_jv_cpu = jax.jit(jax.vmap(fn), device=cpu)
fn_jv_gpu = jax.jit(jax.vmap(fn), device=gpu)

M = 4  # changing M=5 resolves the issue
x_vv = jnp.zeros((M, N, 3))
fn_jvv_gpu = jax.jit(jax.vmap(jax.vmap(fn)), device=gpu)

res_vv_gpu = fn_jvv_gpu(x_vv)
print("Jit (GPU), double vmap: SUCCESS")
res_v = fn_v(x_v)
print("No jit, single vmap: SUCCESS")
res_v_cpu = fn_jv_cpu(x_v)
print("Jit (CPU), single vmap: SUCCESS")
res_v_gpu = fn_jv_gpu(x_v)  # Fails here...
print("Jit (GPU), single vmap: SUCCESS")


