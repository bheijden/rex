import os
# os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"

import jax
import jax.numpy as jnp

# Make sure this is called before jax runs any operations!
jax.config.update("jax_compilation_cache_dir", "./cache-scratch")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_explain_cache_misses", True)

if __name__ == "__main__":

    @jax.jit
    def f(_x):
      return _x + 1

    x = jnp.zeros((2, 2))
    f(x)



