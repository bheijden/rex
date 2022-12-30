import jumpy as jp
import jax
import numpy as onp


def jp_prng(rng):
	return jp.random_split(rng)


def jax_prng(rng):
	return jax.random.split(rng)


jp_prng_jit = jax.jit(jp_prng)
jax_prng_jit = jax.jit(jax_prng)

seed = jp.random_prngkey(0)

jax_key_jit = jax_prng_jit(seed)
jax_key = jax_prng(seed)

jp_key_jit = jp_prng_jit(seed)
jp_key = jp_prng(seed)

print(f"jax: {onp.isclose(jax_key_jit, jax_key).all()}")
print(f"jumpy: {onp.isclose(jp_key, jp_key_jit).all()}")

