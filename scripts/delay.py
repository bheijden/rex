import rex.distributions as dist
import jax
import jax.numpy as jnp
import jax.random as rnd

g = dist.Gaussian(1, 0)
_ = g.info
h = dist.Gaussian(2, 0)
f = dist.Gaussian(1, 0.4 ** 2)
k = dist.Gaussian(3, 0.4 ** 2)

key = rnd.PRNGKey(0)
g.sample(key)
g.pdf(jnp.array(1, dtype="float32"))
g.cdf(jnp.array(1, dtype="float32"))
f.sample(key)
f.pdf(jnp.array(1, dtype="float32"))
f.cdf(jnp.array(1, dtype="float32"))
det_gmm = dist.GMM([g, h], [1, 2])
_ = det_gmm.sample(key)
_ = det_gmm.pdf(jnp.array(1, dtype="float32"))
_ = det_gmm.cdf(jnp.array(1, dtype="float32"))
gmm = dist.GMM([f, k], [1, 2]) + det_gmm
_ = gmm.sample(key)
_ = gmm.pdf(jnp.array(1, dtype="float32"))
_ = gmm.cdf(jnp.array(1, dtype="float32")).block_until_ready()

jit_det_sample = jax.jit(det_gmm.sample, static_argnames="shape")
jit_det_pdf = jax.jit(det_gmm.pdf)
jit_sample = jax.jit(gmm.sample, static_argnames="shape")
jit_pdf = jax.jit(gmm.pdf)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)
ax[0].hist(jit_sample(rnd.PRNGKey(0), shape=10000), density=True, bins=1000)
x = jnp.round(jnp.linspace(0, 5, 1001), decimals=4)
ax[0].plot(x, jit_pdf(x))

ax[1].hist(jit_det_sample(rnd.PRNGKey(0), shape=100), density=True, bins=100)
ax[1].plot(x, jit_det_pdf(x))
plt.show()
