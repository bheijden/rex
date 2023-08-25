import jax
import numpy as onp
import jax.numpy as jnp
from rex.distributions import Gaussian, GMM, Recorded
import jax.random as rnd
import pickle


def test_distributions():
    # Initialize distributions
    det1 = Gaussian(0.005)
    det2 = Gaussian(0.005)
    g1 = Gaussian(0.005, 0.001)
    g2 = Gaussian(0.004, 0.001)
    gmm1 = GMM([g1, g2], [0.5, 0.5])
    gmmdet = GMM([det1, det2], [0.5, 0.5])

    # Test addition of distributions
    g3 = g1 + g2
    gmm2 = g1 + gmm1
    gmm3 = gmm1 + gmm2

    # Test the API
    g1 = g1.from_info(g1.info)
    print(g1)
    print(gmm1)
    _ = gmm1.weights
    _ = gmm1.percentiles
    _ = gmm1.info
    _ = gmm1.vars

    _ = gmm1.sample(gmm1.reset(rnd.PRNGKey(0)))
    _ = gmm1.pdf(0.5)
    _ = gmm1.cdf(0.5)
    _ = g1.sample(g1.reset(rnd.PRNGKey(0)))
    _ = g1.pdf(0.5)
    _ = g1.cdf(0.5)

    # Test quantile (inverse cdf).
    gmm_quantile = GMM([g1], [1.0])
    _ = g1.quantile(jnp.array([[0.5, 0.4]]*2))
    _ = gmm_quantile.quantile(jnp.array([[0.5, 0.4]]*2))
    _ = g1.quantile(0.5)
    _ = gmm_quantile.quantile(0.5)

    # Check deterministic quantile
    _ = gmmdet.quantile(0.5)
    _ = det1.quantile(0.5)

    # Test pickle API
    g1 = pickle.loads(pickle.dumps(g1))
    gmm1 = pickle.loads(pickle.dumps(gmm1))


def test_distribution_recorded():
    # Distribution to be wrapped
    g1 = Gaussian(0.005, 0.001)

    # Prerecorded samples
    recorded_samples = onp.arange(0, 100)

    # Recorded distribution
    rec_g1 = Recorded(g1, recorded_samples)

    # Test the API
    state = rec_g1.reset(rnd.PRNGKey(0))
    jit_sample = jax.jit(rec_g1.sample, static_argnums=1)
    state, samples = jit_sample(state, shape=None)
    state, samples = jit_sample(state, shape=10)
    state, samples = jit_sample(state, shape=1)
    state, samples = jit_sample(state, shape=())
    state, samples = jit_sample(state, shape=(2, 3))

    # Test getattr, setattr
    _ = rec_g1.quantile(jnp.array([[0.5, 0.4]]*2))
    _ = rec_g1.quantile(0.5)

    # Test pickle API
    rec_g1 = pickle.loads(pickle.dumps(rec_g1))
