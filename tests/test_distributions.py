import jax.numpy as jnp
from rex.distributions import Gaussian, GMM
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

    _ = gmm1.sample(rnd.PRNGKey(0))
    _ = gmm1.pdf(0.5)
    _ = gmm1.cdf(0.5)
    _ = g1.sample(rnd.PRNGKey(0))
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


