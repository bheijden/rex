from rex.proto import log_pb2
from typing import Union, List, Tuple
import jax.numpy as jnp  # todo: replace with from brax import jumpy as jp.ndarray?
from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend
tfd = tfp.distributions


class Gaussian:
    def __init__(self, mean: float, var: float = 0, percentile: float = 0.01):
        assert mean >= 0, "Mean must be non-negative"
        assert var >= 0, "var must be non-negative"
        assert percentile > 0, "There must be a truncating percentile > 0."
        self._mean = mean
        self._var = var
        self._std = var ** (1/2)
        self._percentile = percentile
        self._low = tfd.Normal(loc=mean, scale=var).quantile(percentile).tolist()
        self._high = tfd.Normal(loc=mean, scale=var).quantile(1-percentile).tolist()
        if var > 0:
            self._dist = tfd.TruncatedNormal(loc=mean, scale=var, low=self._low, high=self._high)
        else:
            self._dist = tfd.Deterministic(loc=mean)

        # Verify that the delay is always non-negative
        assert self._low >= 0, "Samples should always be positive."

    def __repr__(self):
        return f"Gaussian | {1.0: .2f}*N({self.mean: .4f}, {self.var: .4f}) | percentile={self.percentile}"

    def __add__(self, other: "Distribution"):
        """Summation of two distributions"""
        if isinstance(other, Gaussian):
            mean = self.mean + other.mean
            var = self.var + other.var
            percentile = max(self.percentile, other.percentile)
            return Gaussian(mean, var, percentile=percentile)
        elif isinstance(other, GMM):
            return other + self
        else:
            raise NotImplementedError("Not yet implemented")

    def pdf(self, x: jnp.ndarray):
        return self._dist.prob(x)

    def cdf(self, x: jnp.ndarray):
        return self._dist.cdf(x)

    def sample(self, rng: jnp.ndarray, shape: Union[int, Tuple] = None):
        if shape is None:
            shape = ()
        return self._dist.sample(sample_shape=shape, seed=rng)

    @classmethod
    def from_info(cls, info: Union[log_pb2.GMM, log_pb2.Gaussian]):
        if isinstance(info, log_pb2.GMM):
            assert len(info.gaussians) == 1, "The GMM log should only contain a single Gaussian."
            info = info.gaussians[0]
        mean, var, percentile = info.mean, info.var, info.percentile
        return cls(mean, var, percentile)

    @property
    def info(self) -> log_pb2.GMM:
        info = log_pb2.GMM()
        g = log_pb2.Gaussian(weight=1, mean=self.mean, var=self.var, percentile=self.percentile, low=self.low, high=self.high)
        info.gaussians.append(g)
        return info

    @property
    def percentile(self):
        return self._percentile

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high


class GMM:
    def __init__(self, gaussians: List["Gaussian"], weights: List[float]):
        assert len(gaussians) > 0, "Must specify at least 1 Gaussian."
        assert len(gaussians) == len(weights), "Must provide an equal number of weights and Gaussians"
        assert all([w > 0 for w in weights]), "All weights must be positive."
        self._weights = [w / sum(weights) for w in weights]
        self._gaussians = gaussians

        # Check if distributions are from the same family
        deterministic = [v == 0 for v in self.vars]
        assert all(deterministic) or not any(deterministic), "Either all distributions must be deterministic (ie var=0) or stochastic (var>0)"

        if all(deterministic):
            self._dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=self._weights),
                                               components_distribution=tfd.Deterministic(loc=self.means))
        else:
            self._dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=self._weights),
                                               components_distribution=tfd.TruncatedNormal(loc=self.means,
                                                                                           scale=self.vars,
                                                                                           low=[g.low for g in self._gaussians],
                                                                                           high=[g.high for g in self._gaussians]))

    def __repr__(self):
        msg = " | ".join([f"{w: .2f}*N({m: .4f}, {v: .4f})" for w, m, v in zip(self.weights, self.means, self.vars)])
        return f"GMM | {msg} | percentile={self.percentile}"

    def __add__(self, other: "Distribution"):
        # Convert to GMM
        if isinstance(other, Gaussian):
            other = GMM([other], weights=[1.0])

        # Only compatible with Gaussian or GMM
        if not isinstance(other, GMM):
            raise NotImplementedError("Not yet implemented")

        percentile = max(other.percentile, self.percentile)
        gaussians, weights = [], []
        for w, m, v in zip(self.weights, self.means, self.vars):
            for ow, om, ov in zip(other.weights, other.means, other.vars):
                weights.append(w*ow)
                gaussians.append(Gaussian(m + om, v + ov, percentile=percentile))
        return GMM(gaussians, weights)

    def pdf(self, x: jnp.ndarray):
        return self._dist.prob(x)

    def cdf(self, x: jnp.ndarray):
        return self._dist.cdf(x)

    def sample(self, rng: jnp.ndarray, shape: Union[int, Tuple] = None):
        if shape is None:
            shape = ()
        return self._dist.sample(sample_shape=shape, seed=rng)

    @property
    def info(self) -> log_pb2.GMM:
        info = log_pb2.GMM()
        for w, g in zip(self.weights, self._gaussians):
            ginfo = g.info.gaussians[0]
            ginfo.weight = w
            info.gaussians.append(ginfo)
        return info

    @classmethod
    def from_info(cls, info: log_pb2.GMM):
        weights = []
        gaussians = []
        for g in info.gaussians:
            weights.append(g.weight)
            gaussians.append(Gaussian.from_info(g))
        return cls(gaussians, weights)

    @property
    def percentile(self):
        return max([g.percentile for g in self._gaussians])

    @property
    def weights(self):
        return self._weights

    @property
    def means(self):
        return [g.mean for g in self._gaussians]

    @property
    def vars(self):
        return [g.var for g in self._gaussians]

    @property
    def low(self):
        return min([g.low for g in self._gaussians])

    @property
    def high(self):
        return max([g.high for g in self._gaussians])



Distribution = Union[Gaussian, GMM]
