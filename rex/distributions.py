from functools import reduce
from rex.proto import log_pb2
from typing import Union, List, Tuple, Any
import dill as pickle
import jax
import numpy as onp
import jax.numpy as jnp
import flax.struct as struct
from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend

tfd = tfp.distributions


@struct.dataclass
class EmptyState:
    pass


@struct.dataclass
class DistState:
    rng: jax.random.KeyArray
    state: Any


@struct.dataclass
class RecordedState:
    index: Union[int, jax.typing.ArrayLike]


class Gaussian:
    def __init__(self, mean: float, std: float = 0, percentile: float = 0.01):
        assert std >= 0, "std must be non-negative"
        percentile = max(percentile, 1e-7)
        assert percentile > 0, "There must be a truncating percentile > 0."
        self._mean = mean
        self._std = std
        self._var = self._std**2
        self._percentile = percentile
        self._low = 0.0
        self._high = tfd.Normal(loc=self._mean, scale=self._std).quantile(1 - percentile).tolist()
        assert self._high < onp.inf, "The maximum value must be bounded"
        if std > 0:
            self._dist = tfd.TruncatedNormal(loc=self._mean, scale=self._std, low=self._low, high=self._high)
        else:
            self._dist = tfd.Deterministic(loc=self._mean)

    def __repr__(self):
        return f"Gaussian | {1.0: .2f}*N({self.mean: .4f}, {self.std: .4f}) | percentile={self.percentile}"

    def __add__(self, other: "Distribution"):
        """Summation of two distributions"""
        if isinstance(other, Recorded):
            other = other._dist
        if isinstance(other, Gaussian):
            mean = self.mean + other.mean
            std = (self.var + other.var) ** (1 / 2)
            percentile = max(self.percentile, other.percentile)
            return Gaussian(mean, std, percentile=percentile)
        elif isinstance(other, GMM):
            return other + self
        else:
            raise NotImplementedError("Not yet implemented")

    def __getstate__(self):
        """Used for pickling"""
        args = ()
        kwargs = dict(mean=self.mean, std=self.std, percentile=self.percentile)
        return args, kwargs

    def __setstate__(self, state):
        """Used for unpickling"""
        args, kwargs = state
        self.__init__(*args, **kwargs)

    def pdf(self, x: jax.typing.ArrayLike):
        return self._dist.prob(x)

    def quantile(self, x: jax.typing.ArrayLike):
        """Returns the quantile of the distribution at the given percentile."""
        if isinstance(self._dist, tfd.Deterministic):
            return self.mean
        else:
            return self._dist.quantile(x)

    def cdf(self, x: jax.typing.ArrayLike):
        return self._dist.cdf(x)

    def reset(self, rng: jax.random.KeyArray) -> DistState:
        return DistState(rng=rng, state=EmptyState())

    def sample(self, state: DistState, shape: Union[int, Tuple] = None) -> Tuple[DistState, jax.Array]:
        if shape is None:
            shape = ()
        new_rng, rng_sample = jax.random.split(state.rng)
        samples = self._dist.sample(sample_shape=shape, seed=rng_sample)
        new_state = DistState(rng=new_rng, state=EmptyState())
        return new_state, samples

    @classmethod
    def from_info(cls, info: Union[log_pb2.GMM, log_pb2.Gaussian]):
        if isinstance(info, log_pb2.GMM):
            assert len(info.gaussians) == 1, "The GMM log should only contain a single Gaussian."
            info = info.gaussians[0]
        mean, std, percentile = info.mean, info.std, info.percentile
        return cls(mean, std, percentile)

    @property
    def info(self) -> log_pb2.GMM:
        info = log_pb2.GMM()
        g = log_pb2.Gaussian(weight=1, mean=self.mean, std=self.std, percentile=self.percentile, low=self.low, high=self.high)
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
    def std(self):
        return self._std

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
        deterministic = [v == 0 for v in self.stds]
        assert all(deterministic) or not any(
            deterministic
        ), "Either all distributions must be deterministic (ie std=0) or stochastic (var>0)"

        if all(deterministic):
            self._dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=self._weights),
                components_distribution=tfd.Deterministic(loc=self.means),
            )
        else:
            self._dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=self._weights),
                components_distribution=tfd.TruncatedNormal(
                    loc=self.means,
                    scale=self.stds,
                    low=[g.low for g in self._gaussians],
                    high=[g.high for g in self._gaussians],
                ),
            )

    def __repr__(self):
        msg = " | ".join([f"{w: .2f}*N({m: .4f}, {v: .4f})" for w, m, v in zip(self.weights, self.means, self.stds)])
        return f"GMM | {msg} | percentiles={self.percentiles}"

    def __add__(self, other: "Distribution"):
        if isinstance(other, Recorded):
            other = other._dist

        # Convert to GMM
        if isinstance(other, Gaussian):
            other = GMM([other], weights=[1.0])

        # Only compatible with Gaussian or GMM
        if not isinstance(other, GMM):
            raise NotImplementedError("Not yet implemented")

        gaussians, weights = [], []
        for w, m, v, p in zip(self.weights, self.means, self.vars, self.percentiles):
            for ow, om, ov, op in zip(other.weights, other.means, other.vars, other.percentiles):
                weights.append(w * ow)
                # if p != op:
                #     print(f"WARNING: Percentiles do not match. {p} != {op}. Gaussian with higher percentile will be used.")
                gaussians.append(Gaussian(m + om, (v + ov) ** (1 / 2), percentile=max(p, op)))
        return GMM(gaussians, weights)

    def __getstate__(self):
        """Used for pickling"""
        args = ()
        kwargs = dict(gaussians=self._gaussians, weights=self._weights)
        return args, kwargs

    def __setstate__(self, state):
        """Used for unpickling"""
        args, kwargs = state
        self.__init__(*args, **kwargs)

    def pdf(self, x: jax.typing.ArrayLike):
        return self._dist.prob(x)

    def cdf(self, x: jax.typing.ArrayLike):
        return self._dist.cdf(x)

    def quantile(self, x: jax.typing.ArrayLike):
        """Returns the quantile of the distribution at the given percentile."""
        deterministic = [v == 0 for v in self.stds]
        if all(deterministic):
            return self.means
        else:
            shape = x.shape if isinstance(x, (jax.Array, onp.ndarray)) else ()
            qs = mixture_distribution_quantiles(
                self._dist, jnp.array(x).reshape(-1), N_grid_points=int(1e3), grid_min=self.low, grid_max=self.high
            )
            return qs.reshape(shape)

    def reset(self, rng: jax.random.KeyArray) -> DistState:
        return DistState(rng=rng, state=EmptyState())

    def sample(self, state: DistState, shape: Union[int, Tuple] = None) -> Tuple[DistState, jax.Array]:
        if shape is None:
            shape = ()
        new_rng, rng_sample = jax.random.split(state.rng)
        samples = self._dist.sample(sample_shape=shape, seed=rng_sample)
        new_state = DistState(rng=new_rng, state=EmptyState())
        return new_state, samples

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
    def percentiles(self):
        return [g.percentile for g in self._gaussians]

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
    def stds(self):
        return [g.std for g in self._gaussians]

    @property
    def low(self):
        return min([g.low for g in self._gaussians])

    @property
    def high(self):
        return max([g.high for g in self._gaussians])


class Recorded:
    def __init__(self, dist: "Distribution", samples: jax.typing.ArrayLike):
        super(Recorded, self).__setattr__("_dist", dist)
        super(Recorded, self).__setattr__("_samples", samples)
        # self._dist = dist
        # self._samples = samples

    def __repr__(self):
        return f"Recorded | {self._dist.__repr__}"

    def __getstate__(self):
        """Used for pickling"""
        args = ()
        kwargs = dict(dist=self._dist, samples=self._samples)
        return args, kwargs

    def __setstate__(self, state):
        """Used for unpickling"""
        args, kwargs = state
        self.__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        setattr(self._dist, key, value)

    def __getattr__(self, item):
        return getattr(self._dist, item)

    def reset(self, rng: jax.random.KeyArray) -> DistState:
        return DistState(rng=rng, state=RecordedState(index=0))

    def sample(self, state: DistState, shape: Union[int, Tuple] = None) -> Tuple[DistState, jax.Array]:
        if shape is None:
            shape = ()
            num_samples = 1
        elif isinstance(shape, int):
            num_samples = shape
            shape = (num_samples,)
        elif len(shape) == 0:
            num_samples = 1
            shape = ()
        else:
            # Sum all elements in tuple
            num_samples = reduce(lambda x, y: x * y, shape, 1)

        # Add samples at the end to make sure we have enough
        num_seqs = 1 + num_samples // self._samples.shape[0]
        all_samples = jnp.concatenate([self._samples] * num_seqs + [self._samples[:num_samples]], axis=0)

        # Get samples
        start = state.state.index
        samples = jax.lax.dynamic_slice(all_samples, (start,), (num_samples,))

        # Determine new index
        new_index = (start + num_samples) % self._samples.shape[0]
        new_state = DistState(rng=state.rng, state=RecordedState(index=new_index))
        return new_state, samples.reshape(shape)


Distribution = Union[Gaussian, GMM, Recorded]


import numpy as np


def mixture_distribution_quantiles(dist, probs, N_grid_points: int = int(1e3), grid_min: float = None, grid_max: float = None):
    """More info: https://github.com/tensorflow/probability/issues/659"""
    base_grid = np.linspace(grid_min, grid_max, num=int(N_grid_points))
    shape = (dist.batch_shape, 1) if len(dist.batch_shape) else [1]
    full_grid = np.transpose(np.tile(base_grid, shape))
    cdf_grid = dist.cdf(full_grid)  # this is fully parallelized and even uses GPU
    grid_check = (cdf_grid.min(axis=0).max() <= min(probs)) & (max(probs) <= cdf_grid.max(axis=0).min())
    if not grid_check:
        print(f"Grid min: {grid_min}, max: {grid_max} | CDF min: {cdf_grid.min(axis=0).max()}, max: {cdf_grid.max(axis=0).min()} | Probs min: {min(probs)}, max: {max(probs)}")
        raise RuntimeError("Grid does not span full CDF range needed for interpolation!")

    probs_row_grid = np.transpose(np.tile(np.array(probs), (cdf_grid.shape[0], 1)))

    def get_quantiles_for_one_observation(cdf_grid_one_obs):
        return base_grid[np.argmax(np.greater(cdf_grid_one_obs, probs_row_grid), axis=1)]

    # TODO: this is the main performance bottleneck. uses only one CPU core
    quantiles_grid = np.apply_along_axis(
        func1d=get_quantiles_for_one_observation,
        axis=0,
        arr=cdf_grid,
    )
    return quantiles_grid
