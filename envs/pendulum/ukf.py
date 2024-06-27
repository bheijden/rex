"""https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb"""
from typing import Union, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

from rexv2.base import Base


@struct.dataclass
class UKFState(Base):
    mu: jax.typing.ArrayLike
    sigma: jax.typing.ArrayLike


@struct.dataclass
class SigmaPoints(Base):
    points: jax.typing.ArrayLike
    weights_mu: jax.typing.ArrayLike
    weights_sigma: jax.typing.ArrayLike

    @staticmethod
    def to_points(state: UKFState, alpha: float = None, beta: float = None, kappa: float = None) -> "SigmaPoints":
        """https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/sigma_points.html"""

        n_dim = state.mu.shape[0]
        alpha = alpha or 1.0  # Determines the spread of the sigma points around the mean (usually a small positive value, e.g. 1e-3)
        beta = beta or 2.0  # 2.0 is optimal for a Gaussian distribution
        kappa = kappa or 3.0 - n_dim  # 3-n_dim is optimal for a Gaussian distribution

        # Calculate scaling factor for all off-center points
        lamda = alpha**2 * (n_dim + kappa) - n_dim

        # compute sqrt(sigma)
        U = jnp.linalg.cholesky((lamda + n_dim)*state.sigma).T  # Upper triangular

        # Compute sigma points
        points = jnp.zeros((2 * n_dim + 1, n_dim))  # shape (2n_dim+1, n_dim)
        points = points.at[:].set(state.mu)
        points = points.at[1:(n_dim + 1)].add(U)
        points = points.at[(n_dim + 1):].add(-U)

        c = .5 / (n_dim + lamda)
        weights_mu = jnp.full(2 * n_dim + 1, c)
        weights_sigma = jnp.copy(weights_mu)
        weights_mu = weights_mu.at[0].set(lamda / (n_dim + lamda))
        weights_sigma = weights_sigma.at[0].set(lamda / (n_dim + lamda) + (1 - alpha**2 + beta))
        return SigmaPoints(points=points, weights_mu=weights_mu, weights_sigma=weights_sigma)

    def to_moments(self) -> UKFState:
        """https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/unscented_transform.html#unscented_transform"""
        mu = jnp.dot(self.weights_mu, self.points)
        y = self.points - mu
        sigma = jnp.dot(y.T, jnp.dot(jnp.diag(self.weights_sigma), y))
        return UKFState(mu=mu, sigma=sigma)


@struct.dataclass
class UKFParams(Base):
    """https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb"""
    alpha: float = struct.field(default=None)
    beta: float = struct.field(default=None)
    kappa: float = struct.field(default=None)

    def _unscented_transform(self, points: SigmaPoints, fn: Callable[[jax.Array], jax.Array]) -> Tuple[SigmaPoints, UKFState]:
        # n_points, n_dim_state = points.points.shape
        vmap_fn = jax.vmap(fn, in_axes=0)
        points_trans = points.replace(points=vmap_fn(points.points))
        state_trans = points_trans.to_moments()
        return points_trans, state_trans

    def predict_and_update(self, Fx: Callable[[jax.Array], jax.Array], Qx: Callable[[jax.Array], jax.Array],
                           Gx: Callable[[jax.Array], jax.Array], Rx: Callable[[jax.Array], jax.Array],
                           state: UKFState, y: jax.typing.ArrayLike) -> UKFState:
        # Forward predict to measurement time
        state_fwd = self.predict(Fx, Qx, state)
        # Measurement update
        state_upd = self.update(Gx, Rx, state_fwd, y)
        return state_upd

    def predict(self,  Fx: Callable[[jax.Array], jax.Array], Qx: Callable[[jax.Array], jax.Array],
                state: UKFState) -> UKFState:
        # Generate sigma points
        points = SigmaPoints.to_points(state, self.alpha, self.beta, self.kappa)
        # Forward predict
        points_pred, state_pred = self._unscented_transform(points, Fx)
        # Additive process noise
        state_fwd = state_pred.replace(sigma=state_pred.sigma + Qx(state.mu))
        return state_fwd

    def update(self, Gx: Callable[[jax.Array], jax.Array], Rx: Callable[[jax.Array], jax.Array],
               state: UKFState, y: jax.typing.ArrayLike) -> UKFState:
        # Generate sigma points
        points = SigmaPoints.to_points(state, self.alpha, self.beta, self.kappa)
        # Measurement update
        points_yp, state_yp = self._unscented_transform(points, Gx)
        # Additive measurement noise
        state_yp = state_yp.replace(sigma=state_yp.sigma + Rx(state.mu))
        yp, S = state_yp.mu, state_yp.sigma
        SI = jnp.linalg.inv(S)
        # Compute cross-variance
        dx = points.points - state.mu
        dy = points_yp.points - yp
        Pxy_fn = lambda w, _dx, _dy: w * jnp.outer(_dx, _dy)
        Pxy = jax.vmap(Pxy_fn)(points.weights_sigma, dx, dy).sum(axis=0)
        # Compute gain
        K = jnp.dot(Pxy, SI)
        # Compute error
        e = y - yp
        # Update
        new_mu = state.mu + jnp.dot(K, e)
        new_sigma = state.sigma - jnp.dot(K, jnp.dot(S, K.T))
        state_upd = state.replace(mu=new_mu, sigma=new_sigma)
        return state_upd