"""
This approach is credited to @ericmjl and his dl-workshop. The original file is available here:
https://github.com/ericmjl/dl-workshop/blob/6ef9b7feb60dd5f6a4dbdda4dc899337e583a397/src/dl_workshop/gaussian_mixture.py
"""

import itertools
import jax.numpy as np
from jax.scipy import stats


def loglike_one_component(component_weight, component_mu, log_component_scale, datum):
    """Log likelihood of datum under one component of the mixture.

    Defined as the log likelihood of observing that datum from the component
    (i.e. log of component probability)
    added to the log likelihood of observing that datum
    under the Gaussian that belongs to that component.

    :param component_weight: Component weight, a scalar value between 0 and 1.
    :param component_mu: A scalar value.
    :param log_component_scale: A scalar value.
        Gets exponentiated before being passed into norm.logpdf.
    :returns: A scalar.
    """
    component_scale = np.exp(log_component_scale)
    return np.log(component_weight) + stats.norm.logpdf(datum, loc=component_mu, scale=component_scale)


from jax.scipy.special import logsumexp


def normalize_weights(weights):
    """Normalize a weights vector to sum to 1."""
    return weights / np.sum(weights)


from functools import partial

from jax import vmap


def loglike_across_components(log_component_weights, component_mus, log_component_scales, datum):
    """Log likelihood of datum under all components of the mixture."""
    component_weights = normalize_weights(np.exp(log_component_weights))
    loglike_components = vmap(partial(loglike_one_component, datum=datum))(
        component_weights, component_mus, log_component_scales
    )
    return logsumexp(loglike_components)


def mixture_loglike(log_component_weights, component_mus, log_component_scales, data):
    """Log likelihood of data (not datum!) under all components of the mixture."""
    ll_per_data = vmap(
        partial(
            loglike_across_components,
            log_component_weights,
            component_mus,
            log_component_scales,
        )
    )(data)
    return np.sum(ll_per_data)


# def weights_loglike(log_component_weights, alpha_prior):
#     """Log likelihood of weights under Dirichlet distribution"""
#     component_weights = np.exp(log_component_weights)
#     component_weights = normalize_weights(component_weights)
#     return stats.dirichlet.logpdf(x=component_weights, alpha=alpha_prior)


# def loss_mixture_weights(params, data):
#     """Loss function for first model.
#
#     Takes into account log probability of data under mixture model
#     and log probability of weights under a constant Dirichlet concentration vector.
#     """
#     log_component_weights, component_mus, log_component_scales = params
#     loglike_mixture = mixture_loglike(log_component_weights, component_mus, log_component_scales, data)
#     alpha_prior = np.ones_like(component_mus) * 2
#     loglike_weights = weights_loglike(log_component_weights, alpha_prior=alpha_prior)
#
#     total = loglike_mixture + loglike_weights
#     return -total


def step(i, state, get_params_func, dloss_func, update_func, data):
    """Generic step function."""
    params = get_params_func(state)
    g = dloss_func(params, data)
    state = update_func(i, g, state)
    return state


def make_step_scannable(get_params_func, dloss_func, update_func, data):
    def inner(previous_state, iteration):
        new_state = step(
            i=iteration,
            state=previous_state,
            get_params_func=get_params_func,
            dloss_func=dloss_func,
            update_func=update_func,
            data=data,
        )
        return new_state, previous_state

    return inner


from jax.scipy.stats import norm


# def get_loss(state, get_params_func, loss_func, data):
#     params = get_params_func(state)
#     loss_score = loss_func(params, data)
#     return loss_score


def get_component_norm_pdfs(
    log_component_weights,
    component_mus,
    log_component_scales,
    xmin,
    xmax,
    num_points,
):
    component_weights = normalize_weights(np.exp(log_component_weights))
    component_scales = np.exp(log_component_scales)
    x = np.linspace(xmin, xmax, num_points).reshape(-1, 1)
    pdfs = component_weights * norm.pdf(x, loc=component_mus, scale=component_scales)
    pdf = np.sum(pdfs, axis=1)
    return x, pdfs, pdf


def plot_component_norm_pdfs(
    log_component_weights,
    component_mus,
    log_component_scales,
    xmin,
    xmax,
    num_points,
    ax,
    animated: bool = False,
    plot_comp=True,
    plot_gmm=True,
    kwargs_comp=None,
    kwargs_gmm=None,
):
    x, pdfs, pdf = get_component_norm_pdfs(log_component_weights, component_mus, log_component_scales, xmin, xmax, num_points)
    artists = []
    if plot_comp:
        kwargs_comp = kwargs_comp or itertools.repeat({})  # Default to an empty dict if no generator provided
        for component, kwargs in zip(range(pdfs.shape[1]), kwargs_comp):
            kwargs["label"] = kwargs.get("label", f"comp. {component}")
            a = ax.plot(x, pdfs[:, component], **kwargs, animated=animated)[0]
            artists.append(a)
    if plot_gmm:
        kwargs_gmm = kwargs_gmm or {}
        kwargs_gmm["label"] = kwargs_gmm.get("label", "gmm")
        kwargs_gmm["linestyle"] = kwargs_gmm.get("linestyle", "--")
        kwargs_gmm["color"] = kwargs_gmm.get("color", "red")
        a = ax.plot(x, pdf, **kwargs_gmm, animated=animated)[0]
        artists.append(a)
    return artists


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jax import lax


def animate_training(
    params_for_plotting,
    data_mixture,
    num_frames,
    fig=None,
    ax=None,
    edgecolor: str = None,
    facecolor: str = None,
    bins: int = 40,
    xmin: float = None,
    xmax: float = None,
    num_points: int = 1000,
):
    """Animation function for mixture likelihood."""
    if fig is None:
        assert ax is None, "If fig is None, ax must also be None."
        fig, ax = plt.subplots()

    xmin = xmin or np.min(data_mixture)
    xmax = xmax or np.max(data_mixture)
    (
        log_component_weights_history,
        component_mus_history,
        log_component_scales_history,
    ) = params_for_plotting
    assert num_frames > 0, "num_frames must be positive."
    interval = len(log_component_weights_history) // num_frames
    log_component_weights_history = log_component_weights_history[::interval]
    component_mus_history = component_mus_history[::interval]
    log_component_scales_history = log_component_scales_history[::interval]

    # Draw initial
    w, m, s = log_component_weights_history[0], component_mus_history[0], log_component_scales_history[0]
    artists = plot_component_norm_pdfs(w, m, s, xmin=xmin, xmax=xmax, ax=ax, num_points=num_points, animated=True)
    artists_pdfs, artist_pdf = artists[:-1], artists[-1]

    def init():
        return tuple(artists)

    def update(frame):
        i = int(frame)
        w, m, s = log_component_weights_history[i], component_mus_history[i], log_component_scales_history[i]
        _, pdfs, pdf = get_component_norm_pdfs(w, m, s, xmin, xmax, num_points)
        for component, a in enumerate(artists_pdfs):
            a.set_ydata(pdfs[:, component])
        artist_pdf.set_ydata(pdf)
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=num_frames, blit=True)
    return anim


# def stick_breaking_weights(beta_draws):
#     """Return weights from a stick breaking process.
#
#     :param beta_draws: i.i.d draws from a Beta distribution.
#         This should be a row vector.
#     """
#
#     def weighting(occupied_probability, beta_i):
#         """
#         :param occupied_probability: The cumulative occupied probability taken up.
#         :param beta_i: Current value of beta to consider.
#         """
#         weight = (1 - occupied_probability) * beta_i
#         return occupied_probability + weight, weight
#
#     occupied_probability, weights = lax.scan(weighting, np.array(0.0), beta_draws)
#
#     weights = weights / np.sum(weights)
#     return occupied_probability, weights


from jax import random


# def weights_one_concentration(concentration, key, num_draws, num_components):
#     beta_draws = random.beta(key=key, a=1, b=concentration, shape=(num_draws, num_components))
#     occupied_probability, weights = vmap(stick_breaking_weights)(beta_draws)
#     return occupied_probability, weights


def beta_draw_from_weights(weights):
    def beta_from_w(accounted_probability, weights_i):
        """
        :param accounted_probability: The cumulative probability acounted for.
        :param weights_i: Current value of weights to consider.
        """
        denominator = 1 - accounted_probability
        log_denominator = np.log(denominator)

        log_beta_i = np.log(weights_i) - log_denominator

        newly_accounted_probability = accounted_probability + weights_i

        return newly_accounted_probability, np.exp(log_beta_i)

    final, betas = lax.scan(beta_from_w, np.array(0.0), weights)
    return final, betas


def component_probs_loglike(log_component_probs, log_concentration, num_components):
    """Evaluate log likelihood of probability vector under Dirichlet process.

    :param log_component_probs: A vector.
    :param log_concentration: Real-valued scalar.
    :param num_compnents: Scalar integer.
    """
    concentration = np.exp(log_concentration)
    component_probs = normalize_weights(np.exp(log_component_probs))
    _, beta_draws = beta_draw_from_weights(component_probs)
    # eval_draws = beta_draws[ops.index[:num_components]]
    eval_draws = beta_draws[np.index_exp[:num_components]]
    return np.sum(stats.beta.logpdf(x=eval_draws, a=1, b=concentration))


def joint_loglike(log_component_weights, log_concentration, num_components, component_mus, log_component_scales, data):
    component_probs = np.exp(log_component_weights)
    probs_ll = component_probs_loglike(log_component_weights, log_concentration, num_components)

    mix_ll = mixture_loglike(log_component_weights, component_mus, log_component_scales, data)

    return probs_ll + mix_ll


def make_joint_loss(num_components):
    def inner(params, data):
        (log_component_weights, log_concentration, component_mus, log_component_scales) = params

        ll = joint_loglike(
            log_component_weights,
            log_concentration,
            num_components,
            component_mus,
            log_component_scales,
            data,
        )
        return -ll

    return inner


import matplotlib.animation
from jax.example_libraries.optimizers import adam
from jax import grad, jit
from time import time
import distrax

from rex import base, utils


class GMMEstimator:
    def __init__(self, data, name="GMM", threshold=1e-7, verbose: bool = True):
        self.name = name
        self.data: np.ndarray = data.astype(np.float32)
        self.final_state_norm = None
        self.threshold = threshold
        self.verbose = verbose
        self.is_deterministic = True if self.data.std() < threshold else False
        self._mean = np.mean(data)
        self._std = np.std(data)
        self._data_norm: np.ndarray = (data - data.mean()) / max(data.std(), 1e-7) if not self.is_deterministic else data

    def fit(self, num_steps: int = 100, num_components: int = 2, step_size: float = 0.05, seed: int = 0):
        if self.is_deterministic:
            if self.verbose:
                print(
                    f"{self.name} | Skip because close to deterministic | mean(data)={self.data.mean()} | std(data)={self.data.std()}."
                )
            return

        # Store fit parameters
        self.seed = seed
        self.num_components = num_components
        self.n_components = 2 * num_components  # todo: Why must num_components = 2*n_components?
        self.num_steps = num_steps
        self.step_size = step_size

        # Define loss
        joint_loss = make_joint_loss(num_components=num_components)
        self.joint_loss = jit(joint_loss)
        djoint_loss = grad(joint_loss)

        # Initialization
        key = random.PRNGKey(seed)
        k1, k2, k3, k4 = random.split(key, 4)

        log_component_weights_init = random.normal(k1, shape=(self.n_components,))
        log_concentration_init = random.normal(k2, shape=(1,))
        component_mus_init = random.normal(k3, shape=(self.n_components,))
        log_component_scales_init = random.normal(k4, shape=(self.n_components,))

        params_init = log_component_weights_init, log_concentration_init, component_mus_init, log_component_scales_init

        # Training Loop
        self.adam_init, self.adam_update, self.adam_get_params = adam(self.step_size)
        step_scannable = make_step_scannable(
            get_params_func=self.adam_get_params,
            dloss_func=djoint_loss,
            update_func=self.adam_update,
            data=self._data_norm,
        )
        initial_state = self.adam_init(params_init)

        # Run training
        start = time()
        self.final_state_norm, state_history_norm = lax.scan(step_scannable, initial_state, np.arange(num_steps))
        if self.verbose:
            print(f"{self.name} | Time taken: {time() - start:.2f} seconds.")

        # Store training history
        params_history_norm = self.adam_get_params(state_history_norm)
        (
            self.log_component_weights_history_norm,
            self.log_concentration_history_norm,
            self.component_mus_history_norm,
            self.log_component_scales_history_norm,
        ) = params_history_norm

        # Rescale mu & scale to original data# Rescale
        (
            self.log_component_weights_history,
            self.log_concentration_history,
            self.component_mus_history,
            self.log_component_scales_history,
        ) = self._rescale(params_history_norm)

    def _rescale(self, params):
        log_component_weights, log_concentration, component_mus, log_component_scales = params
        component_mus = component_mus * self._std + self._mean
        log_component_scales = log_component_scales + np.log(self._std)
        return log_component_weights, log_concentration, component_mus, log_component_scales

    def plot_hist(
        self,
        ax: plt.Axes = None,
        edgecolor: str = None,
        facecolor: str = None,
        bins=100,
        xmin: float = None,
        xmax: float = None,
        num_points: int = 1000,
        plot_dist: bool = True,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.data, bins=bins, density=True, label="data", edgecolor=edgecolor, facecolor=facecolor, alpha=0.5)
        if plot_dist and not self.is_deterministic and self.final_state_norm is not None:
            xmin = xmin or np.min(self.data)
            xmax = xmax or np.max(self.data)
            w, _, m, s = self._rescale(self.adam_get_params(self.final_state_norm))
            plot_component_norm_pdfs(w, m, s, xmin=xmin, xmax=xmax, ax=ax, num_points=num_points)
        return ax

    def plot_loss(self, ax: plt.Axes = None, edgecolor: str = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        # Skip if deterministic
        if self.is_deterministic:
            return ax

        assert self.final_state_norm is not None, "You must fit the model before plotting the loss."
        losses = []
        for w, c, m, s in zip(
            self.log_component_weights_history_norm,
            self.log_concentration_history_norm,
            self.component_mus_history_norm,
            self.log_component_scales_history_norm,
        ):
            prm = (w, c, m, s)
            l = self.joint_loss(prm, self._data_norm)
            losses.append(l)

        ax.plot(losses, label="loss", color=edgecolor)
        ax.set(yscale="log")
        return ax

    def plot_normalized_weights(self, ax: plt.Axes = None, edgecolor: str = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        # Skip if deterministic
        if self.is_deterministic:
            return ax

        assert self.final_state_norm is not None, "Must fit model before plotting."
        params_opt = self.adam_get_params(self.final_state_norm)
        log_component_weights_opt = params_opt[0]
        component_weights_opt = np.exp(log_component_weights_opt)
        norm_weights = normalize_weights(component_weights_opt)
        sorted_weights = np.sort(norm_weights)
        ax.stem(sorted_weights, markerfmt="o", linefmt=edgecolor, label="weights")
        return ax

    def animate_training(
        self,
        num_frames: int = 30,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        edgecolor: str = None,
        facecolor: str = None,
        bins: int = 40,
        xmin: float = None,
        xmax: float = None,
        num_points: int = 1000,
    ) -> matplotlib.animation.FuncAnimation:
        assert not self.is_deterministic, "Model must not be deterministic for animating."
        assert self.final_state_norm is not None, "Must train model before animating."

        params_for_plotting = [
            self.log_component_weights_history,
            self.component_mus_history,
            self.log_component_scales_history,
        ]

        anim = animate_training(
            params_for_plotting,
            self.data,
            num_frames,
            fig=fig,
            ax=ax,
            edgecolor=edgecolor,
            facecolor=facecolor,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            num_points=num_points,
        )
        return anim

    def get_dist(self, percentile: float = 0.99) -> base.StaticDist:
        if self.is_deterministic:
            dist = distrax.Deterministic(loc=self.data.mean(dtype="float32"))
            return base.StaticDist.create(dist)
        assert self.final_state_norm is not None, "Must train model before exporting distribution."

        # Get weights
        log_w, _, m, log_s = self._rescale(self.adam_get_params(self.final_state_norm))
        w = normalize_weights(np.exp(log_w))
        indices = np.argsort(w)
        w, s, m = w[indices], np.exp(log_s)[indices], m[indices]

        # Prune weights until percentile
        prune_cum, prune_idx = 0.0, 0
        for i, val in enumerate(w):
            if prune_cum + val < (1 - percentile):
                prune_idx += 1
                prune_cum += val
            else:
                break
        w, s, m = w[prune_idx:], s[prune_idx:], m[prune_idx:]
        w = normalize_weights(w)

        # Define non-truncated gmm
        cdist = distrax.Normal(loc=m, scale=s)
        dist = distrax.MixtureSameFamily(mixture_distribution=distrax.Categorical(probs=w), components_distribution=cdist)
        return base.StaticDist.create(dist)
