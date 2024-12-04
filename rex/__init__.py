import importlib.metadata

import rex.artificial as artificial
import rex.asynchronous as asynchronous
import rex.base as base
import rex.cem as cem
import rex.constants as constants
import rex.evo as evo  # Requires evosax
import rex.gmm_estimator as gmm_estimator
import rex.graph as graph
import rex.jax_utils as jax_utils
import rex.node as node
import rex.open_colors as open_colors
import rex.ppo as ppo  # Requires optax
import rex.rl as rl
import rex.utils as utils


__version__ = importlib.metadata.version("rex-lib")

__all__ = [
    "artificial",
    "asynchronous",
    "base",
    "cem",
    "constants",
    "evo",
    "gmm_estimator",
    "graph",
    "jax_utils",
    "node",
    "open_colors",
    "ppo",
    "rl",
    "utils",
    "__version__",
]
