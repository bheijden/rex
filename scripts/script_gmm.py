import jax
from jax import numpy as np, random, vmap, jit, grad, lax
import matplotlib.pyplot as plt

factor = 0.0001
weights_true = np.array([2, 10, 1, 6])
locs_true = (np.array([-2., -5., 3., 8.]) + 8)*factor
scale_true = np.array([1.1, 2, 1., 1.5])*factor

base_n_draws = 1000
key = random.PRNGKey(42)
keys = random.split(key, 4)

draws = []
for i in range(4):
    shape = int(base_n_draws * weights_true[i]),
    draw = scale_true[i] * random.normal(keys[i], shape=shape) + locs_true[i]
    draws.append(draw)
data_mixture = np.abs(np.concatenate(draws))


import seaborn as sns
from rex.open_colors import ecolor, fcolor
from gaussian_mixture import GMMEstimator
sns.set()

estimator = GMMEstimator(data_mixture)
ax = estimator.plot_hist(edgecolor=ecolor.communication, bins=100)
estimator.fit(num_steps=1000, num_components=5, step_size=0.05, seed=1)
gmm = estimator.get_truncated_gmm()
x = np.linspace(data_mixture.min(), data_mixture.max(), num=1000)
y = gmm.pdf(x)
ax.plot(x, y, color='yellow')

estimator.plot_hist(edgecolor=ecolor.communication, bins=100)
estimator.plot_loss(edgecolor=ecolor.computation)
estimator.plot_normalized_weights(edgecolor=ecolor.communication)
animation = estimator.animate_training()
animation.save("gmm_training.mp4")
plt.show()
