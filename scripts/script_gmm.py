import jax
from jax import numpy as np, random, vmap, jit, grad, lax
import matplotlib.pyplot as plt

factor = 0.001
weights_true = np.array([2, 10, 1, 6])
weights_true_norm = weights_true / np.sum(weights_true)
print(f"{weights_true_norm=}")
locs_true = (np.array([-2., -5., 3., 8.]) + 20)*factor
scale_true = np.array([1.1, 2, 1., 1.5])*factor

base_n_draws = 1000
key = random.PRNGKey(42)
key_draw, keydraw2, key = random.split(key, num=3)
keys_draw = random.split(key_draw, 4)
# keys2_draw = random.split(key_draw, 4)

draws = []
# draws2 = []
for i in range(4):
    shape = int(base_n_draws * weights_true[i]),
    draw = scale_true[i] * random.normal(keys_draw[i], shape=shape) + locs_true[i]
    # draw2 = scale_true[i] * random.normal(keys2_draw[i], shape=shape) + locs_true[i]
    draws.append(draw)
    # draws2.append(draw2)
key_perm, key_perm2, key = random.split(key, num=3)
data_mixture = random.permutation(key_perm, np.abs(np.concatenate(draws)))
# data_mixture2 = random.permutation(key_perm2, np.abs(np.concatenate(draws2)))


import seaborn as sns
from rex.open_colors import ecolor, fcolor
from gaussian_mixture import GMMEstimator
sns.set()

estimator = GMMEstimator(data_mixture)
ax = estimator.plot_hist(edgecolor=ecolor.communication, facecolor=fcolor.communication, bins=100)
estimator.fit(num_steps=500, num_components=2, step_size=0.05, seed=1)
gmm = estimator.get_truncated_gmm()
x = np.linspace(data_mixture.min(), data_mixture.max(), num=1000)
y = gmm.pdf(x)
ax.plot(x, y, color='yellow')

# fig, ax = plt.subplots()
# gmm2 = gmm + gmm
# data_mixture2 += data_mixture
# sns.histplot(data_mixture2, ax=ax, bins=100, stat="density", label="data", edgecolor=ecolor.communication, facecolor=fcolor.communication)
# x = np.linspace(data_mixture.min(), data_mixture.max()*2, num=1000)
# y = gmm2.pdf(x)
# ax.plot(x, y, color='yellow')

estimator.plot_hist(edgecolor=ecolor.communication, facecolor=fcolor.communication, bins=100)
estimator.plot_loss(edgecolor=ecolor.computation)
estimator.plot_normalized_weights(edgecolor=ecolor.communication)
# animation = estimator.animate_training()
# animation.save("gmm_training.mp4")
plt.show()
