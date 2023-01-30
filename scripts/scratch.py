# import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# # Create x, y coords
# nx, ny = 800, 1600
# cellsize = 1.
# x = np.arange(0., float(nx), 1.) * cellsize
# y = np.arange(0., float(ny), 1.) * cellsize
# X, Y = np.meshgrid(x, y)
#
# # dummy data
# Z = (X**2 + Y**2) / 1e6
#
# # Create matplotlib Figure and Axes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# print (X.shape, Y.shape, Z.shape)
#
# # Plot the surface
# ax.plot_surface(X, Y, Z)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# plt.show()

import jumpy.numpy as jp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# import seaborn as sns
# sns.set()
import matplotlib
matplotlib.use("TkAgg")

pi = jp.pi

x = jp.linspace(-pi, pi, 1000)
th, th2 = jp.meshgrid(x, x)

# Wrap
cos_th, sin_th = jp.cos(th), jp.sin(th)
cos_th2, sin_th2 = jp.cos(th2), jp.sin(th2)
thw, thw2 = jp.arctan2(sin_th, cos_th), jp.arctan2(sin_th2, cos_th2)

# Calculate cost
cost = (pi - (thw + thw2)) ** 2
cost = (pi - jp.abs(thw + thw2)) ** 2

# Create matplotlib Figure and Axes
fig = plt.figure()
ax =fig.add_subplot(111)

c = ax.pcolormesh(th, th2, cost, cmap="jet", vmin=cost.min(), vmax=cost.max())
ax.set_xlabel('th')
ax.set_ylabel('th2')
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

plt.show()
