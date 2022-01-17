import numpy as np
import matplotlib.pyplot as  plt
import matplotlib.colors as colors
from matplotlib import cm

line = np.arange(-1,1,0.005)

xx, yy = np.meshgrid(line, line, indexing='ij')

dist_orig = np.sqrt(xx**2 + yy**2)
dist = np.abs(dist_orig-0.5)
dist[dist_orig<0.5] = -dist[dist_orig<0.5]

levels =400
vmin = np.min(dist)
vmax = np.max(dist)
level_boundaries = np.linspace(vmin, vmax, levels + 1)
divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = cm.seismic


fig = plt.figure(figsize=(10,5))
h = plt.pcolormesh(xx, yy, dist, norm=divnorm, cmap=cmap)
plt.axis('scaled')
plt.colorbar()
plt.title("SDF on a disk (center=(0,0), radius=0.5)")
fig.savefig('imgs/sdf.png')

dist_orig = np.sqrt(xx**2 + yy**2)
dist = np.zeros_like(dist_orig)
dist[dist_orig<0.5] = 1
fig2 = plt.figure(figsize=(10,5))

h = plt.pcolormesh(xx, yy, dist, cmap=cmap)
plt.axis('scaled')
plt.colorbar()
plt.title("Occupancy on a disk (center=(0,0), radius=0.5)")
fig2.savefig('imgs/occ.png')
