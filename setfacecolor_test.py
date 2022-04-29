import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def data_for_structure_along_z(N_sec):
    structure_length = 5
    radius = np.ones(N_sec)
    N_sec = N_sec
    z = np.linspace(0, structure_length*10**2, N_sec)
    theta = np.linspace(0, 2 * np.pi, N_sec)
    theta_grid, z_grid = np.meshgrid(theta, z)
    varied_radius = radius*10**2
    x_grid = (np.cos(theta_grid).T * varied_radius).T
    y_grid = (np.sin(theta_grid).T * varied_radius).T
    return x_grid, y_grid, -z_grid, theta
number_sec = 5
x,y,z, theta = data_for_structure_along_z(number_sec)
COL = MplColorHelper('autumn_r', 2, 5)
figure = plt.figure()
ax = figure.add_subplot(1, 3, 3, projection = '3d')

theta_g, p_g1 = np.meshgrid(theta, np.linspace(0, 10,number_sec))
scat = ax.plot_surface(x,y,z, facecolors=COL.get_rgb(p_g1))
theta_g, p_g2 = np.meshgrid(theta, np.linspace(0, 10,number_sec))
scat.set_facecolors(COL.get_rgb(p_g1[:-1, :-1]).reshape((len(x)-1)*(len(y)-1), 4))
ax.set_title('Well defined discrete colors')
plt.show()