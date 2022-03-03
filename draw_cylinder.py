import numpy as np
from matplotlib import cm
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    varied_radius = np.linspace(0,10,50)
    x_grid = (np.cos(theta_grid).T*varied_radius).T + center_x
    y_grid = (np.sin(theta_grid).T*varied_radius).T + center_y
    pressure = np.linspace(0, 1, 50)
    theta_grid, pressure_grid = np.meshgrid(theta, pressure)
    return x_grid,y_grid,z_grid, pressure_grid

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Xc,Yc,Zc, Pc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1)
ax.plot_surface(Zc, Xc, Yc,facecolors=cm.Oranges(Pc), alpha=0.8)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# create some fake data
x = y = np.arange(-4.0, 4.0, 0.02)
# here are the x,y and respective z values
X, Y = np.meshgrid(x, y)
Z = np.sinc(np.sqrt(X*X+Y*Y))
# this is the value to use for the color
V = np.sin(Y)

# create the figure, add a 3d axis, set the viewing angle
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45,60)

# here we create the surface plot, but pass V through a colormap
# to create a different color for each patch
ax.plot_surface(X, Y, Z, facecolors=cm.Oranges(V))
plt.show()