# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:55:42 2021
@author: Faiza
"""
import copy
import sys
import os
import numpy as np
from numpy import save
import PH_helper_functions as func
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from profilehooks import profile
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from tkinter import filedialog as fd
import pandas as pd
from scipy.signal import savgol_filter
import re
import cv2
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import pickle
import matplotlib.lines as lines
from matplotlib.patches import Ellipse
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import sympy as sym

length = 3
radius_end = np.tan(np.radians(5))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
#
loss1 = (1-(A[1:]/A[:-1]))**1
loss1 = np.append(loss1, 0)

length = 3
radius_end = np.tan(np.radians(3))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
loss2 = (1-(A[1:]/A[:-1]))**1
loss2 = np.append(loss2, 0)
length = 3
radius_end = np.tan(np.radians(10))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
loss3 = (1-(A[1:]/A[:-1]))**1
loss3 = np.append(loss3, 0)

plt.figure()
plt.title('Determine loss with version 2')
plt.plot(x,loss1, label = 'loss five')
plt.plot(x, loss2, label = 'loss three')
plt.plot(x, loss3, label = 'loss ten')
plt.legend()
plt.ylabel('loss')
plt.xlabel('axial tube position [cm]')
plt.show()

length = 3
radius_end = np.tan(np.radians(5))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
#
rey1 = 2*r*100/(A)*1.06/0.04

length = 3
radius_end = np.tan(np.radians(3))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
rey2 = 2*r*100/(A)*1.06/0.04
length = 3
radius_end = np.tan(np.radians(10))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
rey3 = 2*r*100/(A)*1.06/0.04

length = 3
radius_end = np.tan(np.radians(1))*length
radius_start = 2
m = (radius_end - radius_start) / length
x = np.linspace(0, 3, int(3/0.2))

r = radius_start +m*x

A = np.pi*r**2
rey4 = 2*r*100/(A)*1.06/0.04


plt.figure()
plt.title('Determine Reynolds')
plt.plot(x,rey4, color = 'b', label = 'minus one')
plt.plot(x,rey1, color = 'g', label = 'minus five')
plt.plot(x, rey2, color = 'r',label = 'minus three')
plt.plot(x, rey3, color = 'k',label = 'minus ten')
plt.legend()
plt.ylabel('reynolds number')
plt.xlabel('axial tube position [cm]')
plt.show()