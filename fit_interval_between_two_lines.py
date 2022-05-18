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
from scipy.interpolate import interp1d
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
from scipy.optimize import curve_fit
from pylab import *
import sympy as sym

point_of_expansion = 10
window_of_quadratic_expansion =0.2
radius_straight = 1
expansion_angle = 2
structure_total_length = 20
point_of_linear_expansion = point_of_expansion+window_of_quadratic_expansion


length_expansion = structure_total_length - point_of_expansion
radius_end = np.tan(np.radians(expansion_angle))*length_expansion + radius_straight

m = (radius_end - radius_straight)/length_expansion
ycross = radius_straight - m*point_of_expansion
value_point_of_linear_expansion = m*(point_of_linear_expansion)+ycross


a,b,c,d = sym.symbols(['a', 'b', 'c', 'd'])
y1 = a*point_of_expansion**3 + b*point_of_expansion**2 + c*point_of_expansion +d
y2 = 3*a*point_of_expansion**2 + 2*b*point_of_expansion+c
y3 = a*point_of_linear_expansion**3 + b*point_of_linear_expansion**2 + c*point_of_linear_expansion +d
y4 = 3*a*point_of_linear_expansion**2 + 2*b*point_of_linear_expansion +c

sol = sym.solve([y1-radius_straight, y2-0, y3-value_point_of_linear_expansion, y4-m], dict=True)
sol = sol[0]



xtrans = np.linspace(point_of_expansion, point_of_linear_expansion, 20)
ytrans = sol[a]*xtrans**3 + sol[b]*xtrans**2+sol[c]*xtrans + sol[d]
xstraight = np.linspace(0, point_of_expansion, 50)
ystraight = np.ones(50)*radius_straight
xexp = np.linspace(point_of_linear_expansion,structure_total_length, 50)
yexp = m*xexp +ycross
x = np.hstack((xstraight, xtrans, xexp))
y = np.hstack((ystraight, ytrans, yexp))
f = interp1d(x, y)







