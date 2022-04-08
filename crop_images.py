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

def get_image_file_names(path):
    file_list = os.listdir(path)
    file_list = sorted(file_list, key=lambda x: float(re.findall("(\d+)", x)[0]))
    file_list = [os.path.join(path, file_list[i]) for i in range(0, len(file_list), 1)]
    return file_list
y_start =355
y_end = 905
x_start = 0
x_end = 715
heart_image_filenames = get_image_file_names(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\heart_side')
for i, image in enumerate(heart_image_filenames):
    image_ = cv2.imread(image)[x_start:x_end, y_start:y_end]
    cv2.imshow('test', image_)
    cv2.imwrite(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\heart_side'+r'\ajd'+str(i)+r'.jpg', image_)


stop = True