# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:17:24 2021
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


if __name__ == '__main__':
    path_to_image =r'C:\Users\Faiza\Desktop\1D_PH_FSI\cardiac_cycle_images'
    loadrun = func.GeometryAndInput()
    pdic, internal_information_of_model, results_integration = func.run_simulation(loadrun.tube_base_radius, loadrun.tube_length, int(loadrun.number_sections), path_to_image,input_filename=loadrun.filename)
    simulation_class = func.VisualizeResults(pdic)
    for i in range(0, len(internal_information_of_model.t_evaluation), 1):
        print(i)
        simulation_class.update_pressure_plot(internal_information_of_model.t_evaluation[i], pdic['interpolated_data'](internal_information_of_model.t_evaluation[i]))