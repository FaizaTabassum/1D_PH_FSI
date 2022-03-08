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
import pickle

if __name__ == '__main__':
    path_to_image =r'C:\Users\Faiza\Desktop\1D_PH_FSI\cardiac_cycle_images'
    simulation_time=3
    scale = 0.00750062 #1 for Pa or 0.00750062 if you want to plot pressure in mmHg
    pressure_title="pressure in [mmgH]" #change title accordingly to scale
    input_definition = func.InputDefinitions()
    save_data_dic = {'radius': input_definition.tube_base_radius, 'tube_length': input_definition.tube_length,
                     'number_sections': input_definition.number_sections}
    with open(input_definition.save_data_path + "\output.npy", 'wb') as f:
        pickle.dump(save_data_dic, f)
    parameter_dic, internal_information_of_model, results_integration = func.run_simulation(input_definition.tube_base_radius, input_definition.tube_length, int(input_definition.number_sections), path_to_image,simulation_time = simulation_time,path_to_input=input_definition.flow_profile_path,path_to_save=input_definition.save_data_path, scale = scale, pressure_title=pressure_title)
    simulation_class = func.VisualizingResults(parameter_dic)
    for i in range(0, len(internal_information_of_model.t_evaluation), 1):
        simulation_class.update_pressure_plot(internal_information_of_model.t_evaluation[i])
    plt.show()