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
    path_to_image =[r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\heart_front', r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\heart_side', r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images'] #please change the path to Images\heart_front, Images_heart_side and Images
    simulation_time=1 #please define time of one heart beat
    pressure_outlet =1000 #please define pressure at outlet
    scale = 0.00750062 #1 for Pa or 0.00750062 if you want to plot pressure in mmHg
    pressure_title="Druck [mmHg]" #change title accordingly to scale

    input_definition = func.InputDefinitions()
    save_data_dic = {'radius': input_definition.tube_base_radius, 'tube_base_radius': input_definition.tube_base_radius_val, 'tube_length': input_definition.tube_length,
                     'number_sections': input_definition.number_sections, 'stenosis_radius_proportion': input_definition.stenosis_radius_proportion, 'stenosis_expansion': input_definition.stenosis_expansion, 'stenosis_position': input_definition.stenosis_position}
    with open(input_definition.save_data_path + "\output.npy", 'wb') as f:
        pickle.dump(save_data_dic, f)

    parameter_dic = func.get_parameter(input_definition.tube_base_radius, input_definition.tube_length,
                                       int(input_definition.number_sections), path_to_image,
                                       simulation_time=simulation_time,
                                       path_to_input=input_definition.flow_profile_path,
                                       path_to_save=input_definition.save_data_path, scale=scale,
                                       pressure_title=pressure_title, pressure_at_outlet=pressure_outlet)

    with open("pressure.pkl", 'rb') as f:
        pressure = pickle.load(f)
    with open("pressure1.pkl", 'rb') as f:
        pressure1 = pickle.load(f)
    with open("pressure2.pkl", 'rb') as f:
        pressure2 = pickle.load(f)
    with open("radius.pkl", 'rb') as f:
        radius = pickle.load(f)
    with open("radius1.pkl", 'rb') as f:
        radius1 = pickle.load(f)
    with open("radius2.pkl", 'rb') as f:
        radius2 = pickle.load(f)
    with open("min_pressure.txt", 'r') as f:
        min_pressure = f.readlines()[0]
    with open("max_pressure.txt", 'r') as f:
        max_pressure = f.readlines()[0]

    simulation_class = func.VisualizingExtendedResults(parameter_dic, pressure, pressure1, pressure2,radius, radius1, radius2, min_pressure,max_pressure)
    simulation_class.update_pressure_plot(20)

