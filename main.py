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
    simulation_time=3 #please define time of one heart beat
    pressure_outlet =0 #please define pressure at outlet
    scale = 0.00750062 #1 for Pa or 0.00750062 if you want to plot pressure in mmHg
    pressure_title="Druck [mmHg]" #change title accordingly to scale



    input_definition = func.ChooseCase().input_definitions
    plt.figure()
    plt.plot(input_definition.tube_base_radius)
    plt.show()
    # save excel file with radius
    # df = pd.DataFrame(input_definition.geometry[0])
    # filepath = 'geometry_z_coordinates.xlsx'
    # df.to_excel(filepath, index=False)
    # df = pd.DataFrame(input_definition.geometry[1])
    # filepath = 'geometry_x_coordinates.xlsx'
    # df.to_excel(filepath, index=False)
    # df = pd.DataFrame(input_definition.geometry[2])
    # filepath = 'geometry_y_coordinates.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # save_data_dic = {'radius': input_definition.tube_base_radius, 'tube_base_radius': input_definition.tube_base_radius_val, 'tube_length': input_definition.tube_length,
    #                  'number_sections': input_definition.number_sections, 'stenosis_radius_proportion': input_definition.stenosis_radius_proportion, 'stenosis_expansion': input_definition.stenosis_expansion, 'stenosis_position': input_definition.stenosis_position}
    # # with open(input_definition.save_data_path + "\output.npy", 'wb') as f:
    #     pickle.dump(save_data_dic, f)

    parameter_dic, internal_information_of_model, results_integration = func.run_simulation(input_definition.tube_base_radius, input_definition.tube_length, int(input_definition.number_sections), path_to_image,simulation_time = simulation_time,path_to_input=input_definition.flow_profile_path,path_to_save=input_definition.save_data_path, scale = scale, pressure_title=pressure_title, pressure_at_outlet=pressure_outlet)

    stop = True
    plt.figure()
    plt.plot(np.linspace(0, parameter_dic['structure_length'], parameter_dic['N_sec']), parameter_dic['interpolated_data'](parameter_dic['t_evaluation'][-1]))
    plt.show()
