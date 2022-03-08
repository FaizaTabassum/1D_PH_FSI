import os
import sys
import math
import numpy as np
import sv
sys.path.append("C:/Users/Faiza/Desktop/stenosis")
import helper_functions
sys.path.pop()


model_name = "r1l8g2f12"
path = r"C:\Users\Faiza\Desktop\1D_PH_FSI\results"
geometric_data = np.load(path+'\output.npy', allow_pickle=True)
radius = geometric_data['radius']
tube_length = geometric_data['tube_length']
number_sections = geometric_data['number_sections']
x = np.linspace(0, tube_length, number_sections)
path_name = model_name + "_path"
segmentations_name = model_name + "_segmentations"
############################################################
############################################################
############################################################
############################################################
############################################################


path_points_array = np.zeros((len(x), 3))
path_points_array[:, 2] = x # make tube along z-axis
path_points_list = path_points_array.tolist()

# make radii list
radii_list = radius.tolist()

if len(radii_list) != len(path_points_list):
    print("Error. Number of points in radius list does not match number of points in path list.")

# create path and segmnetation objects
path = helper_functions.create_path_from_points_list(path_points_list)
segmentations = helper_functions.create_segmentations_from_path_and_radii_list(path, radii_list)
#
# # add path and segmentations to the SimVascular Data Manager (SV DMG) for visualization in GUI
sv.dmg.add_path(name = path_name, path = path)
sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)
