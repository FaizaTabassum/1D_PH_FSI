import os
import sys
import math
import numpy as np
import sv
sys.path.append("C:/Users/Faiza/Desktop/stenosis")
import helper_functions
sys.path.pop()

# Source: adapted from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py

############################################################
############################################################
################## CHANGE THESE ENTRIES ####################
############################################################
############################################################
model_name = "model01"
x0 = -1.5
xf = 1.5
nx = 31 # make this an odd number, so that we will have a segmentation at the midpoint
# make path points list
x = np.linspace(x0, xf, nx)

distance = 0 # for removal of segmentations surrounding the midsection
r_inlet = 0.3
expansion_percentage = 100 # midsection area as a percentage of the inlet area


A_inlet = np.pi*(r_inlet)**2
A_outlet = (expansion_percentage/100)*A_inlet
r_outlet = np.sqrt(A_outlet/np.pi)
m = (r_outlet - r_inlet)/(xf-x0)
r = m*x +r_inlet




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
radii_list = r.tolist()

if len(radii_list) != len(path_points_list):
    print("Error. Number of points in radius list does not match number of points in path list.")
#
# # create path and segmnetation objects
path = helper_functions.create_path_from_points_list(path_points_list)
segmentations = helper_functions.create_segmentations_from_path_and_radii_list(path, radii_list)
#
# # add path and segmentations to the SimVascular Data Manager (SV DMG) for visualization in GUI
sv.dmg.add_path(name = path_name, path = path)
sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)
