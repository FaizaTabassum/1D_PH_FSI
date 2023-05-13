import os
import sys
import math
import numpy as np
import sv
sys.path.append("C:/Users/Faiza/Desktop/stenosis")
import helper_functions
sys.path.pop()

# Source: adapted from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py

def radius(radius_inlet, x, A, sigma, mu):
    # Reference: Sun, L., Gao, H., Pan, S. & Wang, J. X. Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data. Computer Methods in Applied Mechanics and Engineering 361, (2020).

    # controls severity of stenosis/aneurysm; positive for stenosis; negative for aneurysm
    return radius_inlet - A/np.sqrt(2.0*np.pi*sigma**0.5)*np.exp(-1.0*((x - mu)**2/2/sigma**0.5))

############################################################
############################################################
################## CHANGE THESE ENTRIES ####################
############################################################
############################################################
model_name = "model01"
x0 = -1.50
xf = 1.50
nx = 31 # make this an odd number, so that we will have a segmentation at the midpoint
distance = 0 # for removal of segmentations surrounding the midsection

midsection_percentage = 50 # midsection area as a percentage of the inlet area
radius_inlet = 0.3

sigma = 0.010 # controls the spread of the stenosis/aneurysm
mu = 0.0 # controls the center of the stenosis/aneurysm
path_name = model_name + "_path"
segmentations_name = model_name + "_segmentations"
############################################################
############################################################
############################################################
############################################################
############################################################

# make path points list
x = np.linspace(x0, xf, nx)
mid_index = math.floor(len(x)/2.0)
x = np.concatenate((x[0:mid_index - distance], np.array([x[mid_index]]), x[mid_index + 1 + distance:]))
path_points_array = np.zeros((len(x), 3))
path_points_array[:, 2] = x # make tube along z-axis
path_points_list = path_points_array.tolist()

# make radii list
area_midsection = (midsection_percentage/100.0)*np.pi*radius_inlet**2.0
radius_midsection = np.sqrt(area_midsection/np.pi)
A = -1.0*(radius_midsection - radius_inlet)*np.sqrt(2.0*np.pi*sigma**0.5)/np.exp(-1.0*((0.0 - mu)**2/2/sigma**0.5)) # A = 4.856674471372556
print("A = ", A)
radius_array = radius(radius_inlet, x, A, sigma, mu)
radii_list = radius_array.tolist()

if len(radii_list) != len(path_points_list):
    print("Error. Number of points in radius list does not match number of points in path list.")
#
# create path and segmnetation objects
path = helper_functions.create_path_from_points_list(path_points_list)
segmentations = helper_functions.create_segmentations_from_path_and_radii_list(path, radii_list)
#
# # add path and segmentations to the SimVascular Data Manager (SV DMG) for visualization in GUI
sv.dmg.add_path(name = path_name, path = path)
sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)
