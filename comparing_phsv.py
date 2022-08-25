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

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_straight.npy','rb')
straight = pickle.load(file)
radius_straight = straight[3,:]
z_straight = straight[0, :]
z_straight_ = np.linspace(z_straight[0], z_straight[-1], len(z_straight))
pressure_straight = straight[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_straight_extracted.npy','rb')
straight_sv = pickle.load(file)
z_straight_sv= straight_sv[0, :]
pressure_straight_sv = straight_sv[1,:]*0.1

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_one_three.npy','rb')
data = pickle.load(file)
radius_one = data[3,:]
z_one = data[0, :]
z_one_ = np.linspace(z_one[0], z_one[-1], len(z_one))
pressure_one = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_one_three_server.npy','rb')
data_sv = pickle.load(file)
z_one_sv= data_sv[0, :]
pressure_one_sv = data_sv[1,:]*0.1

# file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_three_three.npy','rb')
# data = pickle.load(file)
# radius_three = data[3,:]
# z_three = data[0, :]
# z_three_ = np.linspace(z_three[0], z_three[-1], len(z_three))
# pressure_three = data[1,:]
#
# file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_three_three_server.npy','rb')
# data_sv = pickle.load(file)
# z_three_sv= data_sv[0, :]
# pressure_three_sv = data_sv[1,:]*0.1


file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_five_three.npy','rb')
data = pickle.load(file)
radius_five = data[3,:]
z_five = data[0, :]
z_five_ = np.linspace(z_five[0], z_five[-1], len(z_five))
pressure_five = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_five_three_server.npy','rb')
data_sv = pickle.load(file)
z_five_sv= data_sv[0, :]
pressure_five_sv = data_sv[1,:]*0.1




file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusone_three.npy','rb')
data = pickle.load(file)
radius_minusone = data[3,:]
z_minusone = data[0, :]
z_minusone_ = np.linspace(z_minusone[0], z_minusone[-1], len(z_minusone))
pressure_minusone = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusone_three_v1.npy','rb')
data = pickle.load(file)
radius_minusone1 = data[3,:]
z_minusone1 = data[0, :]
z_minusone1_ = np.linspace(z_minusone1[0], z_minusone1[-1], len(z_minusone1))
pressure_minusone1 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusone_three_v2.npy','rb')
data = pickle.load(file)
radius_minusone2 = data[3,:]
z_minusone2 = data[0, :]
z_minusone2_ = np.linspace(z_minusone2[0], z_minusone2[-1], len(z_minusone2))
pressure_minusone2 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusone_three_v3.npy','rb')
data = pickle.load(file)
radius_minusone3 = data[3,:]
z_minusone3 = data[0, :]
z_minusone3_ = np.linspace(z_minusone3[0], z_minusone3[-1], len(z_minusone3))
pressure_minusone3 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_minusone_three_server.npy','rb')
data_sv = pickle.load(file)
z_minusone_sv= data_sv[0, :]
pressure_minusone_sv = data_sv[1,:]*0.1

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusthree_three.npy','rb')
data = pickle.load(file)
radius_minusthree = data[3,:]
z_minusthree = data[0, :]
z_minusthree_ = np.linspace(z_minusthree[0], z_minusthree[-1], len(z_minusthree))
pressure_minusthree = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusthree_three_v1.npy','rb')
data = pickle.load(file)
radius_minusthree1 = data[3,:]
z_minusthree1 = data[0, :]
z_minusthree1_ = np.linspace(z_minusthree1[0], z_minusthree1[-1], len(z_minusthree1))
pressure_minusthree1 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusthree_three_v2.npy','rb')
data = pickle.load(file)
radius_minusthree2 = data[3,:]
z_minusthree2 = data[0, :]
z_minusthree2_ = np.linspace(z_minusthree2[0], z_minusthree2[-1], len(z_minusthree2))
pressure_minusthree2 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusthree_three_v3.npy','rb')
data = pickle.load(file)
radius_minusthree3 = data[3,:]
z_minusthree3 = data[0, :]
z_minusthree3_ = np.linspace(z_minusthree3[0], z_minusthree3[-1], len(z_minusthree3))
pressure_minusthree3 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_minusthree_three_server.npy','rb')
data_sv = pickle.load(file)
z_minusthree_sv= data_sv[0, :]
pressure_minusthree_sv = data_sv[1,:]*0.1

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusten_three.npy','rb')
data = pickle.load(file)
radius_minusten = data[3,:]
z_minusten = data[0, :]
z_minusten_ = np.linspace(z_minusten[0], z_minusten[-1], len(z_minusten))
pressure_minusten= data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusten_three_v1.npy','rb')
data = pickle.load(file)
radius_minusten1 = data[3,:]
z_minusten1 = data[0, :]
z_minusten1_ = np.linspace(z_minusten1[0], z_minusten1[-1], len(z_minusten1))
pressure_minusten1= data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusten_three_v2.npy','rb')
data = pickle.load(file)
radius_minusten2 = data[3,:]
z_minusten2 = data[0, :]
z_minusten2_ = np.linspace(z_minusten2[0], z_minusten2[-1], len(z_minusten2))
pressure_minusten2= data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusten_three_v3.npy','rb')
data = pickle.load(file)
radius_minusten3 = data[3,:]
z_minusten3 = data[0, :]
z_minusten3_ = np.linspace(z_minusten3[0], z_minusten3[-1], len(z_minusten3))
pressure_minusten3= data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_minusten_three_server.npy','rb')
data_sv = pickle.load(file)
z_minusten_sv= data_sv[0, :]
pressure_minusten_sv = data_sv[1,:]*0.1


file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusfive_three.npy','rb')
data = pickle.load(file)
radius_minusfive = data[3,:]
z_minusfive = data[0, :]
z_minusfive_ = np.linspace(z_minusfive[0], z_minusfive[-1], len(z_minusfive))
pressure_minusfive = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusfive_three_v1.npy','rb')
data = pickle.load(file)
radius_minusfive1 = data[3,:]
z_minusfive1 = data[0, :]
z_minusfive1_ = np.linspace(z_minusfive1[0], z_minusfive1[-1], len(z_minusfive1))
pressure_minusfive1 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusfive_three_v2.npy','rb')
data = pickle.load(file)
radius_minusfive2 = data[3,:]
z_minusfive2 = data[0, :]
z_minusfive2_ = np.linspace(z_minusfive2[0], z_minusfive2[-1], len(z_minusfive2))
pressure_minusfive2 = data[1,:]

file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\results_twenty_two_minusfive_three_v3.npy','rb')
data = pickle.load(file)
radius_minusfive3 = data[3,:]
z_minusfive3 = data[0, :]
z_minusfive3_ = np.linspace(z_minusfive3[0], z_minusfive3[-1], len(z_minusfive3))
pressure_minusfive3 = data[1,:]


file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\results\sv_twenty_two_minusfive_three_server.npy','rb')
data_sv = pickle.load(file)
z_minusfive_sv= data_sv[0, :]
pressure_minusfive_sv = data_sv[1,:]*0.1

# file = open(r'C:\Users\Faiza\Desktop\1D_PH_FSI\simvascular\Output_100Resistance_server\exp_20_2_m5_3_1Dmodel\ROMSimulations\1D_model\1D_model-converted-results_1d\1D_model_pressure.csv','rb')
# pressure_one_sv_1D = np.loadtxt(file, delimiter=",", skiprows=1)[-1,1:]*-0.1
# z_one_sv_1D = np.linspace(0, 20, len(pressure_one_sv_1D))
# plt.figure()
# plt.plot(z_one_sv_1D, pressure_one_sv_1D)
# plt.ylabel('pressure [Pa]')
# plt.xlabel('axial tube position [cm]')
# z_five_sv= data_sv[0, :]
# pressure_five_sv = data_sv[1,:]*0.1



plt.figure()
plt.title('angle minus five degree')
plt.plot(z_minusfive1, pressure_minusfive1, label = 'v1')
plt.plot(z_minusfive2, pressure_minusfive2, label = 'v2')
plt.plot(z_minusfive3, pressure_minusfive3, label = 'v3')
plt.plot(z_minusfive_sv, pressure_minusfive_sv, label = 'sv')
plt.ylabel('pressure [Pa]')
plt.xlabel('tube axial position [cm]')
plt.legend()

plt.figure()
plt.title('angle minus three degree')
plt.plot(z_minusthree1, pressure_minusthree1, label = 'v1')
plt.plot(z_minusthree2, pressure_minusthree2, label = 'v2')
plt.plot(z_minusthree3, pressure_minusthree3, label = 'v3')
plt.plot(z_minusthree_sv, pressure_minusthree_sv, label = 'sv')
plt.ylabel('pressure [Pa]')
plt.xlabel('tube axial position [cm]')
plt.legend()

plt.figure()
plt.title('angle minus ten degree')
plt.plot(z_minusten1, pressure_minusten1, label = 'v1')
plt.plot(z_minusten2, pressure_minusten2, label = 'v2')
plt.plot(z_minusten3, pressure_minusten3, label = 'v3')
plt.plot(z_minusten_sv, pressure_minusten_sv, label = 'sv')
plt.ylabel('pressure [Pa]')
plt.xlabel('tube axial position [cm]')
plt.legend()

plt.figure()
plt.title('angle minus one degree')
plt.plot(z_minusone1, pressure_minusone1, label = 'v1')
plt.plot(z_minusone2, pressure_minusone2, label = 'v2')
plt.plot(z_minusone3, pressure_minusone3, label = 'v3')
plt.plot(z_minusone_sv, pressure_minusone_sv, label = 'sv')
plt.ylabel('pressure [Pa]')
plt.xlabel('tube axial position [cm]')
plt.legend()

plt.figure()



# axs0twin = axs[0].twinx()

# axs0twin.set_ylabel('radius [cm]')

# axs0twin.plot(z_straight, radius_straight, color = 'k')
plt.plot(z_straight_, pressure_straight, color = 'k', label = 'ph straight', alpha = 0.5, linestyle = '-')
# axs[1].plot(z_one, pressure_one, color = 'b', label = 'ph one degree', alpha = 1, linestyle = '--')
# axs[1].plot(z_five, pressure_five, color = 'b', label = 'ph five degree', alpha = 1, linestyle = '-.')
plt.plot(z_minusone3, pressure_minusone3, color = 'b', label = 'ph minus one degree', alpha = 1, linestyle = ':')
plt.plot(z_minusthree3, pressure_minusthree3, color = 'b', label = 'ph minus three degree', alpha = 1, linestyle = '-')
plt.plot(z_minusfive3, pressure_minusfive3, color = 'b', label = 'ph minus five degree', alpha = 1, linestyle = '-.')
plt.plot(z_minusten3, pressure_minusten3, color = 'b', label = 'ph minus ten degree', alpha = 1, linestyle = '--')

# axs[0].set_xticks([])

# ax1twin = axs[1].twinx()
plt.ylabel('pressure [Pa]')
# ax1twin.set_ylabel('radius [cm]')
# ax1twin.plot(z_tto, radius_tto, color = 'k')
plt.plot(z_straight_sv, pressure_straight_sv, color = 'k', label = 'sv straight', alpha = 0.5, linestyle = ':')
# axs[1].plot(z_one_sv, pressure_one_sv, color = 'y', label = 'sv one degree', alpha = 1, linestyle = '--')
# axs[1].plot(z_five_sv, pressure_five_sv, color = 'y', label = 'sv five degree', alpha = 1, linestyle = '-.')

plt.plot(z_minusone_sv, pressure_minusone_sv, color = 'y', label = 'sv minus one degree', alpha = 1, linestyle = ':')
plt.plot(z_minusthree_sv, pressure_minusthree_sv, color = 'y', label = 'sv minus three degree', alpha = 1, linestyle = '-')
plt.plot(z_minusfive_sv, pressure_minusfive_sv, color = 'y', label = 'sv minus five degree', alpha = 1, linestyle = '-.')
plt.plot(z_minusten_sv, pressure_minusten_sv, color = 'y', label = 'sv minus teb degree', alpha = 1, linestyle = '--')

# axs[1].set_xticks([])
plt.legend()
plt.legend()
plt.xlabel('axial tube position [cm]')




plt.figure()
plt.plot(z_minusone, radius_minusone, color = 'b', label = 'ph minus one degree', alpha = 1, linestyle = ':')
plt.plot(z_minusone, -radius_minusone, color = 'b', label = 'ph minus one degree', alpha = 1, linestyle = ':')

plt.plot(z_minusthree, radius_minusthree, color = 'g', label = 'ph minus three degree', alpha = 1, linestyle = '-')
plt.plot(z_minusthree, -radius_minusthree, color = 'g', label = 'ph minus three degree', alpha = 1, linestyle = '-')

plt.plot(z_minusfive, radius_minusfive, color = 'r', label = 'ph minus five degree', alpha = 1, linestyle = '-.')
plt.plot(z_minusfive, -radius_minusfive, color = 'r', label = 'ph minus five degree', alpha = 1, linestyle = '-.')

plt.plot(z_minusten, radius_minusten, color = 'k', label = 'ph minus ten degree', alpha = 1, linestyle = '--')
plt.plot(z_minusten, -radius_minusten, color = 'k', label = 'ph minus ten degree', alpha = 1, linestyle = '--')
plt.ylabel('geometry [cm]')
plt.xlabel('axial tube position [cm]')
plt.legend()
plt.show()





# ax2twin = axs[2].twinx()
# axs[2].set_ylabel('pressure [Pa]')
# ax2twin.set_ylabel('radius [cm]')
# ax2twin.plot(z_ttf, radius_ttf, color = 'k')
# ax2twin.set_ylim(0, 1.6)
# axs[2].set_ylim(-22, 22)
# axs[2].plot(z_ttf_, pressure_ttf, color = 'r', label = 'ph five', alpha = 0.5, linestyle = '-')
# axs[2].plot(z_ttf_sv, pressure_ttf_sv, color = 'r', label = 'sv five', alpha = 1, linestyle = '-.')
# axs[2].set_xticks([])
#
# ax3twin = axs[3].twinx()
# axs[3].set_ylabel('pressure [Pa]')
# ax3twin.set_ylabel('radius [cm]')
# ax3twin.plot(z_ttt, radius_ttt, color = 'k')
# ax3twin.set_ylim(0, 1.6)
# axs[3].set_ylim(-22, 22)
# axs[3].plot(z_ttt_, pressure_ttt, color = 'g', label = 'ph ten', alpha = 0.5, linestyle = '-')
# axs[3].plot(z_ttt_sv, pressure_ttt_sv, color = 'g', label = 'sv ten', alpha = 1, linestyle = '-.')



