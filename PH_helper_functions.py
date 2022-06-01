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
from matplotlib.patches import Ellipse
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import sympy as sym


class BoundaryDefinitions:
    def __init__(self, path_load, out_max):
        self.path_load = path_load
        self.out_max = out_max
        self.scale = 1*10**-6

    @property
    def inp_const(self):
        inp_max = 100 * self.scale
        tinp = 3*10**-3
        return lambda t: (inp_max / 2) * (1 - np.cos(np.pi * t / tinp)) if t < tinp else inp_max

    @property
    def load_input_file(self):
        if ".xlsx" in self.path_load:
            data = pd.read_excel(self.path_load)
            data = data.to_numpy()
        if ".npy" in self.path_load:
            with open(self.path_load, 'rb') as f:
                data = np.load(f)
            data = data[()]
        Q_input = np.array([[cont[0], cont[1]*self.scale] for cont in data if math.isnan(cont[0])==False])
        Q_i = interp1d(Q_input[:, 0], Q_input[:, 1])
        upper_bound = np.max(Q_input[:, 0])
        def f(x):
            if x <= upper_bound:
                return Q_i(x)
            return 0
        return f

    def repeat_input(self, input_time_on):
        if ".xlsx" in self.path_load:
            data = pd.read_excel(self.path_load)
            data = data.to_numpy()
        if ".npy" in self.path_load:
            with open(self.path_load, 'rb') as f:
                data = np.load(f)
            data = data[()]
        if data[-1, 0] >= input_time_on:

            Q_input = np.array([[cont[0], cont[1] * self.scale] for cont in data if math.isnan(cont[0]) == False])
            interpolate_start = interp1d([Q_input[0, 0], Q_input[2, 0], Q_input[4, 0], Q_input[6, 0], Q_input[8, 0], Q_input[10, 0]],
                                         [Q_input[0, 1], Q_input[2, 1], Q_input[4, 1], Q_input[6, 1], Q_input[8, 1], Q_input[10, 1]],
                                         kind=2)
            Q_first =interpolate_start(Q_input[0:10, 0])
            Q_input[0:10, 1] = Q_first
            Q_i = interp1d(Q_input[:, 0], Q_input[:, 1], kind=2)

            upper_bound = np.max(Q_input[:, 0])
            def f(x):
                if x <= upper_bound:
                    return Q_i(x)
                return 0

            return f
        else:
            data = np.array([[cont[0], cont[1] * self.scale] for cont in data if math.isnan(cont[0]) == False])
            sample_rate = data[-1, 0] - data[-2, 0]
            interpolate_start = interp1d(
                [data[0, 0], data[2, 0], data[4, 0], data[6, 0], data[8, 0], data[10, 0]],
                [data[0, 1], data[2, 1], data[4, 1], data[6, 1], data[8, 1], data[10, 1]],
                kind=2)
            data_ = interpolate_start(data[0:10, 0])
            data[0:10, 1] = data_
            time_evaluation_new = np.arange(0, input_time_on+sample_rate, sample_rate)
            repeat_samples = int(len(time_evaluation_new) / data.shape[0])
            input_to_be_repeated = np.reshape(data[:, 1], (1, -1))
            input_repeated = np.repeat(input_to_be_repeated, repeat_samples + 1, axis=0)
            input_repeated = np.reshape(input_repeated, (input_repeated.shape[0] * input_repeated.shape[1],))
            input_repeated =np.hstack((data[0,1], input_repeated))
            input_repeated = input_repeated[0:len(time_evaluation_new)]
            final_input = interp1d(time_evaluation_new,input_repeated)
            upper_bound = np.max(time_evaluation_new)
            def f(x):
                if x <= upper_bound:
                    return final_input(x)
                return 0
            return f

    @property
    def out_const(self):
        out = lambda t: self.out_max
        return out

def get_parameter(radius, tube_length, number_sections, path_to_images,path_to_input, path_to_save, demo = False, simulation_time=1, min_pressure = 0, max_pressure = 0, scale = 1, pressure_title='', pressure_at_outlet=0):
    N_sec = number_sections
    N_nodes = number_sections
    simulation_time = simulation_time
    sample_rate = 1e-03
    path_input_at_inlet = path_to_input
    boundary_definition = BoundaryDefinitions(path_input_at_inlet, pressure_at_outlet)

    t_evaluation = np.ndarray.tolist(np.arange(0, simulation_time, sample_rate))
    # 'inp_entrance': boundary_definition.repeat_input(simulation_time),
    parameter = {
        'min_pressure': min_pressure,
        'max_pressure': max_pressure,
        'geo_dissipation': True,
        'geo_factor': 2.6,
        'vis_factor': 16,
        'vis_dissipation': True,
        'inp_entrance': boundary_definition.input_const,
        'inp_exit': boundary_definition.out_const,
        'structure_length': tube_length * 10 ** -2,
        'radius': radius*10 ** -2,
        'wall_thickness': 3 * 10 ** -4,
        'N_sec': N_sec,
        'N_ypoints_for_pressure_image': N_sec,
        'N_nodes': N_nodes,
        't_evaluation': t_evaluation,
        'sample_time_visualization':40,
        'fluid_density': 1.06 * 10 ** 3,
        'viscosity': 0.004,
        'structure_density': 1.1 * 10 ** 3,
        'poi_rat': 0.4,
        'E_mod': 3 * 10 ** 5,
        'structure_r_dissipation':0,
        'structure_z_dissipation': 0,
        'fluid_bulk_modulus': 2.15 * 10 ** 9,
        'external_force': np.zeros(N_sec),
        'stiffness_k1': 1*10**10,
        'stiffness_k2': -20,
        'stiffness_k3': 1*10**9,
        'requested_data_over_tube': ['fluid_velocity', 'dynamic_pressure', 'static_pressure'],
        'requested_data_at_boundaries': ['pressure', 'flow'],
        'path_to_image': path_to_images,
        'scale': scale,
        'pressure_title':pressure_title
    }
    return parameter


@profile
def run_simulation(radius, tube_length, number_sections, path_to_images,path_to_input, path_to_save, demo = False, simulation_time=1, min_pressure = 0, max_pressure = 0, scale = 1, pressure_title='', pressure_at_outlet=0):
    N_sec = number_sections
    N_nodes = number_sections
    simulation_time = simulation_time
    sample_rate = 1e-03
    path_input_at_inlet = path_to_input
    boundary_definition = BoundaryDefinitions(path_input_at_inlet, pressure_at_outlet)

    t_evaluation = np.ndarray.tolist(np.arange(0, simulation_time, sample_rate))
    parameter = {
        'min_pressure': min_pressure,
        'max_pressure': max_pressure,
        'geo_dissipation': True,
        'geo_factor': 2.6,
        'vis_factor': 16,
        'vis_dissipation': True,
        'inp_entrance': boundary_definition.inp_const,
        'inp_exit': boundary_definition.out_const,
        'structure_length': tube_length * 10 ** -2,
        'radius': radius*10 ** -2,
        'wall_thickness': 3 * 10 ** -4,
        'N_sec': N_sec,
        'N_ypoints_for_pressure_image': N_sec,
        'N_nodes': N_nodes,
        't_evaluation': t_evaluation,
        'sample_time_visualization':40,
        'fluid_density': 1.06 * 10 ** 3,
        'viscosity': 0.004,
        'structure_density': 1.1 * 10 ** 3,
        'poi_rat': 0.4,
        'E_mod': 3 * 10 ** 5,
        'structure_r_dissipation':0,
        'structure_z_dissipation': 0,
        'fluid_bulk_modulus': 2.15 * 10 ** 9,
        'external_force': np.zeros(N_sec),
        'stiffness_k1': 1*10**10,
        'stiffness_k2': -20,
        'stiffness_k3': 1*10**9,
        'requested_data_over_tube': ['fluid_velocity', 'dynamic_pressure', 'static_pressure'],
        'requested_data_at_boundaries': ['pressure', 'flow'],
        'path_to_image': path_to_images,
        'scale': scale,
        'pressure_title':pressure_title
    }

    PHFSI = OneD_PHM(parameter, demo = demo) #this is for flow, if you want to test Pressure, you need to substitute Flow by Pressure
    T = [0, simulation_time]
    x_init = np.concatenate((np.zeros(3*N_sec), parameter['fluid_density']*np.ones(N_nodes))).reshape(-1,)
    sol = solve_ivp(
        fun=lambda t, x0: PHFSI.dynamic_model(t, x0),
        obj=PHFSI,
        t_span=[0, simulation_time],
        t_eval=t_evaluation,
        min_step=sample_rate,
        max_step=sample_rate,
        y0=x_init,
        first_step=None,
        hmax=sample_rate,
        hmin=sample_rate,
        rtol=10 ** (-3),
        atol=10 ** (-6),
        dense_output=False,
        method='BDF',
        vectorized=False
    )
    parameter['min_pressure'] = np.min(PHFSI.total_pressure[:, 0])
    parameter['max_pressure'] = np.max(PHFSI.total_pressure[:, 0])
    total_pressure = PHFSI.total_pressure
    step_time = np.reshape(PHFSI.step_time, (-1,))
    parameter['interpolated_data'] = interp1d(step_time, total_pressure, axis=0)
    parameter['velocity'] = interp1d(step_time, PHFSI.velocity, axis=0)
    return parameter, PHFSI, sol

class InputDefinitions:
    def __init__(self):
        ICON_PLAY = plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\play.png')
        ICON_LOAD = plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\load.png')
        ICON_SAVE = plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\save.png')
        self.filename = ''
        self.fig = plt.figure(constrained_layout = True)
        self.fig.subplots_adjust(bottom=-0.0, top=0.6)
        self.ax_nx = self.fig.add_axes([0.2, 0.84, 0.2, 0.05])
        self.ax_nx.spines['top'].set_visible(True)
        self.ax_nx.spines['right'].set_visible(True)
        self.ax_r = self.fig .add_axes([0.2, 0.78, 0.2, 0.05])
        self.ax_r.spines['top'].set_visible(True)
        self.ax_r.spines['right'].set_visible(True)
        self.ax_l = self.fig .add_axes([0.2, 0.72, 0.2, 0.05])
        self.ax_l.spines['top'].set_visible(True)
        self.ax_l.spines['right'].set_visible(True)
        self.ax_s = self.fig .add_axes([0.2, 0.66, 0.2, 0.05])
        self.ax_s.spines['top'].set_visible(True)
        self.ax_s.spines['right'].set_visible(True)
        self.ax_d = self.fig .add_axes([0.2, 0.60, 0.2, 0.05])
        self.ax_d.spines['top'].set_visible(True)
        self.ax_d.spines['right'].set_visible(True)
        self.ax_w = self.fig .add_axes([0.2, 0.54, 0.2, 0.05])
        self.ax_w.spines['top'].set_visible(True)
        self.ax_w.spines['right'].set_visible(True)
        self.ax_start_button = self.fig.add_axes([0.95, 0.01, 0.05 ,0.05])
        self.ax_save_button = self.fig.add_axes([0.9, 0.01, 0.05, 0.05])
        self.ax_load_button = self.fig.add_axes([0.85, 0.01, 0.05, 0.05])
        self.ax = self.fig.add_axes([0, 0, 0.5, 0.5], projection= '3d')
        self.ax1 = self.fig.add_axes([0.45, 0.5, 0.5, 0.45])
        self.ax2= self.fig.add_axes([0.5, 0.05, 0.4, 0.4])
        self.number_sections = Slider(ax=self.ax_nx, label='#sections ', valmin=10, valmax=200, valinit=35,
                                 valfmt='%d', facecolor='#cc7000', valstep=1)
        self.tube_base_radius = Slider(ax=self.ax_r, label='radius ', valmin=0, valmax=5.0, valinit=1.0,
                                  valfmt=' %1.1f cm', facecolor='#cc7000')
        self.tube_length = Slider(ax=self.ax_l, label='length ', valmin=0, valmax=10,
                             valinit=10, valfmt='%1.1f cm', facecolor='#cc7000')
        self.stenosis_position = Slider(ax=self.ax_s, label='stenosis position ', valmin=-5, valmax=5,
                                   valinit=0, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.stenosis_radius_proportion = Slider(ax=self.ax_d, label='stenosis radius proportion ', valmin=-0.99, valmax=0.99,
                                            valinit=0, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.stenosis_expansion = Slider(ax=self.ax_w, label='stenosis spread ', valmin=0.000001, valmax=5,
                                    valinit=1, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.play_button = Button(ax=self.ax_start_button, label = '', image=ICON_PLAY)
        self.load_button = Button(ax=self.ax_load_button, label="", image = ICON_LOAD)
        self.save_button = Button(ax=self.ax_save_button, label="", image=ICON_SAVE)

        self.x = np.linspace(-self.tube_length.val / 2, self.tube_length.val / 2, self.number_sections.val)

        self.radius = np.ones(self.number_sections.val)*self.tube_base_radius.val
        self.data_for_structure_along_z()
        self.ax.plot_surface(self.geometry[0], self.geometry[1], self.geometry[2],alpha=1)
        self.ax.set_box_aspect((np.ptp(self.geometry[0]), np.ptp(self.geometry[1]), np.ptp(self.geometry[2])))
        self.ax1.imshow(plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\formula1.png'))
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax2.imshow(plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\Aortenatresie.jpeg'))
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel('tube length [cm]')
        self.number_sections.on_changed(self.stenosis)
        self.tube_base_radius.on_changed(self.stenosis)
        self.tube_length.on_changed(self.stenosis)

        self.stenosis_position.on_changed(self.stenosis)

        self.stenosis_radius_proportion.on_changed(self.stenosis)

        self.stenosis_expansion.on_changed(self.stenosis)
        self.play_button.on_clicked(self.start_program)
        self.load_button.on_clicked(self.load_flow_input_file)
        self.save_button.on_clicked(self.save_data)
        plt.show()

    def tube(self, r, l, N):
        x = np.linspace(-l / 2, l / 2, N)
        return 0 * x + r

    def load_flow_input_file(self, val):
        self.flow_profile_path = fd.askopenfilename()

    def save_data(self, val):
        self.save_data_path = fd.askdirectory()

    def handle_close(evt):
        print('Closed Figure!')

    def start_program(self, val):
        self.tube_base_radius_val = copy.copy(self.tube_base_radius.val)
        self.tube_base_radius, self.position, self.window, self.geometry = self.stenosis(2)
        self.tube_length = self.tube_length.val
        self.number_sections = self.number_sections.val
        self.proportion = self.stenosis_radius_proportion.val
        self.flow_profile_path = self.flow_profile_path
        self.save_data_path = self.save_data_path
        plt.close('all')
        return

    def data_for_structure_along_z(self):
        z = np.linspace(0, self.tube_length.val, self.number_sections.val)
        theta = np.linspace(0, 2 * np.pi, self.number_sections.val)
        theta_grid, z_grid = np.meshgrid(theta, z)
        varied_radius = self.radius
        x_grid = (np.cos(theta_grid).T * varied_radius).T
        y_grid = (np.sin(theta_grid).T * varied_radius).T
        self.geometry = [z_grid, x_grid, y_grid]

    def stenosis(self, val):
        self.stenosis_position.valmin = -self.tube_length.val/2
        self.stenosis_position.valmax = self.tube_length.val / 2
        self.stenosis_expansion.valmax = self.tube_length.val
        xs = np.linspace(-self.tube_length.val / 2, self.tube_length.val / 2, int(self.number_sections.val))
        absolute_stenosis_depth = self.tube_base_radius.val * self.stenosis_radius_proportion.val
        self.radius = self.tube_base_radius.val * np.ones(len(xs)) - absolute_stenosis_depth * np.exp(
            -0.5 * (xs - self.stenosis_position.val) ** 2 / self.stenosis_expansion.val ** 2)
        self.data_for_structure_along_z()
        self.ax.cla()
        self.ax.plot_surface(self.geometry[0], self.geometry[1], self.geometry[2])
        self.ax.set_box_aspect((np.ptp(self.geometry[0]), np.ptp(self.geometry[1]), np.ptp(self.geometry[2])))
        self.fig.canvas.draw_idle()
        return self.radius, self.stenosis_position.val, self.stenosis_expansion.val, self.geometry
class ChooseCase:
    def __init__(self):
        self.fig = plt.figure(constrained_layout=True)
        self.fig.subplots_adjust(bottom=-0.0, top=0.6)
        self.ax_straight_expansion = self.fig.add_axes([0.2, 0.84, 0.2, 0.05])
        self.ax_straight_expansion.spines['top'].set_visible(True)
        self.ax_straight_expansion.spines['right'].set_visible(True)
        self.ax_stenosis = self.fig.add_axes([0.2, 0.78, 0.2, 0.05])
        self.ax_stenosis.spines['top'].set_visible(True)
        self.ax_stenosis.spines['right'].set_visible(True)
        self.straight_expansion_button = Button(ax=self.ax_straight_expansion, label='straight expansion')
        self.stenosis_button = Button(ax=self.ax_stenosis, label="stenosis")
        self.straight_expansion_button.on_clicked(self.start_program_straight_expansion)
        self.stenosis_button.on_clicked(self.start_program_stenosis)
        plt.show()
    def start_program_straight_expansion(self, val):
        plt.close('all')
        self.input_definitions = InputDefinitionsStraightExpansion()
        return
    def start_program_stenosis(self, val):
        plt.close('all')
        self.input_definitions = InputDefinitions()
        return

class InputDefinitionsStraightExpansion:
    def __init__(self):
        ICON_PLAY = plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\play.png')
        ICON_LOAD = plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\load.png')
        ICON_SAVE = plt.imread(r'C:\Users\Faiza\Desktop\1D_PH_FSI\Images\save.png')
        self.filename = ''
        self.fig = plt.figure(constrained_layout=True)
        self.fig.subplots_adjust(bottom=-0.0, top=0.6)
        self.ax_number_sections = self.fig.add_axes([0.2, 0.84, 0.2, 0.05])
        self.ax_number_sections.spines['top'].set_visible(True)
        self.ax_number_sections.spines['right'].set_visible(True)
        self.ax_straight_radius = self.fig.add_axes([0.2, 0.78, 0.2, 0.05])
        self.ax_straight_radius.spines['top'].set_visible(True)
        self.ax_straight_radius.spines['right'].set_visible(True)
        self.ax_structure_length = self.fig.add_axes([0.2, 0.72, 0.2, 0.05])
        self.ax_structure_length.spines['top'].set_visible(True)
        self.ax_structure_length.spines['right'].set_visible(True)
        self.ax_point_start_expansion = self.fig.add_axes([0.2, 0.66, 0.2, 0.05])
        self.ax_point_start_expansion.spines['top'].set_visible(True)
        self.ax_point_start_expansion.spines['right'].set_visible(True)
        self.ax_point_end_expansion = self.fig.add_axes([0.2, 0.6, 0.2, 0.05])
        self.ax_point_end_expansion.spines['top'].set_visible(True)
        self.ax_point_end_expansion.spines['right'].set_visible(True)
        self.ax_angle = self.fig.add_axes([0.2, 0.54, 0.2, 0.05])
        self.ax_angle.spines['top'].set_visible(True)
        self.ax_angle.spines['right'].set_visible(True)
        self.ax_window = self.fig.add_axes([0.2, 0.5, 0.2, 0.05])
        self.ax_window.spines['top'].set_visible(True)
        self.ax_window.spines['right'].set_visible(True)
        self.ax_start_button = self.fig.add_axes([0.95, 0.01, 0.05, 0.05])
        self.ax_save_button = self.fig.add_axes([0.9, 0.01, 0.05, 0.05])
        self.ax_load_button = self.fig.add_axes([0.85, 0.01, 0.05, 0.05])
        self.ax = self.fig.add_axes([0, 0, 0.5, 0.5], projection='3d')

        self.number_sections = Slider(ax=self.ax_number_sections, label='#sections ', valmin=10, valmax=200, valinit=100,
                                      valfmt='%d', facecolor='#cc7000', valstep=1)
        self.tube_base_radius = Slider(ax=self.ax_straight_radius, label='radius ', valmin=0.1, valmax=5.0, valinit=1,
                                       valfmt=' %1.1f cm', facecolor='#cc7000')
        self.tube_length = Slider(ax=self.ax_structure_length, label='length ', valmin=0.1, valmax=20,
                                  valinit=10.0, valfmt='%1.1f cm', facecolor='#cc7000')
        self.point_start_expansion = Slider(ax=self.ax_point_start_expansion, label='start expansion ', valmin=0.1, valmax=19.9,
                                        valinit=4.0, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.point_end_expansion = Slider(ax=self.ax_point_end_expansion, label='start expansion ', valmin=0.1,
                                            valmax=19.9,
                                            valinit=7.0, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.expansion_angle = Slider(ax=self.ax_angle, label='expansion angle ', valmin=0,
                                                 valmax=70,
                                                 valinit=1, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.transition_window = Slider(ax=self.ax_window, label='transition window ',valmin=0.1, valmax=20, valinit=0.1,valfmt=' %1.1f cm', facecolor='#cc7000')
        self.play_button = Button(ax=self.ax_start_button, label='', image=ICON_PLAY)
        self.load_button = Button(ax=self.ax_load_button, label="", image=ICON_LOAD)
        self.save_button = Button(ax=self.ax_save_button, label="", image=ICON_SAVE)

        self.x = np.linspace(0, self.tube_length.val, self.number_sections.val)

        self.tube_base_radius_val = np.ones(self.number_sections.val) * self.tube_base_radius.val
        self.radius = self.determine_initial_radius()
        self.data_for_structure_along_z()
        self.ax.plot_surface(self.geometry[0], self.geometry[1], self.geometry[2], alpha=1)
        self.ax.set_box_aspect((np.ptp(self.geometry[0]), np.ptp(self.geometry[1]), np.ptp(self.geometry[2])))
        self.ax.set_xlabel('tube length [cm]')
        self.number_sections.on_changed(self.func_straight_expansion)
        self.tube_base_radius.on_changed(self.func_straight_expansion)
        self.tube_length.on_changed(self.func_straight_expansion)

        self.point_start_expansion.on_changed(self.func_straight_expansion)

        self.expansion_angle.on_changed(self.func_straight_expansion)

        self.transition_window.on_changed(self.func_straight_expansion)
        self.play_button.on_clicked(self.start_program)
        self.load_button.on_clicked(self.load_flow_input_file)
        self.save_button.on_clicked(self.save_data)
        plt.show()

    def determine_initial_radius(self):
        xs = np.linspace(0, self.tube_length.val, int(self.number_sections.val))

        point_start_linear_expansion = self.point_start_expansion.val + self.transition_window.val
        point_end_linear_expansion = self.point_end_expansion.val - self.transition_window.val
        point_start_exp = self.point_start_expansion.val -self.transition_window.val
        point_end_exp = self.point_end_expansion.val + self.transition_window.val

        length_expansion = self.point_end_expansion.val - self.point_start_expansion.val
        radius_end = np.tan(np.radians(self.expansion_angle.val)) * length_expansion + self.tube_base_radius.val

        m = (radius_end - self.tube_base_radius.val) / length_expansion
        ycross = self.tube_base_radius.val - m * self.point_start_expansion.val
        value_point_start_linear_expansion = m * (point_start_linear_expansion) + ycross
        value_point_end_linear_expansion = m * (point_end_linear_expansion) + ycross

        a, b, c, d = sym.symbols(['a', 'b', 'c', 'd'])
        y1_1 = a * point_start_exp ** 3 + b * point_start_exp ** 2 + c * point_start_exp + d
        y1_2 = 3 * a * point_start_exp ** 2 + 2 * b * point_start_exp + c
        y1_3 = a * point_start_linear_expansion ** 3 + b * point_start_linear_expansion ** 2 + c * point_start_linear_expansion + d
        y1_4 = 3 * a * point_start_linear_expansion ** 2 + 2 * b * point_start_linear_expansion + c
        sol_start = sym.solve(
            [y1_1 - self.tube_base_radius.val, y1_2 - 0, y1_3 - value_point_start_linear_expansion, y1_4 - m],
            dict=True)
        sol_start = sol_start[0]
        a, b, c, d = sym.symbols(['a', 'b', 'c', 'd'])
        y2_1 = a * point_end_exp ** 3 + b * point_end_exp ** 2 + c * point_end_exp + d
        y2_2 = 3 * a * point_end_exp ** 2 + 2 * b * point_end_exp + c
        y2_3 = a * point_end_linear_expansion ** 3 + b * point_end_linear_expansion ** 2 + c * point_end_linear_expansion + d
        y2_4 = 3 * a * point_end_linear_expansion ** 2 + 2 * b * point_end_linear_expansion + c


        sol_end = sym.solve([y2_1 - radius_end, y2_2 - 0, y2_3 - value_point_end_linear_expansion, y2_4 - m],
                        dict=True)

        sol_end = sol_end[0]

        xstraight_entrance = np.linspace(0, point_start_exp, 50)
        ystraight_entrance = np.ones(50) * self.tube_base_radius.val

        xtrans_entrance = np.linspace(point_start_exp, point_start_linear_expansion, 50)
        ytrans_entrance = sol_start[a] * xtrans_entrance ** 3 + sol_start[b] * xtrans_entrance ** 2 + sol_start[c] * xtrans_entrance + sol_start[d]

        xexp = np.linspace(point_start_linear_expansion,point_end_linear_expansion, 50)
        yexp = m * xexp + ycross

        xtrans_exit = np.linspace(point_end_linear_expansion, point_end_exp, 50)
        ytrans_exit = sol_end[a] * xtrans_exit ** 3 + sol_end[b] * xtrans_exit ** 2 + sol_end[c] * xtrans_exit + sol_end[d]

        xstraight_exit = np.linspace(point_end_exp, self.tube_length.val, 50)
        ystraight_exit = np.ones(50) * radius_end




        x = np.hstack((xstraight_entrance, xtrans_entrance, xexp, xtrans_exit, xstraight_exit))
        y = np.hstack((ystraight_entrance, ytrans_entrance, yexp, ytrans_exit, ystraight_exit))
        plt.figure()
        plt.plot(x, y)
        f = interp1d(x, y)

        x = np.linspace(0, self.tube_length.val, self.number_sections.val)
        self.radius = f(x)
        return self.radius

    def load_flow_input_file(self, val):
        self.flow_profile_path = fd.askopenfilename()

    def save_data(self, val):
        self.save_data_path = fd.askdirectory()

    def handle_close(evt):
        print('Closed Figure!')

    def start_program(self, val):
        self.tube_base_radius_val = copy.copy(self.tube_base_radius.val)
        self.tube_base_radius, self.position, self.window, self.geometry = self.func_straight_expansion(2)
        self.tube_length = self.tube_length.val
        self.number_sections = self.number_sections.val
        self.proportion = self.expansion_angle.val
        self.flow_profile_path = self.flow_profile_path
        self.save_data_path = self.save_data_path
        plt.close('all')
        return

    def data_for_structure_along_z(self):
        z = np.linspace(0, self.tube_length.val, self.number_sections.val)
        theta = np.linspace(0, 2 * np.pi, self.number_sections.val)
        theta_grid, z_grid = np.meshgrid(theta, z)
        varied_radius = self.radius
        x_grid = (np.cos(theta_grid).T * varied_radius).T
        y_grid = (np.sin(theta_grid).T * varied_radius).T
        self.geometry = [z_grid, x_grid, y_grid]

    def func_straight_expansion(self, val):
        self.point_start_expansion.valmin = 0
        self.point_start_expansion.valmax = self.tube_length.val-0.1
        self.transition_window.valmax = self.tube_length.val-self.point_start_expansion.valmax

        xs = np.linspace(0, self.tube_length.val, int(self.number_sections.val))

        point_start_linear_expansion = self.point_start_expansion.val + self.transition_window.val
        point_end_linear_expansion = self.point_end_expansion.val - self.transition_window.val
        point_start_exp = self.point_start_expansion.val - self.transition_window.val
        point_end_exp = self.point_end_expansion.val + self.transition_window.val

        length_expansion = self.point_end_expansion.val - self.point_start_expansion.val
        radius_end = np.tan(np.radians(self.expansion_angle.val)) * length_expansion + self.tube_base_radius.val

        m = (radius_end - self.tube_base_radius.val) / length_expansion
        ycross = self.tube_base_radius.val - m * self.point_start_expansion.val
        value_point_start_linear_expansion = m * (point_start_linear_expansion) + ycross
        value_point_end_linear_expansion = m * (point_end_linear_expansion) + ycross

        a, b, c, d = sym.symbols(['a', 'b', 'c', 'd'])
        y1_1 = a * point_start_exp ** 3 + b * point_start_exp ** 2 + c * point_start_exp + d
        y1_2 = 3 * a * point_start_exp ** 2 + 2 * b * point_start_exp + c
        y1_3 = a * point_start_linear_expansion ** 3 + b * point_start_linear_expansion ** 2 + c * point_start_linear_expansion + d
        y1_4 = 3 * a * point_start_linear_expansion ** 2 + 2 * b * point_start_linear_expansion + c
        sol_start = sym.solve(
            [y1_1 - self.tube_base_radius.val, y1_2 - 0, y1_3 - value_point_start_linear_expansion, y1_4 - m],
            dict=True)
        sol_start = sol_start[0]
        a, b, c, d = sym.symbols(['a', 'b', 'c', 'd'])
        y2_1 = a * point_end_exp ** 3 + b * point_end_exp ** 2 + c * point_end_exp + d
        y2_2 = 3 * a * point_end_exp ** 2 + 2 * b * point_end_exp + c
        y2_3 = a * point_end_linear_expansion ** 3 + b * point_end_linear_expansion ** 2 + c * point_end_linear_expansion + d
        y2_4 = 3 * a * point_end_linear_expansion ** 2 + 2 * b * point_end_linear_expansion + c

        sol_end = sym.solve([y2_1 - radius_end, y2_2 - 0, y2_3 - value_point_end_linear_expansion, y2_4 - m],
                            dict=True)

        sol_end = sol_end[0]

        xstraight_entrance = np.linspace(0, point_start_exp, 50)
        ystraight_entrance = np.ones(50) * self.tube_base_radius.val

        xtrans_entrance = np.linspace(point_start_exp, point_start_linear_expansion, 50)
        ytrans_entrance = sol_start[a] * xtrans_entrance ** 3 + sol_start[b] * xtrans_entrance ** 2 + sol_start[
            c] * xtrans_entrance + sol_start[d]

        xexp = np.linspace(point_start_linear_expansion, point_end_linear_expansion, 50)
        yexp = m * xexp + ycross

        xtrans_exit = np.linspace(point_end_linear_expansion, point_end_exp, 50)
        ytrans_exit = sol_end[a] * xtrans_exit ** 3 + sol_end[b] * xtrans_exit ** 2 + sol_end[c] * xtrans_exit + \
                      sol_end[d]

        xstraight_exit = np.linspace(point_end_exp, self.tube_length.val, 50)
        ystraight_exit = np.ones(50) * radius_end

        x = np.hstack((xstraight_entrance, xtrans_entrance, xexp, xtrans_exit, xstraight_exit))
        y = np.hstack((ystraight_entrance, ytrans_entrance, yexp, ytrans_exit, ystraight_exit))
        f = interp1d(x, y)

        x = np.linspace(0, self.tube_length.val, self.number_sections.val)
        self.radius = f(x)
        self.data_for_structure_along_z()
        self.ax.cla()
        self.ax.plot_surface(self.geometry[0], self.geometry[1], self.geometry[2])
        self.ax.set_box_aspect((np.ptp(self.geometry[0]), np.ptp(self.geometry[1]), np.ptp(self.geometry[2])))
        self.fig.canvas.draw_idle()
        return self.radius, self.point_start_expansion.val, self.transition_window.val, self.geometry

class OneD_PHM:
    def __init__(self,pdic, demo = False, path_to_image = ''):
        self.radius = pdic['radius']
        self.am1_0_vol = 1
        self.am1_0_pd = 1

        self.calculate_intermediate = True

        self.inp_entrance = pdic['inp_entrance']
        self.inp_exit = pdic['inp_exit']

        self.N_sec = pdic['N_sec']
        self.N_nodes = pdic['N_nodes']
        self.N_ypoints_for_pressure_image = pdic['N_ypoints_for_pressure_image']

        self.external_force = pdic['external_force']
        self.structure_length = pdic['structure_length']
        self.section_length = pdic['structure_length'] / self.N_sec


        self.wall_thickness = pdic['wall_thickness']

        self.poi_rat = pdic['poi_rat']
        self.stiffness_k1 = pdic['stiffness_k1']
        self.stiffness_k2 = pdic['stiffness_k2']
        self.stiffness_k3 = pdic['stiffness_k3']
        self.viscosity = pdic['viscosity']
        self.initial_fluid_density = pdic['fluid_density']
        self.input_shape = np.array([self.inp_entrance(t) for t in pdic['t_evaluation']])

        self.structure_r_stiffness = np.array([(self.poi_rat / ((1 - 2 * self.poi_rat) * (1 + self.poi_rat))) * (self.section_length / np.pi) * (
                    self.stiffness_k1 * (np.exp(self.stiffness_k2 * (self.radius[i]))) + self.stiffness_k3 ) for i in range(self.N_sec)])

        self.structure_z_stiffness = np.array([0.5 * (
                    self.stiffness_k1 * (np.exp(self.stiffness_k2 * (self.radius[i])))+ self.stiffness_k3) * (np.pi * np.square(self.radius[i]) / self.section_length) * (1 / (1 + self.poi_rat)) for i in range(self.N_sec)])


        self.geo_dissipation = pdic['geo_dissipation']
        self.vis_dissipation = pdic['vis_dissipation']
        self.geo_dissipation_factor = pdic['geo_factor']
        # *2 * self.section_length / self.radius
        self.vis_dissipation_factor = pdic['vis_factor']*2 * self.section_length / self.radius

        self.structure_mass = np.multiply(np.pi * pdic['structure_density'] * self.radius,  pdic['wall_thickness'] * self.section_length * np.ones(self.N_sec))
        self.structure_r_dissipation = pdic['structure_r_dissipation'] * np.ones(self.N_sec)
        self.structure_z_dissipation = pdic['structure_z_dissipation'] * np.ones(self.N_sec)  # coupling damping coefficient of structure
        self.node_mass = np.multiply(pdic['fluid_density'] * np.pi * np.square(self.radius),  self.section_length * 10 ** -3 * np.ones(self.N_nodes))


        self.fluid_bulk_modulus = pdic['fluid_bulk_modulus']
        self.t_evaluation = pdic['t_evaluation']
        self.pressure_at_0 = 0
        self.total_pressure = np.zeros((self.N_sec))
        self.static_pressure = np.zeros((self.N_sec))
        self.dynamic_pressure = np.zeros((self.N_sec))
        self.velocity = np.zeros((self.N_sec))
        self.requested_data_over_tube = pdic['requested_data_over_tube']
        self.requested_data_at_boundaries = pdic['requested_data_at_boundaries']
        self.evaluate_timepoint = False
        self.create_structure_requested_data()
        self.demo = demo
        self.min_pressure = pdic['min_pressure']
        self.max_pressure = pdic['max_pressure']
        self.step_time = np.array([0])


    def create_structure_requested_data(self):
        self.requested_data = {}
        for entry in self.requested_data_over_tube:
            self.requested_data[entry] = np.empty((self.N_sec, len(self.t_evaluation)))
        for entry in self.requested_data_at_boundaries:
            self.requested_data[entry] = np.empty((2, len(self.t_evaluation)))

    def dynamic_model(self, t, x):
        u_p = np.concatenate(([float(self.inp_entrance(t))], [self.inp_exit(t)]))
        x = x.reshape(-1, )
        structure_pos = x[0:self.N_sec]
        structure_mom = x[self.N_sec:2 * self.N_sec]
        fluid_mom = x[2 * self.N_sec:3 * self.N_sec]
        fluid_density = x[3 * self.N_sec:]

        A_cross = np.multiply(np.pi * self.radius, (structure_pos + self.radius))

        alpha_1 = A_cross[1:-1] / (A_cross[1:-1] + A_cross[2:])  # check
        alpha_2 = 1 - (A_cross[0:-2] / (A_cross[0:-2] + A_cross[1:-1]))  # check

        V_sec = self.section_length * A_cross[1:-1] - (self.node_mass[2:] / fluid_density[2:]) * alpha_1 - (
                self.node_mass[1:-1] / fluid_density[1:-1]) * alpha_2  # check
        V_sec = np.concatenate((np.reshape(
            self.section_length * A_cross[0] - self.node_mass[1] * (1 / fluid_density[1]) * (
                    A_cross[0] / (A_cross[0] + A_cross[1])) - self.node_mass[0] * (
                    1 / fluid_density[0]) * float(self.am1_0_vol), (1,)), V_sec, np.reshape(
            A_cross[-1] * self.section_length - self.node_mass[-1] * (1 / fluid_density[-1]) * (
                    1 - (A_cross[-2] / (A_cross[-2] + A_cross[-1]))),
            (1,))))  # check

        A_contact_node = np.hstack(
            (np.reshape(self.node_mass[0] / (fluid_density[0] * (self.radius[0] + structure_pos[0])), (1,)),
             self.node_mass[1:] / (
                     fluid_density[1:] * (2 * self.radius[1:] + structure_pos[0:-1] + structure_pos[1:]))))  # check

        A_contact_section = V_sec / (structure_pos + self.radius)  # check

        alpha = A_cross[0:-1] / (A_cross[0:-1] + A_cross[1:])

        dynamic_pressure = np.concatenate((np.reshape(
            (1 * self.initial_fluid_density / 2) * np.square((u_p[0] / A_cross[0])) * self.am1_0_pd - (
                    1 / (self.initial_fluid_density * 2)) * np.square((fluid_mom[0] / V_sec[0])) * self.am1_0_pd,
            (1,)),
                                           (1 / (2 * self.initial_fluid_density)) * np.square(
                                               fluid_mom[0:-1] / V_sec[0:-1]) * alpha - (
                                                   1 / (2 * self.initial_fluid_density)) * np.square(
                                               fluid_mom[1:] / V_sec[1:]) * (
                                                   np.ones(len(alpha)) - alpha)))  # check

        row1 = np.concatenate(
            (np.reshape([self.structure_r_stiffness[0] + self.structure_z_stiffness[0], -self.structure_z_stiffness[0]],
                        (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.structure_z_stiffness[0:-2]), np.zeros((self.N_sec - 2, 2))),
                                axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.structure_r_stiffness[1:-1]) + np.diag(self.structure_z_stiffness[0:-2]) + np.diag(
                 self.structure_z_stiffness[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.structure_z_stiffness[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape(
                [-self.structure_z_stiffness[-2], self.structure_r_stiffness[-1] + self.structure_z_stiffness[-2]],
                (1, 2))),
            axis=1)

        Cspring = np.concatenate((row1, matrix, row2), axis=0)
        C_A_n = np.concatenate(
            (np.concatenate((np.diag(A_contact_node[0:-1]), np.zeros((self.N_sec - 1, 1))), axis=1) + np.concatenate(
                (np.zeros((self.N_sec - 1, 1)), np.diag(A_contact_node[1:])), axis=1),
             np.hstack((np.zeros(self.N_sec - 1), A_contact_node[-1])).reshape(1, -1)), axis=0)
        H_structure_pos = np.dot(Cspring, structure_pos) + A_contact_section * (
                1 / (2 * self.initial_fluid_density)) * np.square(fluid_mom / V_sec) + np.dot(C_A_n, dynamic_pressure)

        H_structure_mom =structure_mom / self.structure_mass

        H_fluid_mom = fluid_mom / (self.initial_fluid_density * V_sec)

        static_pressure = np.array(
            [self.fluid_bulk_modulus * np.log(fluid_density[i] / self.initial_fluid_density) for i in
             range(0, self.N_sec, 1)])

        H_fluid_density = (self.node_mass / (np.square(fluid_density))) * (static_pressure + dynamic_pressure)

        C1 = np.identity(self.N_sec)
        C2 = np.concatenate(
            (np.zeros((1, self.N_sec)),
             np.concatenate((np.identity(self.N_sec - 1), np.zeros((self.N_sec - 1, 1))), axis=1)))
        C1_star = np.reshape(np.concatenate(([1], np.zeros(self.N_sec - 1))), (1, self.N_sec))
        C1_star_help = np.reshape(np.zeros(self.N_sec), (1, self.N_sec))
        C2_star = np.reshape(np.concatenate((np.zeros(self.N_sec - 1), [1])), (1, self.N_sec))

        row1 = np.concatenate(
            (np.reshape(
                [self.structure_r_dissipation[0] + self.structure_z_dissipation[0], -self.structure_z_dissipation[0]],
                (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.structure_z_dissipation[0:-2]), np.zeros((self.N_sec - 2, 2))),
                                axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.structure_r_dissipation[1:-1]) + np.diag(self.structure_z_dissipation[0:-2]) + np.diag(
                 self.structure_z_dissipation[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.structure_z_dissipation[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.structure_z_dissipation[-2],
                                                        self.structure_r_dissipation[-1] + self.structure_z_dissipation[
                                                            -2]], (1, 2))),
            axis=1)

        R_s = np.concatenate((row1, matrix, row2), axis=0)

        loss_g = np.zeros((self.N_sec))
        for k in range(0, self.N_sec - 1, 1):
            if A_cross[k] <= A_cross[k + 1]:
                angle = np.arctan((self.radius[k + 1] - self.radius[k]) / self.section_length)
                if math.degrees(angle) <= 45:
                    loss_g[k] = (np.sin(angle / 2) * (1 - (self.radius[k] / self.radius[k + 1]) ** 2) ** 2)
                else:
                    loss_g[k] = (1 - (A_cross[k] / A_cross[k + 1])) ** 2
            else:
                loss_g[k] = 0.5 * (1 - (A_cross[k + 1] / A_cross[k]))
        loss_g[-1] = 0
        loss_g = self.geo_dissipation_factor * loss_g

        if self.geo_dissipation == False:
            loss_g = loss_g * 0

        loss_vis = np.array(
            [self.viscosity / (2 * (self.radius[i] + structure_pos[i]) * np.abs(
                H_fluid_mom[i]) * self.initial_fluid_density) if H_fluid_mom[i] != 0 else 0 for i
             in range(0, self.N_sec, 1)])
        loss_vis = np.multiply(loss_vis, self.vis_dissipation_factor)

        if self.vis_dissipation == False:
            loss_vis = loss_vis * 0



        R_f_geo = np.diag(A_cross * loss_g * (self.initial_fluid_density / 2) * (
            np.abs(fluid_mom / (self.initial_fluid_density * V_sec))))
        R_f_vis = np.diag(A_cross * loss_vis * (self.initial_fluid_density / 2) * (
            np.abs(fluid_mom / (self.initial_fluid_density * V_sec))))
        R_f = R_f_geo + R_f_vis

        theta_pi = np.diag(A_cross)
        theta_rho = np.diag(np.square(fluid_density) / self.node_mass)
        g_An = np.diag(A_contact_node[1:])
        psi_pi = np.diag(A_contact_section / 2)
        varphi_pi = np.diag(fluid_mom / (self.radius + structure_pos))

        g_An_1 = np.concatenate((g_An, np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1))), axis=1)
        g_An_2 = np.concatenate((np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1)), g_An), axis=1)

        varphi_rho = np.dot(theta_rho, np.concatenate(
            (np.hstack(([A_contact_node[0]], np.zeros(self.N_sec - 1))).reshape(1, -1), g_An_1 + g_An_2)))
        phi = np.dot(theta_pi, np.dot(C1.T - C2.T, theta_rho.T))
        theta = np.dot(theta_pi, np.concatenate((C1_star_help.T, -C2_star.T), axis=1))
        varphi = varphi_rho + np.dot(theta_rho, np.dot((C2 + C1), psi_pi))
        psi = np.dot(np.reshape(np.concatenate((C1_star_help.T, C2_star.T)), (2, self.N_sec)), psi_pi)
        C = np.identity(self.N_sec)

        gamma = np.concatenate((np.dot(theta_rho.T, C1_star.T), np.zeros((self.N_sec, 1))), axis=1)

        J_R_1 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.identity(self.N_sec),
                                np.zeros((self.N_sec, self.N_sec)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        J_R_2 = np.concatenate((-np.identity(self.N_sec), -R_s, np.dot(C.T, varphi_pi.T), np.dot(C.T, varphi.T)),
                               axis=1)
        J_R_3 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi_pi, C), -R_f, phi), axis=1)
        J_R_4 = np.concatenate(
            (np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi, C), -phi.T, np.zeros((self.N_sec, self.N_sec))),
            axis=1)
        J_R = np.concatenate((J_R_1, J_R_2, J_R_3, J_R_4))

        g_P_1 = np.concatenate((np.zeros((self.N_sec, 2)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_2 = np.concatenate((np.dot(C.T, psi.T), np.identity(self.N_sec)), axis=1)
        g_P_3 = np.concatenate((theta, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_4 = np.concatenate((gamma, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P = np.concatenate((g_P_1, g_P_2, g_P_3, g_P_4))
        x_vec = np.concatenate((H_structure_pos, H_structure_mom, H_fluid_mom, H_fluid_density))
        u = np.concatenate((u_p, self.external_force))
        dxdt = np.dot(J_R, x_vec) + np.dot(g_P, u)
        y = np.dot(g_P.T, x_vec)

        if t > self.step_time[-1]:
            self.step_time = np.vstack((self.step_time, t))
            self.pressure_at_0 = np.vstack((self.pressure_at_0, static_pressure[0] + dynamic_pressure[0]))
            self.total_pressure = np.vstack((self.total_pressure, static_pressure+dynamic_pressure))
            self.static_pressure = np.vstack((self.static_pressure, static_pressure))
            self.dynamic_pressure = np.vstack((self.static_pressure, dynamic_pressure))
            self.velocity = np.vstack((self.velocity, H_fluid_mom))

        return dxdt

class PlotResults:
    def __init__(self, sim_results):
        self.pressure = sim_results['interpolated_data']
        self.sample_time = sim_results['t_evaluation']
        self.input_shape = sim_results['inp_entrance']
        self.axial_discretization = np.linspace(0, sim_results['structure_length'], sim_results['N_sec'])

    def plot_pressure(self):
        plt.figure()
        plt.plot(self.axial_discretization, self.pressure(self.sample_time[-1]))



class VisualizingExtendedResults:
    def __init__(self,pdic, pressure, pressure1, pressure2, radius, radius1, radius2, min_pressure, max_pressure):
            self.total_pressure_initial = pressure
            self.total_pressure_intermediate = pressure1
            self.total_pressure_final = pressure2
            self.pressure_title=pdic['pressure_title']
            self.path_to_images = pdic['path_to_image']
            self.heart_front_image_filenames = self.get_image_file_names(pdic['path_to_image'][0])
            self.heart_side_image_filenames = self.get_image_file_names(pdic['path_to_image'][1])
            self.number_of_image_frames = len(self.heart_front_image_filenames)
            self.frame_count = 0
            self.sample_time_vis = pdic['sample_time_visualization']
            self.t_sim = pdic['t_evaluation'][::self.sample_time_vis]
            self.N_sec = pdic['N_sec']
            self.radius_initial = radius
            self.radius_intermediate = radius1
            self.radius_final = radius2
            self.N_ypoints_for_pressure_image = pdic['N_ypoints_for_pressure_image']
            self.structure_length = pdic['structure_length']
            self.section_length = self.structure_length/self.N_sec
            self.t_evaluation = pdic['t_evaluation']
            self.simulation_time = self.t_evaluation[-1]
            self.input_scale = 1*10**6
            self.input_shape = np.array([pdic['inp_entrance'](t)*self.input_scale for t in pdic['t_evaluation']])
            self.input_value =pdic['inp_entrance']
            self.scale = pdic['scale']
            self.min_pressure = float(min_pressure)*self.scale
            self.max_pressure = float(max_pressure)*self.scale
            self.crop_front_x_start = 580
            self.crop_front_x_end = 1110
            self.crop_front_y_start = 0
            self.crop_front_y_end = 715
            self.crop_side_x_start = 800
            self.crop_side_x_end = 2100
            self.crop_side_y_start = 50
            self.crop_side_y_end = 1500
            self.geometry1 = self.data_for_structure_along_z(0, self.radius_initial, np.max(self.radius_initial))
            self.geometry2 = self.data_for_structure_along_z(1, self.radius_intermediate, np.max(self.radius_initial))
            self.geometry3 = self.data_for_structure_along_z(2, self.radius_final, np.max(self.radius_initial))
            self.initialize_demo_figure()




    def data_for_structure_along_z(self, mult, radius, maxvalue):
        z = np.linspace(0, self.structure_length*10**2, self.N_sec)
        theta = np.linspace(0, 2 * np.pi, self.N_sec)
        theta_grid, z_grid = np.meshgrid(theta, z)
        varied_radius = radius*10**2
        x_grid = (np.cos(theta_grid).T * varied_radius).T+mult*4*maxvalue*10**2
        y_grid = (np.sin(theta_grid).T * varied_radius).T
        return x_grid, y_grid, -z_grid

    def get_image_file_names(self, path):
        file_list = os.listdir(path)
        file_list = sorted(file_list, key=lambda x: float(re.findall("(\d+)", x)[0]))
        file_list = [os.path.join(path, file_list[i]) for i in range(0, len(file_list),1)]
        return file_list

    def initialize_demo_figure(self):
        random_inital_pressure = np.linspace(self.min_pressure, self.max_pressure, self.N_sec)
        theta = np.linspace(0, 2 * np.pi, self.N_sec)
        theta_grid, pressure_grid = np.meshgrid(theta, random_inital_pressure)

        self.figure = plt.figure(constrained_layout = True)

        ellipse1 = Arc(xy=(252, 211), width=102, height=40,angle = 0, theta1=0, theta2=180,linestyle='--',
                          edgecolor='b', fc='None', lw=1)
        ellipse2 = Arc(xy=(252, 211), width=102, height=40, angle=0, theta1=180, theta2=360,
                      edgecolor='b', fc='None', lw=1)

        ellipse3= Arc(xy=(160, 266), width=75, height=35, angle=15, theta1=0, theta2=180, linestyle='--',
                       edgecolor='b', fc='None', lw=1)
        ellipse4 = Arc(xy=(160, 266), width=75, height=35, angle=15, theta1=180, theta2=360,
                       edgecolor='b', fc='None', lw=1)
        ellipse5 = Arc(xy=(150, 437), width=75, height=35, angle=15, theta1=0, theta2=180, linestyle='--',
                       edgecolor='b', fc='None', lw=1)
        ellipse6 = Arc(xy=(150, 437), width=75, height=35, angle=15, theta1=180, theta2=360,
                       edgecolor='b', fc='None', lw=1)

        self.figure.add_artist(lines.Line2D([0.188, 0.308], [0.543, 0.723],color='b', lw=1))
        self.figure.add_artist(lines.Line2D([0.2255, 0.353], [0.538, 0.723],color='b', lw=1))

        self.cat_logo = plt.subplot2grid((16, 35), (12, 0), colspan=2, rowspan=1)
        self.flow_profile_fig = plt.subplot2grid((16, 35), (1, 13), colspan=7, rowspan=3)
        self.clock = plt.subplot2grid((16, 35), (3, 10), colspan=2, rowspan=2)
        self.heart_front = plt.subplot2grid((16, 35), (5, 3), rowspan=8, colspan=8)
        self.heart_side = plt.subplot2grid((16, 35), (5, 11), rowspan=8, colspan=8)
        self.dhm_logo = plt.subplot2grid((16, 35), (14, 0), rowspan=2, colspan=4)
        self.mirmi_logo = plt.subplot2grid((16, 35), (14, 4), rowspan=2, colspan=4)
        self.tum_logo = plt.subplot2grid((16, 35), (14, 8), rowspan=2, colspan=6)
        self.tube = plt.subplot2grid((16, 35), (2, 20), rowspan=15, colspan=15, projection='3d')
        self.cat_logo.imshow(cv2.cvtColor(cv2.imread(self.path_to_images[2]+r'\logo-cat-medic.jpg'), cv2.COLOR_BGR2RGB))
        self.cat_logo.axis('off')


        self.flow_profile,  = self.flow_profile_fig.plot(self.t_evaluation, self.input_shape)
        self.set_actual_state_point, = self.flow_profile_fig.plot([0], [self.input_value(0)*self.input_scale], label='toto', ms=5, color='b', marker='o', ls='')
        self.flow_profile_fig.set_ylabel("Blutmenge [ml/s]", fontsize = 20.0)
        self.flow_profile_fig.set_xlabel("Zeit [s]", fontsize = 20.0)
        self.clock.set_aspect('equal')
        self.clock.imshow(
            cv2.cvtColor(cv2.imread(self.path_to_images[2]+ r'\clock.jpg'), cv2.COLOR_BGR2RGB))
        self.clock.axis('off')


        self.heart_front_image = self.heart_front.imshow(
            cv2.cvtColor(cv2.imread(self.heart_front_image_filenames[0]), cv2.COLOR_BGR2RGB))
        self.heart_front.add_patch(ellipse1)
        self.heart_front.add_patch(ellipse2)
        self.heart_front.set_xlabel("Herz (anterior)",  fontsize = 20.0)


        self.heart_side_image = self.heart_side.imshow(
            cv2.cvtColor(cv2.imread(self.heart_side_image_filenames[0]), cv2.COLOR_BGR2RGB))
        self.heart_side.add_patch(ellipse3)
        self.heart_side.add_patch(ellipse4)
        self.heart_side.add_patch(ellipse5)
        self.heart_side.add_patch(ellipse6)
        self.heart_side.set_xlabel("Herz (posterior)", fontsize = 20.0)



        self.norm = matplotlib.colors.Normalize(vmin=self.min_pressure, vmax=self.max_pressure)

        self.figure.add_artist(lines.Line2D([0.4145, 0.608], [0.5, 0.625], color='b', lw=1))
        self.figure.add_artist(lines.Line2D([0.41, 0.608], [0.38, 0.285], color='b', lw=1))


        self.dhm_logo.imshow(
            cv2.cvtColor(cv2.imread(self.path_to_images[2]+r'\dhm.png'), cv2.COLOR_BGR2RGB))
        self.dhm_logo.axis('off')

        self.mirmi_logo.imshow(
            cv2.cvtColor(cv2.imread(self.path_to_images[2]+r'\mirmi.png'), cv2.COLOR_BGR2RGB))
        self.mirmi_logo.axis('off')

        self.tum_logo.imshow(
            cv2.cvtColor(cv2.imread(self.path_to_images[2]+r'\tum.png'), cv2.COLOR_BGR2RGB))
        self.tum_logo.axis('off')

        self.clock.set_xticks([])
        self.clock.set_yticks([])

        self.heart_front.set_xticks([])
        self.heart_front.set_yticks([])

        self.heart_side.set_xticks([])
        self.heart_side.set_yticks([])

        self.cat_logo.set_xticks([])
        self.cat_logo.set_yticks([])

        self.dhm_logo.set_xticks([])
        self.dhm_logo.set_yticks([])

        self.mirmi_logo.set_xticks([])
        self.mirmi_logo.set_yticks([])

        self.tum_logo.set_xticks([])
        self.tum_logo.set_yticks([])
        self.geometry_x_length = len(self.geometry1[0])-1
        self.geometry_y_length = len(self.geometry1[1])-1
        self.color_reshape_nr = self.geometry_x_length*self.geometry_y_length
        self.tube1 = self.tube.plot_surface(self.geometry1[0], self.geometry1[1], self.geometry1[2],
                               facecolors=cm.Reds(self.norm(pressure_grid)), alpha=1,
                               vmin=self.min_pressure, vmax=self.max_pressure)
        self.tube2 = self.tube.plot_surface(self.geometry2[0], self.geometry2[1], self.geometry2[2],
                               facecolors=cm.Reds(self.norm(pressure_grid)), alpha=1,
                               vmin=self.min_pressure, vmax=self.max_pressure)
        self.tube3 = self.tube.plot_surface(self.geometry3[0], self.geometry3[1], self.geometry3[2],
                               facecolors=cm.Reds(self.norm(pressure_grid)), alpha=1,
                               vmin=self.min_pressure, vmax=self.max_pressure)

        self.tube.set_box_aspect((np.ptp(np.hstack((self.geometry1[0], self.geometry2[0], self.geometry3[0]))),
                                  np.ptp(np.hstack((self.geometry1[1], self.geometry2[1], self.geometry3[1]))),
                                  np.ptp(np.hstack((self.geometry1[2], self.geometry2[2], self.geometry3[2])))))
        self.colorMap = m = cm.ScalarMappable(cmap=cm.Reds, norm=self.norm)
        m.set_array([])
        position = self.figure.add_axes([0.7, 0.12, 0.2, 0.02])
        clb = self.figure.colorbar(m, cax=position, orientation='horizontal')
        clb.set_label(self.pressure_title, fontsize = 20, labelpad=10, y=1.05, rotation=0)
        self.tube.set_yticks([])
        self.tube.set_zticks([])
        self.tube.set_xticks([0, 4 * np.max(self.radius_initial) * 10 ** 2, 8 * np.max(self.radius_initial) * 10 ** 2])
        self.tube.set_xticklabels(["Hochgradige \n Engstelle", "Suboptimaler \n Stent", "Optimaler \n Stent"],
                                  fontsize=20, minor=False)

        self.tube.view_init(azim=-76, elev=9)
        plt.subplots_adjust(bottom=0.01)
        plt.subplots_adjust(left=0.025)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(top=0.995)
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")
        plt.draw()

    def update_pressure_plot(self, repeat):
        for j in range(0, repeat, 1):
            for t in self.t_sim:
                if self.frame_count < len(self.heart_front_image_filenames)-1:
                    self.frame_count += 1
                    self.heart_image_front = cv2.cvtColor(cv2.imread(self.heart_front_image_filenames[self.frame_count]), cv2.COLOR_BGR2RGB)
                    self.heart_image_side = cv2.cvtColor(cv2.imread(self.heart_side_image_filenames[self.frame_count]),
                                                          cv2.COLOR_BGR2RGB)

                else:
                    self.frame_count = 0
                    self.heart_image_front = cv2.cvtColor(cv2.imread(self.heart_front_image_filenames[self.frame_count]),
                                                          cv2.COLOR_BGR2RGB)
                    self.heart_image_side = cv2.cvtColor(cv2.imread(self.heart_side_image_filenames[self.frame_count]),
                                                         cv2.COLOR_BGR2RGB)
                theta = np.linspace(0, 2*np.pi, self.N_sec)
                theta_grid, self.pressure_image_initial = np.meshgrid(theta, self.total_pressure_initial(t)*self.scale)
                theta_grid, self.pressure_image_intermediate = np.meshgrid(theta,
                                                                      self.total_pressure_intermediate(t) * self.scale
                                                                        )
                theta_grid, self.pressure_image_final = np.meshgrid(theta,
                                                                      self.total_pressure_final(t) * self.scale)
                self.on_running(t, self.pressure_image_initial,self.pressure_image_intermediate, self.pressure_image_final, self.heart_image_front, self.heart_image_side)

    def on_running(self, t, pmatrix,pmatrix1, pmatrix2, heartimage_front, heartimage_side):
        self.tube.cla()
        self.tube.plot_surface(self.geometry1[0], self.geometry1[1], self.geometry1[2],
                               facecolors=cm.Reds(self.norm(pmatrix)), alpha=1,
                               vmin=self.min_pressure, vmax=self.max_pressure)
        self.tube.plot_surface(self.geometry2[0], self.geometry2[1], self.geometry2[2],
                               facecolors=cm.Reds(self.norm(pmatrix1)), alpha=1,
                               vmin=self.min_pressure, vmax=self.max_pressure)
        self.tube.plot_surface(self.geometry3[0], self.geometry3[1], self.geometry3[2],
                               facecolors=cm.Reds(self.norm(pmatrix2)), alpha=1,
                               vmin=self.min_pressure, vmax=self.max_pressure)
        self.heart_front_image.set_data(heartimage_front)
        self.heart_side_image.set_data(heartimage_side)
        self.set_actual_state_point.set_xdata([t])
        self.set_actual_state_point.set_ydata([self.input_value(t)*self.input_scale])
        self.tube.set_yticks([])
        self.tube.set_zticks([])
        self.tube.set_xticks(
            [0, 4 * np.max(self.radius_initial) * 10 ** 2, 8 * np.max(self.radius_initial) * 10 ** 2])
        self.tube.set_xticklabels(["Hochgradige \n Engstelle", "Suboptimaler \n Stent", "Optimaler \n Stent"],
                                  fontsize=20, minor=False)
        self.figure.canvas.draw_idle()
        # self.tube.set_axis_off()

        plt.pause(0.01)


