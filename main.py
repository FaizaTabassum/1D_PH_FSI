# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:17:24 2021
@author: Faiza
"""
import copy
import sys

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

class constraints:
    def __init__(self, path_load, path_save, out_max):
        self.path_load = path_load
        self.path_save = path_save
        self.out_max = out_max
        self.scale = 1*10**-6

    @property
    def inp_const(self):
        inp_max = 120 * 10**-6
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



@profile
def run_simulation(radius, tube_length, number_sections, input_filename, demo = False, min_pressure = 0, max_pressure = 0):
    N_sec = number_sections
    N_nodes = number_sections
    simulation_time = 1
    sample_rate = 1e-03
    path_save = r'C:\Users\Faiza\Desktop\PH_simvascular\1D_porthamiltonian_FSI\gradual_expansion'
    path_input_at_inlet = input_filename
    cons = constraints(path_input_at_inlet,path_save, 0)

    t_evaluation = np.ndarray.tolist(np.arange(0, simulation_time, sample_rate))

    parameter = {
        'min_pressure': min_pressure,
        'max_pressure': max_pressure,
        'geo_dissipation': True,
        'geo_factor': 2.6,
        'vis_factor': 16,
        'vis_dissipation': True,
        'inp_entrance': cons.repeat_input(simulation_time),
        'inp_exit': cons.out_const,
        'structure_length': tube_length * 10 ** -2,
        'radius': radius*10 ** -2,
        'wall_thickness': 3 * 10 ** -4,
        'N_sec': N_sec,
        'N_ypoints_for_pressure_image': N_sec,
        'N_nodes': N_nodes,
        't_evaluation': t_evaluation,
        'sample_time_visualization':100,
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
        'requested_data_at_boundaries': ['pressure', 'flow']
    }



    PHFSI = func.Model_Flow_Efficient(parameter, demo = demo, path_to_image = r'C:\Users\Faiza\Desktop\PH_simvascular\code\cardiac_cycle_images') #this is for flow, if you want to test Pressure, you need to substitute Flow by Pressure
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
    return np.min(PHFSI.pressure[1:-1]), np.max(PHFSI.pressure[1:-1])
class GeometryAndInput:
    def __init__(self):
        self.filename = ''
        self.fig = plt.figure()
        self.ax = self.fig .add_subplot(111)
        self.fig.subplots_adjust(bottom=0.2, top=0.45)
        self.ax_nx = self.fig .add_axes([0.2, 0.84, 0.2, 0.05])
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
        self.ax_start_button = self.fig.add_axes([0.6, 0.48, 0.2, 0.05])
        self.ax_load_button = self.fig.add_axes([0.6, 0.84, 0.2, 0.05])

        self.number_sections = Slider(ax=self.ax_nx, label='#sections ', valmin=10, valmax=200, valinit=35,
                                 valfmt='%d', facecolor='#cc7000', valstep=1)
        self.tube_base_radius = Slider(ax=self.ax_r, label='radius ', valmin=0, valmax=5.0, valinit=1.0,
                                  valfmt=' %1.1f cm', facecolor='#cc7000')
        self.tube_length = Slider(ax=self.ax_l, label='length ', valmin=0, valmax=10,
                             valinit=5, valfmt='%1.1f cm', facecolor='#cc7000')
        self.stenosis_position = Slider(ax=self.ax_s, label='stenosis position ', valmin=-5, valmax=5,
                                   valinit=0, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.stenosis_radius_proportion = Slider(ax=self.ax_d, label='stenosis gradient ', valmin=0, valmax=1,
                                            valinit=0, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.stenosis_expansion = Slider(ax=self.ax_w, label='stenosis spread ', valmin=0.000001, valmax=5,
                                    valinit=1, valfmt=' %1.1f cm', facecolor='#cc7000')
        self.play_button = Button(ax=self.ax_start_button, label="Play", color='gray')
        self.load_button = Button(ax=self.ax_load_button, label="Load Flow Profile [in cgs]", color='gray')

        self.x = np.linspace(-self.tube_length.val / 2, self.tube_length.val / 2, self.number_sections.val)

        self.vessel = self.tube(self.tube_base_radius.val, self.tube_length.val, self.number_sections.val)
        self.f_d1, = self.ax.plot(self.x, self.vessel, linewidth=2.5, color='k')
        self.f_d2, = self.ax.plot(self.x, -self.vessel, linewidth=2.5, color='k')
        self.f_d1.axes.set_ylim(-1.2 * 5, 1.2 * 5)
        self.f_d2.axes.set_ylim(-1.2 * 5, 1.2 * 5)
        self.f_d2.axes.set_xlim(-1.2 * 5, 1.2 * 5)
        self.number_sections.on_changed(self.stenosis)
        self.tube_base_radius.on_changed(self.stenosis)
        self.tube_length.on_changed(self.stenosis)

        self.stenosis_position.on_changed(self.stenosis)

        self.stenosis_radius_proportion.on_changed(self.stenosis)

        self.stenosis_expansion.on_changed(self.stenosis)
        self.play_button.on_clicked(self.start_program)
        self.load_button.on_clicked(self.load_flow_input_file)
        plt.show()

    def tube(self, r, l, N):
        x = np.linspace(-l / 2, l / 2, N)
        return 0 * x + r

    def load_flow_input_file(self, val):
        self.filename = fd.askopenfilename()

    def handle_close(evt):
        print('Closed Figure!')

    def start_program(self, val):
        self.tube_base_radius = self.stenosis(2)
        self.tube_length = self.tube_length.val
        self.number_sections = self.number_sections.val
        self.filename = self.filename
        plt.close()
        return

    def stenosis(self, val):
        self.stenosis_position.valmin = -self.tube_length.val/2
        self.stenosis_position.valmax = self.tube_length.val / 2
        self.stenosis_expansion.valmax = self.tube_length.val
        xs = np.linspace(-self.tube_length.val / 2, self.tube_length.val / 2, int(self.number_sections.val))
        absolute_stenosis_depth = self.tube_base_radius.val * self.stenosis_radius_proportion.val
        vessel = self.tube_base_radius.val * np.ones(len(xs)) - absolute_stenosis_depth * np.exp(
            -0.5 * (xs - self.stenosis_position.val) ** 2 / self.stenosis_expansion.val ** 2)
        self.f_d1.set_data(xs, vessel)
        self.f_d2.set_data(xs, -vessel)
        self.fig.canvas.draw_idle()
        return vessel

if __name__ == '__main__':
    loadrun = GeometryAndInput()
    min_pressure, max_pressure = run_simulation(loadrun.tube_base_radius, loadrun.tube_length, int(loadrun.number_sections), input_filename=loadrun.filename)
    min_pressure, max_pressure = run_simulation(loadrun.tube_base_radius, loadrun.tube_length,
                                                 int(loadrun.number_sections), input_filename=loadrun.filename, demo = True, min_pressure = min_pressure, max_pressure = max_pressure)