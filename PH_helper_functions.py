# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:55:42 2021
@author: Faiza
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import matplotlib.colors as colors
import cv2
import os
import glob
import re

def calculate_pressure_matrix(tube_radius,pressure_along_z, number_sections):
    max_radius = np.max(tube_radius)
    for i in range(0, len(tube_radius), 1):
        if i == 0:
            new_vec = np.ones(int((tube_radius[i]/max_radius)*number_sections))*pressure_along_z[i]
            number_add_zeros= int(number_sections-len(new_vec))
            p_vec = np.hstack((np.nan*np.ones(number_add_zeros), new_vec))
        else:
            new_vec = np.ones(int((tube_radius[i] / max_radius) * number_sections)) * pressure_along_z[i]
            number_add_zeros = int(number_sections - len(new_vec))
            p_vec = np.vstack((p_vec, np.hstack((np.nan*np.ones(number_add_zeros), new_vec))))
    p_vec =p_vec.T
    p_vec_flip = np.flip(p_vec, axis=0)
    p_matrix = np.vstack((p_vec, p_vec_flip))
    return p_matrix


def intervals_min_max_pressure(array):
    dx1 = np.diff(array)
    dx2 = np.diff(dx1)
    turnpoints = np.array(np.where(dx2[1:] * dx2[:-1] < 0)).reshape(-1, )  # turnpoints in curvature
    extremalpoints = np.array(np.where(dx1[1:] * dx1[:-1] < 0)).reshape(-1, )  # turnpoints in slope
    if len(turnpoints) != 1:
        left_to_right_ind = np.array([turnpoints[i] for i in range(0, len(turnpoints), 1) if
                                      dx2[turnpoints[i] - 1] > 0 and dx2[turnpoints[i] + 1] < 0])
        right_to_left_ind = np.array([turnpoints[i] for i in range(0, len(turnpoints), 1) if
                                      dx2[turnpoints[i] - 1] < 0 and dx2[turnpoints[i] + 1] > 0])
        local_minimum_ind = np.array([extremalpoints[i] for i in range(0, len(extremalpoints), 1) if
                                      dx1[extremalpoints[i] - 1] < 0 and dx1[extremalpoints[i] + 1] > 0])
        if dx1[0] > 0:
            local_minimum_max_ind = np.hstack((int(0), local_minimum_ind))
        if dx1[-1] ==0:
            last_slope_before_zero = np.argwhere(dx1!=0)[-1][0]
            if dx1[last_slope_before_zero]<0:
                if right_to_left_ind[-1] > left_to_right_ind[-1]:
                    local_minimum_ind = np.hstack((local_minimum_ind, int(len(array)-int(1))))
        pressure_max_start = int(local_minimum_max_ind[np.argmax(left_to_right_ind - local_minimum_max_ind)])  # find the one
        pressure_max_end = left_to_right_ind[np.argmax(left_to_right_ind - local_minimum_max_ind)]  # find the one

        pressure_min_start = right_to_left_ind[np.argmax(local_minimum_ind - right_to_left_ind)]  # find the one
        pressure_min_end = int(local_minimum_ind[np.argmax(local_minimum_ind - right_to_left_ind)]) # find the one
    else:
        if dx1[0] > 0:
            pressure_max_start = 0
            pressure_max_end = turnpoints[0]
            pressure_min_start = None
            pressure_min_end = None
        else:
            pressure_min_start = 0
            pressure_min_end = turnpoints[0]
            pressure_max_start = None
            pressure_max_end = None

    return pressure_min_start, pressure_min_end, pressure_max_start, pressure_max_end

class Model_Flow_Efficient:
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
        self.vis_dissipation_factor = pdic['vis_factor'] * 2 * self.section_length / self.radius

        self.structure_mass = np.multiply(np.pi * pdic['structure_density'] * self.radius,  pdic['wall_thickness'] * self.section_length * np.ones(self.N_sec))
        self.structure_r_dissipation = pdic['structure_r_dissipation'] * np.ones(self.N_sec)
        self.structure_z_dissipation = pdic['structure_z_dissipation'] * np.ones(self.N_sec)  # coupling damping coefficient of structure
        self.node_mass = np.multiply(pdic['fluid_density'] * np.pi * np.square(self.radius),  self.section_length * 10 ** -3 * np.ones(self.N_nodes))


        self.fluid_bulk_modulus = pdic['fluid_bulk_modulus']
        self.t_evaluation = pdic['t_evaluation']
        self.pressure = 0
        self.requested_data_over_tube = pdic['requested_data_over_tube']
        self.requested_data_at_boundaries = pdic['requested_data_at_boundaries']
        self.evaluate_timepoint = False
        self.create_structure_requested_data()
        self.demo = demo
        self.min_pressure = pdic['min_pressure']
        self.max_pressure = pdic['max_pressure']

        if self.demo== True:
            self.path_to_heart_images = path_to_image
            self.heart_image_filenames = self.get_image_file_names()
            self.number_of_image_frames = len(self.heart_image_filenames)
            self.frame_count = 0
            self.sample_time_vis = pdic['sample_time_visualization']

            self.t_sim = self.t_evaluation[::self.sample_time_vis]
            self.initialize_demo_figure()
            self.create_structure_requested_data()

    def create_structure_requested_data(self):
        self.requested_data = {}
        for entry in self.requested_data_over_tube:
            self.requested_data[entry] = np.empty((self.N_sec, len(self.t_evaluation)))
        for entry in self.requested_data_at_boundaries:
            self.requested_data[entry] = np.empty((2, len(self.t_evaluation)))


    def get_image_file_names(self):
        file_list = os.listdir(self.path_to_heart_images)
        file_list = sorted(file_list, key=lambda x: float(re.findall("(\d+)", x)[0]))
        file_list = [os.path.join(self.path_to_heart_images, file_list[i]) for i in range(0, len(file_list),1)]
        return file_list

    def initialize_demo_figure(self):

        random_inital_pressure = np.linspace(self.min_pressure, self.max_pressure, self.N_sec)
        self.pressure_image = calculate_pressure_matrix(self.radius, random_inital_pressure, self.N_ypoints_for_pressure_image)

        self.figure, self.ax = plt.subplots(2,2)
        self.figure.subplots_adjust(wspace=0.5)
        self.pressure_curve, = self.ax[0, 1].plot([], [], color = "red")

        if self.path_to_heart_images == '':
            self.ax[0,0].remove()
        else:
            self.heart = self.ax[0,0].imshow(cv2.imread(self.heart_image_filenames[0]), cmap = plt.get_cmap('Blues'))
        self.pressure_in_tube = self.ax[1,1].imshow(self.pressure_image, extent=[-self.structure_length/2, self.structure_length/2, -np.max(self.radius), np.max(self.radius)], cmap = plt.get_cmap('OrRd'))
        self.figure.colorbar(self.pressure_in_tube, ax=self.ax[1,1], orientation="horizontal", pad=0.2)

        self.structure_lengthines_tube_p = self.ax[1,1].plot(np.linspace(-self.structure_length/2, self.structure_length/2, self.N_sec), self.radius, color = "black")
        self.structure_lengthines_tube_n = self.ax[1, 1].plot(np.linspace(-self.structure_length / 2, self.structure_length / 2, self.N_sec), -self.radius, color = "black")

        self.ax[1, 1].set_xlim(-self.structure_length / 2-self.section_length, self.structure_length / 2+self.section_length)
        self.ax[0, 1].set_autoscaley_on(True)
        self.ax[0, 1].set_xlim(-self.structure_length / 2-self.section_length, self.structure_length / 2+self.section_length)
        self.ax[0, 1].set_ylim(self.min_pressure, self.max_pressure)
        # self.ax[0, 1].set_ylim(self.min_pressure, self.max_pressure)
        self.ax[0, 1].grid()
        self.ax[0,1].set_ylabel('pressure [Pa]')
        self.ax[1,0].set_ylabel('flow input [L/s]')
        self.ax[1,1].set_ylabel('tube geometry [m]')
        self.input_shape, = self.ax[1, 0].plot(self.t_evaluation, self.input_shape, color="blue")
        self.state_line, = self.ax[1, 0].plot([0, 0], [self.input_shape.axes.get_ylim()[0], self.input_shape.axes.get_ylim()[1]], color ="g")


    def on_running(self, xdata, ydata,t, pmatrix, heartimage):
        print('here')
        # Update data (with the new _and_ the old points)
        self.pressure_in_tube.set_data(pmatrix)
        self.heart.set_data(heartimage)
        self.pressure_curve.set_data(xdata, ydata)
        self.state_line.set_data([t, t], [self.input_shape.axes.get_ylim()[0], self.input_shape.axes.get_ylim()[1]])
        # Need both of these in order to rescale
        self.ax[0,1].relim()
        self.ax[0,1].autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw_idle()
        plt.pause(0.1)

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
        if self.evaluate_timepoint == True:
            print('dyn p: ', dynamic_pressure[0])
            print('stat p: ', static_pressure[0])
            self.pressure = np.vstack((self.pressure, static_pressure[0] + dynamic_pressure[0]))
            if self.demo == True:
                self.update_pressure_plot(t, static_pressure, dynamic_pressure)
            self.evaluate_timepoint = False
        return dxdt

    def Jacobi(self, t, x):
        return 0

    def update_pressure_plot(self, t, static_pressure, dynamic_pressure):
        if t in self.t_sim:
            if self.frame_count < len(self.heart_image_filenames)-1:
                self.frame_count += 1
                self.heart_image = cv2.imread(self.heart_image_filenames[self.frame_count])

            else:
                self.frame_count = 0
                self.heart_image = cv2.imread(self.heart_image_filenames[self.frame_count])

            pressure_along_z = np.append(static_pressure, 0) + np.append(dynamic_pressure, 0)
            self.pressure_image = calculate_pressure_matrix(self.radius, pressure_along_z, self.N_ypoints_for_pressure_image)
            self.on_running(np.linspace(-self.structure_length/2, self.structure_length/2, self.N_sec+1),pressure_along_z, t, self.pressure_image, self.heart_image)



