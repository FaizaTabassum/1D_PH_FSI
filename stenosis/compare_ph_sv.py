import matplotlib.pyplot as plt
import numpy as np
import numpy
import numpy as np
from numpy import save
from sympy import symbols
import functions as func
import timeit
from scipy.integrate import solve_ivp
from profilehooks import profile
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
import matplotlib.cm as cm

def interpolate_to_n_points(array, n_points):
    import scipy.interpolate
    interpolant = scipy.interpolate.interp1d(np.linspace(0, 1, len(array)), array)
    return interpolant(np.linspace(0, 1, n_points))


def load_p_vel(dir):
    import os.path
    print('axialvel path', repr(os.path.join(dir, 'axial_vel.csv')))
    vz = np.loadtxt(os.path.join(dir, 'axial_vel.csv'), delimiter=',', skiprows=1)[:, 2]
    p = np.loadtxt(os.path.join(dir, 'axial_p.csv'), delimiter=',', skiprows=1)[:, 0]
    # p cgs -> p mks (Pa): divide by 10
    # vz axial cgs -> v avg mks: divide by 100 (cgs -> mks), divide by 2 (vmax w/ parabolic flow profile -> vavg)
    return p / 10, vz / (2 * 100)


def get_p_vel(dir, n_points):
    """Gets the pressure and velocity that are comparable to PH simulation results"""
    p, vel = load_p_vel(dir)
    return interpolate_to_n_points(p, n_points), interpolate_to_n_points(vel, n_points)


def get_p_vel_PH(dir, timepoint):
    arr = np.load(dir, allow_pickle=True)[()]
    return arr.stat_p[:, timepoint], arr.H_i_sec[:, timepoint], arr.A_sec[:, timepoint]

def compare_ph_sv(description, path_ph, path_sv, n_sections):
    p_sv, vel_sv = get_p_vel(path_sv, n_sections)
    p_ph, vel_ph, A = get_p_vel_PH(path_ph, 499)
    plt.figure()
    plt.title(description)
    plt.plot(np.diff(p_sv) / np.diff(p_ph), label= "diffsv/diffph")
    plt.legend()
    plt.figure()
    plt.title(description)
    plt.plot(p_sv, label="sv")
    plt.plot(p_ph, label="ph")
    plt.legend()
    plt.figure()
    plt.title(description)
    plt.plot(p_sv/p_ph, label="psv/pph")
    plt.legend()
    print(np.abs(p_sv[2]-p_ph[2]))


def compare_ph_sv_vel(description, path_ph, path_sv, n_sections):
    p_sv, vel_sv = get_p_vel(path_sv, n_sections)
    p_ph, vel_ph, A = get_p_vel_PH(path_ph, 499)
    print(description)
    print('velph', vel_ph[15])
    print('velfem', vel_sv[15])
    plt.figure()
    plt.title("vel " + description)
    # plt.title(description + ' vel')
    plt.plot(vel_sv, label="sv")
    plt.plot(vel_ph, label="ph")
    # plt.figure()
    # plt.title("flow rate ph")
    # plt.plot(vel_ph*A)

    # plt.plot(np.diff(p_sv) / np.diff(p_ph), label = description)


N1 = 31
N2 = 31
N3 = 31

stenosis = '0_'
radius = "_0.003"
R1 = "0.001"
R2 = "0.002"
R3 = "0.003"
L1 = "0.03"
L2 = "0.04"
L3 = "0.05"
S1 = "50"
S2 = "40"
S3 = "30"
S4 = "100"
l = 3 * 10 ** -2 / 61
print(3 * 10 ** -2 / 61)
ph_results_path = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\analysis\stenosis'
ph_results_path_straight = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\analysis\straight'
sv_results_path = r'C:\Users\Faiza\simvascular_simulation'
ph_results_paths = {
    "straight_rad": [
        f'{ph_results_path_straight}\straight_radius{R1}_stenosis{S4}_length{L1}_sec{N1}.npy',
        f'{ph_results_path_straight}\straight_radius{R2}_stenosis{S4}_length{L1}_sec{N1}.npy',
        f'{ph_results_path_straight}\straight_radius{R3}_stenosis{S4}_length{L1}_sec{N1}.npy',
    ],
    "straight_length": [
        f'{ph_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\straight_radius{R3}_stenosis{S1}_length{L2}_sec{N1}.npy',
        f'{ph_results_path}\straight_radius{R3}_stenosis{S1}_length{L3}_sec{N1}.npy',
    ],
    "straight_N_sec": [
        f'{ph_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N2}.npy',
        f'{ph_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N3}.npy',
    ],
    "stenosis_rad": [
        f'{ph_results_path}\stenosis_radius{R1}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R2}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}.npy',
    ],
    "stenosis_length": [
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L2}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L3}_sec{N1}.npy',
    ],
    "stenosis_N_sec": [
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N2}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N3}.npy',
    ],
    "stenosis_grad": [
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S2}_length{L1}_sec{N1}.npy',
        f'{ph_results_path}\stenosis_radius{R3}_stenosis{S3}_length{L1}_sec{N1}.npy',
    ],
}



sv_results_paths = {
    "straight_rad": [
        f'{sv_results_path}\straight_radius{R1}_stenosis{S4}_length{L1}_sec{N1}',
        f'{sv_results_path}\straight_radius{R2}_stenosis{S4}_length{L1}_sec{N1}',
        f'{sv_results_path}\straight_radius{R3}_stenosis{S4}_length{L1}_sec{N1}',
    ],
    "straight_length": [
        f'{sv_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\straight_radius{R3}_stenosis{S1}_length{L2}_sec{N1}',
        f'{sv_results_path}\straight_radius{R3}_stenosis{S1}_length{L3}_sec{N1}',
    ],
    "straight_N_sec": [
        f'{sv_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\straight_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
    ],
    "stenosis_rad": [
        f'{sv_results_path}\stenosis_radius{R1}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R2}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
    ],
    "stenosis_length": [
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L2}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L3}_sec{N1}',
    ],
    "stenosis_N_sec": [
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
    ],
    "stenosis_grad": [
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S1}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S2}_length{L1}_sec{N1}',
        f'{sv_results_path}\stenosis_radius{R3}_stenosis{S3}_length{L1}_sec{N1}',
    ],
}

GEOMETRY_STEN = "stenosis_grad"

print(sv_results_paths[GEOMETRY_STEN][0])
compare_ph_sv("3", ph_results_paths[GEOMETRY_STEN][0], sv_results_paths[GEOMETRY_STEN][0], 31)
compare_ph_sv("3", ph_results_paths[GEOMETRY_STEN][1], sv_results_paths[GEOMETRY_STEN][1], 31)
compare_ph_sv("3", ph_results_paths[GEOMETRY_STEN][2], sv_results_paths[GEOMETRY_STEN][2], 31)
plt.show()





def to_hex(t):
    return '#' + ''.join(['{:02x}'.format(int(255*x)) for x in t[:3]])

GEOMETRY_STEN = "stenosis_grad"
GEOMETRY_straight = "straight_rad"
GEOMETRY_stenosis_rad = "stenosis_rad"

n_sections = 31
p_sv_50_sten, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_STEN][0], n_sections)
p_ph_50_sten, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_STEN][0], 499)
p_sv_40_sten, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_STEN][1], n_sections)
p_ph_40_sten, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_STEN][1], 499)
p_sv_30_sten, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_STEN][2], n_sections)
p_ph_30_sten, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_STEN][2], 499)


GEOMETRY_straight = "straight_rad"
S1 = "100"
n_sections = 31
p_sv_50_stra, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_straight][0], n_sections)
p_ph_50_stra, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_straight][0], 499)
p_sv_40_stra, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_straight][1], n_sections)
p_ph_40_stra, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_straight][1], 499)
p_sv_30_stra, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_straight][2], n_sections)
p_ph_30_stra, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_straight][2], 499)


p_sv_50_sten_rad, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_stenosis_rad][0], n_sections)
p_ph_50_sten_rad, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_stenosis_rad][0], 499)
p_sv_40_sten_rad, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_stenosis_rad][1], n_sections)
p_ph_40_sten_rad, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_stenosis_rad][1], 499)
p_sv_30_sten_rad, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_stenosis_rad][2], n_sections)
p_ph_30_sten_rad, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_stenosis_rad][2], 499)



cmap1 = cm.get_cmap('Oranges', 10)
cmap2 = cm.get_cmap('Blues', 10)
cmap3 = cm.get_cmap('Greens', 10)
sol = np.load(r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\analysis\analytical_solution'+"/solQ.npy", allow_pickle=True)[()]
fig, axs = plt.subplots(2, 2, figsize = (5.8, 5.8 / (4 / 3)), constrained_layout=True)
axs[0,0].plot(np.linspace(0, 0.03, 31), sol.stat_p[:, -1], color = "b", label = "pressure")
axs[0,0].set_ylabel("pressure [Pa]")
axs[0,0].set_xlabel("axial vessel position [m]")
ax1 = axs[0,0].twinx()
ax1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax1.plot(np.linspace(0, 0.03, 31), sol.A_sec[:, -1] * sol.H_i_sec[:, -1], color = "r", label = "flow")
ax1.set_ylabel("flow [m$^3$/s]")
axs[0,0].legend(loc = 'upper left')
ax1.legend(loc = 'upper right')

axs[0,1].plot(np.linspace(0, 0.03, 31), p_sv_50_sten, label = "SV 50%", color = to_hex(cmap1(9)), linestyle = (0, (1, 1)))
axs[0,1].plot(np.linspace(0, 0.03, 31), p_sv_40_sten, label = "SV 60%", color = to_hex(cmap1(6)), linestyle = (0, (2, 3)))
axs[0,1].plot(np.linspace(0, 0.03, 31), p_sv_30_sten, label = "SV 70%", color = to_hex(cmap1(3)), linestyle = (0, (3, 5)))
axs[0,1].plot(np.linspace(0, 0.03, 31), p_ph_50_sten, label = "PH 50%", color = to_hex(cmap1(9)), alpha = 0.6)
axs[0,1].plot(np.linspace(0, 0.03, 31), p_ph_40_sten, label = "PH 60%", color = to_hex(cmap1(6)),  alpha = 0.6)
axs[0,1].plot(np.linspace(0, 0.03, 31), p_ph_30_sten, label = "PH 70%", color = to_hex(cmap1(3)),  alpha = 0.6)
axs[0,1].set_ylabel("pressure [Pa]")
axs[0,1].set_xlabel("axial vessel position [m]")
axs[0,1].legend(loc = 'upper right')

axs[1,0].plot(np.linspace(0, 0.03, 31), p_sv_50_stra, label = "SV 1mm", color = to_hex(cmap2(9)), linestyle = (0, (1, 1)))
axs[1,0].plot(np.linspace(0, 0.03, 31), p_sv_40_stra, label = "SV 2mm", color = to_hex(cmap2(6)), linestyle = (0, (2, 3)))
axs[1,0].plot(np.linspace(0, 0.03, 31), p_sv_30_stra, label = "SV 3mm", color = to_hex(cmap2(3)), linestyle = (0, (3, 5)))
axs[1,0].plot(np.linspace(0, 0.03, 31), p_ph_50_stra, label = "PH 1mm", color = to_hex(cmap2(9)), alpha = 0.6)
axs[1,0].plot(np.linspace(0, 0.03, 31), p_ph_40_stra, label = "PH 2mm", color = to_hex(cmap2(6)),  alpha = 0.6)
axs[1,0].plot(np.linspace(0, 0.03, 31), p_ph_30_stra, label = "PH 3mm", color = to_hex(cmap2(3)),  alpha = 0.6)
axs[1,0].set_ylabel("pressure [Pa]")
axs[1,0].set_xlabel("axial vessel position [m]")
axs[1,0].legend(loc = 'upper right')

axs[1,1].plot(np.linspace(0, 0.03, 31), p_sv_50_sten_rad, label = "SV 1mm", color = to_hex(cmap3(9)), linestyle = (0, (1, 1)))
axs[1,1].plot(np.linspace(0, 0.03, 31), p_sv_40_sten_rad, label = "SV 2mm", color = to_hex(cmap3(6)), linestyle = (0, (2, 3)))
axs[1,1].plot(np.linspace(0, 0.03, 31), p_sv_30_sten_rad, label = "SV 3mm", color = to_hex(cmap3(3)), linestyle = (0, (3, 5)))
axs[1,1].plot(np.linspace(0, 0.03, 31), p_ph_50_sten_rad, label = "PH 1mm", color = to_hex(cmap3(9)), alpha = 0.6)
axs[1,1].plot(np.linspace(0, 0.03, 31), p_ph_40_sten_rad, label = "PH 2mm", color = to_hex(cmap3(6)),  alpha = 0.6)
axs[1,1].plot(np.linspace(0, 0.03, 31), p_ph_30_sten_rad, label = "PH 3mm", color = to_hex(cmap3(3)),  alpha = 0.6)
axs[1,1].set_ylabel("pressure [Pa]")
axs[1,1].set_xlabel("axial vessel position [m]")
axs[1,1].legend(loc = 'upper right')
plt.show()
#plt.savefig(r'C:\Users\Faiza\Documents\WCB_AbstractSubmission' + '/comparison_sv_ph.pdf')





# compare_ph_sv(
#     '0',
#     ph_results_paths[GEOMETRY][0],
#     sv_results_paths[GEOMETRY][0],
#     N1,
# )
#
# compare_ph_sv(
#     '1',
#     ph_results_paths[GEOMETRY][1],
#     sv_results_paths[GEOMETRY][1],
#     N1,
# )
#
# compare_ph_sv(
#     '2',
#     ph_results_paths[GEOMETRY][2],
#     sv_results_paths[GEOMETRY][2],
#     N1,
# )
# plt.legend()

# compare_ph_sv_vel(
#        'r=0.1',
#        path_ph_c1,
#        sv_results_paths[GEOMETRY]["0.1"],
# )
#
# compare_ph_sv_vel(
#        'r=0.2',
#        path_ph_c2,
#        sv_results_paths[GEOMETRY]["0.2"],
# )
#
# compare_ph_sv_vel(
#        'r=0.3',
#        path_ph_c3,
#        sv_results_paths[GEOMETRY]["0.3"],
# )

# p_sv_01, v_sv_01 = get_p_vel(path_sv, 31)
# p_ph_01, v_ph_01 = get_p_vel_PH(path_PH)
#
# #compare for radius 0.2
# path_PH = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\20211211\solQ0.002.npy'
# path_sv = r'C:\Users\Faiza\simvascular_simulation\stenosis_doubleradius'
#
# p_sv_02, v_sv_02 = get_p_vel(path_sv, 31)
# p_ph_02, v_ph_02 = get_p_vel_PH(path_PH)
#
# #compare for radius 0.2
# path_PH = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\20211211\solQ0.003.npy'
# path_sv = r'C:\Users\Faiza\simvascular_simulation\stenosis_doubleradius'
#
# p_sv_03, v_sv_03 = get_p_vel(path_sv, 31)
# p_ph_03, v_ph_03 = get_p_vel_PH(path_PH)


