import matplotlib.pyplot as plt
import numpy as np

def interpolate_to_n_points(array, n_points):
       import scipy.interpolate
       interpolant = scipy.interpolate.interp1d(np.linspace(0, 1, len(array)), array)
       return interpolant(np.linspace(0, 1, n_points))

def load_p_vel(dir):
       import os.path
       print('axialvel path', repr(os.path.join(dir, 'axial_vel.csv')))
       vz = np.loadtxt(os.path.join(dir, 'axial_vel.csv'), delimiter=',', skiprows=1)[:,2]
       p = np.loadtxt(os.path.join(dir, 'axial_p.csv'), delimiter=',', skiprows=1)[:,0]
       # p cgs -> p mks (Pa): divide by 10
       # vz axial cgs -> v avg mks: divide by 100 (cgs -> mks), divide by 2 (vmax w/ parabolic flow profile -> vavg)
       return p / 10, vz / (2 * 100)

def get_p_vel(dir, n_points):
       """Gets the pressure and velocity that are comparable to PH simulation results"""
       p, vel = load_p_vel(dir)
       return interpolate_to_n_points(p, n_points), interpolate_to_n_points(vel, n_points)

def get_p_vel_PH(dir, timepoint):
       arr = np.load(dir, allow_pickle=True)[()]
       return arr.stat_p[:, timepoint], arr.H_i_sec[:, timepoint]

def get_A_c(dir, timepoint):
    arr = np.load(dir, allow_pickle=True)[()]
    return arr.A_sec[:, timepoint], arr.c_sec


def compare_ph_sv(description, path_ph, path_sv):
       p_sv, vel_sv = get_p_vel(path_sv, 31)
       p_ph, vel_ph, A = get_p_vel_PH(path_ph, 499)
       # plt.figure()
       # plt.title('p ' +description)
       # plt.plot(np.diff(p_sv)/np.diff(p_ph), label = "sv/ph")
       plt.figure()
       plt.title("p " + description)
       plt.plot(p_sv, label="sv")
       plt.plot(p_ph, label="ph")
       print('diff error', np.mean(np.diff(p_sv[1:])/np.diff(p_ph[1:])))
       print('abs error', np.mean(p_sv/p_ph))
def compare_ph_sv_vel(description, path_ph, path_sv):
       p_sv, vel_sv = get_p_vel(path_sv, 31)
       p_ph, vel_ph, A = get_p_vel_PH(path_ph, 499)
       plt.figure()
       plt.title("vel " + description)
       plt.plot(vel_sv, label = "sv")
       plt.plot(vel_ph, label = "ph")


N = '_31.npy'
path_folder = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\paper_abstract_stenosis'
add = '\solQ'
stenosis = '50_'
radius = "_0.003"
l = 3* 10 ** -2/61
print(3* 10 ** -2/61)
ph_results_paths = {
    "straight_radi": {
        "0.1": path_folder + stenosis + str(0.001) + N,
        "0.2": path_folder + stenosis + str(0.002) + N,
        "0.3": path_folder + stenosis + str(0.003) + N,
    },
    "stenosis_rad": {
            "0.1": path_folder + stenosis + str(0.001) + N,
            "0.2": path_folder + stenosis + str(0.002) + N,
            "0.3": path_folder + stenosis + str(0.003) + N,
        },
    "stenosis_grad": {
                "50": path_folder+add + str(50) + str(radius) + N,
                "60": path_folder+add + str(40) + str(radius)  + N,
                "70": path_folder+add + str(30) + str(radius)  + N,
            }
}


#GEOMETRY = "straight"
GEOMETRY = "stenosis_grad"

sv_results_paths = {
       "straight": {
              "0.1": r'C:\Users\Faiza\simvascular_simulation\straight_radius01',
              "0.2": r'C:\Users\Faiza\simvascular_simulation\straight_radius02',
              "0.3": r'C:\Users\Faiza\simvascular_simulation\straight_radius03',
       },
        "stenosis_rad": {
                      "0.1": r'C:\Users\Faiza\simvascular_simulation\stenosis_50percent',
                      "0.2": r'C:\Users\Faiza\simvascular_simulation\stenosis_doubleradius',
                      "0.3": r'C:\Users\Faiza\simvascular_simulation\stenosis_tripleradius',
       },
       "stenosis_grad": {
              "50": r'C:\Users\Faiza\simvascular_simulation\paper_radius003_stenosis50',
              "60": r'C:\Users\Faiza\simvascular_simulation\paper_radius003_stenosis60',
              "70": r'C:\Users\Faiza\simvascular_simulation\paper_radius003_stenosis70',
       }
}

# stenosis 50
plt.figure()
for sten in ["50", "60", "70"]:
    path_ph = ph_results_paths["stenosis_grad"][sten]
    A,c = get_A_c(path_ph, 499)
    l = np.linspace(0, 0.03, 31)

    plt.plot(l,c)
    plt.plot(l,-c)
    plt.xlabel("segment position along vessel length [m]")
    plt.ylabel("radius of vessel [m]")
plt.savefig(path_folder+'/tube.pdf')

p_sv, vel_sv = get_p_vel(path_sv, 31)
p_ph, vel_ph, A = get_p_vel_PH(path_ph, 499)
