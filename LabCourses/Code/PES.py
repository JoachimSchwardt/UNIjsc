"""
Skript zur Auswertung der Photoelektronenspektroskopie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.filters import uniform_filter1d
from matplotlib import rcParams
rcParams["figure.dpi"] = 100

NA = 6.02214076 * 10**(23)          # 1/mol
c = 299762458                       # m/s
u = 1.66 * 10**(-27)                # kg
e = 1.602176634 * 10**(-19)         # As
eps0 = 8.8541878128 * 10**(-12)     # As/Vm

h = 6.62607015 * 10**(-34)          # J s
hbar = 1.054571817 * 10**(-34)      # J s

me = 9.1093837015 * 10**(-31)       # kg
meev = me * c**2 / e                # ev
alpha = e**2/(4*np.pi*eps0*hbar*c)  # Feinstrukturkonstante

def plot_params(ax, xlabel='x', ylabel='y', title='title', vline=None,
                hline=None, gridlines=True, Adjust=0, xlog=0, ylog=0, 
                xlim=None, ylim=None):
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')    
    if vline != None:
        ax.axvline(vline, lw=0.8, c='k')
    if hline != None:
        ax.axhline(hline, lw=0.8, c='k')
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.grid(gridlines)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=15)
    if Adjust:
        plt.subplots_adjust(bottom=0.15)
    plt.show()
    pass

data = np.loadtxt("pes.dat", skiprows=0, unpack=True)
E_val = data[0]
Nrows = len(data)
angle_step = 0.3
angles = np.arange(0, Nrows, angle_step) - 7 * angle_step
angles = np.round(angles, decimals=1)

def upd_guess(y, indizes, size):
    for k, ind in enumerate(indizes):
        yrange = y[ind - size:ind + size + 1]
        delta_ind = np.arange(-size, size+1, 1)[yrange == max(yrange)]
        indizes[k] = delta_ind + ind
    
    return indizes

# for i in range(30, 38, 1):
#     fig, ax = plt.subplots(1, 1, figsize=(18, 10))
#     y = data[i]
#     ax.plot(E_val, y, lw=1.5, c='k', label=i)
#     y_filt = uniform_filter1d(y, size=15)
#     # ax.plot(E_val, y_filt, c='b', lw=1.5)
#     # ax.plot(E_val, y - y_filt, c='g', lw=1.5)
    
#     ind, _ = find_peaks(y - y_filt, prominence=1.5)
#     ind = upd_guess(y, ind, 2)
#     ax.plot(E_val[ind], y[ind], ls='', c='b', marker='x', ms=8, mew=1)
    
#     plot_params(ax, r'$E_{kin}\ /\ $eV', 'Intensität', 
#                 xlim=[E_val[0], E_val[-1] + 0.3])


# Energien der Peaks, steigende Winkel
E_ans = [[47.31, 47.85, 48.435, 48.81, 49.155, 49.725, 50.325], 
         # -1.5
         [47.325, 47.85, 48.45, 48.9, 49.275, 49.725, 50.325],
         [47.295, 47.865, 48.435, 48.69, 48.975, 49.365, 49.755, 50.325],
         [47.325, 47.895, 48.45, 48.78, 49.02, 49.53, 49.785, 49.98, 50.31],
         # -0.6
         [47.31, 47.895, 48.45, 48.84, 49.065, 49.59, 49.8, 
          50.01, 50.175, 50.325],
         [47.295, 47.895, 48.435, 48.855, 49.08, 49.62, 49.83, 
          50.055, 50.235, 50.325],
         [47.295, 47.865, 48.435, 48.87, 49.11, 49.62, 49.86, 
          50.07, 50.235, 50.325],
         # 0.3
         [47.31, 47.895, 48.435, 48.855, 49.095, 49.62, 49.83, 
          50.07, 50.235, 50.325],
         [47.31, 47.895, 48.435, 48.825, 49.065, 49.59, 49.83, 
          50.04, 50.235, 50.325],
         [47.31, 47.895, 48.435, 48.75, 49.05, 49.545, 49.815, 
          49.965, 50.325],
         # 1.2
         [47.28, 47.865, 48.435, 48.99, 49.47, 49.77, 50.325],
         [47.295, 47.88, 48.45, 48.93, 49.335, 49.74, 50.325],
         [47.34, 47.85, 48.435, 48.855, 49.74, 50.325],
         # 2.1
         [47.355, 47.85, 48.42, 48.75, 49.395, 49.725, 50.325],
         [47.355, 47.85, 48.42, 48.63, 49.245, 49.725, 50.325],
         [47.37, 47.655, 48.495, 49.065, 49.725, 50.325],
         # 3.0
         [47.34, 47.625, 48.435, 48.975, 49.725, 50.325],
         [47.325, 47.625, 48.375, 48.87, 49.725, 50.325],
         [47.28, 47.58, 48.3, 48.465, 48.795, 49.02, 49.725, 50.325],
         # 3.9
         [47.22, 47.49, 48.24, 48.435, 48.72, 49.02, 49.725, 50.325],
         [47.13, 47.43, 48.165, 48.405, 48.69, 49.02, 49.725, 50.325],
         [47.085, 47.34, 48.09, 48.39, 48.57, 49.02, 49.725, 50.325],
         # 4.8
         [47.22, 48.03, 48.39, 48.555, 49.02, 49.71, 50.325],
         [47.19, 47.37, 48.0, 48.39, 48.54, 49.035, 49.725, 50.325],
         [47.13, 47.355, 47.985, 48.375, 48.54, 49.035, 49.725, 50.34],
         # 5.7
         [47.205, 48.0, 48.375, 48.54, 49.035, 49.74, 50.34],
         [47.19, 47.985, 48.405, 48.555, 49.035, 49.725, 50.34],
         [47.13, 47.64, 47.94, 48.405, 48.6, 49.035, 49.725, 50.34],
         # 6.6
         [47.01, 47.64, 48.42, 48.63, 49.035, 49.74, 50.34],
         [47.67, 48.42, 48.66, 49.035, 49.74, 50.34],
         [47.805, 48.42, 48.675, 49.035, 49.74, 50.34],
         # 7.5
         [47.835, 48.435, 48.675, 49.035, 49.74, 50.115, 50.34],
         [47.895, 48.495, 48.69, 49.035, 49.74, 50.01, 50.34],
         [48.495, 48.69, 49.035, 49.755, 50.355],
         # 8.4
         [48.48, 48.72, 49.035, 49.755, 50.355],
         [48.495, 48.735, 49.035, 49.74, 50.355],
         [48.495, 48.75, 49.035, 49.74, 50.355]]

## Korrekturen
# 0.6: 50.235
# 3.0: 49.05 -> 48.975
# 3.9: 47.22
# 4.2: 47.43
# 4.5: 47.34
# 4.8: 47.22
# 5.1: 47.19, 47.37
# 5.4: 47.13
# 5.7: 47.205
# 6.0: 47.19
# 6.3: 47.13
# 6.6: 47.01
# 7.2: 47.805
# 7.5: 47.835
# 7.8: 47.895


fig, ax = plt.subplots(1, 1, figsize=(12, 10))
for i in range(1, Nrows, 1):
    y = data[i]
    ax.text(E_val[-1] + 0.05, y[-15] - 4, 
            f'{angles[i]}' + r' $\degree$', fontsize=12)
    ax.plot(E_val, y, lw=1.5, c='k')
    for E in E_ans[i-1]:
        ax.plot(E, y[E_val == E], ls='', marker='x', 
                ms=8, mew=1, c='b')
    
    # # detect peaks (values are already written into E_ans - Array)
    # y_filt = uniform_filter1d(y, size=15)
    # ind, _ = find_peaks(y - y_filt, prominence=1.5)
    # ind = upd_guess(y, ind, 2)
    # ax.plot(E_val[ind], y[ind], ls='', c='b', marker='x', ms=8, mew=1)
    # print([float(str(val)) for val in E_val[ind]])
    
plot_params(ax, r'$E_{kin}\ /\ $eV', 'Intensität', 
            xlim=[E_val[0], E_val[-1] + 0.3])


