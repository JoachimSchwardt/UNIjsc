"""
Created on Sun Dec 20 16:38:26 2020

@author: Joachim
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

with open('VAK2_Auswertung.csv', 'r') as data:
    reader = csv.reader(data)
    rows = list(reader)
    data = np.zeros((31, 16))
    for i in range(31):
        for j in range(16):
            try:
                data[i, j] = float(rows[i+2][j])
            except ValueError:
                pass
            
error_data = np.zeros_like(data)
lp, lh = [1, 8, 18, 28], [5, 15, 25]
error_data[lp, 1] = [2*1e-7, 1e-6, 1e-5, 1e-4]
error_data[lp, 2] = [3*1e-7, 2*1e-6, 3*1e-5, 2*1e-4]
error_data[lp, 3] = [5*1e-6, 5*1e-6, 3*1e-5, 3*1e-4]
error_data[lp, 4] = [1e-6, 3*1e-6, 3*1e-5, 3*1e-4]
error_data[lh, 5] = [2*1e-7, 2*1e-6, 4*1e-5]
error_data[lh, 6] = [5*1e-6, 5*1e-6, 3*1e-5]
error_data[lh, 7] = [5*1e-7, 3*1e-6, 3*1e-5]

error_data[lh, 0] = [5*1e-7, 5*1e-6, 5*1e-5]
error_data[lh, 8] = [5*1e-7, 5*1e-6, 5*1e-5]
for i in range(1, 5, 1):
    error_data[lp, i+8] = error_data[lp, i]
for i in range(5, 8, 1):
    error_data[lh, i+8] = error_data[lh, i]
# eps = 1e-16
# for i in range(16):
#     if i in [1, 8]:
#         error_data[:, i] = 0.1 * 10**(np.floor(np.log10(data[:, i] + eps)))
#     else:
#         error_data[:, i] = 0.2 * 10**(np.floor(np.log10(data[:, i] + eps)))
#     if i in [3, 6, 11, 14]:
#         a = error_data[:, i]
#         error_data[np.where(a <= 2*1e-5), i] = 2*1e-5

# error_data = 0.05 * data
# error_data[:, [2, 5, 10, 13]] *= 4
# error_data[:, [3, 6, 11, 14]] *= 5
# error_data[:, [4, 7, 12, 15]] *= 2

def plot_data(dct):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title(dct['label'], fontsize=20)
    ax.errorbar(data[:, dct['Rref']], data[:, dct['LR']], 
                yerr=error_data[:, dct['LR']], marker='D', mew=1, 
                xerr=error_data[:, dct['Rref']], ms=4, c='g',
                label='LuftRomstedt', lw=0.9, capsize=2)
    ax.errorbar(data[4:, dct['Rref']], data[4:, dct['HR']], 
                yerr=error_data[4:, dct['HR']], marker='D', mew=1, 
                xerr=error_data[4:, dct['Rref'] - 1], ms=4, c='purple',
                label='HeliumRomstedt', lw=0.9, capsize=2)
    ax.errorbar(data[1:, dct['Sref']], data[1:, dct['LS']],
                yerr=error_data[1:, dct['LS']], mew=1, ms=7, 
                xerr=error_data[1:, dct['Sref']], c='b', marker='x', 
                label='LuftSchwardt', lw=0.9, capsize=2)
    ax.errorbar(data[4:, dct['Sref']], data[4:, dct['HS']], 
                yerr=error_data[4:, dct['HS']], marker='x', mew=1, 
                xerr=error_data[4:, dct['Sref'] - 1], ms=7, c='r',
                label='HeliumSchwardt', lw=0.9, capsize=2)
    ax.plot(data[:, dct['Rref']], data[:, dct['Rref']], lw=1, c='k',
            label='Referenz P3')
    ax.grid(True)
    ax.set_xlabel(r'$\log_{10} (P3\ /\ mbar)$', fontsize=18)
    ax.set_ylabel(r'$\log_{10} P\ /\ mbar$', fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=20)
    plt.show()

def main():
    P2dct = {'label' : 'Kaltkathoden-VM P2', 'Sref' : 1, 'Rref' : 9,
             'LS' : 2, 'LR' : 10, 'HS' : 5, 'HR' : 13}
    PXdct = {'label' : 'Glühkathoden-VM PX', 'Sref' : 1, 'Rref' : 9,
             'LS' : 3, 'LR' : 11, 'HS' : 6, 'HR' : 14}
    P9dct = {'label' : 'Kaltkathoden-VM P9', 'Sref' : 1, 'Rref' : 9,
             'LS' : 4, 'LR' : 12, 'HS' : 7, 'HR' : 15}
    plot_data(P2dct)
    plot_data(PXdct)
    plot_data(P9dct)
    
def main2():
    crel_inds = [5, 15, 25]
    crel = np.zeros((3, 6))
    for i, inds in enumerate(crel_inds):
        crel[i, 0:3] = data[inds, 2:5] / data[inds, 5:8]
        crel[i, 3:6] = data[inds, 10:13] / data[inds, 13:16]
    print(crel) 
    counter, cvals = 0, []
    for i, cval in enumerate(crel.flatten()):
        if i in [6, 8, 9, 11, 13, 14, 16, 17]:
            counter += 1
            cvals.append(cval)
    print(cvals)
    crelavg = sum(cvals)/counter
    csigma = np.sqrt(np.sum((cvals - crelavg)**2) / (counter - 1))
    print(crelavg, csigma)
    print(1/0.17)
    
def main3():
    L, deltaL = 22, 1
    D, deltaD = 1.5, 0.1
    V3, deltaV3 = 63.5, 0.7
    V2geo = L * np.pi/4 * D**2
    p1, deltap1 = np.array([2.65, 4.8, 5]), 0.1
    p3, deltap3 = np.array([0.00588, 0.00748, 0.00718]), 0.0002
    p3v = np.array([1e-5, 1.4*1e-5, 3*1e-6])
    deltaV2geo = V2geo * np.sqrt((deltaL/L)**2 + (2*deltaD/D)**2)
    print(V2geo, deltaV2geo)
    V2 = 1000 * V3 * (p3 - p3v) / (p1 - p3)
    deltaV2 = V2 * np.sqrt((deltaV3/V3)**2 + (deltap1/p1)**2 + 
                           (deltap3/p3)**2)
    print(V2, deltaV2, p1*V2)
    
def main4():
    # ppr = np.array([110, 24, 10, 6.3, 4.1, 3.9, 1]) * 1e-11
    # pps = np.array([81, 47, 6.5, 54, 3.3, 17, 7.3]) * 1e-11
    # ptotr, ptots = 1.68 * 1e-9, 2.2 * 1e-9
    # print(sum(ppr), ptotr, sum(pps), ptots)
    # print(ppr / sum(ppr))
    # print(pps / sum(pps))
    
    ppr = np.array([880, 140, 40, 9.2, 6.5, 6, 4.8, 1.8, 1]) * 1e-9
    pps = np.array([2700, 470, 120, 27, 26, 21, 4.7]) * 1e-9
    ptotr, ptots = 1.12 * 1e-6, 3.8 * 1e-6
    print(sum(ppr), ptotr, sum(pps), ptots)
    print(ppr / sum(ppr))
    print(pps / sum(pps))
    
def main5():
    Vrez = 12.6
    S = 100
    T = 60 * np.array([60, 35])
    Xar, Yar = np.array([0.037, 0.036]), np.array([0.66, 0.55])
    PE, PG = np.array([1.12, 3.8]) * 1e-6, np.array([0.574, 2]) * 1e-6
    Qauf = Xar * S * T * PE
    Qab = Yar * PG * Vrez
    print(Qauf, Qab, Qab/Qauf)

if __name__ == "__main__":
    # main()
    # main2()
    # main3()
    # main4()
    main5()
    # print(0.48*25 / (0.37*40) * 0.42 * 1.3 / 8)
    # print(6.5*0.77 / (18*0.9) * 0.8 * 0.16 / 0.53)
    # print(2.5*0.77 / 17.5 * 60 / 8)
    # print(0.25*3 / 1.3)