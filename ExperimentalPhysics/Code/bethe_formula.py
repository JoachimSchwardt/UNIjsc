#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://link.springer.com/article/10.1186/2251-7235-6-3/tables/1
https://www-nds.iaea.org/amdc/ame2016/mass16.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=1)


def bethe(Z, A, aV=15.8, aS=-18.3, aC=-0.714, aA=-23.2, delta=12.0):
    """Computes the Bethe-Weizsäcker semi-empirical mass formula"""
    N = A - Z
    volume = aV * A
    surface = aS * A**(2/3)
    coulomb = aC * Z**2 * A**(-1/3)
    asymmetry = aA * (N - Z)**2 / A
    pairing = delta * A**(-1/2)
    indx = (A % 2 != 0)
    pairing[indx] = 0.0
    indx = ((A % 2 == 0) & (N % 2 != 0))
    pairing[indx] *= -1        
    
    B = volume + surface + coulomb + asymmetry + pairing
    return B / A

def get_data(slines):
    element_indx = np.zeros(len(slines))
    atomic_numbers = np.zeros((len(slines), 3))   # N, Z, A
    binding = np.zeros(len(slines))               # E_B / A in keV
    for i, line in enumerate(slines):
        split = line.split()
        for el_indx, char in enumerate(split):
            try:
                int(char)
            except ValueError:
                break
        element_indx[i] = el_indx
        atomic_numbers[i] = split[el_indx-3:el_indx]
        try:
            offset = 0
            try:
                float(split[el_indx+1])
            except ValueError:
                offset = 1
                
            binding[i] = float(split[el_indx + 3 + offset])
        except ValueError:
            binding[i] = np.inf
        
    indx = (binding == np.inf)
    return atomic_numbers[~indx], binding[~indx]

def arghist(array, length, max_estimate=50):
    bucket_count = int(np.max(array) // length) + 1
    indx_array = np.zeros((bucket_count, max_estimate), dtype=int)
    indx_table = np.zeros(bucket_count, dtype=int)
    for i in range(array.shape[0]):
        indx = int(array[i] // length)
        indx_array[indx, indx_table[indx]] = i
        indx_table[indx] += 1
    return indx_array, indx_table

def get_plot_data(atomic, EBE):
    N, Z, A = atomic.T
    
    array, table = arghist(A, 1, 100)
    indx = np.zeros(table.shape, dtype=int)
    for i in range(table.shape[0]):
        ebe_val = EBE[array[i, :table[i]]]
        try:
            indx_val = np.argmax(ebe_val)
            indx[i] = array[i, indx_val]
        except ValueError:
            indx[i] = -1
            continue
    
    indx = indx[indx > 0]
    return N[indx], Z[indx], A[indx], EBE[indx]

def main():
    # with open("experimental-binding-energy-data.txt") as f:
    #     lines = f.readlines()[1:]    # remove header
    #     data = np.array([float(value) for line in lines 
    #                      for value in line.split(" \t")[1:]])
    #     data = data.reshape((-1, 5))
        
    # # Protons, Nucleons, energy (model1), energy (LDM), energy exp.  
    # Z, A, EBM, EBL, EBE = data.T    
    # N = A - Z
    
    with open("nds-iaea-amdc-ame2016-mass-table.txt") as f:
        lines = f.readlines()[39:]    # remove header
        atomic, EBE = get_data(lines)
        # N, Z, A = atomic.T
        N, Z, A, EBE = get_plot_data(atomic, EBE)
        EBE *= 1e-3    # convert to MeV
        
        
    aV = 15.75
    aS = -17.8
    aC = -0.711
    aA = -23.7
    delta = 11.18
    
    magic_numbers = [28, 50, 82, 126]
    magic_AZ = []
    magic_AN = []
    
    for i in range(A.shape[0]):
        if Z[i] in magic_numbers:
            magic_AZ.append(i)
        if N[i] in magic_numbers:
            magic_AN.append(i)
            
    _, indx_AZ = np.unique(Z[magic_AZ], return_index=True)
    _, indx_AN = np.unique(N[magic_AN], return_index=True)
    magic_AZ = np.array(magic_AZ)[indx_AZ]
    magic_AN = np.array(magic_AN)[indx_AN]
    
    model = bethe(Z, A, aV, aS, aC, aA, delta)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$A$")
    ax.set_ylabel(r"$\frac{E_B\,/\,A}{\mathrm{MeV}}$")
    ax.axis([A.min(), A.max(), 7.3, 9.1])
    
    ax.plot(A, EBE, ls='', marker='o', c='b', label='data')
    ax.plot(A, model, c='k', lw=0.8, label='model')
    
    eps = 0.15
    for i in magic_AN:
        ha = 'center'
        xytext = [A[i], EBE[i] + eps]
        if N[i] == 28:
            ha = 'right'
        if N[i] == 126:
            ha = 'left'
        ax.annotate(f'N={int(N[i])}', 
                    xy=(A[i], EBE[i] + 0.25*eps), 
                    xycoords='data',
                    xytext=xytext, 
                    textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.0, 
                                    headlength=0.7, headwidth=1.5, width=0.2),
                    horizontalalignment=ha, 
                    verticalalignment='bottom',)
    for i in magic_AZ:
        ha = 'center'
        xytext = [A[i], EBE[i] + eps]
        if Z[i] == 28:
            ha = 'center'
        if Z[i] == 82:
            xytext[1] += 0.12
        # if i in magic_AN:
        #     ax.text(A[i], EBE[i] + eps + 0.1, f'Z={int(Z[i])}', 
        #             transform=ax.transData,
        #             horizontalalignment='center', 
        #             verticalalignment='bottom',)
        #     continue        
        ax.annotate(f'Z={int(Z[i])}', 
                    xy=(A[i], EBE[i] + 0.25*eps), 
                    xycoords='data',
                    xytext=xytext, 
                    textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.0, 
                                    headlength=0.7, headwidth=1.5, width=0.2),
                    horizontalalignment='center', 
                    verticalalignment='bottom',)
    
    ax.legend()
    special.polish(fig, ax)
    return 0


if __name__ == "__main__":
    print(__doc__)
    main()