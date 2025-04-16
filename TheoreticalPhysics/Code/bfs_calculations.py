#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:12:29 2023

@author: joachim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import mpl_special


def complex_to_string(zval, dec=0):
    """Convert complex number to string"""
    zval = np.round(zval, decimals=dec)
    string = r""
    if zval.real == 0:
        if zval.imag == 0:
            return r"0"
    elif zval.real < 0:
        string += fr"-{-zval.real}"
    else:
        string += fr"{zval.real}"

    if zval.imag < 0:
        string +=  fr"-{-zval.imag}\i"
    elif zval.imag > 0:
        string +=  fr"{zval.imag}\i"
    if dec == 0:
        string = string.replace(".0", "")
        string = string.replace(r"1\i", r"\i")
    return string


def matrix_to_latex(arr, dec=0):
    string = r"\begin{pmatrix} "
    for row in arr:
        string += r" & ".join([complex_to_string(elem, dec) for elem in row]) + r" \\ "
    string += r" \end{pmatrix}"
    return string


def get_pauli(ind):
    """Pauli-matrices for given index"""
    if ind == 0:
        return np.array([[1, 0], [0, 1]])
    elif ind == 1:
        return np.array([[0, 1], [1, 0]])
    elif ind == 2:
        return np.array([[0, -1j], [1j, 0]])
    elif ind == 3:
        return np.array([[1, 0], [0, -1]])
    else:
        raise ValueError(f"Wrong index {ind}, must be one of 0,1,2,3!")


def get_u_t():
    return np.kron(get_pauli(0), get_pauli(2) * 1j)


def get_u_trans():
    return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def get_j_matrix(ind):
    """Spin 3/2 J_ind matrices"""
    if ind == 0:
        diag = np.array([np.sqrt(3) / 2, 1, np.sqrt(3) / 2])
        return np.diag(diag, k=1) + np.diag(diag, k=-1)
    elif ind == 1:
        diag = np.array([np.sqrt(3) / 2, 1, np.sqrt(3) / 2]) * 1j
        return -np.diag(diag, k=1) + np.diag(diag, k=-1)
    elif ind == 2:
        return np.diag([3, 1, -1, -3]) / 2
    else:
        raise ValueError(f"Wrong index {ind}, must be one of 0,1,2!")


def get_gamma(ind):
    if ind == 0:
        return np.kron(get_pauli(0), get_pauli(0))
    elif ind == 1:
        return np.kron(get_pauli(1), get_pauli(0))
    elif ind == 2:
        return np.kron(get_pauli(3), get_pauli(0))
    elif ind == 3:
        return np.kron(get_pauli(2), get_pauli(1))
    elif ind == 4:
        return np.kron(get_pauli(2), get_pauli(2))
    elif ind == 5:
        return np.kron(get_pauli(2), get_pauli(3))
    else:
        raise ValueError(f"Wrong index {ind}, must be one of 0,1,2,3,4,5!")
        
def get_eta(ind, u_t=None):
    j_x = get_j_matrix(0)
    j_y = get_j_matrix(1)
    j_z = get_j_matrix(2)
    if u_t is None:
        u_trans = get_u_trans()
        u_t = u_trans @ get_u_t() @ u_trans
        
    if ind == 0:
        return u_t
    elif ind == 1:
        return ((j_x@j_x - j_y@j_y) / np.sqrt(3)) @ u_t        #eta x^2-y^2
    elif ind == 2:
        return ((2*j_z@j_z - j_x@j_x - j_y@j_y) / 3) @ u_t     #eta 3z^2-r^2
    elif ind == 3:
        return ((j_y@j_z + j_z@j_y) / np.sqrt(3)) @ u_t        #eta yz
    elif ind == 4:
        return ((j_x@j_z + j_z@j_x) / np.sqrt(3)) @ u_t        #eta xz
    elif ind == 5:
        return ((j_x@j_y + j_y@j_x) / np.sqrt(3)) @ u_t        #eta xy
    else:
        raise ValueError(f"Wrong index {ind}, must be one of 0,1,2,3,4,5!")


def list_pauli_products(mode=None):
    # for mu in range(4):
    #     sigma_mu = get_pauli(mu)
    #     for nu in range(4):
    #         sigma_nu = get_pauli(nu)
    #         arr = np.kron(sigma_mu, sigma_nu)
    #         string = fr"\hat{{s}}_{mu} \otimes \hat{{\sigma}}_{nu} &= "
    #         string += matrix_to_latex(arr, dec=0)
    #         print(string, end=" ")
    #         if nu < 3:
    #             print(", &&&")
    #     if mu < 3:
    #         print(r"\\")
    for mu in range(4):
        sigma_mu = get_pauli(mu)
        print(fr"$\mu = {mu}$ & ", end="")
        for nu in range(4):
            sigma_nu = get_pauli(nu)
            arr = np.kron(sigma_mu, sigma_nu)
            if mode == "U_T":
                u_t = get_u_t()
                arr = u_t @ arr @ u_t.T
            string = matrix_to_latex(arr, dec=0)
            print(f"${string}$", end=" ")
            if nu < 3:
                print("&", end=" ")
            else:
                print(r"\\")


def list_spin_products():
    j_x = get_j_matrix(0)
    j_y = get_j_matrix(1)
    j_z = get_j_matrix(2)
    prod = [(j_x@j_x - j_y@j_y) / (np.sqrt(3)),
            (2*j_z@j_z - j_x@j_x - j_y@j_y) / 3,
            (j_y@j_z + j_z@j_y) / (np.sqrt(3)),
            (j_x@j_z + j_z@j_x) / (np.sqrt(3)),
            (j_x@j_y + j_y@j_x) / (np.sqrt(3)), 
            ]
    u_trans = get_u_trans()
    for ind, product in enumerate(prod):
        print(matrix_to_latex(u_trans.T @ product @ u_trans), 
              matrix_to_latex(get_gamma(ind+1)))
        print("DIFF:", matrix_to_latex(u_trans.T @ product @ u_trans-get_gamma(ind+1)))


def list_projection_hamiltonian():
    j_list = [get_j_matrix(ind) for ind in range(3)]
    eta = [get_eta(ind, u_t=np.eye(4)) for ind in range(6)]
    j_name_list = ["j_x", "j_y", "j_z"]
    for ind in range(6):
        print(ind, "eta_" + ["s", "x^2-y^2", "3z^2-r^3", "yz", "xz", "xy"][ind])
        for i_1, j_1 in enumerate(j_list):
            for i_2, j_2 in enumerate(j_list):
                trace = np.trace(eta[ind].T.conj() @ (j_1@j_2))
                print(j_name_list[i_1], j_name_list[i_2], trace)
        print("1_4", np.trace(eta[ind].T.conj()), end="\n\n")


# def plot_bfs_kzkxiky_state():
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
    



def main():
    print(__doc__)
    # list_pauli_products(mode="U_T")
    # list_spin_products()
    # list_projection_hamiltonian()
    return 0

if __name__ == "__main__":
    main()
