# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:03:46 2023

@author: Carl
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

def make_H_0(N, J_x, J_y, J_z, tau_real, tau_imag, d_real, d_imag, c_real, c_imag):
    H_0 = np.zeros((N,2,2),dtype=complex)

    k = np.linspace(0,2*np.pi,N,endpoint=False)
    for i in range(N):
        H_0[i] = (-(J_x + tau_real*np.cos(k[i]) - tau_imag*np.sin(k[i]))*sigma_x
         - (J_y + d_real*np.cos(k[i]) - d_imag*np.sin(k[i]))*sigma_y -
         (J_z + c_real*np.cos(k[i]) - c_imag*np.sin(k[i]))*sigma_z)
    return H_0


def read_data_set(data_path):
    data_set = {
        "model": "null",
        "Nsites": 0,
        "orbital": 0,
        "Nt": 0,
        "Ntau": 0,
        "h": 0.0,
        "j_x": 0.0,
        "j_y": 0.0,
        "j_z": 0.0,
        "tau_real": 0.0,
        "tau_imag": 0.0,
        "d_real": 0.0,
        "d_imag": 0.0,
        "c_real": 0.0,
        "c_imag": 0.0,
        "U_a": 0.0,
        "U_b": 0.0,

        "Solveorder": 0,
        "MatsMaxIter": 0,
        "beta": 0.0,
        "MuChem": 0.0,
        "MatsMaxErr": 0.0,

        "BootstrapMaxIter": 0,
        "BootstrapMaxErr": 0.0,
        "CorrectorSteps": 0,
        "CorrectorStepsErr": 0.0,

        "filename": "null",
        }
    file1 = open(data_path,"r")
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split('=')
    data_set["Nsites"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["orbital"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["Nt"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["Ntau"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["h"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["j_x"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["j_y"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["j_z"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["tau_real"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["tau_imag"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["d_real"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["d_imag"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["c_real"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["c_imag"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["U_a"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["U_b"] = float(first_line[1])
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split('=')
    data_set["Solveorder"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["MatsMaxIter"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["beta"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["MuChem"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["MatsMaxErr"] = float(first_line[1])
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split('=')
    data_set["BootstrapMaxIter"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["BootstrapMaxErr"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["CorrectorSteps"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["CorrectorStepsErr"] = float(first_line[1])
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split(' ')
    first_line = file1.readline().split(':')
    data_set["filename"] = first_line[1]

    file1.close()

    return data_set

def time_fouriertrafo(se_t, t_values, delta_omega, N_omega, meanfield_part):
    omega = np.linspace(-N_omega*delta_omega,N_omega*delta_omega,2*N_omega +1)
    h = t_values[1]
    T,Omega = np.meshgrid(t_values, omega)

    EXP = np.exp(1j*Omega*T)

    se_w = np.einsum('abde,cb->cade',se_t,EXP)*h + meanfield_part[np.newaxis,:,:,:]


    return np.einsum('abcd->bacd', se_w)

def effective_hamilton_k_w(se_w,h_nI):
    h_eff = np.einsum('abcd->bacd', se_w ) + h_nI[np.newaxis,:,:,:]

    return np.einsum('abcd->bacd', h_eff )

def make_retarded_green_k_w(h_eff,omega_values,k_values):
    retarded_green_k_w_array = np.zeros((np.shape(h_eff)),dtype=complex)


    W,K = np.meshgrid(omega_values,k_values)
    Omega_array = W[:,:,np.newaxis,np.newaxis]*np.eye(2)[np.newaxis,np.newaxis,:,:]


    inv_G = Omega_array - h_eff


    det_k_w = inv_G[:,:,0,0]*inv_G[:,:,1,1]-inv_G[:,:,1,0]*inv_G[:,:,0,1]
    # print(np.shape(det_k_w))
    retarded_green_k_w_array[:,:, 0,0] = 1/det_k_w[:,:]*inv_G[:,:,1,1]
    retarded_green_k_w_array[:,:, 1,1] = 1/det_k_w[:,:]*inv_G[:,:,0,0]
    retarded_green_k_w_array[:,:, 1,0] = -1/det_k_w[:,:]*inv_G[:,:,1,0]
    retarded_green_k_w_array[:,:, 0,1] = -1/det_k_w[:,:]*inv_G[:,:,0,1]


    return retarded_green_k_w_array

def get_data(path):
    N_omega = 200           # 500
    delta_omega = 0.0125    # 0.005
    N_time = 200            # 2500

    path_data_set = glob.glob(path + r"data*.txt")
    path_meanfield = glob.glob(path + r"meanfield_Sigma*.npy")
    path_ret_Sigma = glob.glob(path + r"ret_Sigma*.npy")
    dict_data = read_data_set(path_data_set[0])

    selfenergy_array = np.load(path_ret_Sigma[0])[:, :N_time]
    meanfield_array = np.load(path_meanfield[0])

    k_values = np.linspace(0,2*np.pi,dict_data['Nsites'],endpoint=False)

    t_values = np.linspace(0,dict_data['h']*(dict_data['Nt']-1), dict_data['Nt'])[:N_time]
    omega_values = np.linspace(-N_omega*delta_omega, N_omega*delta_omega,2*N_omega +1)

    # selfenergy in momentum-frequency space
    selfenergy_k_w = time_fouriertrafo(selfenergy_array, t_values, delta_omega,
                                       N_omega, meanfield_array[:,0])


    ssh1 = make_H_0(dict_data['Nsites'],dict_data["j_x"],dict_data["j_y"],
                       dict_data["j_z"], dict_data["tau_real"],
                       dict_data["tau_imag"], dict_data["d_real"],
                       dict_data["d_imag"], dict_data["c_real"],
                       dict_data["c_imag"])

    # effective hamiltonian in momentum-frequency space
    h_eff_k_w = effective_hamilton_k_w(selfenergy_k_w,ssh1)

    # retarded Green's function in momentum-frequency space
    ret_green_k_w = make_retarded_green_k_w(h_eff_k_w, omega_values, k_values)

    # spectral function
    A = np.zeros(np.shape(ret_green_k_w),dtype=complex)

    A[:,:,0,0] = 1j*(ret_green_k_w[:,:,0,0] - np.conjugate(ret_green_k_w[:,:,0,0]))
    A[:,:,1,1] = 1j*(ret_green_k_w[:,:,1,1] - np.conjugate(ret_green_k_w[:,:,1,1]))
    A[:,:,1,0] = 1j*(ret_green_k_w[:,:,1,0] - np.conjugate(ret_green_k_w[:,:,0,1]))
    A[:,:,0,1] = 1j*(ret_green_k_w[:,:,0,1] - np.conjugate(ret_green_k_w[:,:,1,0]))
    data = {"k_values" : k_values,
            "omega_values" : omega_values,
            "t_values" : t_values,
            "h_eff_k_w" : h_eff_k_w,
            "ret_green_k_w" : ret_green_k_w,
            "A" : A, 
            "dict_data" : dict_data
        }
    return data

def main():
    data = get_data(r'data_joachim/N80_MPI_SBA_UA1.500000_UB-1.400000_h0.030000/')
    A = data["A"]
    k_values = data["k_values"]
    omega_values = data["omega_values"]
    fig, axs = plt.subplots(nrows=1, ncols=1)
    dset1 = axs.imshow(np.abs(np.transpose(A[:,::-1,0,0]+A[:,::-1,1,1])),
                       aspect='auto',extent=[k_values[0],k_values[-1],
                                             omega_values[0], omega_values[-1]])
    dset1.set_clim(0,8)
    plt.show()

if __name__=='__main__':
    main()