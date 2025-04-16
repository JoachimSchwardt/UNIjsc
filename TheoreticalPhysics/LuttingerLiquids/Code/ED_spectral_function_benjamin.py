# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:41:35 2024

@author: Carl
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import glob

sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])


def make_H_0(N, x0, y0, z0, xr, xi, yr, yi, zr, zi):
    H_0 = np.zeros((N,2,2),dtype=complex)
    k = np.linspace(0,2*np.pi,N,endpoint=False)
    for i in range(N):
        H_0[i] = ((x0 + xr*np.cos(k[i]) + xi*np.sin(k[i]))*sigma_x
          + (y0 + yr*np.cos(k[i]) + yi*np.sin(k[i]))*sigma_y +
          (z0 + zr*np.cos(k[i]) + zi*np.sin(k[i]))*sigma_z)
    return H_0

def read_data_set_SSHC(data_path):
    
    

    data_set = {
        "L": 0,
        "x0": 0.0,
        "xr": 0.0,
        "yi": 0.0,
        "Ua": 0.0,
        "Ub": 0.0,
        "pbc": False,
        "t_end": 0.0,
        "Nt": 0.0,
        "site_b": 0,
        "beta": 0.0,
        "timecode": 0.0,
        }
    file1 = open(data_path,"r")
    first_line = file1.readline().split('=')
    data_set["L"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["x0"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["xr"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["yi"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["Ua"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["Ub"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["pbc"] = bool(first_line[1])
    first_line = file1.readline().split('=')
    data_set["t_end"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["Nt"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["site_b"] = int(first_line[1])
    first_line = file1.readline().split('=')
    data_set["beta"] = float(first_line[1])
    first_line = file1.readline().split('=')
    data_set["timecode"] = int(first_line[1])
    
    file1.close()
    
    return data_set

def time_fouriertrafo(g_t, delta_omega, N_omega, delta_t):
    omega = np.linspace(-N_omega*delta_omega, N_omega *
                        delta_omega, 2*N_omega + 1)
    Nt = np.shape(g_t)[0]
    t_values = np.linspace(0, delta_t*Nt, Nt)

    T, Omega = np.meshgrid(t_values, omega)

    EXP = np.exp(1j*Omega*T)

    g_w = np.einsum('b...,cb->c...', g_t, EXP)*delta_t - 1 / \
        2 * g_t[0]*delta_t - 1/2 * g_t[-1]*delta_t

    return g_w

def fourier_x_to_k(g, N_k, mid):

    N_data = np.shape(g)[0]
    k_values = np.linspace(0, 2*np.pi, N_k, endpoint=False)
    x_values = np.arange(N_data)-mid
    print(x_values)
    X, K = np.meshgrid(x_values, k_values)
    print(np.shape(X))
    EXP = np.exp(-1j*K*X)
    g_k = np.einsum('ab..., ca->cb...', g, EXP)

    return g_k

def main():
    
    # Fouriertrafo parameters
    gamma = 0.2
    delta_omega = 0.009
    N_omega = 500
    
    omega_values = np.linspace(-N_omega*delta_omega,N_omega*delta_omega,2*N_omega +1)
    
    # paths to the data
    path_to_data = "/home/bm/Documents/Uni/Promotion/Projekte/EPs_from_Luttinger_Liquid/Data_Carl/Comp_methods/SSHC2/ED"
    path0 = glob.glob(path_to_data + "/G_fT_SSHC_*_s0")[0]
    path1 = glob.glob(path_to_data + "/G_fT_SSHC_*_s1")[0]
    
    data_set = read_data_set_SSHC(path0 + '/datafile_fT.txt')
    
    # read out main parameters
    t_end = data_set['t_end']
    Nt = data_set['Nt']
    Nk = int(data_set['L']/2)
    g_ret0 = np.load(path0 + '/G_ret_fT.npy')
    g_ret1 = np.load(path1 + '/G_ret_fT.npy')
    
    t_values, delta_t = np.linspace(0,t_end, Nt ,endpoint=True,retstep=True)
    k_values = np.linspace(0, 2*np.pi, Nk, endpoint=False)
    
    # construct green's function
    g_ret = np.zeros((Nt,Nk, 2,2), dtype=complex)
    g_ret[:,:,0,0] = g_ret0[:,::2]
    g_ret[:,:,1,0] = g_ret0[:,1::2]
    g_ret[:,:,0,1] = g_ret1[:,0::2]
    g_ret[:,:,1,1] = g_ret1[:,1::2]
    
    
    #fouriertransformation in space
    g_ret_x = np.einsum('ab...->ba...',g_ret)
    g_ret_k = fourier_x_to_k(g_ret_x, Nk, 0)
    g_ret_k = np.einsum('ab...->ba...',g_ret_k)
    
    #fouriertransformation in time, window denotes the damping for reducing the errors 
    window = np.exp(-gamma*t_values)
    g_ret_d = g_ret_k*window[:,np.newaxis,np.newaxis,np.newaxis]
    g_ret_w_k = time_fouriertrafo(g_ret_d, delta_omega, N_omega, delta_t)
    
    # spectral function
    A = 1j/(2*np.pi) *(g_ret_w_k-np.conjugate(np.einsum('abcd->abdc',g_ret_w_k)))
    # inverse greensfunction
    g_ret_inv_w_k = np.linalg.inv(g_ret_w_k)
    # non-interacting hamiltonian
    h0 = make_H_0(Nk, data_set["x0"], 0.0, 0.0, data_set["xr"], 0.0, 0.0, data_set["yi"], 0.0, 0.0)
    # effective hamiltonian
    h_eff = (omega_values[:,np.newaxis,np.newaxis,np.newaxis]*np.eye(2)[np.newaxis,np.newaxis,:,:]
             - g_ret_inv_w_k + 1j*gamma * np.eye(2)[np.newaxis,np.newaxis,:,:])
    # selfenergy
    selfenergy = h_eff - h0
    
    #blochvector of h_eff
    bloch_d = np.zeros((N_omega*2+1,Nk, 4), dtype=complex)
    
    bloch_d[:,:,0] = (h_eff[:,:,0,0] + h_eff[:,:,1,1])/2
    bloch_d[:,:,3] = (h_eff[:,:,0,0] - h_eff[:,:,1,1])/2
    bloch_d[:,:,1] = (h_eff[:,:,1,0] + h_eff[:,:,0,1])/2
    bloch_d[:,:,2] = 1j*(h_eff[:,:,1,0] - h_eff[:,:,0,1])/2
    
    # eigenvectors of h_eff 
    ev_plus = bloch_d[:,:,0] + np.sqrt(bloch_d[:,:,1]**2+bloch_d[:,:,2]**2+bloch_d[:,:,3]**2)
    ev_minus = bloch_d[:,:,0] - np.sqrt(bloch_d[:,:,1]**2+bloch_d[:,:,2]**2+bloch_d[:,:,3]**2)
    
    
    # plot spectral function A
    fig, axs = plt.subplots(nrows=1, ncols=1)
    dset1 = axs.imshow((np.real(A[:,:,0,0] + A[:,:,1,1])),aspect='auto',extent=[0,np.pi*2,omega_values[0],omega_values[-1]])
    axs.set_xlabel(r'$ x $',fontsize=20.0)
    axs.set_ylabel(r'$ t $',fontsize=20.0)
    fig.colorbar(dset1,ax=axs)
    
    # plot eigenvalues at omega_for_plot, omega_for_plot= N_omega corresponds to w=0
    omega_for_plot = N_omega
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(k_values, np.real(ev_plus[omega_for_plot]))
    axs.plot(k_values, np.real(ev_minus[omega_for_plot]))
    axs.plot(k_values, np.imag(ev_plus[omega_for_plot]))
    axs.plot(k_values, np.imag(ev_minus[omega_for_plot]))
    
    # plot selfenergy at omega_for_plot
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].plot(k_values,np.real(selfenergy[omega_for_plot,:,0,0]))
    axs[0,0].plot(k_values,np.imag(selfenergy[omega_for_plot,:,0,0]))
    
    axs[1,0].plot(k_values,np.real(selfenergy[omega_for_plot,:,1,0]))
    axs[1,0].plot(k_values,np.imag(selfenergy[omega_for_plot,:,1,0]))
    
    axs[0,1].plot(k_values,np.real(selfenergy[omega_for_plot,:,0,1]))
    axs[0,1].plot(k_values,np.imag(selfenergy[omega_for_plot,:,0,1]))
    
    axs[1,1].plot(k_values,np.real(selfenergy[omega_for_plot,:,1,1]))
    axs[1,1].plot(k_values,np.imag(selfenergy[omega_for_plot,:,1,1]))
   
    
    plt.show()
    
if __name__=='__main__': 
    main()