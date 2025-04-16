#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-version der hamilton'schen Ratsche
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from time import perf_counter
import mpl_special


@nb.njit
def get_xpot_prime(x, L, a, v_0, alpha):
    """Ableitung des ortsabhängigen Anteils des Potentials"""
    phi = 2 * np.pi * x / L
    fac = -np.sin(phi) + 2 *  a * np.cos(2 * phi)
    return 2 * np.pi / L * v_0 * fac + alpha


@nb.njit
def get_tpot(t, tau, t_off):
    """Zeitabhängiger Anteil des Potentials"""
    return (t % tau) > tau * t_off


@nb.njit
def diffusion(x, t, tau, t_off, L, a, v_0, alpha, dt, D):
    """Führt eine Iteration der Ratschen-Gleichung aus"""
    x_rand = np.random.normal(0, 1, x.size)
    pot_prime = get_xpot_prime(x, L, a, v_0, alpha) * get_tpot(t, tau, t_off)
    x += np.sqrt(2 * D * dt) * x_rand - dt * pot_prime


@nb.njit
def simulate(x0, L, a, v_0, alpha, num_part,
             tau, theta, num_tau, num_steps, D, num_iter):
    """Simuliert für 'num_part' Teilchen die Ratschen-Gleichung für 
    num_tau * tau * num_steps Schritte."""
    x_mean = np.zeros(num_iter)
    t_off = tau / (1 + theta)       # Zeitdauer im Zustand 'Aus' je Periode
        
    # endpoint und retstep werden von numba nicht unterstützt -> objmode
    with nb.objmode(t="f8[:]", dt="f8"):
        t, dt = np.linspace(0.0, num_tau * tau, num_tau * tau * num_steps,
                            endpoint=False, retstep=True)
        
    for x_i in range(num_iter):
        x = np.full(num_part, x0)
    
        for i in range(t.size):
            diffusion(x, t[i], tau, t_off, L, a, v_0, alpha, dt, D)

        x_mean[x_i] = np.mean(x)

    return x_mean


@nb.njit(parallel=True)
def simulate_parallel(x_mean, a_arr, theta_arr,
                      x0, L, v_0, alpha, num_part,
                      tau, num_tau, num_steps, D, num_iter):
    """Paralleles Ausführen der Funktion 'simulate'
    für den gegebenen Parameterraum"""
    for theta_i in nb.prange(theta_arr.size):
        for a_i in range(a_arr.size):
            val = simulate(x0, L, a_arr[a_i], v_0, alpha, num_part,
                           tau, theta_arr[theta_i], num_tau, num_steps, D, 
                           num_iter)
            x_mean[theta_i, a_i, :] = val
            
            
def plot_parspace(theta, a, x, **kwargs):
    """Visualisierung des Parameterraums"""
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$a$")
    
    extent = [np.min(theta), np.max(theta), np.min(a), np.max(a)]
    ax.axis(extent)
    img = ax.imshow(x, extent=extent, origin='lower', cmap=kwargs['cmap'])
    cbar = mpl_special.colorbar(img, ax, label=r"$\langle x \rangle$")
    mpl_special.embed_ylabel(cbar.ax)
    mpl_special.polish(fig, ax)
    
    
def plot_parspace3d(theta, a, x, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$a$")
    ax.set_zlabel(r"$\langle x \rangle$")
    shape = x.shape
    ax.plot_surface(theta.reshape(shape), a.reshape(shape), x, 
                    cmap=kwargs['cmap'])
    
    
def plot_cpp_data():
    path = ("/home/joachim/Documents/UNI/Programmierung/ComputationalPhysics/"
            "Programme/hamilton_ratchet/ratchet_store_random_float/")
    cmap = 'jet'
    data = np.loadtxt(path + "results_96x100.txt", skiprows=3, delimiter=',')
    a, theta, x = data.T
    ind = np.argmax(x)
    print(f"{a[ind] = }, {theta[ind] = }, {x[ind] = }")
    shape = (96, 100)
    plot_parspace(theta, a, x.reshape(shape), cmap=cmap)
    # plot_parspace3d(theta, a, x.reshape(shape), cmap=cmap)


def main():
    """Berechnet den Parameterraum für die Hamilton'sche Ratsche"""
    print(__doc__)
    x0 = 0.0                        # Anfangsort
    L = 1.5                         # räumliche Periode
    v_0 = -0.2                      # Potentialhöhe
    alpha = 0.0                     # Kippwinkel

    tau = 30                        # Periodendauer
    num_steps = 100                 # Anzahl Schritte pro Zeiteinheit

    D = 0.008                       # Diffusionskonstante

    # Parameter relevant für die Performance
    num_tau = 5                     # Anzahl an Perioden
    num_part = 10000                # Anzahl an Teilchen/Realisierungen
    num_threads = 16                 # Anzahl threads für multiprocessing

    # Definition des zu untersuchenden Parameterraums
    a_min = 0.0                     # minimaler Asymmetriefaktor
    a_max = 1.0                     # maximaler      -||-
    a_num = 1 * num_threads        # Anzahl Punkte

    theta_min = 0.0                 # minimales Verhältnis 't_an / t_aus'
    theta_max = 2.0
    theta_num = 1
    
    num_iter = 1                   # Anzahl an Iterationen für jeden Parameter

    a_arr = np.linspace(a_min, a_max, a_num)
    theta_arr = np.linspace(theta_min, theta_max, theta_num)

    # Simulation
    x_mean = np.zeros((theta_arr.size, a_arr.size, num_iter))
    simulate_parallel(x_mean, a_arr, theta_arr,
                      x0, L, v_0, alpha, num_part,
                      tau, num_tau, num_steps, D, num_iter)
    return theta_arr, a_arr, x_mean


if __name__ == "__main__":
    plot_cpp_data()
    # t_start = perf_counter()
    # theta_arr, a_arr, x_mean = main()
    # t_end = perf_counter()
    # print(f"Time: {t_end - t_start:.2f} seconds")
