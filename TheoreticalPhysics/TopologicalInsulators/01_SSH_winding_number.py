#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 16:42:11 2021

@author: joachim
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=True)

def d_vector(k, phi1, phi2, phi3, t1=1.0, t2=1.0, t3=1.0):
    dx = t1 * np.cos(phi1) + t2 * np.cos(phi2 + k) + t3 * np.cos(phi3 - k)
    dy = t1 * np.sin(phi1) + t2 * np.sin(phi2 + k) + t3 * np.sin(phi3 - k)
    return dx, dy

def cos_phi(k, phi1, phi2, phi3, t1=1.0, t2=1.0, t3=1.0):
    dx, dy = d_vector(k, phi1, phi2, phi3, t1, t2, t3)
    return dx / np.sqrt(dx*dx + dy*dy)

def sin_phi(k, phi1, phi2, phi3, t1=1.0, t2=1.0, t3=1.0):
    dx, dy = d_vector(k, phi1, phi2, phi3, t1, t2, t3)
    return dy / np.sqrt(dx*dx + dy*dy)

def phi(k, phi1, phi2, phi3, t1=1.0, t2=1.0, t3=1.0):
    return np.arccos(cos_phi(k, phi1, phi2, phi3, t1, t2, t3))

def main():
    PATH = "../TopIns_Latex/"
    t1, t2, t3 = 1.0, 1.0, 0.1
    alpha_c = np.arcsin((1 - t3) * np.sqrt(0.5 + t3 / 4))
    phi1 = np.array([0.0, np.pi/6, alpha_c, np.pi/3, np.pi/2]) 
    phi2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
    phi3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
    label = [(f"$\\phi_1={phi1[i]:.3f}$, "
             + f"$\\phi_2={phi2[i]:.3f}$, "
             + f"$\\phi_3={phi3[i]:.3f}$") 
             for i in range(phi1.shape[0])]
    
    k = np.linspace(-np.pi, np.pi, 500)
    phi_val = np.array([phi(k, phi1[i], phi2[i], phi3[i], t1, t2, t3)
                        for i in range(phi1.shape[0])])
    
    colors = special.Colors()
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\phi(k)$")
    ax.set_title(f"$|t_1|={t1}$, $|t_2|={t2}$, $|t_3|={t3}$")
    ax.set_xlim(np.min(k), np.max(k))
    ax.set_ylim(0.0, np.pi)
    for i in range(phi1.shape[0]):
        ax.plot(k, phi_val[i], c=colors.get_color(), label=label[i])
    
    ax.legend()
    special.format_ticklabels(ax, major_den=4, axis='x')
    special.format_ticklabels(ax, major_den=4, axis='y')
    special.polish(fig, ax, False)
    plt.savefig(PATH + "01_WindingNumber_phi_v2.png")
    
    # cos_phi_val = np.array([cos_phi(k, phi1[i], phi2[i], phi3[i], t1, t2, t3)
    #                         for i in range(phi1.shape[0])])
    # sin_phi_val = np.array([sin_phi(k, phi1[i], phi2[i], phi3[i], t1, t2, t3)
    #                         for i in range(phi1.shape[0])])
    
    # height = special.figsize_height
    # fig, ax = plt.subplots(figsize=(1.1  *height, height))
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$y$")
    # ax.set_title(f"$|t_1|={t1}$, $|t_2|={t2}$, $|t_3|={t3}$")
    # ax.axis([-1.05, 1.05, -1.05, 1.05])
    # colors.ctr = 0
    # scale = np.linspace(0.7, 1.0, cos_phi_val.shape[0])
    # for i in range(phi1.shape[0]):
    #     ax.plot(scale[i] * cos_phi_val[i], scale[i] * sin_phi_val[i], 
    #             c=colors.get_color(), label=label[i], 
    #             ls='', marker='o', ms=3, mew=1)
    
    # ax.legend()
    # special.polish(fig, ax, False)
    
    d_val = np.array([d_vector(k, phi1[i], phi2[i], phi3[i], t1, t2, t3)
                      for i in range(phi1.shape[0])])
    dx_val, dy_val = d_val[:, 0], d_val[:, 1]
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$d_x$")
    ax.set_ylabel(r"$d_y$")
    ax.axhline(0, c='k')
    ax.axvline(0, c='k')
    ax.set_title(f"$|t_1|={t1}$, $|t_2|={t2}$, $|t_3|={t3}$")
    colors.ctr = 0
    for i in range(phi1.shape[0]):
        ax.plot(dx_val[i], dy_val[i], 
                c=colors.get_color(), label=label[i], 
                ls='', marker='o', ms=3, mew=1)
    
    ax.legend()
    special.polish(fig, ax, False)
    plt.savefig(PATH + "01_WindingNumber_dvector_v2.png")


if __name__ == "__main__":
    print(__doc__)
    main()