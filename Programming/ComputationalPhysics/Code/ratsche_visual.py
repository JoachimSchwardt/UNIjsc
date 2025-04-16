#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load data from HDF5 files and visualize as 2D and 3D heatmaps

@author: joachim
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import mpl_special


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
            "Programme/hamilton_ratchet/ratchet_dimensionless/")
    data = h5py.File(path + "results_96x100.hdf5", 'r')
    a = np.array(data['a'])
    theta = np.array(data['theta'])
    x = np.array(data['xmean'])
    xmean = np.mean(x, axis=0)
    ind = np.unravel_index(np.argmax(xmean), xmean.shape)
    print(f"Optimal result: {a[ind] = }, {theta[ind] = }, {xmean[ind] = }")
    cmap = 'jet'
    plot_parspace(theta, a, xmean, cmap=cmap)
    plot_parspace3d(theta, a, xmean, cmap=cmap)


def main():
    """Berechnet den Parameterraum für die Hamilton'sche Ratsche"""
    print(__doc__)
    plot_cpp_data()
    return 0


if __name__ == "__main__":
    main()
