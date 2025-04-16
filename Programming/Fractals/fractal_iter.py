#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First draft of a fractal engine for iterative fractals (Mandelbrot etc.)
"""

import functools
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import mpl_special


def get_zarray(extent, shape=(100, 200)):
    """Return a 2d-array of complex numbers in the given 'extent'-rectangle
    """
    real = np.linspace(extent[0], extent[1], shape[1])
    imag = np.linspace(extent[2], extent[3], shape[0])#[::-1]
    z_arr = np.outer(np.ones(shape[0]), real) + 1j * np.outer(imag, np.ones(shape[1]))
    return z_arr.flatten()


def map_julia(z, c=0j):
    return z*z + c


def iteration(mapping, z, *args, n_iter=20, abort_val=4):
    for ctr in range(n_iter):
        z = mapping(z, *args)
        if z.real**2 + z.imag**2 > abort_val:
            break
    return ctr


# def array_iteration(mapping, z_arr, *args, n_iter=20, abort_val=4):
#     mask = np.ones(z_arr.size, dtype=bool)
#     ctr_arr = np.zeros(z_arr.size, dtype=int)
#     for ctr in range(n_iter):
#         z_arr[mask] = mapping(z_arr[mask], *args)
#         mask[mask] = (z_arr[mask].real**2 + z_arr[mask].imag**2 < abort_val)
#         ctr_arr += mask

#     return ctr_arr


def array_iteration(mapping, z_arr, c_arr, n_iter=20, abort_val=4):
    mask = np.ones(c_arr.size, dtype=bool)
    ctr_arr = np.zeros(c_arr.size, dtype=int)
    for _ in range(n_iter):
        z_arr[mask] = mapping(z_arr[mask], c_arr[mask])
        mask[mask] = (z_arr[mask].real**2 + z_arr[mask].imag**2 < abort_val)
        ctr_arr += mask

    return ctr_arr


def get_mandelbrot(shape=(100, 200), extent=(-2, 1, -1.2, 1.2), n_iter=20):
    c_arr = get_zarray(extent, shape)
    z_arr = np.zeros_like(c_arr)
    ctr_arr_mandelbrot = array_iteration(map_julia, z_arr, c_arr, n_iter)
    return ctr_arr_mandelbrot.reshape(shape)


def plot_fractal(ctr_arr, extent, cmap="viridis"):
    # figsize = mpl_special.set_figsize()
    # fig, ax = plt.subplots(figsize=(figsize[1], figsize[1]))
    fig, ax = plt.subplots(figsize=(4.3, 4.5))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img = ax.imshow(ctr_arr, extent=extent, cmap=cmap, origin="lower")

    fig.tight_layout()
    # plt.show()
    return fig, ax, img

class InteractiveFractal:
    """Interactive plot of a given fractal"""

    def __init__(self, get_fractal, shape=(100, 200), extent=(-2, 1, -1.2, 1.2),
                 n_iter=40, cmap="viridis"):
        self.shape = shape
        self.extent = extent
        self.n_iter = n_iter
        self.n_iter_change = 10
        self.cmap = cmap
        self.get_fractal = get_fractal
        self.ctr_arr = get_fractal(shape, extent, n_iter)
        self.fig, self.ax, self.img = plot_fractal(self.ctr_arr, extent, cmap)


    def update(self):
        """Update the fractal plot using the current state parameters"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        t_start = perf_counter()
        self.ctr_arr = self.get_fractal(self.shape, self.extent, self.n_iter)
        t_comp = perf_counter()
        print(f"Computation of {self.ctr_arr.size} values for {self.n_iter} iterations "
              f"completed in {t_comp - t_start:.2f} seconds!")
        self.img.set_extent(self.extent)
        self.img.set_clim(1, self.n_iter)
        self.img.set_data(self.ctr_arr)
        self.fig.canvas.draw()
        t_end = perf_counter()
        print(f"Visualization update done in {t_end - t_comp:.2f} seconds!")


    def keypress(self, event):
        update_bool = False
        if event.key == '+':
            self.n_iter += self.n_iter_change
            update_bool = True
        if event.key == '-':
            self.n_iter -= self.n_iter_change
            update_bool = True

        if event.key == 'p':
            self.n_iter_change *= 10

        if event.key == 'm':
            self.n_iter_change //= 10

        if self.n_iter < 10:
            self.n_iter = 10
            print("Number of iterations is bounded from below by 10!")

        if update_bool:
            print(f"Number of iterations updated to {self.n_iter}")
            self.update()


    def onrelease(self, event):
        mode = event.canvas.toolbar.mode
        if mode == "zoom rect" and event.inaxes:
            self.update()
        elif mode == "pan/zoom" and event.inaxes:
            self.update()


    def start(self):
        self.fig.canvas.mpl_connect('key_press_event',
                                    functools.partial(self.keypress))
        self.fig.canvas.mpl_connect('button_release_event',
                                    functools.partial(self.onrelease))
        plt.show()


def main():
    print(__doc__)
    # z_arr = get_zarray(-1+1j, 1-1j)
    # c_arr = np.full_like(z_arr, 0.0)
    # ctr_arr_julia = array_iteration(map_julia, z_arr, c_arr)

    # shape = (1000, 2000)
    # extent = [-2, 1, -1.2, 1.2]
    # ctr_arr_mandelbrot = get_mandelbrot(shape, extent=extent, n_iter=10)
    # plot_fractal(ctr_arr_mandelbrot, extent=extent, cmap="gnuplot2_r")

    ifrac = InteractiveFractal(get_mandelbrot, shape=(800, 1600), n_iter=1000)
    ifrac.start()

    return 0

if __name__ == "__main__":
    main()
