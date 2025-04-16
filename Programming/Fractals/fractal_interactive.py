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
import fractal_cpp


def get_mandelbrot(shape=(100, 200), extent=(-2, 1, -1.2, 1.2), n_iter=20):
    """
    Current time complexity is T = (0.14 + 2.6e-5 * n_iter) sec
     --> Theoretical runtime for 10000 iterations is 0.26 sec 
         (matches C++, roughly 0.28 sec)
    """
    c_shape = fractal_cpp.Shape(shape[1], shape[0])     # shape[1] == width
    c_extent = fractal_cpp.Extent(extent[0], extent[1], extent[2], extent[3])
    ctr_arr = np.array(fractal_cpp.get_fractal(c_shape, c_extent, n_iter))
    return ctr_arr


def plot_fractal(ctr_arr, extent, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(4.3, 4.5))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img = ax.imshow(ctr_arr, extent=extent, cmap=cmap, origin="lower", 
                    interpolation="antialiased")

    plt.get_current_fig_manager().window.showMaximized()
    fig.tight_layout()
    # plt.show()
    return fig, ax, img

class InteractiveFractal:
    """Interactive plot of a given fractal"""

    def __init__(self, get_fractal, shape=(100, 200), extent="auto",
                 n_iter=40, cmap="viridis"):
        self.shape = shape
        if extent == "auto":
            ratio = shape[1] / shape[0]
            width = 3
            height = width / ratio
            extent = [-2, -2 + width, -height/2, height/2]
        self.extent = extent
        self.n_iter = n_iter
        self.n_iter_change = 100
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

        if self.n_iter < 1:
            self.n_iter = 1
            print("Number of iterations is bounded from below by 1!")

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
        # self.fig.canvas.mpl_connect('button_press_event',
        #                             functools.partial(self.onpress))
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
    
    ifrac = InteractiveFractal(get_mandelbrot, shape=(1008, 1792), n_iter=100)
    ifrac.start()

    return 0

if __name__ == "__main__":
    main()
