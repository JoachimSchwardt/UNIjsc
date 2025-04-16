#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the 2D Ising model using the metropolis algorithm.
A left mouse button click inside the right subplot will select a tuple (tau, m)
    Here 'tau' is the effective temperature and 'm' the average magnetization.
    This will generate a random state of spins with the selected 'm'.

A left mouse button click inside the left subplot will run 'num_steps = 10'
    steps of the metropolis algorithm.
"""

import functools
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import mpl_special


@njit
def _extern_metropolis_step(spins, tau):
    """Compute a metropolis step for the given spins and effective temperature
    """
    rsize, csize = spins.shape
    rand = np.random.uniform(0.0, 1.0, size=(rsize, csize))
    row_ind = np.random.randint(0, rsize, size=(rsize, csize))
    col_ind = np.random.randint(0, csize, size=(rsize, csize))
    for r_i in range(rsize):
        for c_i in range(csize):
            row, col = row_ind[r_i, c_i], col_ind[r_i, c_i]
            spin = spins[row, col]
            delta_h = spin * (spins[row, (col + 1) % csize]
                              + spins[row, (col - 1) % csize]
                              + spins[(row + 1) % rsize, col]
                              + spins[(row - 1) % rsize, col])

            if delta_h < 0 or rand[row, col] < np.exp(-2 * delta_h / tau):
                spins[row, col] *= -1


# def powerlaw_space(xmin, xmax, num, power=0.5, endpoint=True):
#     """Return 'num' values between 'xmin' and 'xmax'
#     following a power-law distribution"""
#     xval = np.linspace(xmin, xmax, num, endpoint=endpoint)
#     xval *= ((xval - xmin) / (xval[-1] - xmin))**power
#     return xval


# def get_ising2d_theory(tau_min=0.01, tau_max=4.0, num=1000, power=0.1):
#     """Return the theoretical result for the 2d Ising model"""
#     tau_c = 1 / np.arcsinh(1)
#     tau = np.zeros(num)
#     mag = np.zeros(num)
#     tau[:-2] = powerlaw_space(tau_min, tau_c, num - 2, power=power,
#                               endpoint=False)
#     tau[-2] = tau_c
#     tau[-1] = tau_max
#     mag[:-2] = (1 - (1 / np.sinh(tau_c / tau[:-2]))**4)**(1 / 8)
#     return tau, mag


def get_ising2d_theory(tau_min=0.01, tau_max=4.0, num=200):
    """Return the theoretical result for the 2d Ising model"""
    tau_c = 2 / np.arcsinh(1)
    tau = np.zeros(num)
    mag = np.zeros(num)
    mag[:-1] = np.linspace(0.0, 1.0, num=num - 1)[::-1]
    tau[0] = 0.0
    tau[-1] = tau_max
    tau[1:-1] = tau_c / np.arcsinh(1 / (1 - mag[1:-1]**8)**(1 / 4))
    return tau, mag


class Ising2D:
    """Simulate the 2d Ising model"""
    def __init__(self, size=50, tau=1.0):
        self.steps = 10    # number of metropolis steps per click
        self.size = size
        self.spins = np.ones((size, size), dtype=int)
        self.tau = tau     # effective temperature
        self.mag = 1.0     # average magnetization
        self.fig = None
        self.ax = None
        self.marker = None
        self.img = None
        self.title = r"$\tau = {:.3f}$"#" and $\langle m \rangle = {:.3f}$""
        self.axtitle = None
        self.colors = {'cmap' : 'viridis', 'marker' : 'blue'}


    def __getitem__(self, ind):
        """Return the spins at the given index"""
        return self.spins[ind[0], ind[1]]


    def init_state(self, mag=0.0):
        """Number of spins with (-1) is (1 - mag) / 2"""
        shape = self.spins.shape
        self.spins = np.ones(self.spins.size, dtype=int)
        self.spins[:int(self.spins.size * (1 - mag) / 2)] = -1
        np.random.shuffle(self.spins)
        self.spins = self.spins.reshape(shape)
        self.compute_mag()


    def compute_mag(self):
        """Update the average magnetization"""
        self.mag = np.mean(self.spins)


    def metropolis_step(self):
        """Execute a metropolis step for every spin"""
        _extern_metropolis_step(self.spins, self.tau)


    def click(self, event):
        """TODO"""
        mode = event.canvas.toolbar.mode
        if event.button == 1 and event.inaxes == self.ax[0] and mode == '':
            for _ in range(self.steps):
                self.metropolis_step()
                self.compute_mag()
                self.img.set_data(self.spins)
                self.marker.set_ydata([self.mag])
                self.fig.canvas.flush_events()
                self.fig.canvas.draw()

        elif event.button == 1 and event.inaxes == self.ax[1] and mode == '':
            self.tau = event.xdata
            self.mag = event.ydata
            self.init_state(self.mag)
            self.img.set_data(self.spins)
            self.marker.set_xdata([self.tau])
            self.marker.set_ydata([self.mag])
            self.axtitle.set_text(self.title.format(self.tau))
            self.fig.canvas.draw()


    def setup_figure(self):
        """Create a figure and subplots"""
        fig, ax = plt.subplots(ncols=2)
        ax[0].tick_params(axis='both', which='both', bottom=False, left=False,
                          labelbottom=False, labelleft=False)
        ax[0].set_aspect(1.0)
        self.img = ax[0].imshow(self.spins, cmap=self.colors['cmap'])
        ax[1].set_xlabel(r"$\tau$")
        ax[1].set_ylabel(r"$\langle m \rangle$")
        self.axtitle = ax[1].set_title(self.title.format(self.tau))

        tau, mag = get_ising2d_theory()
        ax[1].set_xlim(tau[0], tau[-1])
        ax[1].plot(tau, mag, c='k', label=r"$\langle m \rangle_\mathrm{theo.}$")
        ax[1].plot(tau, -mag, c='k')
        self.marker = ax[1].plot(self.tau, self.mag, ls='', marker='o',
                                 c=self.colors['marker'])[0]
        ax[1].legend()

        self.fig = fig
        self.ax = ax


    def show(self):
        """Run the simulation"""
        plt.connect("button_press_event", functools.partial(self.click))
        mpl_special.embed_labels(self.fig, self.ax)


def main():
    print(__doc__)
    ising = Ising2D(size=500, tau=1.0)
    ising.init_state(0.5)
    ising.setup_figure()
    ising.show()
    return 0


if __name__ == "__main__":
    main()
