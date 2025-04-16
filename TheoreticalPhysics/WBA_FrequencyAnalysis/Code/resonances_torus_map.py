#!/usr/bin/env python

"""Resonance dynamics for a map on a 2-torus."""

from __future__ import division, print_function

import numpy as np
# import matplotlib
# matplotlib.use("GtkAgg")
import matplotlib.pylab as plt


###############################################################################
# Main
###############################################################################

def map_on_2torus(nu_1, nu_2, title="", res=None):      # pylint: disable=R0914
    """Illustrate resonances of a map on a 2-torus."""

    # omega_1 = 2.0 * np.pi * nu_1
    # omega_2 = 2.0 * np.pi * nu_2

    if res is not None:
        k_1, k_2, m = res
        result = k_1 * nu_1 + k_2 * nu_2 - m
        print("Check resonance condition:", result)

    t_max = 2
    times = np.arange(t_max)

    # --- Dynamics:
    phi_1_t = nu_1 * times % 1.0#(2.0*np.pi)
    phi_2_t = nu_2 * times % 1.0#(2.0*np.pi)

    # Plot of the dynamics.
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    markersize = 10
    lines = ax.plot(phi_1_t, phi_2_t, ls="", marker=".",
                     markersize=markersize)

    # --- Also plot linear connection.
    # Note: this is not always correct. FIXME: When?
    # t_max_lin = 100
    # times = np.linspace(0.0, t_max_lin, t_max_lin*100)
    # phi_1_t = omega_1 * times % (2.0*np.pi)
    # phi_2_t = omega_2 * times % (2.0*np.pi)
    # plt.plot(phi_1_t, phi_2_t, ls="", marker=".", markersize=1,
    #          color="red")

    ax.set_xlabel(r"$\varphi_1$")
    ax.set_ylabel(r"$\varphi_2$")
    ax.set_xlim(0.0, 1.0)#2.0*np.pi)
    ax.set_ylim(0.0, 1.0)#2.0*np.pi)
    ax.set_title(title)

    # We abuse plt to keep the counter of the part to be plotted:
    plt.part = 3

    def key_press(event):
        """Quit on space or 'q'."""

        if event.key == ' ':
            plt.close("all")
        elif event.key.lower() == 'c' or event.key.lower() == 'd':
            times = np.arange(plt.part)
            phi_1_t = nu_1 * times % 1.0#(2.0*np.pi)
            phi_2_t = nu_2 * times % 1.0#(2.0*np.pi)

            lines[0].set_data(phi_1_t, phi_2_t)
            plt.draw()

            plt.part = plt.part + 1
            if event.key == 'C':
                plt.part = plt.part + 9
            if event.key == 'd':
                plt.part = plt.part + 99
            if event.key == 'D':
                plt.part = plt.part + 999

    plt.connect('key_press_event', key_press)

    plt.show()


def main(num):
    """Call the examples."""
    print("Usage:")
    print("- <Space> for next example")
    print("- 'c' evolve in time")
    print("- 'C' evolve in time (faster)")
    print("- 'd' evolve in time (even faster)")
    print("- 'D' evolve in time (even^2 faster)")

    if num == 0:# Example 0:
        nu_1 = 1.0/np.sqrt(2.0)
        nu_2 = 1.0/np.sqrt(3.0)
        map_on_2torus(nu_1, nu_2, title="No resonance")

    if num == 1:# Example 1a:
        k_1 = 0
        k_2 = 4
        m = 1
        nu_1 = 1.0/3.0*np.sqrt(2)
        nu_2 = 1.0/4.0
        map_on_2torus(nu_1, nu_2, title="0:4:1 (uncoupled)", res=(k_1, k_2, m))

    if num == 2:# Example 1b:
        k_1 = 3
        k_2 = 0
        m = 1
        nu_1 = 1.0/3.0
        nu_2 = 1.0/4.0*np.sqrt(2)
        map_on_2torus(nu_1, nu_2, title="3:0:1 (uncoupled)", res=(k_1, k_2, m))

    if num == 3:# Example 2:
        k_1 = -3
        k_2 = 4
        m = 0
        nu_1 = 1.0/3.0*np.sqrt(2)
        nu_2 = 1.0/4.0*np.sqrt(2)
        map_on_2torus(nu_1, nu_2, title="-3:4:0 (coupled)", res=(k_1, k_2, m))

    if num == 4:# Resonance with m != 0
        k_1 = 3
        k_2 = -5
        m = 1
        nu_1 = 1.0/3.0*np.sqrt(2)
        nu_2 = (np.sqrt(2)-1)/5.0
        map_on_2torus(nu_1, nu_2, title="3:-5:1 (coupled, m!=0)",
                      res=(k_1, k_2, m))

    if num == 5:# This fills the same curve, but in a different ordering!
        k_1 = 3
        k_2 = -5
        m = 0
        nu_1 = 1.0/3.0*np.sqrt(2)
        nu_2 = np.sqrt(2)/5.0
        map_on_2torus(nu_1, nu_2, title="3:-5:0 (coupled, same curve)",
                      res=(k_1, k_2, m))

    if num == 6:# Double resonance
        k_1 = -3
        k_2 = 4
        m = 0
        nu_1 = 1.0/3.0
        nu_2 = 1.0/4.0
        map_on_2torus(nu_1, nu_2, title="Double resonance", res=(k_1, k_2, m))
        
    if num == 7:#complex
        k_1 = 3
        k_2 = 5
        m = 1
        phi = 0.3
        nu = np.exp(1j*phi*0.5*np.pi)
        nu_1, nu_2 = nu.real, nu.imag
        nu *= m / (k_1*nu_1 + k_2*nu_2)
        map_on_2torus(nu.real, nu.imag, title="Complex 1", res=(k_1, k_2, m))


###############################################################################
# Call `main()` if run from the command line
###############################################################################

if __name__ == "__main__":
    main(7)

###############################################################################
