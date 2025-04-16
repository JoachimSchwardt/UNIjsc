#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCIPY integration benchmarks
"""

from time import perf_counter
import numpy as np
from scipy.integrate import ode, odeint, solve_ivp, RK45
from numba import njit
from rk_integrator import rk_int
from hamilton_integrator_cpp import hamilton_odeint

import rk_integrator_cpp
def rk_int_cpp(func, t_0_values, y_0, atol=1e-8, max_iterations=100):
    """max_iterations == maximal number of iterations per time step"""
    steps = t_0_values.size
    t_values = np.zeros(steps, dtype=np.float64)
    y_current = np.zeros(y_0.size, dtype=np.float64)
    y_next = np.zeros(y_0.size, dtype=np.float64)
    y_previous = np.zeros(y_0.size, dtype=np.float64)
    temp = np.zeros(y_0.size, dtype=np.float64)
    y_values = np.zeros((steps, y_0.size), dtype=np.float64)
    y_values[0] = y_0
    #func, t_0_values, t_values, y_values, 
    #y_current, y_next, y_previous, tau_init, atol, temp
    rk_integrator_cpp.rk_int(func, t_0_values, t_values, y_values, 
                             y_current, y_next, y_previous, atol, 
                             max_iterations, temp)
    return t_values, y_values

@njit
def fkt(t, x):
    return t - x

def y_analytic(t, t0=0, x0=0):
    return (x0 - t0 + 1) * np.exp(-(t-t0)) + t - 1

def abs_rel_error(numeric, analytic):
    return np.abs((numeric - analytic) / analytic)

def benchmark():
    """Absolute relative error and timing for different solvers."""
    x0 = 0.3
    t0 = 0.5
    t_final = 5.0
    t = np.linspace(t0, t_final, 50)
    y_ana = y_analytic(t_final, t0=t0, x0=x0)

    # odeint
    t_start = perf_counter()
    y_odeint = odeint(fkt, x0, t, tfirst=True)[:, 0]
    err_odeint = abs_rel_error(y_odeint[-1], y_ana)
    t_end = perf_counter()
    print(f"ODEINT: {err_odeint:.3e} in {(t_end - t_start)*1e6:.2f} us")

    # solve_ivp
    t_start = perf_counter()
    y_solve_ivp_res = solve_ivp(fkt, (t0, t_final), [x0,], method='LSODA',
                                rtol=1e-8, atol=1e-8)
    _, y_solve_ivp = y_solve_ivp_res.t, y_solve_ivp_res.y[0]
    err_solve_ivp = abs_rel_error(y_solve_ivp[-1], y_ana)
    t_end = perf_counter()
    print(f"SOLVE_IVP: {err_solve_ivp:.3e} in {(t_end - t_start)*1e6:.2f} us")

    # ode
    t_start = perf_counter()
    ode_obj = ode(fkt).set_integrator('LSODA').set_initial_value(x0, t0)
    y_ode = np.array([ode_obj.integrate(t[i])[0] for i in range(1, t.size)
                      if ode_obj.successful()])
    err_ode = abs_rel_error(y_ode[-1], y_ana)
    t_end = perf_counter()
    print(f"ODE: {err_ode:.3e} in {(t_end - t_start)*1e6:.2f} us")


import functools
from matplotlib import pyplot as plt


def ham(p, x):
    return 0.5 * p**2 + x**4 - x**2 + 0.05 * x

@njit    # ?!? breaks all the ODE-solvers. ODEINT and SOLVE_IVP work though...
def abl(t, y):
    return np.array([y[1], -4.0 * y[0]**3 + 2.0 * y[0] - 0.05])

class Methods:
    def __init__(self):
        self.methods = {0 : 'ODEINT', 
                        1 : 'SOLVE_IVP',
                        2 : 'MY_RK4',
                        3 : 'MY_CPP_RK4',
                        4 : 'CPP_ONLY_RK4',}
        #self.methods = {0 : 'ODEINT',
        #                1 : 'SOLVE_IVP',
        #                2 : 'ODE_LSODA',
        #                3 : 'ODE_VODE',
        #                4 : 'ODE_DOPRI5',
        #                5 : 'ODE_DOP853',
        #                # 6 : 'RK45',
        #                }
        self.num_methods = len(self.methods.keys())
        self.counter = 0

    def get_method(self):
        return self.methods[self.counter]

    def increment_counter(self):
        self.counter = (self.counter + 1) % self.num_methods

def switch_solver(event, methods):
    if event.key == 'm':
        methods.increment_counter()
        print(f"Current method: {methods.get_method()}")

def neue_trajektorie(event, ax_phasenraum, ax_energie, zeiten, relf, absf,
                     methods):
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax_phasenraum and mode == '':
        y0 = np.array([event.xdata, event.ydata])  # Startpunkt (x,p)
        E0 = ham(y0[1], y0[0])               # Bestimme Energie zu Beginn
        if E0 <= 0.5:                        # Teste Energie
            method = methods.get_method()
            t_start = perf_counter()
            if method == 'ODEINT':
                y_t = odeint(abl, y0, zeiten, rtol=relf, atol=absf, tfirst=True)
                t_t = zeiten
            elif method == 'SOLVE_IVP':
                res = solve_ivp(abl, (zeiten[0], zeiten[-1]), y0,
                                method='LSODA', rtol=1e-8, atol=1e-8)
                y_t = res.y.T
                t_t = res.t
            elif method == 'ODE':
                ode_obj = (ode(abl)
                           .set_integrator('LSODA')
                           .set_initial_value(y0, zeiten[0]))
                y_t = np.array([ode_obj.integrate(zeiten[i])
                                for i in range(1, zeiten.size)
                                if ode_obj.successful()])
                t_t = zeiten[1:]
            elif method == 'MY_RK4':
                t_t, y_t = rk_int(abl, zeiten, y0, atol=absf)
            elif method == 'MY_CPP_RK4':
                t_t, y_t = rk_int_cpp(abl, zeiten, y0, atol=absf)
            elif method == 'CPP_ONLY_RK4':
                y_t = hamilton_odeint(abl, zeiten, y0, atol=absf)
                t_t = zeiten
            elif method == 'ODE_LSODA':
                ode_obj = (ode(abl)
                           .set_integrator('LSODA')
                           .set_initial_value(y0, zeiten[0]))
                y_t = np.array([ode_obj.integrate(zeiten[i])
                                for i in range(1, zeiten.size)
                                if ode_obj.successful()])
            elif method == 'ODE_VODE':
                ode_obj = (ode(abl)
                           .set_integrator('vode', method='bdf')
                           .set_initial_value(y0, zeiten[0]))
                y_t = np.array([ode_obj.integrate(zeiten[i])
                                for i in range(1, zeiten.size)
                                if ode_obj.successful()])
            elif method == 'ODE_DOPRI5':
                ode_obj = (ode(abl)
                           .set_integrator('dopri5', method='bdf')
                           .set_initial_value(y0, zeiten[0]))
                y_t = np.array([ode_obj.integrate(zeiten[i])
                                for i in range(1, zeiten.size)
                                if ode_obj.successful()])
            elif method == 'ODE_DOP853':
                ode_obj = (ode(abl)
                           .set_integrator('dop853', method='bdf')
                           .set_initial_value(y0, zeiten[0]))
                y_t = np.array([ode_obj.integrate(zeiten[i])
                                for i in range(1, zeiten.size)
                                if ode_obj.successful()])
            # elif method == 'RK45':
            #     ode_obj = RK45(abl, zeiten[0], y0, zeiten[-1],
            #                    rtol=1e-8, atol=1e-8)
            #     y_t = np.zeros((zeiten.size, 2))
            #     y_t[0] = y0
            #     for i in range(1, zeiten.size):
            #         try:
            #             ode_obj.step()
            #             y_t[i] = ode_obj.y
            #         except RuntimeError:
            #             break
            t_end = perf_counter()
            print(f"{method}: new orbit in {(t_end - t_start)*1e3:.2f} ms.")

            # zeichne im Phasenraum-Plotbereich
            ax_phasenraum.plot(y_t[:, 0], y_t[:, 1])   # Zeichne Trajektorie
            # zeichne im Energiefehlerplotbereich
            ax_energie.plot(t_t, ham(y_t[:, 1], y_t[:, 0]) - E0)
            event.canvas.draw()

def main():
    zeiten = np.linspace(0.0, 100.0, 2000)       # Zeiten fuer DGL-Integration
    relf = 1e-8                                  # rel. Fehler, ok ohne Zoom
    absf = 1e-8                                  # abs. Fehler, ok ohne Zoom

    fig = plt.figure()
    # Phasenraum-Plotbereich:
    ax_phasenraum = fig.add_subplot(1, 2, 1, autoscale_on=False, aspect='equal')
    ax_phasenraum.set_title("Phasenraum")
    ax_phasenraum.set_xlabel("$x$")
    ax_phasenraum.set_ylabel("$p$")
    ax_phasenraum.axis([-1.5, 1.5, -1.5, 1.5])

    # Energiefehler-Plotbereich:
    ax_energie = fig.add_subplot(1, 2, 2)
    ax_energie.set_title("Fehler in Energie")
    ax_energie.set_xlabel("$t$")
    ax_energie.set_ylabel("$H(t)-H(0)$")
    ax_energie.set_xlim(0, np.max(zeiten))
    # ax_energie.axis([0, np.max(zeiten), -5e-3, 5e-3])

    # plt.subplots_adjust(wspace=0.40)        # mehr Platz zw. den Plotbereichen

    print("Bewegung eines Teilchens im asymmetrischen Doppelmuldenpotential.")
    print("Maus-Interaktion im Phasenraum:")
    print("linke Maustaste: Startpunkt auswaehlen (Phasenraumbereich)")

    methods = Methods()
    # Bei Mausklick soll die Funktion neue_trajektorie aufgerufen werden,
    # wobei die zusaetzlichen Paramater beim Aufruf mit uebergeben werden:
    klick_funktion = functools.partial(
        neue_trajektorie, ax_phasenraum=ax_phasenraum, ax_energie=ax_energie,
        zeiten=zeiten, relf=relf, absf=absf, methods=methods)
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    fig.canvas.mpl_connect('key_press_event', functools.partial(switch_solver,
                                                                methods=methods))

    plt.show()


if __name__ == "__main__":
    main()
