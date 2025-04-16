#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple step-size adaptive implementation of the Runge-Kutta 4 method.
    y_{n+1} = y_n + tau/6 * (k_1 + 2*k_2 + 2*k_3 + k_4),
    t_{n+1} = t_n + tau,
    k_1 = f(t_n, y_n),
    k_2 = f(t_n + tau/2, y_n + tau * k_1/2),
    k_3 = f(t_n + tau/2, y_n + tau * k_2/2),
    k_4 = f(t_n + tau, y_n + tau * k_3)
"""

import numpy as np
from numba import njit
from scipy.integrate import odeint

@njit
def rk_step(func, t_n, y_n, tau):
    """Execute a single step of the RK4 method."""
    k_1 = func(t_n, y_n)
    k_2 = func(t_n + tau/2, y_n + tau * k_1/2)
    k_3 = func(t_n + tau/2, y_n + tau * k_2/2)
    k_4 = func(t_n + tau, y_n + tau * k_3)
    y_new = y_n + tau/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
    return y_new

@njit
def estimate_error(func, y_n, t_n, y_n_plus1, tau):
    """Estimate the error of a single step due to the current size."""
    y_n_plus1_minus1 = rk_step(func, t_n + tau, y_n_plus1, -tau)
    return np.max(np.abs(y_n_plus1_minus1 - y_n))

@njit
def adapt_stepsize(err, tau, min_update_ratio=2.0, atol=1e-8):
    """Adapt the stepsize 'tau' based on the current error estimate."""
    abs_err_ratio = err / atol
    if abs_err_ratio < 1 / min_update_ratio:
        tau_new = min_update_ratio * tau
    elif abs_err_ratio < 1.0:
        tau_new = tau
    elif abs_err_ratio < min_update_ratio:
        tau_new = tau / min_update_ratio
    else:
        tau_new = tau / (min_update_ratio * abs_err_ratio**(0.25))    # expected error is h**4

    redo_step = tau_new < tau
    return tau_new, redo_step

@njit
def _rk_int_step(func, t_initial, t_final, y_initial, tau=None, atol=1e-8):
    """Integrate an ODE from t_initial to t_final starting from y_initial (SINGLE STEP)."""
    if tau is None:
        tau = t_final - t_initial     # try integrating in one step

    t_current = t_initial
    y_current = y_initial
    while t_current < t_final:
        redo_step = True
        while redo_step:    # iteratively adjust step size to match desired precision
            y_next = rk_step(func, t_current, y_current, tau)
            err = estimate_error(func, y_current, t_current, y_next, tau)
            tau_new, redo_step = adapt_stepsize(err, tau, atol=atol)
            if redo_step:
                tau = tau_new

        t_current += tau
        y_current = y_next

    return t_current, y_current

@njit
def _rk_int(func, t_values, y_0, tau_init=None, atol=1e-8):
    """Integrate an ODE for given initial value y_0 for the times in 't_values'."""
    y_res = np.zeros((t_values.size, y_0.size))
    t_res = np.zeros(t_values.size)
    y_res[0] = y_0
    t_res[0] = t_values[0]
    for i in range(1, t_values.size):
        t_initial = t_res[i-1]
        t_final = t_values[i]
        y_initial = y_res[i-1]
        t_new, y_new = _rk_int_step(func, t_initial, t_final, y_initial, tau_init, atol=atol)
        t_res[i] = t_new
        y_res[i] = y_new

    return t_res, y_res

def rk_int(func, t_values, y_0, tau_init=None, atol=1e-8):
    """Wrapper for the RK4-integrator"""
    y_0 = np.asarray(y_0)
    if y_0.ndim == 0:
        y_0 = np.expand_dims(y_0, 0)
    return _rk_int(func, t_values, y_0, tau_init, atol)

def main():
    """Test RK integrator"""
    print(__doc__)
    @njit
    def fkt(t, x):
        return -x + t

    def y_analytic(t, t_0=0, x_0=0):
        return (x_0 - t_0 + 1) * np.exp(-(t-t_0)) + t - 1

    def abs_rel_error(numeric, analytic):
        return np.abs((numeric - analytic) / analytic)

    mode = "single"
    # mode = "many"
    x_0 = 0.3
    t_0 = 0.5
    t_final = 5.0
    t_values = np.linspace(t_0, t_final, 50)
    if mode == "single":
        t_rk, y_rk = rk_int(fkt, t_values, x_0)
        print("Error: ", abs_rel_error(y_rk[-1,0], y_analytic(t_rk[-1], t_0, x_0)))

        # ODEINT comparison:
        y_odeint = odeint(fkt, x_0, t_values, tfirst=True)[:, 0]
        print("Error: ", abs_rel_error(y_odeint[-1], y_analytic(t_final, t_0, x_0)))

    if mode == "many":
        # function calls per run:
        # v100: _rk_int_step : 49, err_est : 116, rk_step : 430, fkt : 1720, ~6 sec
        # v101: _rk_int_step : 49, err_est :  49, rk_step :  98, fkt :  392, ~1.5 sec
        for _ in range(1000):
            t_rk, y_rk = rk_int(fkt, t_values, x_0)

if __name__ == "__main__":
    main()
