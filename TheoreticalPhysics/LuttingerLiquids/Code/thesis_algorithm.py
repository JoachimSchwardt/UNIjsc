#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:21:18 2024

@author: ag_budich1
"""

import numpy as np
from time import perf_counter as pc
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
import thesis_toolkit as tlk
import thesis_gf as gf


def get_mmln(num_params):
    return f"{num_params['mmaxp']}.{num_params['mmax']}.{num_params['lmax']}.{num_params['numkp']}"
def get_ep_delta_string():
    return r"$\Delta E_{\mathrm{EP}}$"
def get_ep_offset_string():
    return r"$-\langle E_{\mathrm{EP}}\rangle$"
def get_label(arg):
    if arg is None:
        raise ValueError
    if arg == "beta":
        return r"$\beta$"
    elif arg == "vf":
        return r"$v_{\mathrm{F}}$"
    elif arg == "u_a":
        return r"$U_{\mathrm{A}}$"
    else:
        return f"${arg}$"

def get_par_string(arg, mod=".3f"):
    if arg is None:
        raise ValueError
    if isinstance(arg, (np.ndarray, list)):
        par = f"{arg[0]:{mod}}.{arg[-1]:{mod}}"
    elif isinstance(arg, dict):
        par = get_mmln(arg)
    else:
        par = f"{arg:{mod}}"
    return par

def get_pars_string(args, labels, mod=".3f"):
    pars = "_".join([label + get_par_string(arg, mod) for (label, arg) in zip(labels, args)])
    return pars

def find_iterable_argument(args):
    for indx, arg in enumerate(args):
        if isinstance(arg, (np.ndarray, list)):    # find the (assumed only) iterable argument
            iter_arg = arg                         # (e.g. 'beta' is array-like and others scalar)
            iter_indx = indx
            break
    return iter_arg, iter_indx

def get_modified_iterable_argument(args, indx):
    args_i = []
    for arg in args:
        if isinstance(arg, (np.ndarray, list)):
            args_i.append(arg[indx])
        else:
            args_i.append(arg)
    return args_i


def get_energies(green, omega_vals=None):
    """(green, omega) -> (energies)"""
    if omega_vals is None:
        if green.ndim == 4:
            omega_vals = np.zeros(green.shape[1])
        elif green.ndim == 3:
            omega_vals = np.array([0])
        else:
            raise ValueError(f"No default omega values for GF of shape {green.shape}")
    hamilton = tlk.hamilton_from_green(omega_vals, green)
    if hamilton.ndim == 4:
        energies = np.array([[tlk.eigenvalues_2x2(hamilton[i_k, i_w])
                              for i_k in range(hamilton.shape[0])]
                             for i_w in range(hamilton.shape[1])])
    elif hamilton.ndim == 3:
        energies = np.array([tlk.eigenvalues_2x2(hamilton[i_k])
                             for i_k in range(hamilton.shape[0])])
    elif hamilton.ndim == 2:
        energies = tlk.eigenvalues_2x2(hamilton)
    else:
        raise ValueError(f"{hamilton.ndim = } not supported for conversion to eigenvalues")
    return np.squeeze(energies)

def get_energies_from_heff(h_eff):
    return np.array([tlk.eigenvalues_2x2(h_eff[i]) for i in range(h_eff.shape[0])])

def energies_to_ep_size(energies):
    """(E_+, E_-) -> (ep_delta, ep_offset)"""
    ep_delta = np.abs((energies[1] - energies[0]).imag) / 2
    ep_offset = np.mean(energies)
    return ep_delta, ep_offset

def ep_search(green_function, k_min=0, k_max=1, omega=0, args=None,
              ktol=1e-4, etol=1e-15, max_iter=80, verbose=False):
    """Search for the EP using interval splitting"""
    ktol_old = ktol     # back etol to recover from clamping (difficulties if given ktol < 1e-15)
    etol_old = etol
    def energy_function(k_val):
        green = green_function([k_val], [omega], *args)
        energies = get_energies(green, [omega])
        return energies

    e1_min, e2_min = energy_function(k_min)
    e1_max, e2_max = energy_function(k_max)
    if np.abs(e1_min.real) > np.max([etol, 1e-15]) or np.abs(e1_max.real) < np.max([etol, 1e-15]):
        raise ValueError("EP is not within initial bounds!")
    for i in range(max_iter):
        k_mean = (k_min + k_max) / 2
        if ktol < 1e-15:
            if k_max > 1e-8:
                ktol = 1e-15
            else:
                ktol = ktol_old
        if etol < 1e-15:
            if k_max > 1e-8:
                etol = 1e-15
            else:
                etol = etol_old
        if (k_max - k_min) < ktol:
            if verbose:
                print(f"Iteration {i} for {args=}")
            return k_mean
        e1_mean, e2_mean = energy_function(k_mean)
        if np.abs(e1_mean.real) < etol:
            k_min = k_mean
        else:
            k_max = k_mean
    print(f"Warning: Error tolerance {ktol:.2e} not achieved in {max_iter} iterations. "
          f"Final uncertainty approximately {k_max - k_min:.2e}.")
    return k_mean

def ep_search_iter(green_function, k_min=0, k_max=1, omega=0, args=None, verbose=False,
                   ktol=1e-5, etol=1e-15, max_iter=80, initial_buffer=1, ep_retry_buffer=30):
    """(gf, k_min, k_max, omega, args) -> (k_ep)"""
    iter_arg, _ = find_iterable_argument(args)
    k_epv = np.zeros(len(iter_arg))
    for k_epi in range(k_epv.size):
        args_i = get_modified_iterable_argument(args, k_epi)
        ep_in_bounds = False
        ep_retry_ctr = 0
        while not ep_in_bounds:
            try:
                k_ep = ep_search(green_function, k_min, k_max, omega, args_i, ktol, etol, max_iter, verbose)
                ep_in_bounds = True
            except ValueError:
                if verbose:
                    print(f"Warning: EP not in bounds [{k_min = :.3e}, {k_max = :.3e}] at indx {k_epi}, "
                          f"retrying with larger bounds ({args_i = })")
                k_min /= 2
                k_max *= 2
                ep_retry_ctr += 1
            if ep_retry_ctr >= ep_retry_buffer:
                raise RuntimeError(f"EP retry buffer overrun at {args_i = }")
        k_epv[k_epi] = k_ep
        # if k_epi == 0:
        #     k_min = k_ep * (1 - initial_buffer)
        #     k_max = k_ep * (1 + initial_buffer)
        # else:
        #     delta_k = k_epv[k_epi] - k_epv[k_epi-1]
        #     k_min, k_max = sorted([k_ep * (1 - np.sign(delta_k)*initial_buffer), k_ep + 2*delta_k])
        k_min = 0
        k_max = k_ep * (1 + initial_buffer)
        if k_epi % 50 == 0:
            print(f"Finished iteration {k_epi}")
    return k_epv

def ep_size_iter(green_function, omega=0, args=None, **kwargs):
    iter_arg, _ = find_iterable_argument(args)
    ep_deltav = np.zeros(len(iter_arg))                    # imaginary energy gap of EP at k=0
    ep_offsetv = np.zeros(len(iter_arg), dtype=complex)    # average value of the two branches
    for epi in range(ep_deltav.size):
        args_i = get_modified_iterable_argument(args, epi)
        green = green_function([0], omega, *args_i, **kwargs)
        energies = get_energies(green)
        if np.any(np.imag(energies) > 0):
            print("Warning: Positive imaginary part in energy eigenvalue at {args_i = }")
        ep_deltav[epi], ep_offsetv[epi] = energies_to_ep_size(energies)
        if epi % 50 == 10:
            print(f"Finished iteration {epi}")
    return ep_deltav, ep_offsetv

def ep_size_num(u_a_num, u_b_num=None, N=None):
    """(u_a, u_b) -> (ep_delta, ep_offset)"""
    u_a_num = np.asarray(u_a_num)
    if u_b_num is None:
        u_b_num = np.zeros_like(u_a_num)
    u_b_num = np.asarray(u_b_num)
    ep_deltav = []
    ep_offsetv = []
    for (u_a_val, u_b_val) in zip(u_a_num, u_b_num):
        k_vals, omega_vals, _, h_eff = gf.green_numeric(u_a_val, u_b_val, N)
        energies = tlk.eigenvalues_2x2(h_eff[k_vals.size//2, omega_vals.size//2])
        ep_delta, ep_offset = energies_to_ep_size(energies)
        ep_deltav.append(ep_delta)
        ep_offsetv.append(ep_offset)
    return ep_deltav, ep_offsetv

def get_weights(x_vals, decay=0, exponent=4):
    return 1 / (1 + decay * x_vals**exponent)

def get_weight_func(decay=50, exponent=4):
    def getter(x_vals):
        return get_weights(x_vals, decay, exponent)
    return getter

def ll_par_fit(energies_num, k_vals, omega=0, K=0.7, g=0, beta=5, alpha=None, w=np.pi, a=1,
               num_params={"mmax" : 5, "mmaxp" : 5, "lmax" : 3, "numkp" : 48},
               weight_func=get_weights, mute_ptr_warnings=False):
    """Fit the LL PT EP to given numerical data -> (K, v, g)"""
    bounds_K = (0.3, 1 - 5e-3)
    bounds_v = (0.1, 10)
    bounds_g = (0, np.inf)
    bounds_beta = (0.5, 15)
    if beta is None:
        beta = 5
        fixed_beta = False
        bounds = [bounds_K, bounds_v, bounds_g, bounds_beta]
    else:
        fixed_beta = True
        bounds = [bounds_K, bounds_v, bounds_g]
    diff = np.abs(np.diff(energies_num[:, 0].real))
    try:
        argmin = next(i for i, value in enumerate(diff) if value < diff[0] / 10)
    except StopIteration:
        argmin = np.argmin(diff)
    if argmin < 4:
        raise IndexError("Probably energy window too narrow to extract v")
    v = diff[argmin - 2] / np.abs(k_vals[argmin - 1] - k_vals[argmin - 2])
    k_max = np.abs(k_vals[argmin - 4]) + 1e-2
    indx = tlk.arg_restrict(k_vals, -k_max, k_max)
    k_vals = k_vals[indx]
    energies_num = energies_num[indx]

    indx_center = np.where(np.abs(k_vals) < 1e-8)[0][0]
    ep_delta, ep_offset = energies_to_ep_size(energies_num[indx_center])
    # if ep_delta > 1e-5:
    #     g = 3
    # def local_minimizer(x):
    #     K, g = x
    #     w_val = tlk.get_w(w, alpha, beta, v)
    #     args = (beta, [K], v, g, a, w_val, num_params)
    #     [ep_delta_val], [ep_offset_val] = ep_size_iter(gf.green_perturbative, args=args,
    #                                                    mute_ptr_warnings=mute_ptr_warnings)
    #     return np.abs(ep_delta_val - ep_delta)**2 + np.abs(ep_offset_val - ep_offset)**2
    # result = minimize(local_minimizer, [K, g], bounds=[bounds_K, bounds_g],
    #                   tol=1e-8, method="SLSQP")
    # K, g = result.x
    # print(f"LL par fit: Local Optimization -> {K = :.4f}, {v = :.3f}, {g = :.2e}, {k_max = :.3f}")
    # print(f"LL par fit: Least squares error {result.fun : .2e} after {result.nit} iterations")

    energies_num_real_max = np.max(energies_num.real, axis=1)
    energies_num_imag_max = np.max(energies_num.imag, axis=1)
    energies_num_real_min = np.min(energies_num.real, axis=1)
    energies_num_imag_min = np.min(energies_num.imag, axis=1)
    weights = weight_func(k_vals * a)
    def global_minimizer(x):
        if fixed_beta:
            K, v, g = x
            beta_val = beta
        else:
            K, v, g, beta_val = x
        w_val = tlk.get_w(w, alpha, beta_val, v)
        args = (beta_val, K, v, g, a, w_val, num_params)
        green = gf.green_perturbative(k_vals, omega, *args, mute_ptr_warnings=mute_ptr_warnings)
        energies = get_energies(green, omega)
        energies_real_max = np.max(energies.real, axis=1)
        energies_imag_max = np.max(energies.imag, axis=1)
        energies_real_min = np.min(energies.real, axis=1)
        energies_imag_min = np.min(energies.imag, axis=1)
        abs_diff_sqr_real_max = np.abs(energies_real_max - energies_num_real_max)**2
        abs_diff_sqr_imag_max = np.abs(energies_imag_max - energies_num_imag_max)**2
        abs_diff_sqr_real_min = np.abs(energies_real_min - energies_num_real_min)**2
        abs_diff_sqr_imag_min = np.abs(energies_imag_min - energies_num_imag_min)**2
        abs_diff_sqr = (abs_diff_sqr_real_max + abs_diff_sqr_imag_max
                        + abs_diff_sqr_real_min + abs_diff_sqr_imag_min)
        return np.sum(abs_diff_sqr * weights)
    indx = np.abs(k_vals) <= 1/2
    indx4 = np.concatenate((indx, indx, indx, indx))
    ydata = np.concatenate((energies_num_real_max, energies_num_imag_max,
                            energies_num_real_min, energies_num_imag_min))
    def fit(k, K, v, g):
        w_val = tlk.get_w(w, alpha, beta, v)
        args = (beta, K, v, g, a, w_val, num_params)
        green = gf.green_perturbative(k, omega, *args, mute_ptr_warnings=mute_ptr_warnings)
        energies = get_energies(green, omega)
        res = np.concatenate((np.max(energies.real, axis=1), np.max(energies.imag, axis=1),
                              np.min(energies.real, axis=1), np.min(energies.imag, axis=1)))
        return res
    if fixed_beta:
        result = minimize(global_minimizer, [K, v, g], bounds=bounds, tol=1e-8, method="SLSQP")
        K, v, g = result.x
    else:
        result = minimize(global_minimizer, [K, v, g, beta], bounds=bounds, tol=1e-8, method="SLSQP")
        K, v, g, beta = result.x
    print(f"LL par fit: Global Optimization -> {K = :.4f}, {v = :.3f}, {g = :.2e}, {beta = :.3f}")
    print(f"LL par fit: Least squares error {result.fun : .2e} after {result.nit} iterations")
    print(result)
    if fixed_beta:
        par, cov = curve_fit(fit, k_vals[indx], ydata[indx4], p0=(K,v,g))
        K, v, g = par
        return K, v, g, beta, cov
    return K, v, g, beta

def ll_par_fit_gf(green_num, k_vals, omega=0, K=0.7, g=0, beta=5, vf=0.5, alpha=None, w=np.pi, a=1,
                  num_params={"mmax" : 5, "mmaxp" : 5, "lmax" : 3, "numkp" : 48},
                  weight_func=get_weight_func(5, 2), mute_ptr_warnings=False):
    """Fit the LL PT GF to given numerical data -> (K, v, g)
    green_num must be in LR-basis and be 3d (i.e. frequency already selected)"""
    weights = weight_func(k_vals)
    def func_g0(x):
        K, v = x
        w_mod = tlk.get_w(w, alpha, beta, v)
        green = gf.green_perturbative(k_vals, [omega], beta, K=K, v=v, w=w_mod, order=0)
        error = np.abs(green[:, 0, 0, 0] - green_num[:, 0, 0])
        return np.sum(error * weights)
    result = minimize(func_g0, [K, vf], bounds=[(0.3, 1.0), (0.01, 100)], tol=1e-8, method="SLSQP")
    K, v = result.x
    print(f"LL par fit: G0 Optimization -> {K = :.4f}, {v = :.3f}")
    print(f"LL par fit: Least squares error {result.fun : .2e} after {result.nit} iterations")

    def func_g1(x):
        [g] = x
        w_mod = tlk.get_w(w, alpha, beta, v)
        green = gf.green_perturbative(k_vals, [omega], beta, K=K, v=v, g=g, w=w_mod, order=1,
                                      mute_ptr_warnings=mute_ptr_warnings)
        error = np.abs(green[:, 0, 0, 1] - green_num[:, 0, 1])
        return np.sum(error * weights)
    result = minimize(func_g1, [g], tol=1e-8, method="SLSQP")
    [g] = result.x
    print(f"LL par fit: G1 Optimization -> {g = :.3e}")
    print(f"LL par fit: Least squares error {result.fun : .2e} after {result.nit} iterations")
    def fit(k, K, v):
        w_mod = tlk.get_w(w, alpha, beta, v)
        green = gf.green_perturbative(k_vals, [omega], beta, K=K, v=v, g=g, w=w_mod, order=1,
                                      mute_ptr_warnings=mute_ptr_warnings)
        return reduce_complex_array(green[:, 0, 0, 1])
    indx = np.abs(k_vals) <= 1/2
    indx2 = np.append(indx, indx)
    ydata = reduce_complex_array(green_num[:, 0, 1])
    par, cov = curve_fit(fit, k_vals[indx], ydata[indx2], p0=(K,v,g))
    print(par, cov)
    K, v, g = par
    return K, v, g, cov

def ll_par_ua_eq_ub_gf(u_a=0.8, beta=1, K=0.7, v=0.5, w=np.pi, alpha=None,
                       weight_func=get_weight_func(5, 2)):
    """estimate the LL parameters for given (u_a==u_b) -> (K, v)
    if 'alpha' is 'None' use the given 'w', else 'alpha' takes precedence and 'w = w(alpha)'."""
    k_num, omega_num, green_num, _ = gf.green_numeric(u_a, u_a, beta=beta)
    green_num_lr = tlk.basis_ab_to_lr(green_num)
    weights = weight_func(k_num)
    def function(x):
        K, v = x
        w_mod = tlk.get_w(w, alpha, beta, v)
        green = gf.green_perturbative(k_num, [0], beta, K=K, v=v, w=w_mod, order=0)
        error = np.abs(green[:, 0, 0, 0] - green_num_lr[:, omega_num.size//2, 0, 0])
        return np.sum(error * weights)
    result = minimize(function, [K, v], bounds=[(0.3, 1-5e-3), (0.01, 100)], tol=1e-8, method="SLSQP")
    K, v = result.x
    print(f"LL par fit: Global Optimization -> {K = :.4f}, {v = :.3f}")
    def fit(k, K, v):
        w_mod = tlk.get_w(w, alpha, beta, v)
        green = gf.green_perturbative(k, [0], beta, K=K, v=v, w=w_mod, order=0)
        return reduce_complex_array(green[:,0,0,0])
    indx = np.abs(k_num) <= 1/2
    indx2 = np.append(indx, indx)
    ydata = reduce_complex_array(green_num_lr[:, omega_num.size//2, 0, 0])
    par, cov = curve_fit(fit, k_num[indx], ydata[indx2], p0=(K,v))
    print(par, cov)
    K, v = par
    return K, v, tlk.get_w(w, alpha, beta, v), cov

def reduce_complex_array(arr):
    return np.append(arr.real, arr.imag)


# RENORMALIZATION GROUP FLOW EQUATIONS


def rg_beta_ope(couplings, l_val):
    c2, c4, K, v = couplings
    c2p = (1 - K) * c2 - 2*np.pi*c2*c4
    c4p = (2 - 4*K) * c4 + 2*np.pi*K**2*c2**2
    Kp = -16*np.pi**2*K**2*c4**2 - 8*np.pi**2*(1-K)**2*K**2 * c2**2
    vp = v*8*np.pi**2*K**2*(2-K)*(1 - K) * c2**2
    return np.array([c2p, c4p, Kp, vp])

def rg_beta_ms(couplings, l_val):
    c2, c4, K, v = couplings
    c2p = (1 - K) * c2 - 8*np.pi*c2*c4
    c4p = (2 - 4*K) * c4
    Kp = 128*np.pi**2 * K**3 * (c2**2 - 16 * c4**2)
    vp = -128*np.pi**2 * K**3 * v * c2**2
    return np.array([c2p, c4p, Kp, vp])

def get_l_vals(l_max=1, l_count=200):
    l_vals = np.linspace(0, l_max, l_count)
    return l_vals

def get_rg_flow(rg_beta, couplings0, l_vals, **kwargs):
    couplings = odeint(rg_beta, couplings0, l_vals, **kwargs)
    return couplings

def get_g20g40(u_a=1.5, u_b=0.0, a=1, v0=0.5):
    u_minus = (u_a - u_b) / 2
    u_plus = (u_a + u_b) / 2
    g20 = -u_minus*a / (4*np.pi**2*v0)
    g40 = -u_plus*a / (4*np.pi**2*v0)
    return g20, g40

def get_rg_flow_h2h4(l_vals, vf=0.5, u_a=1.5, u_b=-1.0, a=1, K_0=1, method="ope", magnitude=True, **kwargs):
    rg_method = globals()["rg_beta_" + method]
    v0 = vf / K_0
    g20, g40 = get_g20g40(u_a, u_b, a, v0)
    couplings = get_rg_flow(rg_method, [g20, g40, K_0, v0], l_vals, **kwargs)
    if magnitude:
        return np.abs(couplings)
    return couplings

# SPECIAL FUNCTIONS and ERROR ANALYSIS

def get_rel_err(measurement, true_value):
    return np.abs(1 - measurement/true_value)

def s_b1s_arg(args):
    """(k, omega, b, a, s, mmax, lmax) -> (s_b1s)"""
    arg, _ = find_iterable_argument(args)
    arg = np.asarray(arg)
    result = np.zeros(arg.size, dtype=complex)
    for i in range(arg.size):
        args_i = get_modified_iterable_argument(args, i)
        result[i] = gf.s_b1s(*args_i)
    return result

def err_sb1s_arg(args, mmax=30, lmax=30):
    """(k, omega, b, a, s, mmax, lmax) -> (rel_error)"""
    result = s_b1s_arg(args)
    args_limit = (*args[:5], mmax, lmax)
    limit = gf.s_b1s(*args_limit)
    return get_rel_err(result, limit)
