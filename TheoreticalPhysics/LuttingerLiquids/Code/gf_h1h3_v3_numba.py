#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green's Function for the H1H3 model
"""

import numpy as np
import green_toolkit
from gamma import _gamma_rel, _gamma_real
from numba import njit, objmode, prange
from gf_h1h3_v3 import green_order_0

@njit(fastmath=True)
def _j_bn(k, omega, b, n=0):
    """Integral e^(i_omega_n tau - I k x) csc b+n csc b"""
    prefactor = np.pi * 2**(2*b-2+n) * _gamma_real(1-b) / _gamma_real(b+n)
    gamma_p = _gamma_rel(b/2 - 1j*(omega+k)/4, 1 - b/2 - 1j*(omega+k)/4)
    gamma_m = _gamma_rel((b+n)/2 - 1j*(omega-k)/4, 1 - (b+n)/2 - 1j*(omega-k)/4)
    return prefactor * np.exp(1j*np.pi/2 * n) * gamma_p * gamma_m

@njit(fastmath=True, error_model="numpy")
def alternating_range(mmax):
    mv = np.zeros(2*mmax + 1, dtype=np.int64)
    for i in range(2*mmax + 1):
        if i % 2:
            mv[i] = i//2+1
        else:
            mv[i] = -i//2
    return mv

@njit(fastmath=True, error_model="numpy")
def s_b1s(k, omega, b=0.3, a=1.0, s=1, mmax=9, lmax=5, rtol=1e-6, verbose=False):
    """Integral e^(i_omega_n tau - I k x) csc b+1 csc b sum cot(tau + I sx + I s_a a)"""
    prefactor = 2**(2*b-1)*np.sin(np.pi*b)*_gamma_real(1-b)**2
    res = 0
    mv = alternating_range(mmax)
    if s == +1:
        for mi in range(2*mmax+1):
            m = mv[mi]
            delta = (_j_bn(2j*m-omega-k, 2j*m, b, n=0)
                     * np.exp(-a*(2*m+1j*omega) * np.sign(2*m-omega.imag)))
            res += delta
            for ell in [+1, -1]:
                for l in range(lmax+1):
                    delta = (-2j * (-1)**l/_gamma_real(l+1) * prefactor
                             /_gamma_real(1-b-l) * _gamma_real(b+l+m) /_gamma_real(1+l+m)
                             * np.exp(-np.abs(4*l+2*b+2*m)*a
                                      - 1j*ell*k*a*np.sign(4*l+2*b+2*m))
                             / (2j*b+2j*m + 4j*l + ell * (2j*m - omega - k)))
                    res += delta
                    # if res != 0 and np.abs(delta / res) < rtol:
                    #     if verbose:
                    #         print("Accuracy ", rtol, " obtained at l =", l, " of ", lmax)
                    #     break
            # if res != 0 and np.abs(delta / res) < rtol:
            #     if verbose:
            #         print("Accuracy ", rtol, " obtained at m =", m, " of ", mmax)
            #     break
    elif s == -1:
        for mi in range(2*mmax+1):
            m = mv[mi]
            delta = (_j_bn(2j*m-omega+k, 2j*m, b-1, n=2)
                      * np.exp(-a*(2*m+1j*omega) * np.sign(2*m-omega.imag)))
            res += delta
            for ell in [+1, -1]:
                for l in range(lmax+1):
                    delta = (-2j * (-1)**l/_gamma_real(l+1)*(b-1)/b * prefactor
                              /_gamma_real(1+ell-b-l) * _gamma_real(b+l+m)
                              /_gamma_real(1-ell +l+m + 1e-14)    # regul. for 1/inf == nan
                              * np.exp(-np.abs(4*l+2*b+2*m-2*ell)*a
                                      + 1j*ell*k*a*np.sign(4*l+2*b+2*m-2*ell))
                              / (2j*b+2j*m + 4j*l + ell * (2j*m - omega + k - 2j)))
                    res += delta
                    # if res != 0 and np.abs(delta / res) < rtol:
                    #     if verbose:
                    #         print("Accuracy ", rtol, " obtained at l =", l, " of ", lmax)
                    #     break
            # if res != 0 and np.abs(delta / res) < rtol:
            #     if verbose:
            #         print("Accuracy ", rtol, " obtained at m =", m, " of ", mmax)
            #     break
    return res * 2/np.sinh(a)

@njit(fastmath=True, error_model="numpy")
def simpson_njit(y, x):
    """https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data"""
    n = x.shape[0] - 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1] - x[i]
    result = 0.0
    for i in range(1, n, 2):
        h0, h1 = h[i - 1], h[i]
        hph, hdh, hmh = h1 + h0, h1 / h0, h1 * h0
        result += (hph / 6) * ((2 - hdh) * y[i - 1] + (hph**2 / hmh) * y[i] + (2 - 1 / hdh) * y[i + 1])

    if n % 2 == 1:
        h0, h1 = h[n - 2], h[n - 1]
        result += y[n]     * (2 * h1 ** 2 + 3 * h0 * h1) / (6 * (h0 + h1))
        result += y[n - 1] * (h1 ** 2 + 3 * h1 * h0)     / (6 * h0)
        result -= y[n - 2] * h1 ** 3                     / (6 * h0 * (h0 + h1))
    return result

@njit(fastmath=True, error_model="numpy")
def get_integration_region(peak, width, num_points):
    """
    # peaks are at k'==0 and k'~ -k beta v/pi
    #   -- if separation large enough it is better to split integration
    #   -- separation check looks for peak distance compared to expected width ~10*K_-
    """
    if np.abs(peak) > 2*width:
        mode = 2            # 2 peaks
        vals1 = np.linspace(-width, width, num_points//2)
        vals2 = np.linspace(-width + peak, width + peak, num_points - num_points//2)
        if peak > 0:
            vals = np.concatenate((vals1, vals2))
        else:
            vals = np.concatenate((vals2, vals1))
    else:
        mode = 1            # 1 peak
        if np.abs(peak) > width:
            mode = 3        # 2 peaks relatively close together
        sign = -1 if peak < 0 else 1
        v_max = sign * width + peak
        v_min = -sign * width
        vals = np.linspace(v_min, v_max, num_points)
    return vals, mode

@njit(fastmath=True, error_model="numpy")
def check_peak_resolution(integrand, kp_mode, numkp, peak_tolerance=5e-2):
    # sanity check of the peak detection
    #  (1 vs. 2 peaks, are the maxima sufficiently large compared to the edge values)
    abs_integrand = np.abs(integrand)
    issue_warning = False
    if kp_mode == 1:
        kp_argmax = np.argmax(abs_integrand)
        peak_left = (abs_integrand[0] / abs_integrand[kp_argmax] > peak_tolerance)
        peak_right = (abs_integrand[-1] / abs_integrand[kp_argmax] > peak_tolerance)
        if (peak_left or peak_right):
            issue_warning = True
    elif kp_mode >= 2:
        kp_max1 = np.max(abs_integrand[:numkp//2])
        kp_max2 = np.max(abs_integrand[numkp//2:])
        peak1_left = (abs_integrand[0] / kp_max1 > peak_tolerance)
        peak2_right = (abs_integrand[-1] / kp_max2 > peak_tolerance)
        if kp_mode == 3:    # peaks too close together, do not check middle
            peak1_right = False
            peak2_left = False
        else:
            peak1_right = (abs_integrand[numkp//2] / kp_max1 > peak_tolerance)
            peak2_left = (abs_integrand[numkp//2] / kp_max2 > peak_tolerance)
        if (peak1_left or peak1_right or peak2_left or peak2_right):
            issue_warning = True
    return issue_warning

@njit(fastmath=True, error_model="numpy", parallel=True, nogil=True)
def _green_order_1_single(k, omega=0.0, beta=1.5, K=1.3, v=0.5, a=1.0, w=3,
                          numkp=31, mmaxp=8, mmax=9, lmax=5, delta=0, peak_tolerance=1e-1, kp_sigma=12):
    """Fast implementation of the numerical integration for the first order GF perturbation"""
    M = (1/K + K - 2) / 4
    K_m = (K - 1)/2
    prefactor = w**(2*M + K - 1) / (8*np.pi**4) * beta
    #kpv = np.linspace(-kp, kp, numkp)
    res = 0
    for s in [+1, -1]:
        integrand_kpv = np.zeros(numkp, dtype=np.complex128)
        kpv, kp_mode = get_integration_region(k * beta*v/np.pi, kp_sigma*K_m, numkp)

        # actual calculation
        for kpi in prange(numkp):
            kp = kpv[kpi]
            for m in range(-mmaxp, mmaxp+1):
                for sp in [+1, -1]:
                    integrand_kpv[kpi] += (_j_bn(kp, 2j*m, M - K_m, n=0)
                                          * s_b1s(sp*(beta*v/np.pi * k - kp),
                                                  beta/np.pi * (omega + 1j*delta) - 2j*m,
                                                  K_m, np.pi*a / beta/v, s, mmax, lmax)
                                          * _j_bn(sp*(kp - beta*v/np.pi * k),
                                                  beta/np.pi * (omega + 1j*delta) - 2j*m, K_m, n=1))
        res += simpson_njit(integrand_kpv, kpv) * prefactor * (s * K + 1)
        if check_peak_resolution(integrand_kpv, kp_mode, numkp, peak_tolerance):
            print("Warning: Simpson integration insufficiently peaked "
                  "(k=", k, ", omega=", omega, ", s=", s, "; ", kp_mode, "peak)")
    return res

@njit(fastmath=True, error_model="numpy", parallel=True, nogil=True)
def _green_order_1(k_vals, omega=0.0, beta=1.5, K=1.3, v=0.5, a=1.0, w=3,
                   numkp=31, mmaxp=8, mmax=9, lmax=5, delta=0, peak_tolerance=1e-1, kp_sigma=12):
    """Fast implementation of the numerical integration for the first order GF perturbation"""
    M = (1/K + K - 2) / 4
    K_m = (K - 1)/2
    prefactor = w**(2*M + K - 1) / (8*np.pi**4) * beta
    #kpv = np.linspace(-kp, kp, numkp)
    res = np.zeros(k_vals.shape[0], dtype=np.complex128)
    for ki in prange(k_vals.shape[0]):
        k = k_vals[ki]
        for s in [+1, -1]:
            integrand_kpv = np.zeros(numkp, dtype=np.complex128)
            kpv, kp_mode = get_integration_region(k * beta*v/np.pi, kp_sigma*K_m, numkp)

            # actual calculation
            for kpi in range(numkp):
                kp = kpv[kpi]
                for m in range(-mmaxp, mmaxp+1):
                    for sp in [+1, -1]:
                        integrand_kpv[kpi] += (_j_bn(kp, 2j*m, M - K_m, n=0)
                                              * s_b1s(sp*(beta*v/np.pi * k - kp),
                                                      beta/np.pi * (omega + 1j*delta) - 2j*m,
                                                      K_m, np.pi*a / beta/v, s, mmax, lmax)
                                              * _j_bn(sp*(kp - beta*v/np.pi * k),
                                                      beta/np.pi * (omega + 1j*delta) - 2j*m, K_m, n=1))
            res[ki] += simpson_njit(integrand_kpv, kpv) * prefactor * (s * K + 1)
            if check_peak_resolution(integrand_kpv, kp_mode, numkp, peak_tolerance):
                print("Warning: Simpson integration insufficiently peaked "
                      "(k=", k, ", omega=", omega, ", s=", s, "; ", kp_mode, "peak)")
    return res

def green_order_1(k_vals, omega=0.0, beta=1.5, K=1.3, v=1.0, a=1, w=3, num_params=None, delta=0):
    """First order perturbation theory for SP Green's function"""
    k_vals = np.asarray(k_vals)
    numkp = num_params["numkp"]
    mmaxp, mmax, lmax = num_params["mmaxp"], num_params["mmax"], num_params["lmax"]
    if k_vals.size == 1:
        g_vals = np.array([_green_order_1_single(k_vals[0], omega, beta, K, v, a, w, numkp, mmaxp, mmax, lmax, delta)])
    elif k_vals.size < 4:
        g_vals = _green_order_1(k_vals, omega, beta, K, v, a, w, numkp, mmaxp, mmax, lmax, delta)
    elif k_vals.size > 4 and not np.allclose(k_vals[::-1], -k_vals):
        print("Asymmetric k-values in 'green_order_1_shift_res', comp. cost *= 2!")
        g_vals = _green_order_1(k_vals, omega, beta, K, v, a, w, numkp, mmaxp, mmax, lmax, delta)
    else:
        g_vals = _green_order_1(k_vals[:k_vals.size//2+1], omega, beta, K, v, a, w,
                                numkp, mmaxp, mmax, lmax, delta)
        g_vals = np.concatenate((g_vals, g_vals[k_vals.size//2 - (k_vals.size % 2)::-1]))
    green = np.array([np.array([[0, g_vals[i]], [g_vals[i], 0]])
                      for i in range(g_vals.size)])
    return green

def green_perturbative(k_vals, omega_vals=0, beta=1.5, K=1.3, v=0.5, g=0.2, a=1.0, w=3,
                       num_params=None, delta=1e-14):
    """Perturbative calculation of the SP Green's function
    num_params :: dictionary, {"order" : 1, "mmax" : 2, "mmin" : -2, "lmax" : 1}
    """
    omega_vals = np.asarray(omega_vals)
    k_vals = np.asarray(k_vals)
    green = np.zeros((k_vals.size, omega_vals.size, 2, 2), dtype=complex)
    if num_params is None:
        num_params = {"order" : 1, "mmax" : 2, "mmaxp" : 0, "lmax" : 2, "numkp" : 21}
    for i_omega, omega in enumerate(omega_vals):
        green_vals = green_order_0(k_vals, omega, beta, K, v, w, delta)
        if num_params["order"] >= 1:
            green_vals += -g * green_order_1(k_vals, omega, beta, K, v, a, w, num_params, delta)
        green[:, i_omega] = green_vals
    return green


def get_free_K0(u_plus, vf, a=1, factor=1):
    return 1 / np.sqrt(1 + 2 * u_plus * a / np.pi / vf * factor)


def green_eff_ua(k_vals, omega_vals, beta=1, K=1, v=0.5, w=np.pi):
    k_vals = np.asarray(k_vals)
    omega_vals = np.asarray(omega_vals)
    args = (beta, K, v, w)
    green = np.zeros((k_vals.size, omega_vals.size, 2, 2), dtype=complex)
    for i_omega, omega in enumerate(omega_vals):
        green[:, i_omega] = green_order_0(k_vals, omega, *args)
    U = green_toolkit.U
    green_ab = U @ green @ U
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green_ab)
    self_energy = np.copy(hamilton)
    for (ia, ib) in zip([0, 1, 1], [1, 0, 1]):
        self_energy[:, :, ia, ib] = 0
    hamilton_eff = np.zeros_like(self_energy)
    hamilton_eff += self_energy
    for (ia, ib) in zip([0, 1], [1, 0]):
        for iw in range(omega_vals.size):
            hamilton_eff[:, iw, ia, ib] += v * K * k_vals
    green_eff = green_toolkit.green_from_hamilton(omega_vals, hamilton_eff)
    return green_eff


def main():
    print(__doc__)
    k=[0.001]; omega=[0]; beta=np.pi; K=1.3; v=1.0; g=0.03; w=np.pi
    #green_perturbative(k, omega, beta, K, v, g, model="density a shift", alpha=alpha)
    print(green_perturbative(k, omega, beta, K, v, g, w=w))
    # [-1.05562884e+00-46.10705909j  9.26608840e-09+11.6801032j ]
    #    [ 9.26608840e-09+11.6801032j   1.05562884e+00-46.10705909j]
    return 0


if __name__ == "__main__":
    main()
