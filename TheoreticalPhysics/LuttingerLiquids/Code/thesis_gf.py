#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green's Function for the H1H3 model

k = -1; omega = 0; s = 1; beta = 6; K = 1.6; v = 7; w =  0.15; mmaxp =  5; mmax =  5; lmax =  3
numkp=48; kp_sigma=15; delta=0; a=1; peak_tolerance=1e-1
M = get_M(K); K_m = get_Km(K)
integrand_kpv = np.zeros(numkp, dtype=np.complex128)
kpv, kp_mode = get_integration_region(k * beta*v/np.pi, kp_sigma*np.abs(K_m), numkp)
for kpi in prange(numkp):
    kp = kpv[kpi]
    integrand_kpv[kpi] = _green_order_1_integrand(
        kp, M, K_m, beta*v/np.pi * k, beta/np.pi * (omega + 1j*delta), s,
        np.pi*a / beta/v, mmaxp, mmax, lmax)
#tlk.plot_complex(plt.subplots()[1], kpv, integrand_kpv)
check_peak_resolution(integrand_kpv, kp_mode, numkp, peak_tolerance)
"""

import numpy as np
from numba import njit, prange
import thesis_toolkit as tlk
from gamma import _gamma_rel, _gamma_real
from greens_j import get_data


def green_numeric(u_a=1.5, u_b=1.5, N=None, beta=None):
    """(u_a, u_b) -> (k, omega, green, h_eff)"""
    if N is None:
        path = f"data_joachim/N*SBA_UA{u_a:.6f}_UB{u_b:.6f}_h0.030000"
    else:
        path = f"data_joachim/N{N}*SBA_UA{u_a:.6f}_UB{u_b:.6f}_h0.030000"
    if beta is not None:
        if beta == 1:
            path += f"_beta{beta}"
    path += "/"
    data = get_data(path)
    k_vals = data["k_values"] - np.pi
    omega_vals = data["omega_values"]
    green = data["ret_green_k_w"]
    h_eff = data["h_eff_k_w"]
    if k_vals.size == 80:
        green = np.roll(green, -40, axis=0)
        h_eff = np.roll(h_eff, -40, axis=0)
    return k_vals, omega_vals, green, h_eff


@njit(fastmath=True)
def _j_bn(k, omega, b, n=0):
    """Integral e^(i_omega_n tau - I k x) csc b+n csc b"""
    prefactor = np.pi * 2**(2*b-2+n) * _gamma_real(1-b) / _gamma_real(b+n)
    gamma_p = _gamma_rel(b/2 - 1j*(omega+k)/4, 1 - b/2 - 1j*(omega+k)/4)
    gamma_m = _gamma_rel((b+n)/2 - 1j*(omega-k)/4, 1 - (b+n)/2 - 1j*(omega-k)/4)
    return prefactor * np.exp(1j*np.pi/2 * n) * gamma_p * gamma_m


@njit(fastmath=True, error_model="numpy")
def j_bn(k, omega, b, n=0):
    size = k.shape[0]
    res = np.zeros(size, dtype=np.complex128)
    for i in range(size):
        res[i] = _j_bn(k[i], omega, b, n)
    return res


def green_order_0(k_vals, omega=0.0, beta=1.1, K=0.6, v=1.0, w=3, delta=0):
    """Zeroth order perturbation theory for SP Green's function"""
    k_vals = np.asarray(k_vals)
    M = (K + 1/K - 2) / 4
    prefactor = -beta/(2*np.pi**2) * w**(2*M)
    g_rr = prefactor * j_bn(-beta*v/np.pi * k_vals, beta/np.pi*(omega + 1j * delta), M, n=1)
    g_ll = prefactor * j_bn(beta*v/np.pi * k_vals, beta/np.pi*(omega + 1j * delta), M, n=1)
    green = np.array([np.array([[g_rr[i], 0], [0, g_ll[i]]])
                      for i in range(k_vals.size)])
    return green


# @njit(fastmath=True, error_model="numpy")
# def s_b1s(k, omega, b=0.3, a=1.0, s=1, mmax=9, lmax=5):
#     """Integral e^(i_omega_n tau - I k x) csc b+1 csc b sum cot(tau + I sx + I s_a a)"""
#     sm1 = (s == -1)     # s is minus 1 --> some parameters change
#     coeff = -1j * np.pi*s * 4**b * _gamma_real(1-b + sm1) / _gamma_real(b + (s==-1))
#     res = 0
#     for m in range(-mmax, mmax+1):
#         res += (_j_bn(k+s*(omega-2j*m), 2j*m, b - sm1, 1-s)
#                 * np.exp((-1j*omega - 2*m) * a * np.sign(2*m - omega.imag)))
#         for ell in [+1,-1]:
#             for l in range(lmax+1):
#                 kl = k + 2j*ell * (2*l+m+b-sm1*ell)
#                 res += ((-1)**l / _gamma_real(1+l) * _gamma_rel(b+l+m, 1-b+sm1*ell - l)
#                         / _gamma_real(1-sm1*ell + l + m + 1e-14) * coeff
#                         * np.exp(1j*kl*a*np.sign(kl.imag)) / (kl + s*(omega - 2j*m))*ell)
#     return res * 2/np.sinh(a)
@njit(fastmath=True, error_model="numpy")
def s_b1s(k, omega, b=0.3, a=1.0, s=1, mmax=9, lmax=5):
    """Integral e^(i_omega_n tau - I k x) csc b+1 csc b sum cot(tau + I sx + I s_a a)"""
    sm1 = (s == -1)     # s is minus 1 --> some parameters change
    coeff = -1j * 4**b * _gamma_real(1-b + sm1) / _gamma_real(b + sm1) * np.sin(np.pi* b)
    res = 0
    for m in range(-mmax, mmax+1):
        res += (_j_bn(k+s*(omega-2j*m), 2j*m, b - sm1, 1-s)
                * np.exp((-1j*omega - 2*m) * a * np.sign(2*m - omega.imag)))
        for ell in [+1,-1]:
            for l in range(lmax+1):
                kl = k + 2j*ell * (2*l+m+b-sm1*ell)
                ratio = _gamma_rel(b+l+m, 1-sm1*ell + l + m + 1e-14) * _gamma_rel(b-sm1*ell + l, 1+l)
                res += ratio * coeff* np.exp(1j*kl*a*np.sign(kl.imag)) / (kl + s*(omega - 2j*m))*ell
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
        if v_min > v_max:
            v_min, v_max = v_max, v_min
        vals = np.linspace(v_min, v_max, num_points)
    return vals, mode

@njit(fastmath=True, error_model="numpy")
def check_peak_resolution(integrand, kp_mode, numkp, peak_tolerance=5e-2, peak_ratio_limit=5):
    """    
    sanity check of the peak detection
    (1 vs. 2 peaks, are the maxima sufficiently large compared to the edge values)
    """
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
        bias = 0    # 0 if peaks of similar size, +1 if right peak much larger, -1 if left larger
        if kp_max1 / kp_max2 > peak_ratio_limit:
            bias = -1
        elif kp_max2 / kp_max1 > peak_ratio_limit:
            bias = 1
        peak1_left = (abs_integrand[0] / kp_max1 > peak_tolerance) and (bias != 1)
        peak2_right = (abs_integrand[-1] / kp_max2 > peak_tolerance) and (bias != -1)
        if kp_mode == 3:    # peaks too close together, do not check middle
            peak1_right = False
            peak2_left = False
        else:
            peak1_right = (abs_integrand[numkp//2 - 1] / kp_max1 > peak_tolerance) and (bias != 1)
            peak2_left = (abs_integrand[numkp//2] / kp_max2 > peak_tolerance) and (bias != -1)
        if (peak1_left or peak1_right or peak2_left or peak2_right):
            issue_warning = True
    # if issue_warning:
    #     print(integrand)
    #     raise RuntimeError
    return issue_warning

@njit
def get_M(K):
    return (1/K + K - 2) / 4
@njit
def get_Km(K):
    return (K - 1) / 2

@njit
def print_simpson_warning(k, omega, s, kp_mode, beta, K, v, w, mmaxp, mmax, lmax, delta):
    print("Warning: Simpson integration insufficiently peaked "
          "(k =", k, "; omega =", omega, "; s =", s,
          "; beta =", beta, "; K =", K, "; v =", v, "; w = ", w, "; mmaxp =", mmaxp,
          "; mmax =", mmax, "; lmax =", lmax, "; delta =", delta, "# ", kp_mode, "peak", ")")

@njit(fastmath=True, error_model="numpy")
def _green_order_1_integrand(kp, M, K_m, k, omega, s, a, mmaxp, mmax, lmax):
    value = 0
    for m in range(-mmaxp, mmaxp+1):
        for sp in [+1, -1]:
            j_b0_part = _j_bn(kp, 2j*m, M - K_m, n=0)
            j_b1_part = _j_bn(sp*(kp - k), omega - 2j*m, K_m, n=1)
            s_b1s_part = s_b1s(sp*(k - kp), omega - 2j*m, K_m, a, s, mmax, lmax)
            value += j_b0_part * j_b1_part * s_b1s_part
    return value

@njit(fastmath=True, error_model="numpy", parallel=True, nogil=True)
def _green_order_1_single(k, omega=0.0, beta=1.5, K=1.3, v=0.5, a=1.0, w=3,
                          numkp=48, mmaxp=3, mmax=3, lmax=2, delta=0, 
                          peak_tolerance=1e-1, kp_sigma=15):
    """Fast implementation of the numerical integration for the first order GF perturbation"""
    M = get_M(K)
    K_m = get_Km(K)
    prefactor = w**(2*M + K - 1) / (8*np.pi**4) * beta
    #kpv = np.linspace(-kp, kp, numkp)
    res = 0
    for s in [+1, -1]:
        integrand_kpv = np.zeros(numkp, dtype=np.complex128)
        kpv, kp_mode = get_integration_region(k * beta*v/np.pi, kp_sigma*np.abs(K_m), numkp)

        # actual calculation
        for kpi in prange(numkp):
            kp = kpv[kpi]
            integrand_kpv[kpi] = _green_order_1_integrand(
                kp, M, K_m, beta*v/np.pi * k, beta/np.pi * (omega + 1j*delta), s,
                np.pi*a / beta/v, mmaxp, mmax, lmax)
        res += simpson_njit(integrand_kpv, kpv) * prefactor * (s * K + 1)
        if check_peak_resolution(integrand_kpv, kp_mode, numkp, peak_tolerance):
            print_simpson_warning(k, omega, s, kp_mode, beta, K, v, w, mmaxp, mmax, lmax, delta)
    return res

@njit(fastmath=True, error_model="numpy", parallel=True, nogil=True)
def _green_order_1(k_vals, omega=0.0, beta=1.5, K=1.3, v=0.5, a=1.0, w=3,
                   numkp=31, mmaxp=8, mmax=9, lmax=5, delta=0, peak_tolerance=1e-1, kp_sigma=15):
    """Fast implementation of the numerical integration for the first order GF perturbation"""
    M = get_M(K)
    K_m = get_Km(K)
    prefactor = w**(2*M + K - 1) / (8*np.pi**4) * beta
    #kpv = np.linspace(-kp, kp, numkp)
    res = np.zeros(k_vals.shape[0], dtype=np.complex128)
    for ki in prange(k_vals.shape[0]):
        k = k_vals[ki]
        for s in [+1, -1]:
            integrand_kpv = np.zeros(numkp, dtype=np.complex128)
            kpv, kp_mode = get_integration_region(k * beta*v/np.pi, kp_sigma*np.abs(K_m), numkp)

            # actual calculation
            for kpi in range(numkp):
                kp = kpv[kpi]
                integrand_kpv[kpi] = _green_order_1_integrand(
                    kp, M, K_m, beta*v/np.pi * k, beta/np.pi * (omega + 1j*delta), s,
                    np.pi*a / beta/v, mmaxp, mmax, lmax)
            res[ki] += simpson_njit(integrand_kpv, kpv) * prefactor * (s * K + 1)
            if check_peak_resolution(integrand_kpv, kp_mode, numkp, peak_tolerance):
                print_simpson_warning(k, omega, s, kp_mode, beta, K, v, w, mmaxp, mmax, lmax, delta)
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
    green = np.array([np.array([[0, -1j * g_vals[i]], [1j * g_vals[i], 0]])
                      for i in range(g_vals.size)])
    return green

def green_perturbative(k_vals, omega_vals=0, beta=1.5, K=1.3, v=0.5, g=0.2, a=1.0, w=3,
                       num_params=None, order=1, delta=0, ratio_warning_threshold=0.8,
                       mute_ptr_warnings=False, use_omega_symmetry=False):
    """Perturbative calculation of the SP Green's function
    num_params :: dictionary, {"order" : 1, "mmax" : 2, "mmin" : -2, "lmax" : 1}
    """
    omega_vals = tlk.asarray(omega_vals)
    k_vals = tlk.asarray(k_vals)
    green = np.zeros((k_vals.size, omega_vals.size, 2, 2), dtype=complex)
    size_half = omega_vals.size // 2
    display_progress_info = (k_vals.size * omega_vals.size > 2<<14) and omega_vals.size > 30
    if display_progress_info:
        print(f"Computing GF for {beta = :.4f}, {K = :.4f}, {v = :.4f}, {g = :.4f}, {K = :.4f}, "
              f"{w = :.4f}, {delta = :.2e}, {num_params = }, {k_vals.size = }, {omega_vals.size = }")
    if num_params is None:
        num_params = {"mmax" : 3, "mmaxp" : 3, "lmax" : 2, "numkp" : 48}
    for i_omega, omega in enumerate(omega_vals):
        green_vals = green_order_0(k_vals, omega, beta, K, v, w, delta)
        if order >= 1:
            green_vals += g * green_order_1(k_vals, omega, beta, K, v, a, w, num_params, delta)
        green[:, i_omega] = green_vals
        if display_progress_info and i_omega % 10 == 0:
            print(f"Finished iteration ({i_omega+1} / {omega_vals.size})...")
        if use_omega_symmetry and i_omega == size_half-1:
            if (omega_vals.size % 2 == 0 and
                np.allclose(omega_vals[:size_half-1:-1], -omega_vals[:size_half])):
                green[:, :size_half-1:-1] = -tlk.sigma_x @ green[:, :size_half].conj() @ tlk.sigma_x
                print("Info: Exploited symmetry to obtain G(omega > 0)")
                break
            else:
                print("Warning: Symmetry could not be used because frequency values are asymmetric!")
    if not mute_ptr_warnings and ratio_warning_threshold is not None and order > 0:
        warning = check_ptr(green, k_vals, omega_vals, ratio_warning_threshold)
        parameters = f"{beta=:.2f}, {K=:.3f}, {v=:.3f}, {g=:.3e}, {a=:.2f}, {w=:.3f}, {num_params=}"
        if warning is not None:
            print(warning + parameters)
    return green

def get_ptr(green):
    """Compute the Perturbation Theory Ratio (PTR) of a given Green's Function"""
    if green.ndim == 4:
        ratio = np.abs(green[:,:,0,1] / green[:,:,0,0])
    elif green.ndim == 3:
        ratio = np.abs(green[:,0,1] / green[:,0,0])
    elif green.ndim == 2:
        ratio = np.abs(green[0,1] / green[0,0])
    else:
        raise ValueError(f"PTR undefined for {green.ndim = }")
    return ratio

def check_ptr(green, k_vals, omega_vals, ratio_warning_threshold):
    """"""
    ratio = get_ptr(green)
    argmin, argmax, rmean = np.argmin(ratio), np.argmax(ratio), np.mean(ratio)
    warning = f"PTR_Warning ({ratio_warning_threshold:.2f}): "
    if ratio.flat[argmin] > ratio_warning_threshold:
        specifier = f"All values ({rmean:.2f}): "
    elif rmean > ratio_warning_threshold:
        specifier = f"Average ({rmean:.2f}): "
    elif ratio.flat[argmax] > ratio_warning_threshold:
        indx = np.unravel_index(argmax, ratio.shape)
        if ratio.ndim == 2:
            k_max, omega_max = k_vals[0], omega_vals[0]
        elif ratio.ndim == 3:
            k_max, omega_max = k_vals[indx[0]], omega_vals[0]
        elif ratio.ndim == 4:
            k_max, omega_max = k_vals[indx[0]], omega_vals[indx[1]]
        else:
            raise ValueError(f"{ratio.ndim = } not supported for PTR")
        r_max = ratio[indx]
        specifier = f"Maximum ({r_max:.2f}): k={k_max:.4f}, omega={omega_max:.2f}, "
    else:
        specifier = None

    if specifier is not None:
        return warning + specifier
    return None


def get_free_K0(u_plus, vf, a=1, factor=1):
    return 1 / np.sqrt(1 + 2 * u_plus * a / np.pi / vf * factor)


def green_eff_ua(k_vals, omega_vals, beta=1, K=1, v=0.5, vf=None, w=np.pi):
    if vf is None:
        vf = v
    k_vals = np.asarray(k_vals)
    omega_vals = np.asarray(omega_vals)
    args = (beta, K, v, w)
    green = np.zeros((k_vals.size, omega_vals.size, 2, 2), dtype=complex)
    for i_omega, omega in enumerate(omega_vals):
        green[:, i_omega] = green_order_0(k_vals, omega, *args)
    green_ab = tlk.basis_lr_to_ab(green)
    hamilton = tlk.hamilton_from_green(omega_vals, green_ab)
    self_energy = np.copy(hamilton)
    for (ia, ib) in zip([0, 1, 1], [1, 0, 1]):
        self_energy[:, :, ia, ib] = 0
    hamilton_eff = np.zeros_like(self_energy)
    hamilton_eff += self_energy
    for (ia, ib) in zip([0, 1], [1, 0]):
        for iw in range(omega_vals.size):
            hamilton_eff[:, iw, ia, ib] += vf * k_vals
    green_eff = tlk.green_from_hamilton(omega_vals, hamilton_eff)
    return green_eff


def main():
    print(__doc__)
    k=[0.001]; omega=[0]; beta=np.pi; K=1.3; v=1.0; g=0.03; w=np.pi
    print(green_perturbative(k, omega, beta, K, v, g, w=w))
    return 0


if __name__ == "__main__":
    main()
