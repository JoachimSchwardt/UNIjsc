# -*- coding: utf-8 -*-
"""
Tests for the Naff variants using orbits from a 2D standard map
"""

import Naff_var
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
from std_map import _std_map

def main(q0=0.1, p0=0.05, MapToCircle=False):
    special.setup(UseTex=1, dpi=100)    
    colors = special.Colors()
    Nmin, Nmax, NN = 5.0, 14.0, 100
    Narr = Naff_var.N_arr(Nmin, Nmax, NN).astype(int)
    N_draw = 4096
    K = 0.7
    q, p = _std_map(q0, p0, Narr[-1], K)
    # print(Naff_var.naff1d_gauss(q+1j*p, Offset=True))
    # return
    z = q - 0.5 + 1j*p
    
    nu1 = np.array([Naff_var.naff1d(z[:N]) for N in Narr])
    f1 = nu1[-1]
    diff1 = np.abs(nu1 - f1)
    diff1[diff1 < 1e-16] = 1e-16
    diff1corr = np.copy(diff1)
    indx = (diff1 > np.abs(1 - nu1 - f1))
    diff1corr[indx] = 1 - diff1[indx]
    
    nu2 = np.array([Naff_var.naff1d_approx(z[:N]) for N in Narr])
    f2 = nu2[-1]
    diff2 = np.abs(nu2 - f2)
    diff2[diff2 < 1e-16] = 1e-16
    
    nu3 = np.array([Naff_var.naff1d_gauss(z[:N], MapToCircle=MapToCircle) 
                    for N in Narr])
    f3 = nu3[-1]
    diff3 = np.abs(nu3 - f3)
    diff3[diff3 < 1e-16] = 1e-16
    
    z2 = q + 1j * p
    nu4 = np.array([Naff_var.naff1d_gauss(z2[:N], Offset=True, 
                                          MapToCircle=MapToCircle) 
                    for N in Narr])
    f4 = nu4[-1]
    diff4 = np.abs(nu4 - f4)
    diff4[diff4 < 1e-16] = 1e-16
    
    nu5 = np.array([Naff_var.naff1d_gauss(z[:N], MapToCircle=MapToCircle, 
                                          Extrapolate=True) 
                    for N in Narr])
    f5 = nu5[-1]
    diff5 = np.abs(nu5 - f5)
    diff5[diff5 < 1e-16] = 1e-16
    
    # nu6 = np.array([Naff_var.naff1d_gauss(z[:N], Extrapolate=True) 
    #                 for N in Narr])
    # f6 = nu6[-1]
    # diff6 = np.abs(nu6 - f6)
    # diff6[diff6 < 1e-16] = 1e-16
    
    print(f"hann={f2}, gauss={f3}, gauss-shift={f4}, gauss-ext={f5}")
    
    weights = np.exp(-140 * (np.arange(Narr[-1]) / Narr[-1] - 0.5)**2)
    fwba = np.sum(weights * p) / np.sum(weights)
    print(f"WBA: {fwba}")
    
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title(f"$K={K}$ and $N={N_draw}$")
    ax[0].plot(q[:N_draw], p[:N_draw], ls='', marker='o', c='k', ms=2)
    ax[0].axis([0.0, 1.0, -0.5, 0.5])
    ax[0].set_xlabel(r"$q$")
    ax[0].set_ylabel(r"$p$")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(Narr[0], Narr[-1])
    ax[1].set_xlabel(r"$N$")
    ax[1].set_ylabel(r"$|\nu_N - \nu|$")
    # ax[1].plot(Narr, diff1, c=colors.get_color(), 
    #            label=r'$\nu_{\mathrm{hann}}$')
    ax[1].plot(Narr, diff2, c=colors.get_color(), 
                label=r'$\nu_{\mathrm{hann-approx}}$')
    # ax[1].plot(Narr, diff1corr, ls=':', c=colors.colors[0], 
    #            label=r'$\nu_{\mathrm{hann-corr}}$')
    ax[1].plot(Narr, diff3, c=colors.get_color(), 
                label=r'$\nu_{\mathrm{gauss-manual-shift}}$')
    ax[1].plot(Narr, diff4, c=colors.get_color(), 
                label=r'$\nu_{\mathrm{gauss-fft-shift}}$')
    ax[1].plot(Narr, diff5, c=colors.get_color(), 
                label=r'$\nu_{\mathrm{gauss-extrapolate}}$')
    # ax[1].plot(Narr, diff6, c=colors.get_color(), 
    #             label=r'$\nu_{\mathrm{gauss-ext.-epsilon}}$')
    
    ax[1].legend()
    special.polish(fig, ax, SetCaptions=False)
    

    
def richardson_naff(q0 = 0.6, p0 = 0.05):
    def richardson(func, n, K=4):
        return np.sum([(-1)**(K-j) * np.math.comb(K, j) * func(j+n) * (j+n)**K 
                       for j in range(K+1)]) / np.math.factorial(K)
    
    Nmin, Nmax, NN = 5.0, 8.0, 30
    Narr = Naff_var.N_arr(Nmin, Nmax, NN).astype(int)
    K = 0.7
    q, p = _std_map(q0, p0, Narr[-1], K)
    z = q - 0.5 + 1j*p
    
    qt, pt = _std_map(q0, p0, 2**14, K)
    zt = qt - 0.5 + 1j*pt
    
    alpha = 140.0
    N = z.shape[0]
    t = np.arange(N)
    weights = np.exp(-alpha * (t / N - 0.5)**2) 
    abs_fft = np.abs(np.fft.fft(weights * z))
    ind = np.argmax(abs_fft)
    if abs_fft[ind - 1] > abs_fft[ind + 1]:  
        ind -= 1
    # k_arr = np.array([k for k in range(-5, 6, 1) if k != 0])
    k_arr = np.array([[k, -k] for k in range(1, 6, 1)]).flatten()
    R_k = np.array([abs_fft[ind] / abs_fft[(ind + k) % N] for k in k_arr])
    nu_k = ind/N + k_arr/(2*N) - alpha/(2*k_arr*N*np.pi**2) * np.log(R_k)
    nu_k[nu_k > 0.5] = 1.0 - nu_k[nu_k > 0.5]

    f = Naff_var.naff1d_gauss(zt)
    if f > 0.5:
        f = 1.0 - f
    print(f"True frequency: {f = }\n")
    
    for i, k in enumerate(k_arr):
        print(f"{k = }: {nu_k[i] - f = }")
        
    def richardson_arr(arr, n, K=4):
        return np.sum([(-1)**(K-j) * np.math.comb(K, j) 
                       * arr[j+n-1] * (j+n)**K 
                       for j in range(K+1)]) / np.math.factorial(K)
    
    
    # k_arr = np.arange(1, 4, 1)
    # R_k = np.array([abs_fft[ind] / abs_fft[(ind + k) % N] for k in k_arr])
    # nu_val = ind/N + k_arr/(2*N) - alpha/(2*k_arr*N*np.pi**2) * np.log(R_k)
    # nu_val[nu_val > 0.5] = 1.0 - nu_val[nu_val > 0.5]
    
    # eps = (nu_val[1] - nu_val[0])**2 / (nu_val[2] - 2*nu_val[1] + nu_val[0])
    # nu = nu_val[0] - eps
    
        
    # rarr = []
    # arr = nu_k[1:] - nu_k[0]
    # arr = arr[::-1]
    # arr = nu_k[::-1]
    # for k in range(1, 5, 1):
    #     rarr.append([])
    #     for n in range(1, arr.shape[0] + 1 - k, 1):
    #         val = richardson_arr(arr, n, k)
    #         print(f"{k=}, {n=}: {val-f:.2e}")
    #         rarr[k-1].append(val)
            
    # #### TEST OF RICHARDSON EXTRAPOLATION FOR BASEL PROBLEM   
    # rarr = []
    # arr = np.array([np.sum([1/j**2 for j in range(1, n, 1)]) 
    #                 for n in range(1, 11, 1)])
    # tval = np.pi**2/6
    # for k in range(1, 5, 1):
    #     rarr.append([])
    #     for n in range(1, arr.shape[0] + 1 - k, 1):
    #         val = richardson_arr(arr, n, k)
    #         print(f"{k=}, {n=}: {val-tval:.2e}")
    #         rarr[k-1].append(val)
    # plt.plot(np.abs(arr - tval))
    # for elem in rarr:
    #     plt.plot(np.abs(np.array(elem) - tval))
        
    """
from scipy.optimize import curve_fit
def exxp(x, a, offset, sigma, mu=0.0):
    return a * np.exp((x - mu)**2 / sigma) + offset
def hypx(x, a, b, offset):
    return (1 + (b*x)**a)**(1/a) - 1 + offset

k_abs = np.array([np.abs(k) + 0.5 * (k < 0.0) for k in k_arr])
kc = np.linspace(np.min(k_abs), np.max(k_abs), 200)
par, cov = curve_fit(exxp, k_abs, nu_k, p0=[nu_k[-1], nu_k[0], k_abs.shape[0], 1e-10])
plt.plot(k_abs, nu_k, ls='', marker='x', ms=9, mew=1)
plt.plot(kc, exxp(kc, *par), label='exxp')
plt.legend()


##### NEW PLOT #####

Nmin, Nmax, NN = 5.0, 8.0, 30
Narr = Naff_var.N_arr(Nmin, Nmax, NN).astype(int)
q0 = 0.6; p0 = 0.05; K = 0.7
colors = special.Colors()
for delta in np.linspace(0.0, -0.04, 5):
    q, p = _std_map(q0, p0 + delta, Narr[-1], K)
    z = q - 0.5 + 1j*p
    
    qt, pt = _std_map(q0, p0 + delta, 2**14, K)
    zt = qt - 0.5 + 1j*pt
    
    alpha = 140.0
    N = z.shape[0]
    t = np.arange(N)
    weights = np.exp(-alpha * (t / N - 0.5)**2) 
    abs_fft = np.abs(np.fft.fft(weights * z))
    ind = np.argmax(abs_fft)
    if abs_fft[ind - 1] > abs_fft[ind + 1]:  
        ind -= 1
    # k_arr = np.array([k for k in range(-5, 6, 1) if k != 0])
    k_arr = np.array([[k, -k] for k in range(1, 6, 1)]).flatten()
    R_k = np.array([abs_fft[ind] / abs_fft[(ind + k) % N] for k in k_arr])
    nu_k = ind/N + k_arr/(2*N) - alpha/(2*k_arr*N*np.pi**2) * np.log(R_k)
    nu_k[nu_k > 0.5] = 1.0 - nu_k[nu_k > 0.5]

    f = Naff_var.naff1d_gauss(zt)
    if f > 0.5:
        f = 1.0 - f
    print(f"True frequency: {f = }\n")
    abs_diff = np.abs(nu_k - f)
    x = np.arange(abs_diff.shape[0])
    a, b = np.polyfit(x, np.log(abs_diff), deg=1)
    plt.plot(abs_diff, label=delta, ls='', marker='x', c=colors.get_color())
    plt.plot(x, np.exp(a*x + b), c=colors.prev_color())
plt.legend()



def get_eps(nu_val):
    return (nu_val[1] - nu_val[0])**2 / (nu_val[2] - 2*nu_val[1] + nu_val[0])

nu0, nu1, nu2 = nu_k[0], nu_k[2], nu_k[4]
val = np.array([nu0, nu1, nu2])
nval = np.copy(val)

eps = get_eps(nval)
nu = nval[0] - eps
nval = np.array([nu, nval[0], nval[1]])
print(nu - f)
    """

if __name__ == "__main__":
    print(__doc__)
    q0 = 0.6 
    # p0 = 0.35
    # main(q0, p0, MapToCircle=True)
    p0 = 0.05
    main(q0, p0, MapToCircle=False)
    # richardson_naff(q0 = 0.6, p0 = 0.05)