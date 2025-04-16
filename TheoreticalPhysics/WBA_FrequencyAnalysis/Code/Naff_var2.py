# -*- coding: utf-8 -*-
"""
Numerical analysis of fundamental frequencies -- different methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special

def signal_title(flist, alist):
    """Returns the raw string for the symbolic expression of the signal"""
    txt = r"$z_t=\sum_k a_k\mathrm{e}^{2\pi\mathrm{i}\nu_k t}$"
    flist = [np.round(fval, special.dig) for fval in flist]
    txt += f"\n$\\nu_k \\simeq {flist}\ \\mathrm{{and}}\ a_k = {alist}$"
    return txt

def WBA(arr, w='ccinf', alpha=140):
    """WBA of a time series; weights are 'gauss' or 'ccinf'."""
    N = arr.shape[0]
    t = np.arange(1, N, 1) / N
    if w == 'ccinf':
        weights = np.exp(-1 / (t * (1 - t)))
    elif w == 'gauss':
        weights = np.exp(-alpha * (t - 0.5)**2)
    _sum = np.sum(weights)
    return np.sum(weights * arr[1:]) / _sum % 1.0

def map_arctan2(z):
    from WBA_core import embedding
    phi = np.arctan2(z.imag, z.real) / (2 * np.pi)
    phidiff = phi[1:] - phi[:-1]
    embedding(phidiff)
    return phidiff
    
def N_arr(Nmin, Nmax, NN):
    """Logarithmic distribution of 'NN' integers in [2**Nmin ... 2**Nmax]"""
    res = 2**np.linspace(Nmin, Nmax, NN)
    return np.unique(res.astype(np.uint32))

def signal(f1, f2, a1=1.0, a2=0.0, N=300):
    """Test signal with two frequencies"""
    t = np.arange(N)
    z1 = a1 * np.exp(1j * 2*np.pi * f1 * t)
    z2 = a2 * np.exp(1j * 2*np.pi * f2 * t)
    return z1 + z2

def signal_list(flist, alist, N=300):
    """Test signal with a list of frequencies"""
    t = np.arange(N)
    z = np.sum([alist[n] * np.exp(1j * 2*np.pi * flist[n] * t) 
                for n in range(len(flist))], axis=0)
    return z

def _weights(N, ident='hanning', alpha=140):
    """Identifier can be (hanning, gauss, none)"""
    t = np.arange(N)
    if ident == 'none':
        return np.ones(N)
    elif ident == 'gauss':
        return np.exp(-alpha * (t / N - 0.5)**2) 
    elif ident == 'hanning':
        return 2 / N * np.sin(np.pi * t / N)**2
    else:
        print("Unknown identifier in _weights()!")
        raise NotImplementedError

def naff1d(z, Approx=True, weights='hanning', alpha=140):
    """Naff 1d -- weights (hanning, gauss, none)"""
    N = z.shape[0]
    w = _weights(N, ident=weights, alpha=alpha)
    zfft = np.fft.fft(w * z)
    abs_fft = np.abs(zfft)
    ind = np.argmax(abs_fft)
    ind -= (abs_fft[ind-1] > (abs_fft[(ind + 1) % N] + 1e-6))
    if weights == 'none':
        # R = abs_fft[ind] / abs_fft[(ind+1) % N]
        R = np.real(zfft[ind] / zfft[(ind+1) % N])
        if Approx:
            nu = (ind - 1 / (R - 1)) / N
        else:
            nu = (ind / N
                  + np.arctan(np.sin(np.pi / N) / (np.cos(np.pi / N) - R)))
            
    if weights == 'hanning':
        R = np.real(zfft[ind] / zfft[(ind+1) % N])
        if Approx:
            nu = ind / N + 1 / N * (2 + R) / (1 - R)
        else:
            def nu_analytic_A(R, c):                  
                """Helper function to determine the frequency"""
                result = (-(R + c) * (R - 1) 
                          + np.sign(R + c) * np.sqrt(c**2*(R + 1)**2 
                                                     - 2*R*(2*c**2 - c - 1)))
                return result / (R**2 + 1 + 2*R*c)
            nu = (ind / N + 1 / (2*np.pi) 
                  * np.arcsin(nu_analytic_A(-R, np.cos(2*np.pi / N))
                              * np.sin((2*np.pi) / N)) ) % 1.0
            
    if weights == 'gauss':
        R = abs_fft[ind] / abs_fft[(ind+1) % N]
        if Approx:
            nu = ind / N + 1 / (2*N) - alpha / (2 * np.pi**2 * N) * np.log(R)
        else:
            print("Warning, no exact solution know for gaussian weights!")
            raise NotImplementedError
    
    return nu

# def naff1d_hann(z):
#     """Naff 1d with hanning-window weights"""
#     N = z.shape[0]
#     t = np.arange(N)
#     weights = 2 / N * np.sin(np.pi * t / N)**2
#     zfft = np.fft.fft(weights * z)
#     ind = np.argmax(np.abs(zfft))
#     def nu_analytic_A(R, c):                  
#         """Helper function to determine the frequency"""
#         result = - (R + c)*(R - 1) \
#             + np.sign(R + c) * np.sqrt(c**2*(R + 1)**2 
#                                        - 2*R*(2*c**2 - c - 1))
#         return result / (R**2 + 1 + 2*R*c)
#     R = np.real(zfft[ind] / zfft[ind + 1])
#     nu = (ind / N + 1 / (2*np.pi) 
#           * np.arcsin(nu_analytic_A(- R,
#                                     np.cos(2*np.pi / N))
#                       * np.sin((2*np.pi) / N))) % 1.0
#     return nu

# def naff1d_approx(z):
#     """Naff 1d with hanning-window weights; approximation"""
#     N = z.shape[0]
#     t = np.arange(N)
#     weights = 2 / N * np.sin(np.pi * t / N)**2
#     zfft = np.fft.fft(weights * z)
#     ind = np.argmax(np.abs(zfft))
#     R = np.real(zfft[ind] / zfft[ind + 1])
#     nu = ind / N + 1 / N * (2 + R) / (1 - R)
#     return nu

# def gauss(t, alpha):
#     """Gaussian-window weights"""
#     N = t.shape[0]
#     res = np.exp(-alpha * (t / N - 0.5)**2)
#     return res * np.sqrt(alpha / np.pi)

        #### SUPERSEEDED by _remove_peak function
        # c0 = np.sum(z)
        # corr = np.abs(c0) * np.exp(-np.pi**2 / alpha * np.arange(J)**2)
        # abs_fft[:J] -= corr
        # # correction for the 'negative frequency' section (J-1 terms)
        # abs_fft[-J+1:] -= corr[:0:-1]

def _remove_peak(fft, ind, nu, N, alpha=140, J=30):
    """Remove a gaussian peak from a FFT at a given frequency"""
    ampl = fft[ind]
    phase = nu * N - ind
    for j in range(-J+1, J, 1):
        corr_exp = -np.pi**2 / alpha * (j**2 - 2 * phase * j)
        fft[ind+j] -= ampl * np.exp(corr_exp)

def naff1d_gauss(z, alpha=140, Offset=False):
    """Naff 1d with gauss-window weights; approximation"""
    N = z.shape[0]
    t = np.arange(N)
    weights = np.exp(-alpha * (t / N - 0.5)**2) 
    abs_fft = np.abs(np.fft.fft(weights * z))
    
    # correct a potential constant offset of the input signal
    # it is much better to perform this after the FFT (else O(1/N) error)
    if Offset:
        J = 30    # number of corrections until 1e-16 error
        if J > N//2:    # avoid index error for small 'N'
            J = N//2
        _remove_peak(abs_fft, ind=0, nu=0.0, N=N, J=J)   # c0 contribution
    
    # find the global maximum of the FFT
    ind = np.argmax(abs_fft)
    if ind == (N-1):   # avoid index error at 'abs_fft[ind + 1]'
        ind -= 1
    elif abs_fft[ind - 1] > abs_fft[ind + 1]:   # 'ind' closer to true peak
        ind -= 1
    R = abs_fft[ind] / abs_fft[ind + 1]
    nu = ind / N + 1 / (2*N) - alpha / (2 * np.pi**2 * N) * np.log(R)
    return nu

# from numba import njit, objmode
# with objmode(fft='c16[:]'):
#    fft = np.fft.fft(z * weights)

def naffnd_gauss(z, n_freq=1, alpha=140, J=30, 
                 ReturnCoeff=False, Offset=False):
    """
    NaffND with gaussian weights.
    Computes the first 'n_freq' fundamental frequencies of the complex-valued
        input signal 'z', as well as their respective complex amplitudes.
    Parameters:
        n_freq == number of frequenices to be computed until max(fft) < 1e-15
                  (iterative process is aborted if the fft becomes 'flat')
        alpha == gaussian weights given by exp(-alpha * (t / N - 0.5)**2)
                 where t = 0,...,N-1 with 'N' being the signal length
                 (alpha = 140 leads to weights being about 1e-16 at t = 0)
        J == for the '2*J - 1' points closest to the relevant peak in the fft,
             the respective fft-value is subtracted based on an approximate
             analytical formula
             (J = 23 would be enough for machine precision at the outer parts)
        ReturnCoeff == whether to return the computed frequency amplitudes
        Offset == whether to compute, return and adjust a signal offset
    """
    N = z.shape[0]
    t = np.arange(N)    # t = 0,...,N-1
    weights = np.exp(-alpha * (t / N - 0.5)**2)
    fft = np.fft.fft(weights * z)   # need complex-valued fft for coeff.
    abs_fft = np.abs(fft)           # only need absolute values for freqs.
    
    # frequencies and their respective amplitudes
    nu_arr = np.zeros(n_freq + Offset)
    c_arr = np.zeros(n_freq + Offset, dtype=np.complex128)     
    
    if J > N//2:
        J = N//2
    if Offset:    # compute constant offset and adjust fft spectrum
        c_arr[n_freq] = np.sqrt(alpha / np.pi) * fft[0] / N    
        _remove_peak(abs_fft, ind=0, nu=0.0, N=N, J=J)
    
    for i in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-15:   
            break   # abort if fft-peak too small -> nu == 0
            
        # choose 'ind' closer to true peak (with 1e-6 tolerance)
        if abs_fft[ind - 1] > abs_fft[(ind + 1) % N] + 1e-6: 
            ind -= 1
        R = abs_fft[ind] / abs_fft[(ind + 1) % N]    # avoid index error
            
        # Ratio 'R' of largest neighboring fft-values to compute 'nu'
        R = abs_fft[ind] / abs_fft[ind + 1]
        nu = ind / N + 1 / (2*N) - alpha / (2 * np.pi**2 * N) * np.log(R)
        nu_arr[i] = nu
        
        # compute the complex value of the frequency amplitude
        phase = nu * N - ind
        c = np.sqrt(alpha / np.pi) * fft[ind] / N * np.exp(-1j*np.pi * phase)
        c_arr[i] = c * np.exp(np.pi**2 / alpha * phase**2)
        
        # remove the previous peak from the fft spectrum
        _remove_peak(abs_fft, ind, nu, N, alpha=alpha, J=J)
    if ReturnCoeff:
        return nu_arr, c_arr
    return nu_arr

def linear(x, a, b):
    """Linear function with offset 'b' and slope 'a'"""
    return b + a * x

def compute_and_plot_variant(ax, z, freq, Narr, colors, 
                             Approx=True, weights='hanning', ShowPoly=True):
    """Returns the absolute difference, the slope 'a' and offset 'b'"""
    nu = np.array([naff1d(z[:N], Approx=Approx, weights=weights) 
                   for N in Narr])
    diff = np.abs(nu - freq)
    diff[diff < 1e-16] = 1e-16
    ls = '-'
    label = f"$\\nu_{{\\mathrm{{{weights}}}}}$"
    if Approx:
        ls = '--'
        label = f"$\\nu_{{\\mathrm{{{weights}-approx}}}}$"
    ax.plot(Narr, diff, c=colors.get_color(), ls=ls,
            label=label)
    if ShowPoly:
        a, b = np.polyfit(np.log(Narr), np.log(diff), deg=1)
        print(f"Naff with {weights}-weights has slope {a} and offset {b}.")
        ax.plot(Narr, np.exp(linear(np.log(Narr), a, b)), 
                ls=ls, c=colors.prev_color())
        lin1xpos = Narr[(3*Narr.shape[0]) // 5]
        lin1ypos = np.exp(linear(np.log(lin1xpos), a, b))
        ax.text(lin1xpos, lin1ypos, f"$\\sim N^{{{a:.1f}}}$", 
                ha='left', va='bottom')
        return diff, a, b
    return diff


from scipy.optimize import brentq
from scipy.special import comb

def naff_g(z, alpha=140.0):
    N = z.shape[0]
    x = np.arange(N) / N
    weights = np.exp(-alpha * (x - 0.5)**2)
    fft = np.fft.fft(weights * z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)
    
    # choose 'ind' closer to true peak (with 1e-6 tolerance)
    if abs_fft[ind - 1] > abs_fft[(ind + 1) % N] + 1e-6: 
        ind -= 1
    
    R = abs_fft[ind] / abs_fft[(ind + 1) % N]   
    
    def Fj_g(eps, N=N):
        """Proportional to the abs-value of modeled Fourier coefficients"""
        return np.exp(-np.pi**2/alpha * N**2 * eps**2)
    
    def root(eps, R, ak=ak, N=N):
        Fj = Fj_g(eps, N)
        Fj_plus1 = Fj_g(eps - 1/N, N)
        return Fj - R * Fj_plus1
    
    delta = 1e-3/N
    eps = brentq(root, delta, 1/N - delta, args=(R))
    nu = ind/N + eps
    return nu

def hann_coeff(M):
    """Returns the cosine series coefficients for the M-th hanning window"""
    ak = np.zeros(M+1)
    ak[0] = comb(2*M, M, exact=True)
    for k in range(1, M+1, 1):
        ak[k] = 2 * (-1)**k * comb(2*M, M-k, exact=True)    
    return ak / (4**M)
    

ak = np.array([0.21557895, -0.41653158, 0.277263158, 
               -0.083578947, 0.006947368])
ak = np.array([0.5, -0.5])
ak = np.array([3, -4, 1]) / 8.0
ak = hann_coeff(14)
ak = np.array([1, -1.942604, 1.340318, -0.440811, 0.043097])
def naff_cos(z, ak=ak):
    N = z.shape[0]
    #ak /= N
    x = np.arange(N) / N
    weights = np.sum([ak[k] * np.cos(2*np.pi*x*k) 
                      for k in range(ak.shape[0])], axis=0)
    fft = np.fft.fft(weights * z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)
    
    # choose 'ind' closer to true peak (with 1e-6 tolerance)
    flag = False
    if abs_fft[ind - 1] > abs_fft[(ind + 1) % N] + 1e-6: 
        flag = True
    
    if flag:
        R = abs_fft[ind] / abs_fft[(ind - 1) % N] 
    else:
        R = abs_fft[ind] / abs_fft[(ind + 1) % N] 
    
    def Fj_cos(eps, ak=ak, N=N):
        res = 2.0 * (ak[0] * (eps**2 - 1/N**2) + ak[1] * eps**2)
        fac = eps * (eps - 1/N) * (eps + 1/N)
        corr = 0.0
        for k in range(2, ak.shape[0], 1):
            corr += ak[k] * (1 / (eps - k/N) + 1 / (eps + k/N))
        
        return res + fac * corr
    
    def root(eps, R, ak=ak, N=N):
        Fj = Fj_cos(eps, ak, N)
        Fj_plus1 = Fj_cos(eps - 1/N, ak, N)
        fac = (eps + 1/N) / (eps - 2/N)
        return R * np.abs(fac) - np.abs(Fj / Fj_plus1)
    
    delta = 1e-12 #1e-3/N
    a = delta
    b = 1/N - delta
    if root(a, R) * root(b, R) < 0.0:
        eps = brentq(root, a, b, args=(R), xtol=1e-15)
        # print(f"success positive eps = {eps:.2e}", flag)
        # print("Success: R = ", R, N)
    elif root(-a, R) * root(-b, R) < 0.0:
        eps = brentq(root, -b, -a, args=(R), xtol=1e-15)
        # print(f"success negative eps = {eps:.2e}", flag)
        # print("Success: R = ", R, N)
    else:
        eps = 0.0
        print("failure warning, eps set to zero")
        print("Failure: R = ", R, N)
    # print()
    
    # if flag:
    #     eps = brentq(root, -b, -a, args=(R))
    #     print(f"success negative eps = {eps:.2e}", flag)
    #     print("Success: R = ", R, N)
    # else:
    #     eps = brentq(root, a, b, args=(R))
    #     print(f"success positive eps = {eps:.2e}", flag)
    #     print("Success: R = ", R, N)
    # eps = brentq(root, delta, 1/N - delta, args=(R))
    
    if flag:
        nu = ind/N - eps
    else:
        nu = ind/N + eps
    
    return nu
    
    
def binary_root(f, a, b, args=(), eps=1e-12, maxiter=100):
    """Assuming f(a) * f(b) < 0, searches for a root with binary intervals"""
    sa = np.sign(f(a))
    sb = np.sign(f(b))
    if sa == sb:
        raise ValueError("f(a) and f(b) must have opposite signs!")
    
    for ctr in range(maxiter):
        m = (a+b) / 2           # midpoint
        mval = f(m)
        sm = np.sign(mval)      # sign at midpoint
        if sm == sa:
            a = m
        else:
            b = m
        
        if np.abs(mval) < eps:
            return m
    
    print(f"Warning, after {maxiter} steps the error was {mval}, "
          + f"the accuracy of {eps} could not be achieved!")
    return m
    
def main():
    special.setup(UseTex=False, dpi=50)
    f1 = np.sqrt(2) - 1
    f2 = 0.12345
    Nmin, Nmax, NN = 5.0, 14.0, 100
    Narr = N_arr(Nmin, Nmax, NN)
    z = signal(f1, f2, a2=0.8, N=Narr[-1])
    flist = [f1, f2, 0.6, 0.8, 0.2341, 0.456]
    alist = [1, 0.8, 0.6, 0.3, 0.7, 0.8]
    # alist = [1, 0.4, 0.2, 0.1, 0.08, 0.05]
    # flist = [f1, 0.381, 0.522]
    # alist = [1, 0.3, 0.1]
    z = signal_list(flist, alist, N=Narr[-1])
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(Narr[0], Narr[-1])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$|\nu_N - \nu|$")
    # ax.plot(z.real, z.imag, ls='', marker='o', mew=1, ms=3, c='b')
    colors = special.Colors()
    
    compute_and_plot_variant(ax, z, f1, Narr, colors, 
                             Approx=False, weights='hanning', ShowPoly=True)
    compute_and_plot_variant(ax, z, f1, Narr, colors,
                             Approx=True, weights='hanning', ShowPoly=False)
    
    nu = np.array([naff_cos(z[:N]) for N in Narr])
    diff = np.abs(nu - f1)
    diff[diff < 1e-16] = 1e-16
    #a7, b7 = np.polyfit(np.log(Narr), np.log(diff7), deg=1)
    ax.plot(Narr, diff, c=colors.get_color(),
            label=r"$\nu_{\mathrm{cos}}$")
    
    nu3 = np.array([naff1d_gauss(z[:N]) for N in Narr])
    diff3 = np.abs(nu3 - f1)
    diff3[diff3 < 1e-16] = 1e-16
    imin = 35#42#35
    imax = imin + 10#6#10
    a3, b3 = np.polyfit(np.log(Narr)[imin:imax], 
                        np.log(diff3)[imin:imax], deg=1)
    ax.plot(Narr, diff3, c=colors.get_color(), 
            label=r"$\nu_{\mathrm{gauss-approx}}$")
    Narr_draw = Narr[imin-3:imax+2]
    ax.plot(Narr_draw, np.exp(linear(np.log(Narr_draw), a3, b3)),
            c=colors.prev_color(), ls='--')
    lin3xpos = Narr_draw[Narr_draw.shape[0] // 2]
    lin3ypos = np.exp(linear(np.log(lin3xpos), a3, b3))
    ax.text(lin3xpos, lin3ypos, f"$\\sim N^{{{a3:.1f}}}$", 
            ha='left', va='center')
    
    # nu4 = np.array([WBA(map_arctan2(z[:N]), w='ccinf') for N in Narr])
    # diff4 = np.abs(nu4 - f1)
    # indx = (diff4 > np.abs(1 - nu4 - f1))
    # diff4[indx] = np.abs(1 - nu4[indx] - f1)
    # diff4[diff4 < 1e-16] = 1e-16
    # ax.plot(Narr, diff4, c=colors.get_color(), 
    #         label=r"$\nu_{\mathrm{WBA-}C_c^\infty}$")
    
    # nu5 = np.array([WBA(map_arctan2(z[:N]), w='gauss') for N in Narr])
    # diff5 = np.abs(nu5 - f1)
    # indx = (diff5 > np.abs(1 - nu5 - f1))
    # diff5[indx] = np.abs(1 - nu5[indx] - f1)
    # diff5[diff5 < 1e-16] = 1e-16
    # ax.plot(Narr, diff5, c=colors.get_color(), 
    #         label=r"$\nu_{\mathrm{WBA-gauss}}$")
    
    # nu6 = np.array([naff1d_nw(z[:N], Approx=0) for N in Narr])
    # diff6 = np.abs(nu6 - f1)
    # diff6[diff6 < 1e-16] = 1e-16
    # a6, b6 = np.polyfit(np.log(Narr), np.log(diff6), deg=1)
    # ax.plot(Narr, diff6, c=colors.get_color(),
    #         label=r"$\nu_{\mathrm{no-weights}}$")
    # ax.plot(Narr, np.exp(linear(np.log(Narr), a6, b6)), 
    #         ls='--', c=colors.prev_color())
    # print("No weights: slope and offset", a6, b6)
    
    # nu7 = np.array([naff1d_nw(z[:N], Approx=1) for N in Narr])
    # diff7 = np.abs(nu7 - f1)
    # diff7[diff7 < 1e-16] = 1e-16
    # a7, b7 = np.polyfit(np.log(Narr), np.log(diff7), deg=1)
    # ax.plot(Narr, diff7, c=colors.get_color(),
    #         label=r"$\nu_{\mathrm{no-weights-approx}}$")
    # ax.plot(Narr, np.exp(linear(np.log(Narr), a7, b7)), 
    #         ls='--', c=colors.prev_color())
    # print("No weights: slope and offset", a7, b7)
    
    ax.set_title(signal_title(flist, alist))
    ax.legend()
    special.polish(fig, ax, SetCaptions=False)
    

    
if __name__ == "__main__":
    print(__doc__)
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    main()
    
    #pragma omp parallel for
    
    """
def FFT(p):
    N = p.shape[0]
    if N == 1:
        return p
    y_even = FFT(p[::2])
    y_odd = FFT(p[1::2])
    omega = np.exp(2j*np.pi / N)
    y = np.zeros(N, dtype=np.complex128)
    for i in range(N // 2):
        y[i] = y_even[i] + omega**i * y_odd[i]
        y[i+N//2] = y_even[i] - omega**i * y_odd[i]
    return y

def bitReverse(x, log2n):
    n = 0
    for i in range(log2n):
        n <<= 1
        n |= (x & 1)
        x >>= 1
    return n
            
def FFTiter(a):
    n = a.shape[0]
    log2n = int(np.log2(n))
    A = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        A[i] = a[bitReverse(i, log2n)]
    
    for s in range(1, log2n + 1, 1):
        m = 1 << s
        m2 = m >> 1
        w = 1.0
        wm = np.exp(1j * np.pi / m2)   # change sign in exp() for np.fft.fft
        for j in range(m2):
            for k in range(j, n, m):
                t = w * A[k + m2]
                u = A[k]
                A[k] = u + t
                A[k + m2] = u - t
            w *= wm
    return A
    
import time
def call(func, *args):
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    return end - start

plist = [np.random.randint(0, 23414, size=2**N) for N in range(16)]
tlist = [call(FFT, pval) for pval in plist]
tlistnp = [call(np.fft.fft, pval) for pval in plist]
tlistiter = [call(FFTiter, pval) for pval in plist]

nvals = 2**np.arange(16)
plt.plot(nvals, tlist, marker='x', ms=7, mew=1, c='b')
plt.plot(nvals, tlistnp, marker='x', ms=7, mew=1, c='orange')
plt.plot(nvals, tlistiter, marker='x', ms=7, mew=1, c='g')
    """
