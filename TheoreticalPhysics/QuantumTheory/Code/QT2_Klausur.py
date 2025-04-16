# -*- coding: utf-8 -*-
"""
Plots and analysis of a part of the quantum theory 2 exam.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import rc, rcParams, special
rcParams["figure.dpi"] = 100
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)

def f1(theta, kappa=1.0):
    """Scattering amplitude in first Born-approximation (delta potential)"""
    return np.full(theta.shape[0], kappa)

def f2(theta, x=0.01, kappa=1.0):
    """Scattering amplitude in first Born-approximation (double delta pot.)"""
    A = x * (np.cos(theta) - 1)
    return kappa / 2 * (1 + np.exp(-1j * A))

def f2cos(cos_theta, x=0.01, kappa=1.0):
    """Scattering amplitude in first Born-approximation (double delta pot.)"""
    A = x * (cos_theta - 1)
    return kappa / 2 * (1 + np.exp(-1j * A))

def sigma(theta, f=f1, args=[]):
    """Differential cross section for a scattering amplitude f."""
    return np.abs(f(theta, *args))**2

def plot_cross_sections():
    kappa = 0.5         # V_0*m / pi, potential strength parameter
    N = 300             # number of points for linspace
    dig = 3             # number of digits to round
    x_low = 0.1
    x_high = 2*np.pi
    
    colors = special.Colors()
    
    theta = np.linspace(0, np.pi, N)
    cos_theta = np.cos(theta)[::-1]         # reverse to an increasing array
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlabel(r"$\cos(\theta)$")
    ax.set_ylabel(r"$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}$")
    sigma1 = sigma(cos_theta, f=f1, args=[kappa])
    ax.plot(cos_theta, sigma1, c=colors.get_color(), lw=special.lw,
            label=f"$|f_1(\\theta)|^2$ for $\\kappa={kappa}$")
    sigma2_low = sigma(cos_theta, f=f2cos, args=[x_low, kappa])
    ax.plot(cos_theta, sigma2_low, c=colors.get_color(), lw=special.lw,
            label=f"$|f_2(\\theta)|^2$ for $ka={round(x_low, dig)}$")
    sigma2_high = sigma(cos_theta, f=f2cos, args=[x_high, kappa])
    ax.plot(cos_theta, sigma2_high, c=colors.get_color(), lw=special.lw,
            label=f"$|f_2(\\theta)|^2$ for $ka={round(x_high, dig)}$")
    ax.axis([cos_theta[0], cos_theta[-1], 0, kappa**2 * 1.05])
    ax.legend(fontsize=special.fs)
    ax.tick_params(labelsize=special.tls)
    fig.tight_layout()
    fig.canvas.draw()
    special.embed_labels(ax, SetCaptions=False)
    plt.show()
    
###############################################################################
# Radial Schroedinger equation
###############################################################################

from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.ndimage.filters import uniform_filter1d
import cv2
from IndexingMethods import arr2int

def u_numeric(fname='Potential1.png'):
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    
    # Define range of blue color 
    lower_limit = np.array([170,120,0])
    upper_limit = np.array([200,130,255])
    
    # Generate mask for the blue pixels 
    blue_color_mask = cv2.inRange(img, lower_limit, upper_limit)
    blue_color_mask = cv2.flip(blue_color_mask, 0)
    # blue_color_mask = cv2.blur(blue_color_mask, (6, 6))
    # blue_color_mask = cv2.Canny(blue_color_mask, 500, 500)
    contours, hierarchy = cv2.findContours(blue_color_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # print(contours, len(contours))
    # cv2.drawContours(blue_color_mask, contours, -1, (255,255,0), 1)
    # cv2.imshow('window_name', blue_color_mask)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 
    
    graph = np.array([[x, y] for cont in contours for [x,y] in cont[:,0]]).T        
    x, y = graph
    indx = np.argsort(x)
    x, y = x[indx], y[indx]
    y -= y[0]
    y = y / np.max(np.abs(y))
    # filter to unique x values
    indx = []
    m = 1
    for i in range(0, x.shape[0] - m, m):
        if x[i+m] > x[i]:
            indx.append(i)
    
    multiples = [1]
    for i in range(1, x.shape[0], 1):
        if x[i] > x[i-1]:
            multiples.append(1)
        else:
            multiples[-1] += 1
            
    y_avg = np.zeros(len(multiples))
    ctr = 0
    for i, mul in enumerate(multiples):
        new_ctr = ctr + mul
        y_avg[i] = np.mean(y[ctr:new_ctr])
        ctr = new_ctr
        
    x_avg = np.unique(x)
    # plt.plot(x, y)
    # plt.plot(x[indx], y[indx])
    # plt.plot(x_avg, y_avg)
    
    # indxArray, indxTable = arr2int(x, bLen=2, bSizeEst=20)
    # y_avgb = np.zeros(indxArray.shape[0])
    # x_avgb = np.linspace(0, x[-1], indxArray.shape[0])
    # for i in range(indxArray.shape[0]):
    #     indx = indxArray[i, :indxTable[i]]
    #     if indx.shape[0] > 0:
    #         y_avgb[i] = np.mean(y[indx])
        
    # plt.plot(x_avgb, y_avgb)
        
    return x_avg, y_avg
    
def yukawa_fit(rho, V0, rho0):
    """fit function for the yukawa potential"""
    return V0 * np.exp(-rho / rho0) / rho
    
def coulomb_fit(rho, V0):
    """fit function for the yukawa potential"""
    return V0 / rho

def analyze_v(fname='Potential1.png'):
    """Analysis of the potential plots"""
    N = 500
    rho = np.linspace(0.001, 22*np.pi, N)
    x, y = u_numeric(fname)
    x_rho = (x - x[0]) * (rho[-1] - rho[0]) / x[-1]
    max_indx = int(x_rho.shape[0] * 14 / 22)
    x_rho, y = x_rho[:max_indx], y[:max_indx]
    
    uSpline = UnivariateSpline(x_rho, y, k=5, s=0.1)
    rho = rho[(N * 1) // 35:(N * 14) // 22]
    drho = rho[1] - rho[0]
    u = uSpline(rho)
    v_num = (uSpline(rho, nu=2) + u) / u
    
    # # u'' = (sec(z)**2 * z')' = sec(z)**2 * z'' + 2tan(z)sec(z)**2 * (z')**2
    # z = np.arctan(u)  
    # v_tan = ((1/u[1:-1] + u[1:-1]) * np.diff(z, 2)
    #          + 2 * (1 + u[1:-1]**2) * np.diff(z[:-1], 1)**2) / drho**2 + 1
    # v_num = (np.diff(u, 2) / (rho[1] - rho[0])**2) / u[1:-1] + 1
    
    # indx = (rho < 12 * np.pi)
    # v_fit = uniform_filter1d(v_num[indx], size=17)
    # par, var = curve_fit(yukawa_fit, rho[indx], v_fit, p0=[21])
    # print(f"V0 = {par[0]} +- {np.sqrt(var[0, 0])}")
    # print(f"rho0 = {par[1]} +- {np.sqrt(var[1, 1])}")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(rho, u)
    ax.plot(rho[1:-1], np.diff(u, 2) / drho**2)
    ax.plot(rho, v_num)
    # ax.plot(rho[1:-1], v_tan)
    # ax.plot(rho, uSpline(rho, nu=1))
    # ax.plot(rho, uSpline(rho, nu=2))
    # ax.plot(rho[indx], v_fit)
    # ax.plot(rho[indx], yukawa_fit(rho[indx], *par))
    plt.show()
    
def fourier_series(f, xmax, N=200):
    """Fourier coefficients for a discrete function
    u_0 = sum_k c_k * exp(i2 pi n x) 
    c_k = 1/L int_0^L u_0(x) * exp(-i2 pi n x) dx
    """
    c_n = []
    for n in range(N):
        c_val = f * np.exp(-2j*np.pi * np.linspace(0, 1, f.shape[0]) * n)
        c_n.append(np.sum(c_val) / f.shape[0])
    return c_n
    
    
def analyze_v_fourier(fname='Potential1.png', ShowPotential=0, ShowFourier=0):
    """Fourier series of u_0 to approximate the potential"""
    x, y = u_numeric(fname)
    m = 5
    max_indx = int(y.shape[0] * m / 22)
    y = y[:max_indx]
    y = uniform_filter1d(y, size=10)
    xmax = m * np.pi
    N = y.shape[0] - 1
    Npoints = y.shape[0]
    c_n = fourier_series(y, xmax, N=N)
    rho = np.linspace(0, xmax, Npoints)
    y_f = np.zeros(Npoints, dtype=np.complex128)
    for n in range(N):
        y_f += c_n[n] * np.exp(2j*np.pi * rho * n / xmax)
    
    if ShowFourier:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(rho, y)
        ax.plot(rho, y_f.real)
        plt.show()
    
    # compute the potential y''/y + 1
    y_f2diff = np.zeros(Npoints, dtype=np.complex128)
    for n in range(N):
        y_f2diff -= ((2*np.pi*n/xmax)**2 * c_n[n] * 
                     np.exp(2j*np.pi * rho * n / xmax))
    v_f = y_f2diff / y_f + 1
    min_indx = (v_f == np.max(v_f))
    rho_min = rho[min_indx][0]
    indx = (rho >= rho_min)
    v_f = v_f[indx]
    rho = rho[indx]
    par, var = curve_fit(coulomb_fit, rho, v_f.real, p0=[21])
    print(f"V0 = {par[0]} +- {np.sqrt(var[0, 0])}")
    # print(f"rho0 = {par[1]} +- {np.sqrt(var[1, 1])}")
    if ShowPotential:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(rho, v_f.real)
        ax.plot(rho, coulomb_fit(rho, *par))
        plt.show()
    return par[0] 

from scipy.special import binom

def upper_factorial(k, n):
    """Computes (k+n)! / k!"""
    return np.prod(np.arange(k+1, k+n+1, 1))

def taylor_coeff_dgl(u_k, N=10, J=2, n=1, alpha=0.3):
    """
    Computes the first 'N' coefficients of the potential 'v(x)', which 
    satisfies the linear differential equation
        u^(J)(x) / u(x) = v(x)
    for a given solution 'u(x) = sum_k u_k * x**k / k!' with 'u_0 = 0'.
    Note that writing 'v(x) = sum_k v_k * x**k / k!' leads 
    to the recursive solution
        v_k = (1 / ((k+1)*u_1) * 
               (u_{k+J+1} - sum_{j=0}^{k-1} binom(k+1,j) * v_k * u_{k+1-j}))
    Returns a 1-D-array with 'N' entries, corresponding to the 'v_k'.
    
    Coefficients below 'alpha' are set to zero to avoid runaway behavior.
    """
    v_k = np.zeros(N-J-1)
    c_k = u_k# / u_k[1]
    v_k[0] = c_k[J+1-n]
    for k in range(1, N-J-1, 1):
        # if k < 10: print([binom(k+1, j)# * v_k[j] * c_k[k+1-j] 
        #                     for j in range(0, k-1, 1)])
        prev_sum = np.sum([binom(k+1, j) * v_k[j] * c_k[k+1-j] 
                            for j in range(0, k-1, 1)])
        v_k[k] = (c_k[k+J+1-n] - prev_sum) / (k+1-n)
        print(v_k[k])
        if np.abs(v_k[k]) < alpha:
            v_k[k] = 0
    return v_k

def power_series_dgl(a_k, N=10, J=2, n=1):
    """
    Computes the first 'N' coefficients of the potential 'v(x)', which 
    satisfies the linear differential equation
        u^(J)(x) / u(x) = v(x)
    for a given solution 'u(x) = sum_k a_k * x**k' with 'a_0 = 0'.
    Note that writing 'v(x) = sum_k b_k * x**(k - n)' leads 
    to the recursive solution
        b_k = 1 / a_1 * (a_{k+J+1-n} * (k+1+J-n)! / (k+1-n)! 
                         - sum_{j=0}^{k-1} b_j * a_{k+1-j}))
    Returns a 1-D-array with 'N' entries, corresponding to the 'b_k'.
    """
    b_k = np.zeros(N-J)
    c_k = a_k / a_k[1]
    b_k[0] = c_k[J+1-n]
    for k in range(1, N-J-1, 1):
        # print(b_k[k-1])
        prev_sum = np.sum([b_k[j] * c_k[k+1-j] for j in range(0, k-1, 1)])
        b_k[k] = c_k[k+J+1-n] / upper_factorial(k+1-n, J) - prev_sum
    return b_k

def taylor_series(x, a_k):
    """Computes the Taylor series for a given set of coefficients 'a_k'."""
    res = 0
    for k in range(a_k.shape[0]):
        # print(a_k[k] * x**k / np.math.factorial(k))
        res = res + a_k[k] * x**k / np.math.factorial(k)
    return res

def power_series(x, a_k):
    """Computes the Taylor series for a given set of coefficients 'a_k'."""
    res = 0
    for k in range(a_k.shape[0]):
        res = res + a_k[k] * x**k
    return res

def coeff_func(x, y, N=10, alpha=0.1):
    """Returns the best-fit Taylor-series coefficients for x-y-data."""
    p = np.polynomial.polynomial.polyfit(x, y, deg=N)
    p = np.array([p[k] * np.math.factorial(k) for k in range(p.shape[0])])
    # for i in range(p.shape[0]):
        # if np.abs(p[i]) * alpha > 1:
        #     p[i] = 0
    return p

def fft_diff(x, y, nu=1, NonUniform=False):
    """Computes the 'nu' derivative of a set of poits (x, y) using fft."""
    N = y.shape[0]
    if NonUniform:
        new_x = np.linspace(x[0], x[-1], N)
        ySpline = UnivariateSpline(x, y, k=5, s=0.1)
        new_y = ySpline(new_x)
        x, y = new_x, new_y
        
    yfft = np.fft.fft(y)
    kappa = 2*np.pi * np.arange(-N//2, N//2) / (x[-1] - x[0])
    # kappa = np.pi * np.linspace(-1, 1, N, endpoint=False) #/ (x[-1] - x[0])
    # print(x[0], x[-1])
    kappa = np.fft.fftshift(kappa)
    ydiff = np.fft.ifft((1j*kappa)**nu * yfft)
    
    if NonUniform:
        return ydiff.real, x, y
    return ydiff.real

def v_fftdiff(fname='Potential1.png'):
    """Potential v using fft-differentiation"""
    x, y = u_numeric(fname)
    fig, ax = plt.subplots(figsize=(16, 9))
    y_max = np.max(np.abs(y))
    ax.set_ylim(-1.05 * y_max, 1.05 * y_max)
    ax.plot(x, y)
    ydiff, new_x, new_y = fft_diff(x, y, nu=2, NonUniform=True)
    ydiff *= ((x[-1] - x[0]) / (22*np.pi))**2
    ax.plot(new_x, new_y)
    v = np.abs(ydiff / new_y) + 1
    ax.plot(new_x, ydiff)
    ax.plot(new_x, v)
    # print(v[v.shape[0]//2])
        

def v_power_series(fname='Potential1.png'):
    """Approximation of the potential using Taylor-series."""
    x, y = u_numeric(fname)
    m = 7
    max_indx = int(y.shape[0] * m / 22)
    
    Npoints = 500
    rho = np.linspace(0, m*np.pi, Npoints)
    # rho = np.linspace(0, 0.99, Npoints)
    
    x = x[:max_indx] * rho[-1] / x[max_indx]
    y = y[:max_indx]
    
    # x, y = rho, np.exp(-rho/(14*np.pi)) / (rho + 1e-2)
    # x, y = rho, np.sin(rho)
    
    N = 150
    p = coeff_func(x, y, N=N)
    # print(p)
    # print([np.diff(y, n)[n//2] / (x[1] - x[0])**n for n in range(N)])
    p = np.zeros(N)
    # p[1::4] = 1
    # p[3::4] = -1
    # p += 0.1 * np.random.uniform(-1, 1, size=N)
    # print(p)
    
    #sinc
    for i in range(p.shape[0]):
        if i % 2 == 0:
            p[i] = (-1) ** (i // 2) / (i+1)
    
    # p = (-1)**np.arange(N)
    v_k = taylor_coeff_dgl(p, N=100, n=0)
    # print(v_k)
    # # raise Warning
    v_taylor = taylor_series(rho, v_k) + 1
    # print(v_taylor)
    # b_k = power_series_dgl(p, N, n=0)
    # # v_taylor = power_series(rho, b_k) + 1# / rho
    # v_taylor = power_series(rho, np.array([k*(k-1)*p[k] for k in range(2, p.shape[0], 1)])) / power_series(rho, p)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_ylim(np.min(y)*1.05, np.max(y)*1.05)
    ax.plot(x, y)
    ax.plot(rho, taylor_series(rho, p))
    ax.plot(rho, v_taylor)
    ax.plot(rho, np.exp(-rho/(14*np.pi)) / (rho + 1e-2))
    plt.show()
    return v_k
    

def v_gen(code='heaviside', V0=0.4, rho0=14*np.pi):
    """
    V0 == potential strength; sign determines the sign of the full potential
    rho0 == potential range
    code can be 'heaviside', 'coulomb', 'yukawa' or 'gauss'."""
    if code == 'heaviside':
        def v(rho):
            if rho > rho0:
                return 0
            return V0
    elif code == 'coulomb':
        def v(rho):
            if rho > rho0:
                return 0
            return V0 / rho
    elif code == 'yukawa':
        def v(rho):
            if rho > rho0:
                return 0
            return V0 * np.exp(-rho / rho0) / rho
    elif code == 'gauss':
        def v(rho):
            # if rho > rho0:
            #     return 0
            return V0 * np.exp(-(rho / rho0)**2 * 4.5)
    else:
        print("No valid code, aborting programm!")
        raise TypeError
    return v

def u_l_gradient(u_vec, rho=0.0, l: int=0, v=v_gen()):
    """
    u_l'' = [v(rho) - 1 + l*(l+1) / rho**2] * u =: v_mod * u    # rho = k*r
    rewriting as u = [u1, u2] with u2 := u1' leads to
    u' = [u1', u2'] = [u2, u1''] = [[0, 1], [v_mod, 0]] * [u1, u2]
    """
    v_mod = v(rho) - 1 + l*(l+1) / rho**2
    return np.array([u_vec[1], v_mod * u_vec[0]])

def u_l(rho, l: int=0, v=v_gen()):
    """u_l(rho = k*r) as a function of 'l' and the potential 'v'."""
    u0_vec = np.array([0, 0.0001])
    u_vals = odeint(u_l_gradient, u0_vec, rho, args=(l, v))
    return u_vals

def plot_u_l(code='yukawa', V0=21.5, ShowPotential=0):
    """Plot u_l(rho = k*r) for a given potential"""
    N = 500     # number of points for the rho-array
    l = 0       # orbital angular momentum quantum number    
    v = v_gen(code=code, V0=V0)     # potential for SE
    
    rho = np.linspace(0.001, 22*np.pi, N)
    u_vals = u_l(rho, l, v)[:, 0]
    u_max = np.max(np.abs(u_vals))
    print(f"Maximal value of u_l is {u_max}")
    u_vals /= u_max
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlabel(r"$\rho$")
    # ax.set_ylabel(r"$u_l(\rho)$") 
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(2 * np.pi))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(special.multiple_formatter()))
    
    ax.spines['left'].set_position('zero')
    
    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    # remove y-ticklabels 
    ax.yaxis.tick_left()
    
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    
    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    
    ax.plot(rho, u_vals, c='b', lw=special.lw)
    if ShowPotential:
        v_rho = v(rho)
        drho = rho[1] - rho[0]
        v_num = (np.diff(u_vals, 2) / drho**2) / u_vals[1:-1] + 1
        ax.plot(rho, v_rho / np.max(np.abs(v_rho)), 
                c='orange', lw=special.lw)
        ax.plot(rho[1:-1], v_num / np.max(np.abs(v_rho)), 
                c='g', lw=special.lw)
        
    ax.axis([0.0, np.max(rho), -1.05, 1.05])
    ax.tick_params(labelsize=special.tls)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False) 
    ax.tick_params(axis='x', which='major', direction='inout', length=20, 
                   width=1.2, labelbottom=True) 
    ax.tick_params(axis='x', which='minor', direction='in', length=10, 
                   width=0.9, labelbottom=True) 
    fig.tight_layout()
    fig.canvas.draw()
    special.embed_labels(ax, SetCaptions=False, labelAxis='x')
    plt.setp(ax.get_xticklabels()[1], visible=False)    # remove '0' label
    plt.show()
    return ax
    
if __name__ == "__main__":
    print(__doc__)
    # plot_cross_sections()
    # x, y = u_numeric()
    # b_k = v_power_series()
    v_fftdiff()
    # V0 = analyze_v_fourier(fname='Potential2.png', ShowPotential=1)
    # V0 = 14.25      # coulomb
    # V0 = 23.7       # yukawa
    # ax = plot_u_l(code='yukawa', V0=V0, ShowPotential=0)
    # V0 = -2.05        # gauss
    # ax = plot_u_l(code='gauss', V0=V0, ShowPotential=1)