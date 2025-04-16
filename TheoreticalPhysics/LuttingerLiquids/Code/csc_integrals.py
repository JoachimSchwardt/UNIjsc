#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Everything about csc integrals

#csc b+3 csc b
m=2; omm=1j*(2*m); b=0.1
c_0 = lambda k,om,b: (2**(b-1)/special.gamma(b) * np.abs(special.gamma(b/2 + 1j*k/2))**2
                      * np.exp(-k*np.pi/2) * (np.exp(k*np.pi+om*np.pi) - 1) / (om + k))
# c_quad(lambda tau: np.exp(om*tau) * c_quad(lambda x: np.exp(-1j*k*x)  / np.sin(tau+1j*x)**(b),
#                                            -300, 300)[0], 0, np.pi), c_0(k,om,b)
kpv = np.linspace(-10, 10, 400)
val = 0
for m in range(-200, 201):
    omm=1j*2*m
    integrand = lambda kp: c_0(kp, omm, 2) * j_0(k-kp,om-omm,np.pi,b) / (2*np.pi**2)
    val += integrate.simpson(integrand(kpv), kpv)
#c_quad(lambda kp: c_0(kp, omm, 2) * j_0(k-kp,om-omm,np.pi,b), -L, L)
print(val, j_1p(k,om,np.pi,b) * -(b+3), c_0(k,om,b+3))  # --> decent approximation!


#csc 1+b csc b
k=0.85; n=3; om=1j*(2*n+1); b=1.6; a=1+b; L=np.inf
coshln = lambda x: -np.log(2) + np.abs(x) + np.log(1 + np.exp(-2*np.abs(x)))
integrand = lambda x, k: (np.exp(special.loggamma((a+1j*k)/2 - 1j*x/2)
                                 + special.loggamma((a-1j*k)/2 + 1j*x/2)
                                 + special.loggamma(b/2 - 1j*x/2)
                                 + special.loggamma(b/2 + 1j*x/2)
                                 + coshln(np.pi/2 * k - np.pi*x)) / (1j*(2*n+1) + k - 2*x))
# print(c_quad(lambda tau: np.exp(1j*(2*n+1)*tau)
#              * c_quad(lambda x: np.exp(-1j*k*x) / np.sin(tau+1j*x)**a / np.sin(tau-1j*x)**b,
#                       -10, 10)[0], 0, np.pi))
print(c_quad(integrand, -L, L, k)[0] / (-special.gamma(a) * special.gamma(b)
                                        * np.pi / 2**(a+b-2)))
print(j_0(k,om,np.pi,b))

#csc 3+b csc b
k=0.85; n=3; om=1j*(2*n+1); b=0.1; a=3+b; L=np.inf
print(c_quad(integrand, -L, L, k)[0] / (-special.gamma(a) * special.gamma(b)
                                        * np.pi / 2**(a+b-2)))
print((2*j_0(k,om,np.pi,2+b) - j_0(k+2j,om+2j,np.pi,2+b) - j_0(k-2j,om-2j,np.pi,2+b))/4)


#J1+ var
K=1.1; KP=(K+1)/2; KM=(K-1)/2; N=(1/K-K)/2
x=0.53; tau=0.78; beta=2.3; u=1.2; alpha=0.9; sa=-1; ell=1; L=5
xp=0.34; taup=1.22; k=0.43; omega=0.15; delta=1e-16j
n=2; om=1j*(2*n+1)*np.pi/beta;
print(c_quad(lambda tau: np.exp(om*tau)
             * c_quad(lambda x: np.exp(-1j*k*x) / np.sin(np.pi/beta/u*(u*tau+1j*x))**KP
                      / np.sin(np.pi/beta/u*(u*tau-1j*x))**KM
                      / np.tan(np.pi/beta/u*(u*tau+1j*ell*(x+sa*alpha))), -L, L)[0], 0, beta))
mm=50
print(c_quad(lambda kp: np.sum([(j_0(k+kp,om-2j*m*np.pi/beta,beta,KM,u)
                                 * -u/np.pi * np.exp(-1j*kp*sa*alpha)
                                 / (2j*m*np.pi/beta -u*ell*kp)
                                 + j_0(k-kp,om-2j*m*np.pi/beta,beta,KM,u)
                                 * -u/np.pi * np.exp(1j*kp*sa*alpha)
                                 / (2j*m*np.pi/beta +u*ell*kp))
                                for m in range(-mm, mm+1)]), 0, L))

n=1; om=1j*2*n; b=0.3; k=0.43
print(c_quad(lambda tau: np.exp(om*tau) * c_quad(lambda x: np.exp(-1j*k*x) / np.sin(tau+1j*x)**(b+1)
                                                 / np.sin(tau-1j*x)**(b-1), -30, 30)[0], 0, np.pi))
print(-(l_0(k+2j,om+2j,np.pi,b+1) - 2*l_0(k,om,np.pi,b+1) + l_0(k-2j,om-2j,np.pi,b+1))/4)
print((j_0(k+1j,om+1j,np.pi,b) - j_0(k-1j,om-1j,np.pi,b))/2j)
import sympy as sy
def j_0(k, i_omega_n, beta=sy.pi, b=0.5, v=1):
    k_p = beta * (i_omega_n + v*k) / (4*sy.pi)
    k_m = beta * (i_omega_n - v*k) / (4*sy.pi)
    prefactor = 1j * v * beta**2 / sy.pi * sy.gamma(1-b) / sy.gamma(1+b) * 2**(2*b-1)
    gamma_p = sy.gamma(b/2 - I*k_p) / sy.gamma(1-b/2 - I*k_p)
    gamma_m = sy.gamma((1+b)/2 - I*k_m) / sy.gamma((1-b)/2 - I*k_m)
    return prefactor * gamma_p * gamma_m

def l_0(k, i_omega_n, beta=sy.pi, b=0.5, v=1):
    k_p = beta * (i_omega_n + v*k) / (4*sy.pi)
    k_m = beta * (i_omega_n - v*k) / (4*sy.pi)
    prefactor = v * (beta / sy.pi)**2 * sy.sin(sy.pi*b) * sy.gamma(1-b)**2 * 2**(2*b-2)
    gamma_p = sy.gamma(b/2 - I*k_p) / sy.gamma(1-b/2 - I*k_p)
    gamma_m = sy.gamma(b/2 - I*k_m) / sy.gamma(1-b/2 - I*k_m)
    return prefactor * gamma_p * gamma_m
#progress csc b+1 csc b-1
#gamma(-b/2 - I k/4 - I w/4 + 1/2)*gamma(b/2 - I k/4 - I w/4 + 1/2)
# - gamma(-b/2 - I k/4 - I w/4 + 3/2)*gamma(b/2 - I*k/4 - I*w/4 - 1/2)
# == (-1 + b) Gamma[1/2 - b/2 - I/4 k - I/4 w] Gamma[-1/2 + b/2 - I/4 k - I/4 w]    # (WAlpha)
(2**(2*b-2)*sy.sin(sy.pi*b)*sy.gamma(1 - b)**2 *sy.gamma(b/2 - I*(om+k)/4 - 1/2)
 *sy.gamma(b/2 - I*(om-k)/4 + 1/2)/(sy.gamma(-b/2 - I*(om+k)/4 + 1 + 1/2)
                                    *sy.gamma(-b/2 - I*(om-k)/4 + 1 - 1/2))*(b-1)/b
 ).evalf(subs={k:0.43, om:2j, b:0.3})
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate
from special_functions import c_quad
from green_functions_matsubara import j_0, l_0
import mpl_special


SIZE = 30
DELTA = 1e-13

def integrand_csc(x, tau, a, b):
    return 1 / np.sin(tau + 1j*x)**a / np.sin(tau - 1j*x)**b


def integrand_gamma(kp, k, tau, a, b):
    prefactor = 2**(a+b-2) / special.gamma(a) / special.gamma(b) / (2*np.pi)
    gamma_log_a = np.log(np.abs(special.gamma(a/2 + 1j*(k-kp)/2)))
    gamma_log_b = np.log(np.abs(special.gamma(b/2 + 1j*kp/2)))
    exp_arg = kp * (np.pi - 2*tau) + 2 * gamma_log_a + 2 * gamma_log_b
    return prefactor * np.exp(k * (tau - np.pi/2)) * np.exp(exp_arg)


def integral_csc(k, tau, a, b):
    return np.real(c_quad(lambda x: np.exp(-1j*k*x) * integrand_csc(x, tau, a, b), -SIZE, SIZE))


def integral_gamma(k, tau, a, b):
    upper = SIZE
    while integrand_gamma(upper, k, tau, a, b) > DELTA:
        upper *= 2
    lower = -SIZE
    while integrand_gamma(lower, k, tau, a, b) > DELTA:
        lower *= 2
    return integrate.quad(lambda kp: integrand_gamma(kp, k, tau, a, b), lower, upper)


def plot_tau(k, tauv, a, b):
    yv = np.array([integral_gamma(k, tau, a, b)[0] for tau in tauv])
    _, ax = plt.subplots()
    ax.set_yscale("log")
    line = ax.plot(tauv, yv, ls='', marker='o')
    # zv = yv[0] * (tauv / tauv[0])**(-0.13 * k) * np.exp(k * tauv)
    # p = 1 + 4e-4
    # #zv = yv[0] * (1 + 1200 * (tauv - tauv[0]))**(-(a+b) * np.abs(k)) * np.exp(2*k * tauv)
    # zv = yv[0] * (1 + 1000 * (tauv - tauv[0]))**(-(a+b) * np.abs(k)) * (1 - tauv / np.pi)**(-1.7*np.abs(k)) * np.exp(k*tauv)
    # #zv = yv[0] * ((p - np.exp(-k*tauv)) / (p - np.exp(-k*tauv[0])))**(-0.7) * np.exp(2*k*tauv)
    # ax.plot(tauv, zv, alpha=0.5, c=line[0].get_color())

    #int tauv csc gamma
    a = 1.2
    b = 0.3
    k = -0.43
    tauv = np.concatenate((np.geomspace(1e-4, 1e-1, 10), np.linspace(0.15, np.pi-0.15, 9),
                           np.pi - np.geomspace(1e-4, 1e-2, 5)[::-1]))
    yv = np.array([integral_gamma(k, tau, a, b)[0] for tau in tauv])
    #plot tauv csc gamma
    _, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    line = ax.plot(tauv, yv, ls='', marker='o')
    zv = yv[0] * (1 + 1000 * (tauv - tauv[0]))**(-(a+b) * np.abs(k)) * (1 - tauv / np.pi)**(-1.7*np.abs(k)) * np.exp(k*tauv)
    ax.plot(tauv, zv, alpha=0.5, c=line[0].get_color())


def mc_int(func, domain, n_points=10**4, args=(), dist=np.random.uniform):
    domain = np.asarray(domain)
    assert(domain.shape[1] == 2)
    xv = np.array([dist(domain[i, 0], domain[i, 1], size=n_points)
                   for i in range(domain.shape[0])])
    fv = func(xv, *args)
    domain_size = np.prod([domain[i, 1] - domain[i, 0] for i in range(domain.shape[0])])
    return np.mean(fv) * domain_size


def test():
    """MC Integration test; works up to ~5% error with 1e8 samples"""
    k=0.43; n=2; beta=np.pi; u=1.0; om=1j*(2*n+1)*np.pi/beta; K=0.6; N=(1/K-K)/2;
    a=N/2; b=(K-1)/2; L=500; n_m=30; ell=1; xm=10
    prefactor = 1 / (2*np.pi*beta)
    def integrand_kw(kp, i_omega_m, k, i_omega_n, a, b):
        return (l_0(kp, i_omega_m, beta, a, u)
                * j_0(k-kp, i_omega_n - i_omega_m, beta, b, u)
                * j_0(kp-k, i_omega_n - i_omega_m, beta, b, u)) * prefactor
    integral_kwv = np.array([c_quad(integrand_kw, -L, L, args=(1j*2*m*np.pi/beta, k, om, a, b))
                             for m in range(-n_m, n_m+1)])
    integral_kw = np.sum(integral_kwv[:,0])
    def integrand_csc(xp, x, taup, tau):
        return (np.exp(om*tau-1j*k*x) / (np.sin(tau)**2 + np.sinh(x)**2)**a
                / (np.sin(taup)**2 + np.sinh(xp)**2)**b
                / (np.sin(tau-taup)**2 + np.sinh(x-xp)**2)**b
                / np.sin(taup+1j*xp*ell) / np.sin(tau-taup - 1j*(x-xp)*ell))
    integral_csc = mc_int(lambda y: integrand_csc(y[1], y[0], y[2], y[3]),
                          [[-xm, xm], [-xm, xm], [0, np.pi], [0, np.pi]], n_points=10**8)
    print(integral_kw, integral_csc)


def test_j1_integral_dev():
    n=2; b=0.3; k=0.43; xi=0.1; L=2
    om = 1j*(2*n+1)
    # ana_res = c_quad(lambda tau: np.exp(om*tau)
    #                  * c_quad(lambda x: np.exp(-1j*k*x) / np.tan(tau+1j*x+1j*xi)
    #                           / np.sin(tau+1j*x)**(1+b) / np.sin(tau-1j*x)**b, -L, L)[0], 0, np.pi)
    # print(ana_res)  #(14.23-0.568j)
    # integrand = lambda x,tau: np.log(1 / np.tan(tau+1j*x+1j*xi/2)
    #                                  / np.sin(tau+1j*x-1j*xi/2)**(1+b) / np.sin(tau-1j*x+1j*xi/2)**b)
    # #integrand = lambda x,tau: -np.log(np.sin(tau+1j*x))
    # #f = lambda x,tau: -(b+1)*np.log(tau+1j*x-1j*xi/2) - b*np.log(tau-1j*x+1j*xi/2) - np.log(tau+1j*x+1j*xi/2)
        #+ (b+1)/6*(tau+1j*x-1j*xi/2)**2 + b/6*(tau-1j*x+1j*xi/2)**2 - 1/3*(tau+1j*x+1j*xi/2)**2
    # f = lambda x,tau: np.log((tau**2+(x-xi/2)**2)**(-b)/(tau**2-x**2+2j*tau*x+xi**2/4))
    # xm=0.5; taum=0.3
    # xv=np.linspace(-xm,xm,200)
    # tauv=np.linspace(-taum,taum,200)
    # x2,tau2=np.meshgrid(xv,tauv)
    # z2=integrand(x2,tau2)
    # y2=f(x2,tau2)
    # fig=plt.figure()
    # ax=fig.add_subplot(projection="3d")
    # ax.plot_surface(x2,tau2,z2.real)
    # ax.plot_surface(x2,tau2,y2.real)
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$\tau$")
    # ax.set_zlabel(r"$f(x,\tau)$")
    mybeta=lambda x,y:special.gamma(x)*special.gamma(y)/special.gamma(x+y)
    def integrand(kp, mmax=150, rtol=1e-2):
        term1 = np.pi/2 * j_0(k-kp,om+kp,np.pi,b)/np.tanh(kp*np.pi/2)
        pre_term2 = - np.pi**2*2**(2*b)/special.gamma(1+b)
        term2 = 0
        for m in range(mmax):
            pre1 = -b*(-1)**m/special.gamma(1+m)/special.gamma(1-b-m)
            delta1 = mybeta(-b, 1/2-m+1j*(k-kp)/2)
            ratio1 = (1+2/(np.exp(np.pi*(2j*b+om+k-kp))-1)) / (2j*(2*m+b)+om + k)
            pre2 =  (-1)**m/special.gamma(1+m)/special.gamma(-b-m)
            delta2 = mybeta(1-b, -1/2-m-1j*(k-kp)/2)
            ratio2 = (1+2/(np.exp(np.pi*(2j*b+om-k+kp))-1)) / (2j*(2*m+1+b)+om -k+2*kp)
            delta_full = pre1 * ratio1 * delta1 + pre2 * ratio2 * delta2
            term2 += delta_full
            if np.abs(delta_full / (term1 / pre_term2 + term2)) < rtol:
                break
        if m == mmax-1:
            print(f"Precision {rtol} not achieved in {mmax} steps;"
                  f" final error was {np.abs(delta_full / (term1 / pre_term2 + term2)):.3e}"
                  f" (par: k={k}, kp={kp}, omega={om}, b={b})")
        return term1 + pre_term2 * term2

def test_gf1_integral():
    from numba import njit, prange, objmode
    from gf_h1h3_v3 import green_perturbative
    k=0.02; beta=4.2; v=1.6; a=1.0; w=1.3; K=1.4; om=1j*np.pi/beta
    def integrand(x,xp,tau,taup,k,om,K,beta,v,w=1,a=1):
        z_p = np.pi/beta * (tau + 1j*x/v)
        z_m = np.pi/beta * (tau - 1j*x/v)
        zp_p = np.pi/beta * (taup + 1j*xp/v)
        zp_m = np.pi/beta * (taup - 1j*xp/v)
        M=(K+1/K-2)/4
        K_p = (K+1)/2
        K_m = (K-1)/2
        xi=np.pi*a/beta/v
        prefactor = np.pi**2*v/(2*beta*v)**3 * w**(2*M+2*K_m) * np.exp(-1j*k*x) * np.exp(om*tau)
        term1 = 1 / (np.sin(z_p) * np.sin(z_m))**(M-K_m) / np.sin(zp_p)**K_p / np.sin(zp_m)**K_m
        term2 = 1 / np.sin(z_p - zp_p)**K_m / np.sin(z_m - zp_m)**K_p
        term3 = ( (1+K) * (1/np.tan(z_m-zp_m + 1j*xi) + 1/np.tan(zp_p + 1j*xi)
                           + 1/np.tan(z_m-zp_m - 1j*xi) + 1/np.tan(zp_p - 1j*xi))
                 + (1-K) * (1/np.tan(z_p-zp_p + 1j*xi) + 1/np.tan(zp_m + 1j*xi)
                            + 1/np.tan(z_p-zp_p - 1j*xi) + 1/np.tan(zp_m - 1j*xi)) )
        return prefactor * term1 * term2 * term3
    ana_res = green_perturbative([k],[om],beta,K,v,g=1,a=a, w=w,
                                 num_params={"order" : 1, "mmax" : 10, "mmaxp" : 8, "lmax" : 5,
                                             "kp" : 4, "numkp" : 501})[0][0][0,1]
    xm=2
    func = lambda y: integrand(y[0],y[1],y[2],y[3],k,om,K,beta,v,w,a)
    domain = [[-xm,xm],[-xm,xm],[0,beta],[0,beta]]
    @njit(parallel=True)
    def get_mc_res(n_iter=100):
        mc_res = np.zeros(n_iter, dtype=np.complex128)
        for i in prange(n_iter):
            with objmode():
                mc_res[i] = mc_int(func, domain, n_points=10**6)
            if (i % 10) == 0:
                print(f"finished iteration {i}...")
        return np.sum(mc_res) / n_iter
    print("finished compilation; preliminary result : ", get_mc_res(1))
    mc_res=get_mc_res(100)
    print(ana_res)
    print(mc_res)
    # 08.01.2024 :: 
    # ana_res = (8.406920586111546e-16+6.324317578236352j)
    # xm=3
    # (0.05+5.786j)
    # (-0.02+5.988j)
    # (-0.01+5.524j)
    # (-0.29+5.987j)
    # (0.17+6.081j)
    # xm=2
    # (-0.05+6.205j)
    # (-0.03+5.999j)
    # (0.09+6.028j)
    # (0.14+6.049j)
    # (0.005+6.015j)



def main():
    # print(__doc__)
    x = 0.43
    tau = 0.78
    k = 0.043
    kv = np.linspace(-5, 5, 100)
    tauv = np.append(np.geomspace(1e-4, 1e-1, 30), np.linspace(0.15, 3.14, 30))
    omega = 0.15
    # u = 1.2
    # beta = 2.6
    # alpha = 1e-5
    K = 0.6
    a = 1.2
    b = 0.3
    # xp = 0.67
    # tp = 0.23
    # test_j1_integral_dev()
    # test_gf1_integral()

    return 0


if __name__ == "__main__":
    main()
