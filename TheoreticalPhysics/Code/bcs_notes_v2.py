# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 21:46:52 2022

@author: Joachim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import dblquad


def get_k(n):
    k = np.linspace(0.0, np.pi, n)
    return k

    
def get_delta0(k):
    n = k.size
    delta0 = np.zeros(n*n)
    for ix in range(n):
        for iy in range(n):
            delta0[ix*n + iy] = np.cos(k[ix]) + np.cos(k[iy])
            
    return delta0


def get_j_k(k, j0=1.0):
    n = k.size
    j_k = np.zeros(n*n)
    for ix in range(n):
        for iy in range(n):
            j_k[ix*n + iy] = np.cos(k[ix]) + np.cos(k[iy])
            
    return j0/2 * j_k


def get_eps_k(j_k, mu=1.0):
    eps = 0.5 * j_k + mu
    return eps
    

def func(delta, j_k, eps_k):
    """Root function for the gap-equation"""
    f_val = np.ones_like(delta)
    for i in range(delta.size):
        ratio = delta / np.sqrt(eps_k**2 + delta**2)
        f_val[i] += np.sum((j_k[i] + j_k) * ratio) / (2 * delta[i])
        
    return f_val


def func_prime(delta, j_k, eps_k):
    n2 = delta.size         # size is n**2
    fp_val = np.zeros((n2, n2))
    for i in range(n2):
        for j in range(n2):
            ratio = eps_k[j]**2 / (eps_k[j]**2 + delta[j]**2)**(3/2)
            fp_val[i, j] = ratio * (j_k[i] + j_k[j]) / 2
            if i == j:
                ratio = np.sum((j_k[i] + j_k) * eps_k**2 / np.sqrt(eps_k + delta**2))
                fp_val[i, i] -= ratio / (2 * delta[i]**2)
    
    return fp_val


# sp.optimize.newton(func)
# n = 5
# j0 = 1.0
# mu = 2.0
# k = get_k(n)
# j_k = get_j_k(k, j0)
# eps_k = get_eps_k(j_k, mu)
# # delta0 = get_delta0(k)
# # delta0 = np.ones(n*n)
# delta0 = np.random.uniform(3.0, 6, size=n*n)
# print(np.linalg.norm(func(delta0, j_k, eps_k), ord=np.inf))

# # delta = root(func, x0=delta0, jac=func_prime, args=(j_k, eps_k))
# delta = root(func, x0=delta0, jac=func_prime, args=(j_k, eps_k),
#              method="lm")
# print(np.linalg.norm(func(delta.x, j_k, eps_k), ord=np.inf))


def cos_k(kx, ky):
    return np.cos(kx) + np.cos(ky)

def f_func(fg, kx, ky, mu_eff):
    c_k = cos_k(kx, ky)
    # return (1 + ((c_k + mu_eff) / (c_k * fg[0] + fg[1]))**2)**(-0.5)
    fg_c_k = c_k * fg[0] + fg[1]
    return fg_c_k / np.sqrt(fg_c_k**2 + (c_k + mu_eff)**2)


def g_func(fg, kx, ky, mu_eff):
    c_k = cos_k(kx, ky)
    # return c_k * (1 + ((c_k + mu_eff) / (c_k * fg[0] + fg[1]))**2)**(-0.5)
    fg_c_k = c_k * fg[0] + fg[1]
    return fg_c_k * c_k / np.sqrt(fg_c_k**2 + (c_k + mu_eff)**2)


def implicit_func(fg, kx, ky, mu_eff):
    arg1 = fg[0] + np.sum(f_func(fg, kx, ky, mu_eff))
    arg2 = fg[1] + np.sum(g_func(fg, kx, ky, mu_eff))
    return np.array([arg1, arg2])


def implicit_vec(fvec, gvec, kx, ky, mu_eff):
    return np.array([[implicit_func([fvec[i], gvec[j]], kx, ky, mu_eff)
                      for j in range(gvec.size)]
                     for i in range(fvec.size)])


def implicit_jac(fg, kx, ky, mu_eff):
    return #TODO



def get_delta_fg(fg, kx, ky, j0):
    return -j0 /4 * (cos_k(kx, ky) * fg[0] + fg[1])


def check_delta(delta, kx, ky, j0, mu):
    res = np.copy(delta)
    j_k = j0 * cos_k(kx, ky) / 2
    eps_k = j_k / 2 + mu
    ratio = delta / np.sqrt(eps_k**2 + delta**2)
    for i in range(kx.shape[0]):
        for j in range(kx.shape[1]):
            res[i, j] += 0.5 * np.sum(ratio * (j_k + j_k[i, j]))
    
    return res


n = 50
j0 = 1.0
mu = -5.0
mu_eff = 4*mu / j0
k = np.linspace(-np.pi, np.pi, n)
kx, ky = np.meshgrid(k, k)

# f = np.linspace(-2475.0, -2477.0, 40)
# g = np.linspace(101, 103, 40)
f = np.linspace(-1554.0, -1555.0, 40)
g = np.linspace(1463, 1464, 40)
im_vals = implicit_vec(f, g, kx, ky, mu_eff)
arg1, arg2 = im_vals[:, :, 0], im_vals[:, :, 1]


# fplot, gplot = np.meshgrid(f, g)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(fplot, gplot, arg1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
# surf = ax.plot_surface(fplot, gplot, arg2, cmap=plt.cm.plasma, linewidth=0, antialiased=False)

###[-1554.37806548  1463.82949211] for n=50, -pi..pi, j0=1.0, mu=-5.0
fg0 = np.array([-1554.4, 1463.8])
fg_par = root(implicit_func, x0=fg0, args=(kx, ky, mu_eff), method="lm")
print(fg_par.x, implicit_func(fg_par.x, kx, ky, mu_eff))

delta = get_delta_fg(fg_par.x, kx, ky, j0)
check = check_delta(delta, kx, ky, j0, mu)
print("2-norm of the implicit function at 'delta':", np.linalg.norm(check))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(kx, ky, delta, cmap=plt.cm.plasma, linewidth=0,
                       antialiased=False)
