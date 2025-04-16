"""FKP-Plots und mehr"""

from scipy.integrate import quad as integrate 
from scipy.special import factorial
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

NA = 6.02214076 * 10**(23) # 1/mol


me = 9.1093837015 * 10**(-31) # kg
mp = 1.67262192369 * 10**(-27) # kg
mn = 1.67492749804 * 10**(-27) # kg

c = 299762458 # m/s
u = 1.66 * 10**(-27) # kg
e = 1.602176634 * 10**(-19) # As
mu0 = 1.25663706212 * 10**(-6) # Vs/Am
eps0 = 8.8541878128 * 10**(-12) # As/Vm
G = 6.67430 * 10**(-11) # m^3/kg s^2

T0 = -273.15 # °C
k = 1.380649 * 10**(-23) # J/K
sigma = 5.670374419 * 10**(-8) # W/m^2 K^4
b = 2.897771955 * 10**(-3) # m K
h = 6.62607015 * 10**(-34) # J s
hbar = 1.054571817 * 10**(-34) # J s
g = 9.81 # m/s^2

# zusammengesetzte Größen
R = NA * k # Gaskonstante
alpha = e**2/(4*np.pi*eps0*hbar*c) # Feinstrukturkonstante
ER = alpha**2 * me * c**2 / 2 # Rydberg-Energie in J
a0 = hbar / (me * c * alpha)  # Bohrradius in m
muB = hbar * e / (2 * me)     # Bohrmagneton in T/J


    
def fplot_single(functions, x, args=None, label=['label'], lw=[1], c=['b'],
                 ls=['-'], xlabel='x', ylabel='y', title='title', vline=None,
                 hline=None, gridlines=True, xlog=True, ylog=True):
    try:
        nf = len(functions)
    except TypeError:
        functions = [functions]
        nf = len(functions)
    if len(ls) == 1:
        ls = ls * nf
    if len(lw) == 1:
        lw = lw * nf
    if len(label) != nf:
        label = [f'{i+1}' for i in range(nf)]
    if len(c) != nf:
        c = [[i/nf, 0.3 * i/nf, 1 - i/nf, 1] for i in range(nf)]
        
        
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(title, fontsize=14)
    if xlog == True:
        ax.set_xscale('log')
    if ylog == True:
        ax.set_yscale('log')
    for i, f in enumerate(functions):
        try:
            y = f(x, args[i])
        except TypeError:
            y = f(x)
        ax.plot(x, y, lw=lw[i], ls=ls[i], c=c[i], label=label[i])
    
    if vline != None:
        ax.axvline(vline, lw=0.8, c='k')
    if hline != None:
        ax.axhline(hline, lw=0.8, c='k')
    ax.grid(gridlines)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    # plt.subplots_adjust(bottom=0.15)
    plt.show()
    return

def fplot(f, x, label=[''], nrows=1, ncols=1, yscale='', c=['b'], lw=[1],
          ls=['-'], coord_lines=True, coord_grid=True, xlabel='', ylabel='',
          args=[], simplePlot=False):
    if nrows != int(nrows):
        error_nrows = ("nrows must be an integer but was {}!\n".format(nrows)
                       + "It was converted to nrows={}".format(int(nrows)))
        print(error_nrows)
        nrows = int(nrows)
    if ncols != int(ncols):
        error_ncols = ("ncols must be an integer but was {}!\n".format(ncols)
                       + "It was converted to ncols={}".format(int(ncols)))
        print(error_ncols)
        ncols = int(ncols)
    
    fig, axis = plt.subplots(nrows, ncols, figsize=(15, 10))
    if nrows*ncols == 1:
        axis = [axis]
        
    for i in range(0, nrows*ncols):
        # draw coordinate lines
        if coord_lines == True:
            axis[i].axvline(0, lw=0.8, c='k')
            axis[i].axhline(0, lw=0.8, c='k')
        
        # draw grid lines
        axis[i].grid(coord_grid)
        
        # set xy labels and tick params
        axis[i].set_xlabel(xlabel, fontsize=22)
        axis[i].set_ylabel(ylabel, fontsize=22)
        axis[i].tick_params(labelsize=16)
        # axis[i].plot([100, 200, 300, 600, 1200],[0.2, 0.28, 0.3, 0.322,0.339],
        #              marker='x', ls='', mew=1, c='k', ms=8, label='Daten')
        for j in range(len(f)):
            if simplePlot == False:
                axis[i].plot(x, f[j](x, *args[j]), lw=lw[j], 
                             ls=ls[j], c=c[j], label=label[j])
            elif simplePlot == True:
                axis[i].plot(x, f[j](x))
        
        axis[i].legend(fontsize=20)
    plt.show()

def U(r, eps, s, a12=1, a6=1):
    return 4*eps*(a12*(s/r)**(12) - a6*(s/r)**6)

def f(T, x, m, T0):
    return 3*k / m * (x * T0/T * 1 / np.sinh(x * T0/T))**2

def fconst(x, const):
    return const + 0*x

# T0 = 300
# T = np.linspace(0.01, 5, 400) * T0
# x = np.linspace(0.2, 1, 5)
# label = [r'$c(T)$ bei x=' + str(round(i, 3)) for i in x] + [r'$3k_B/m$']
# mGe = 72.61 * 1000 * u
# args = [[x[i], mGe, T0] for i in range(5)] + [[3*k / mGe]]
# color = [[1-i, 0, i] for i in np.arange(0, 1, 0.2)] + ['k']
# lw, ls = [1.5]*5 + [1.2], ['-']*5 + ['--']
# func = [f]*5 + [fconst]
# # print(args)
# fplot(func, T, label=label, xlabel=r'$T\ /\ K$', ylabel=r'$c\ /\ \frac{J}{g\cdot mol}$', c=color, args=args, lw=lw, ls=ls)
    
def g(E):
    result = []
    for Eval in E:
        if Eval <= 1:
            result.append(Eval / 2) 
        elif Eval > 1:
            result.append((4*Eval - 2 - Eval**2) / (4 - 2*Eval))
    return result

def g2(E, N):
    result = []
    for Eval in E:
        if Eval <= 1:
            result.append(Eval**N / factorial(N)) 
        elif Eval > 1:
            result.append(1 - (2 - Eval)**N * (1 - 1 / factorial(N)))
    return result
x = [2, 3, 4, 5, 6]
label=[str(i) for i in x]
args = [[i] for i in x]
color = [[1-i, 0, i] for i in np.arange(0, 1, 0.2)]

E = np.linspace(0, 2, 300)
fplot([g2]*5, E, label=label, lw=[1]*5, ls=['-']*5, c=color, args=args)
# def f(x, a):
#     return np.tanh(x + a)
# eps = 4.683 * 1e-22     # in J
# s = 2.801 * 1e-10       # in m

# a6bcc, a12bcc = 12.253, 9.1141
# a6hcp, a12hcp = 14.4549, 12.1323
# a6fcc, a12fcc = 14.4539, 12.1319
# r0bcc = s * (2*a12bcc/a6bcc)**(1/6)
# r0hcp = s * (2*a12hcp/a6hcp)**(1/6)
# r0fcc = s * (2*a12fcc/a6fcc)**(1/6)

# print(r0bcc/s, U(r0bcc, eps, s, a12bcc, a6bcc) / eps)
# print(r0hcp/s, U(r0hcp, eps, s, a12hcp, a6hcp) / eps)
# print(r0fcc/s, U(r0fcc, eps, s, a12fcc, a6fcc) / eps)

# A6 = 0
# A12 = 0
# for i in range(1, 100, 1):
#     A6 += 4*i**(-2) + 2*i**(-3)
#     A12 += 4*i**(-2) + 2*i**(-6)
    
# print(A6, A12)

# x = np.linspace(0.9, 2.5, 300) * s
# fplot([U], x, label='U(r)', args=[(eps, s, a6bcc, a12bcc)])

# a = 1    
# x = np.linspace(-3, 3, 200)
# fplot([f, f], x, args=[[a], [-a]])