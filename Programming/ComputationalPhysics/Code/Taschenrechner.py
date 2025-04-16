
from scipy.integrate import quad as integrate 
from scipy.special import factorial
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from numba import njit, float64, int32
import numpy as np
import time
import timeit
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["figure.dpi"] = 50

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


# @njit('float64[:, :](int32[:])', cache=True)
def looper(T):
    _wgths = np.zeros((len(T), max(T)))
    for i in range(len(T)):
        t = np.arange(1, T[i], 1, dtype=np.float64) / T[i]
        _gtT = np.exp( -(t * (1 - t))**(-1))
        _sum = np.sum(_gtT)
        _wgths[i, 0:T[i]-1] = _gtT / _sum
    return _wgths

T = (10**np.linspace(1, 5, 11, dtype=np.float64)).astype(np.int32)
def yielder(T):
    for i in range(len(T)):
        t = np.arange(1, T[i], 1, dtype=np.float64) / T[i]
        _gtT = np.exp( -(t * (1 - t))**(-1), dtype=np.float64)
        _sum = np.sum(_gtT, dtype=np.float64)
        yield _gtT / _sum
        
def wrapper(T):
    return np.fromiter(yielder(T), np.float64, len(T))

def f(a, *args):
    return a, args


# @njit('f8[:](f8, f8, f8)')
# def arange(nmin, nmax, nstep):
#     res_length = int((nmax - nmin) // nstep)
#     res = np.zeros(res_length, dtype=np.float64)
    
#     counter = nmin
#     for i in range(res_length):
#         res[i] = counter
#         counter += nstep
#     return res

# def zipit(n=1000):
#     return sum(zip(range(1, n + 1, 1), range(-1, -1-n, -1)), (0,))

# def npit(n=1000):
#     ans = np.zeros(2*n + 1, dtype=int)
#     x1 = np.arange(1, n+1, 1, dtype=int)
#     ans[1:] = np.dstack((x1, -x1)).flatten()
#     return ans

# def npit3(n=1000):
#     ans = np.zeros(2*n + 1, dtype=int)
#     x1 = np.arange(1, n+1, 1, dtype=int)
#     ans[1::2] = x1
#     ans[2::2] = -x1
#     return ans

# def npit2(n=1000):
#     x1 = np.arange(1, n+1, 1, dtype=int)
#     return np.dstack((x1, -x1)).flatten()
# # %timeit -n 100 -r 20 npit3(10000)
# x1 = np.arange(1, 10, 1, dtype=int)
# # print(npit3(10)[:3])
# def cross(a, b):
#     return [a[i % 3] * b[(i+1) % 3] - a[(i+1) % 3] * b[i % 3] 
#             for i in [1, 2, 3]]

# a1 = 0.5*np.array([-1, 1, 1])
# a2 = 0.5*np.array([1, -1, 1])
# a3 = 0.5*np.array([1, 1, -1])
# b1 = cross(a2, a3) / np.dot(a1, cross(a2, a3))
# print(b1)

# def epsilon(kx, ky):
#     return 2 - np.cos(kx * np.pi) - np.cos(ky * np.pi)

# kx, ky = np.linspace(0, 1, 200), np.linspace(0, 1, 200)
# plt.plot(kx, epsilon(kx, 0))
# plt.plot(kx + 1, epsilon(1, ky))
# plt.plot(kx * np.sqrt(2) + 2, epsilon(1, 1) - epsilon(kx, ky))
# x = np.outer(np.ones(7), np.arange(-3, 4, 1))
# y = np.outer(np.arange(-3, 4, 1), np.ones(7))
# t = np.linspace(0, 2*np.pi, 200)
# r = 3.5 / 1.5
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.set_aspect(1.0)
# ax.plot(x, y, ls='', marker='o', ms=5, mew=1, c='k')
# ax.plot([0, -1], [0, 1], lw=1.5, c='b')
# ax.plot(r*np.sin(t), r*np.cos(t), c='r')
# ax.plot(r*np.sin(t) - 1, r*np.cos(t) + 1, c='r')
# plt.show()

# def parabel(k, x, y, z, E):
#     return ((k - x)**2 + y**2 + z**2) * E

# k = np.linspace(0, 1, 200)
# a = 3.62 * 1e-10
# a = 4.05 * 1e-10
# E = 8*np.pi**2 * hbar**2 / (me * a**2 * e)
# plt.plot(k, parabel(k, 0, 0, 0, E))
# plt.axhline(11.6)
# plt.plot(k, parabel(k, 1, 0, 0, E))
# plt.plot(k, parabel(k, 0.5, 0.5, 0.5, E))

# PhoneNumber = "3662277"
# words = ["foo", "bar", "baz", "foobar", "emo", "cap", "car", "cat"]

# dct_num = {'a' : 2, 'b' : 2, 'c' : 2, 'd' : 3, 'e' : 3, 'f' : 3, 'g' : 4, 
#             'h' : 4, 'i' : 4, 'j' : 5, 'k' : 5, 'l' : 5, 'm' : 6, 'n' : 6, 
#             'o' : 6, 'p' : 7, 'q' : 7, 'r' : 7, 's' : 7, 't' : 8, 'u' : 8, 
#             'v' : 8, 'w' : 9, 'x' : 9, 'y' : 9, 'z' : 9}

# def word_to_int(word, dct=dct_num):
#     word_int = "".join([str(dct_num[ltr]) for ltr in word])
#     return word_int

# def word_in_number(number, word_lst, dct):
#     ans_lst = []
#     for word in word_lst:
#         if word_to_int(word, dct=dct) in number:
#             ans_lst.append(word)
            
#     return ans_lst

# print(word_in_number(PhoneNumber, words, dct_num))



# n = 10**5 + 1
# t1 = time.time()
# a = [i for i in range(1, n, 1)]
# a[2:n:3] = ['Fizz'] * int(n/3)
# a[4:n:5] = ['Buzz'] * int(n/5)
# a[14:n:15] = ['FizzBuzz'] * int(n/15)
# t2 = time.time()
# # print(a)
# a = [[[x,"Buzz"],["Fizz","FizzBuz"]][x%3==0][x%5==0] 
#      for x in range(1,101)]

# words = ["Fizz", "Buzz"]
# keys = [3, 5]
# iterations = 100
# numbers = range(1, iterations + 1, 1)
# t1 = time.time()
# rule = lambda word, num, key: word if num % key == 0 else ""

# msgs = ["".join([rule(word, num, key) for [word, key] in zip(words, keys)]) 
#         for num in numbers]
# a = [msg if msg != "" else num for [num, msg] in zip(numbers, msgs)]
# t2 = time.time()
# print(t2 - t1)
# print(a)

# for num in range(1, iterations, 1):
#     msg = "".join([rule(word, num, key) 
#                    for [word, key] in zip(words, keys)])
#     print(msg if msg != "" else num)
# a = []
# for i in range(1, n, 1):
#     if i % 3 + i % 5 == 0:
#         a.append('FizzBuzz')
#     elif i % 3 == 0:
#         a.append('Fizz')
#     elif i % 5 == 0:
#         a.append('Buzz')
#     else:
#         a.append(i)
# t3 = time.time()
# print(t2-t1, t3-t2)
# # print(a)

# Taschenrechner
# Msol = 2 * 1e30
# ly = 3600 * 24 * 365.25 * c
# NGalaxy = np.array([1, 100, 1500, 30000, 5*1e13]) * 1e9
# MGalaxy = NGalaxy * Msol
# RSy = 2*G*MGalaxy / (c**2 * ly)
# print(RSy, ly / 1e15)

# def IRF(x, a=1, b=0, c=0):
#     return c + (1-c) / (1 + np.exp(a * (b - x)))

# def logistic(t, t0, p0, p1, a):
#     return p0 + (p1 - p0) / (1 + np.exp(a * (t0 - t)))


# def ping_logistic(t, tval, pval, aval):
#     tbreak = (tval[:-1] + tval[1:]) / 2
#     if len(tbreak) == 1:
#         return logistic(t, tval, *pval, aval)
#     else:
#         ans = np.zeros_like(t)
#         ans[t <= tbreak[0]] = logistic(t[t <= tbreak[0]], tval[0], 
#                                        pval[0], pval[1], aval[0])
#         tbreak[-1] = max(t)
#         for i in range(1, len(tbreak), 1):
#             ind = ((t <= tbreak[i]) & (t > tbreak[i-1]))
#             ans[ind] = logistic(t[ind], tval[i], pval[i], 
#                                 pval[i+1], aval[1])
#     return ans

# t = np.linspace(0, 120, 500)
# tval = np.array([50, 65, max(t)])
# pval = [10, 1000, 10]
# aval = [1, 1]
# plt.plot(t, ping_logistic(t, tval, pval, aval))

# a = [1, 5]
# b = [-0.5, 2]
# c = [0.4, 0.2]
# xmin, xmax = -5, 5
# fig, axes = plt.subplots(1, 2, figsize=(18, 10))
# x = np.linspace(xmin, xmax, 300)
# for [ax, a, b, c] in zip(axes, a, b, c):
#     ax.plot(x, IRF(x, a, b, c), c='b', lw=1.5, 
#             label=r'Chance als Funktion von Fähigkeit $\theta$')
#     ax.axvline(b, c='r', ls='dashed', lw=1.5, 
#                label=f'Item Schwierigkeit = {b}')
#     ax.axhline(c, c='purple', ls='dashed', lw=1.5, 
#                label=f'Minimale Lösungswahrscheinlichkeit = {c}')
#     ax.plot(x, a * (1 - c) * (x - b) / 4 + (1+c)/2, c='g', 
#             ls='dashed', lw=1.5, label=f'Item Diskrimination = {a}')
# for ax in axes:
#     ax.axis([xmin, xmax, -0.1, 1.1])
#     ax.axvline(0, c='k', lw=1)
#     ax.axhline(0, c='k', lw=1)
#     ax.axhline(1, c='k', lw=1)
#     ax.set_xlabel(r'Fähigkeit $\theta$', fontsize=18)
#     ax.set_ylabel(r'Lösungswahrscheinlichkeit', fontsize=18)
#     ax.grid(True)
#     ax.legend(fontsize=15)
#     ax.tick_params(labelsize=15)
# plt.tight_layout()
# # plt.get_current_fig_manager().window.showMaximized()
# # plt.savefig("ItemResponseFunction.jpg", dpi=300)
# plt.show()

# ans = alpha / me * hbar/c
# E = 59.54 * 1e3
# ans = E / (1 + E / (me*c**2 / e) * (1 - np.cos(np.pi/3)))
# aval = np.array([3.62, 4.05]) * 1e-10
# nval = np.array([1, 3]) * 4
# for i in range(len(aval)):
#     kval = (3*np.pi**2*nval[i])**(1/3) / aval[i]
#     Eval = (hbar * kval)**2 / (2*me)
#     print(round(kval*1e-9, 3), "&", round(Eval / e, 3), "&", 
#             round(Eval / k * 1e-3, 3), "&", round(hbar*kval/me *1e-6, 3),
#           "\\\ \hline")
#     print(2*np.pi/aval[i] * np.sqrt(3)/2 * 1e-9)

# mW, mZ = 80.4, 91.2
# alpha = 1/137
# theta = np.arccos(mW/mZ) * 180/np.pi
# gprime = np.sqrt(4*np.pi*alpha) * mZ/mW
# vw = 2/gprime * np.sqrt(mZ**2 - mW**2)
# gw = 2*mW / vw
# print(theta, gprime, vw, gw)

# L = 35314
# n = 512
# a = 0.5

# m = int(L / ((1-a) * n)) + 1

# array = np.arange(0, L, 1)
# array = np.append(array, np.zeros(n))
# matrix = np.zeros((m, n))

# for k in range(0, m, 1):
#     j = int(k * n * (1-a))
#     matrix[k, :] = array[j:n + j]

# print(np.max(matrix), np.max(array))

# yk = sum n=1 N x(n) cos(pi/N (n-0.5) (k-1))

# def discreteCosineTransform(x):
#     N = len(x)
#     y = np.zeros(N)
#     k = np.arange(0, N, 1, dtype=int)
    
#     for n in range(0, N, 1):
#         y += x[n] * np.cos(np.pi/N * (n+0.5) * k)
#     return y[0:N]

# x1 = np.random.uniform(0, 1, 20)
# x2 = np.linspace(0, 1, 20)
# x3 = np.ones(20)

# y1 = discreteCosineTransform(x1)
# y2 = discreteCosineTransform(x2)
# y3 = discreteCosineTransform(x3)
# x, y = np.arange(1, 21, 1), [y1, y2, y3]

# fig, axis = plt.subplots(1, 3, figsize=(15, 10))
# for i in range(0, 3, 1):
#     axis[i].plot(x, y[i], ls='', marker='o', mew=2, ms=2)
# plt.show()

# t_frame = 32
# n = 256

# X = np.ones((137, n))
# freq = 1000/t_frame * np.arange(0, n, 1)
# fc_Mel = np.linspace(283, 2072, 17)
# fc_Hz = 700 * (10**(fc_Mel / 2595) - 1)
# h = np.zeros(len(fc_Hz)-2)


# for k in range(0, len(fc_Hz)-2):
#     h[k]= 2/(fc_Hz[k+2]-fc_Hz[k])
    
# def melFilterBank(X, f_Hz, fc_Mel, h):
#     fc_Hz = 700 * (10**(fc_Mel / 2595) - 1)
#     X_filt = np.zeros(len(h))
#     for i in range(0, len(fc_Hz)-2, 1):
#         i_links = (f_Hz < fc_Hz[i+1]) & (f_Hz > fc_Hz[i])
#         f_Hz_links = f_Hz[i_links]
        
#         g_links = (h[i] * (f_Hz_links - fc_Hz[i]) / 
#                    (fc_Hz[i+1] - fc_Hz[i]))
#         X_filt[i] += sum([*map(lambda x,y: x*y, g_links, 
#                                X[1, i_links])])
        
#         i_rechts = (f_Hz < fc_Hz[i+2]) & (f_Hz > fc_Hz[i+1])
#         f_Hz_rechts = f_Hz[i_rechts]
#         g_rechts = (h[i] - h[i] * (f_Hz_rechts - fc_Hz[i+1]) / 
#                     (fc_Hz[i+2] - fc_Hz[i+1]))
#         X_filt[i] += sum([*map(lambda x,y: x*y, g_rechts, 
#                                X[1, i_rechts])])
        
#     return X_filt

# print(melFilterBank(X, freq, fc_Mel, h))

# for i in range(1, 15, 1):
#     print(fc_Hz[i], fc_Hz[i-1])

# def implicit(f, x, args):
#     """Solve f(x, a) == 0 for an array of x-values."""
#     return [fsolve(f, x[i], [args[0][i], *args[1]]) for i in range(len(x))]
# def func(x, args):
#     [beta, J] = args
#     return x - np.tanh(beta * J * x)
# def ifunc(beta, J):
#     args = [J]
#     x0 = np.ones_like(beta)
#     return implicit(func, x0, [beta, args])

# def free_energy(beta, J):
#     deter = 1 - 4*np.tanh(beta * J)
#     deter[deter < 0] = 0
#     return -np.log(1 + np.sqrt(deter)) / beta

# # beta = np.linspace(1, 2, 501)
# beta = np.linspace(0.1, 1, 501)
# T = 1/beta
# J = 1

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# # x = ifunc(beta, J)
# # ax.plot(beta, x, lw=1, c='b')
# # ax.plot(beta, np.tanh(beta * J), lw=1, c='r')
# # ax.plot(T, x, lw=1, c='b')
# # ax.plot(T, T*np.sqrt(3*(1/T - 1)), lw=1, c='r')
# # ax.plot(T, (3 - 3*T)**0.5, lw=1, c='g')

# ax.plot(T, free_energy(beta, J), lw=1, c='b')

# # x = np.linspace(0, 1, 201)
# # ax.plot(x, x, lw=1, c='k')
# # beta = np.arange(1.1, 3, 0.2)
# # for i, b in enumerate(beta):
# #     ax.plot(x, np.tanh(x * J * b), lw=1, 
# #             c=[0, i/len(beta), 1 - i/len(beta), 1])

# ax.grid(True)
# plt.show()


# import scipy.fft as fft

# def filter(t, y, f_limit, high_pass=True, low_pass=False):    
#     """t sind Zeiten, y(t) zugehoerige Amplitudenwerte.
#        benoetigt import scipy.fft as fft"""     
#     N = len(y)          # length of discrete signal
#     T = t[1] - t[0]     # time resolution assuming equal spacing of 't'
    
#     yf = fft.fft(y)[:N//2]    # first half of coefficients 
#     xf = fft.fftfreq(N, T)[:N//2]    # corresponding frequencies to 'yf'
    
#     yf_abs = 2/N * np.abs(yf)        # normalized coefficient 

#     if high_pass == True:
#         yf_abs[xf < f_limit] = 0     # remove coefficients below 'f_limit'
#     elif low_pass == True:
#         yf_abs[xf > f_limit] = 0     # remove coefficients above 'f_limit'
#     else:
#         print("Atleast one filter mode must be True, returns 'y'.")
#         return y
    
#     # 'manual' inverse fft, sum over amplitudes 'yf_abs' and freqeuncies
#     y_filt = np.sum([yf_abs[i] * np.sin(xf[i] * 2*np.pi*x) 
#                      for i in range(len(xf))], axis=0)
#     return y_filt

# N = 600             # length of discrete signal

# T = 1 / 800         # signal is in [0, N*T]-time array

# x = np.linspace(0, N*T, N, endpoint=False)

# # example signal with frequencies 26, 12, 24 Hz and amplitudes 1, 0.5, 0.333
# y = np.sum([1/(i + 1) * np.sin(f * 2*np.pi*x) 
#             for [i, f] in enumerate([36, 12, 24])], axis=0)


# yf = fft.fft(y)[:N//2]    # first half of coefficients (rest is redundant)
# xf = fft.fftfreq(N, T)[:N//2]    # corresponding frequencies to 'yf'
# yf_abs = 2/N * np.abs(yf)        # normalized coefficient absolute values

# f_limit = 30                     # maximum allowed frequency for high pass
# indizes = (xf < f_limit)         # indizes of freq-array where 'f < f_limit'


# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# ax[0].plot(xf, yf_abs)          # original signal in frequency domain

# yf_abs[indizes] = 0             # remove coefficients below 'f_limit'

# ax[0].plot(xf, yf_abs)          # manipulated signal in frequency domain

# # 'manual' inverse fft, sum over amplitudes 'yf_abs' and freqeuncies 'xf'
# y_filt = np.sum([yf_abs[i] * np.sin(xf[i] * 2*np.pi*x) 
#                  for i in range(len(xf))], axis=0)

# ax[1].plot(x, y)                            # original signal in time domain
# ax[1].plot(x, high_pass(x, y, 35))
# # ax[1].plot(x, y_filt)                       # result from inverse fft
# ax[1].plot(x, np.sin(36 * 2*np.pi*x))       # expected result (only 36 Hz)

# plt.show()

# # Transfermatrix berechnen
# T = np.zeros((4, 4))
# for i1, a in enumerate([1, -1]):
#     for i2, b in enumerate([1, -1]):
#         for i3, c in enumerate([1, -1]):
#             for i4, d in enumerate([1, -1]):
#                 value = c*a + d*b + 2*c*d
#                 print(f"a={a}, b={b}, c={c}, d={d}", value)
#                 T[2*i1+i3, 2*i2+i4] = value

# T = np.exp(T)
# print(np.linalg.det(T))

# size = 10**6
# Nbins = 100
# x = np.random.uniform(0, 1, size)

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# # ax.hist(x, bins=Nbins, density=True, color='b')
# ax.axhline(1, c='k', lw=1)
# plt.show()
# avg = sum(x) / size
# sigma = np.sqrt(sum((x - avg)**2)) / size
# print(f"average={avg}, theoretical value is {1/2}")
# print(f"deviation is {sigma}, theoretical value is {1/12}")

# phi = np.linspace(-np.pi, np.pi, 300)
# b = np.linspace(0.2, 0.99, 5)
# x = np.linspace(-1, 1, 300)
# def rphi(phi, b, RS=1):
#     return (RS / b) / (b + np.cos(phi) - 0.5 * b * np.cos(phi)**2)
# ab = np.sqrt(1 + 2*b) - 1

# def c_elem(nf):
#     return [[i/nf, 0.3 * i/nf, 1 - i/nf, 1] for i in range(nf)]
# for i, bval in enumerate(b):
#     plt.plot(phi, rphi(phi, bval), c=c_elem(len(b))[i], label=i)
# for i, bval in enumerate(ab):
#     plt.plot(phi, rphi(phi, bval), c=c_elem(len(b))[i], label=i)
# plt.legend()
# plt.ylim(0, 10)
    
# b = np.linspace(0.01, 0.99, 300)
# ab = np.sqrt(1 + 2*b) - 1
# xb = (1 - np.sqrt(1 + 2 * ab**2)) / (ab)
# plt.plot(b, 360/np.pi * np.arccos(xb) - 180)
# plt.plot(b, 360/np.pi * np.arccos(-b) - 180)
# plt.plot(b, 360/np.pi * b)


