# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:37:58 2019

@author: Joachim
"""
import numpy as np
from matplotlib import pyplot as plt

def f1(x, a, R):
    c = -a * R**2 - 1 / (4*a)
    return a*x**2 + c

def f2(x, R):
    return np.sqrt(R**2 - x**2)

def A(a, R):
    return abs(1/(6*a**2) * np.sqrt(a**2*R**2-1/4) * (4*a**4*R**4\
               - 14*a**2*R**2 - 11/4)) - R**2 * np.pi/2
 
def Pee(x, E=1e6,  delta_m=7.37*10**(-5), c=3*1e8, h=4.136*10**(-15), theta=np.arcsin(np.sqrt(0.297))):
    return 1-np.sin(2*theta)**2*np.sin(1000*np.pi*delta_m*x/(2*E*h*c))**2

def Pee_avg(theta=np.arcsin(np.sqrt(0.297))):
    return 1-0.5*np.sin(2*theta)**2

def sin_2theta_m(E=0, const=0.241, theta=np.arcsin(np.sqrt(0.297))):
    #return np.sin(2*theta) / np.sqrt(np.sin(2*theta)**2+(np.cos(2*theta)-(E-E)*const/delta_m)**2)
    return 1/np.sqrt(1+(np.cos(2*theta)/np.sin(2*theta)-E*const)**2)

def theta_m(E=0, const=0.241, theta=np.arcsin(np.sqrt(0.297))):
    return 0.5*np.arcsin(1/np.sqrt(1+(np.cos(2*theta)/np.sin(2*theta)-E*const)**2))

def sin_theta_m(E=0, const=0.241, theta=np.arcsin(np.sqrt(0.297))):
    return 1/np.sqrt(2)*1/np.sqrt(1+(np.cos(2*theta)/np.sin(2*theta)-E*const)**2+\
                     (np.cos(2*theta)/np.sin(2*theta)-E*const)*np.sqrt(1+(np.cos(2*theta)/np.sin(2*theta)-E*const)**2))

def cos_theta_m(E=0, const=0.241, theta=np.arcsin(np.sqrt(0.297))):
    return np.sqrt(1-sin_theta_m(E, const, theta)**2)

def hilfsfkt(E=0, const=0.241, theta=np.arcsin(np.sqrt(0.297))):
    const *= np.sin(2*theta)
    return np.sin(2*theta) * np.sqrt(np.sin(2*theta)**2 + 2*(np.cos(2*theta) - E*const)**2+2*(np.cos(2*theta)-E*const)*np.sqrt((np.cos(2*theta)-E*const)**2 + np.sin(2*theta)**2))/(np.sin(2*theta)**2 + (np.cos(2*theta)-E*const)**2 + (np.cos(2*theta)-E*const)*np.sqrt((np.cos(2*theta)-E*const)**2+np.sin(2*theta)**2))

c_0 = 2.99792458*1e8 #m / s
e = 1.602176565*10**(-19) #J / eV
h_bar = 1.054571726 *10**(-34) #Js
G_F = (h_bar*c_0)**3*4.5437957*1e14 #J**(-2) from https://en.wikipedia.org/wiki/Fermi%27s_interaction
N_e = 6.4*1e31 #m**(-3) from https://physics.stackexchange.com/questions/192338/are-the-electrons-at-the-centre-of-the-sun-degenerate-or-not
theta = np.arcsin(np.sqrt(0.297)) #degrees
delta_m2 = 7.37*10**(-5)*e**2 #J**2

const = 2*np.sqrt(2)*G_F*N_e*1e6*e/(delta_m2*np.sin(2*theta))
#print(const)

#print(np.arcsin(np.sqrt(0.297))*180/np.pi)
#print(np.sin(np.pi/180*76.5))
# print(sin_theta_m(10))
# print(theta_m(0)*180/np.pi)       

# def G_ae(f=1):
#     v0 = 2*1e5
#     f1 = 4# * 2*np.pi
#     f2 = 1.5 * 1e6# * 2*np.pi
#     return v0 / ((1+1j*f/f1)*(1+1j*f/f2)) 

# def G_gs(w=1):
#     v0 = -31.106
#     a = 1.5656 * 1e5
#     b = 1.4755 * 1e12
#     return v0 / (1+1j*w / a - w**2 / b) 

# def G_gs2(w=1):
#     v0 = -31.106
#     a2 = 1.5925 * 1e5
#     b2 = 9.26525 * 1e6
#     return v0 / ((1+1j*w / a2)*(1 + 1j*w / b2)) 

# f = 10**np.linspace(0, 8, 200)

# a = (1.8+56) / (10**(106/20) * 1.8 + 1.8 + 56)*(1/4 + 1/(1.5*1e6))/(2*np.pi)
# b = a / (4+1.5*1e6) * 1 / (2*np.pi)
# print(a, b)
# c = 1/(a/2 -np.sqrt(a**2 / 4 - b))
# d = 1/(a/2 +np.sqrt(a**2 / 4 - b))
# c = (a/(2*b) - np.sqrt(a**2/(4*b**2) - 1/b))
# d = (a/(2*b) + np.sqrt(a**2/(4*b**2) - 1/b))
# x = 5000
# print((1+x/a+x**2/b), (1+x/c)*(1+x/d))
# print(c, d)

t = np.linspace(0.0001, 1, 300)
# m = np.arange(0, 12, 0.01)
fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, xscale='log', yscale='log')
ax = fig.add_subplot(1, 1, 1)
# ax.axis([1e0, 1e7, 1e-2, 1e6])
# ax.axis([0, 1, -0.5, 0])

ax.plot(t, -np.log(t))

# ax.plot(np.log10(f), 20*np.log10(abs(G_ae(f))), c='b', lw=0.7)
# ax.plot(np.log10(f), 20*np.log10(abs(G_gs(f))), c='r', lw=0.7)
# ax.plot(np.log10(f), 20*np.log10(abs(G_gs2(f))), c='g', lw=0.7)
# ax.plot(f, f**(-40) + 100)
# ax.plot(f, abs(G_ae(f)), c='b', lw=0.7)
# ax.plot(f, abs(G_gs(f)), c='r', lw=0.7)
# ax.plot(f, abs(G_gs2(f)), c='g', lw=0.7)
# ax.axvline(4)
# ax.axvline(1e5 * 1.5656)

# plt.xlim(min(m), max(m))
# plt.ylim(0, 1.2)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#plt.xlabel("x / km", fontsize=20)
# plt.xlabel("E / MeV", fontsize=20)

#plt.plot(t, Pee(t), c="blue")
#plt.axhline(Pee_avg(), c='red')

# plt.plot(m, sin_2theta_m(m), c="blue", label="sin 2theta_m")
# plt.plot(m, sin_theta_m(m), c='lime', label="sin theta_m")
# plt.plot(m, cos_theta_m(m), c='firebrick', label="cos theta_m")

#plt.plot(m, 2*sin_theta_m(m)*cos_theta_m(m), c='purple')
#plt.plot(m, hilfsfkt(m), c='black')
#plt.axhline(np.sin(2*np.pi/180 * 76.5), c="red")
#plt.axhline(sin_theta_m(10))

plt.grid(True, which='major', lw=0.5, c='k', alpha=0.4, ls='-')
plt.grid(True, which='minor', lw=0.4, alpha=0.6, ls='--')
plt.legend(fontsize=20)
plt.show()

# print("Flächengröße zwischen den Graphen: {}".format(A(a, R)))
