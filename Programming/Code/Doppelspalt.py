# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:37:36 2019

@author: Joachim
"""

import scipy as sc
from matplotlib import pyplot as plt

def wave(x, y=0, a=0, t=0, k=1, w=1):
    return sc.cos(2*sc.pi*k*sc.sqrt(x**2+(y-a)**2)-w*t)

def wave_approx(x, y=0, a=0, t=0, k=1, w=1):
    return 2*sc.cos(k*x-w*t+k*(y**2+a**2)/(2*x))*sc.cos(k*a*y/x)
#    try:
#        return 2*sc.cos(k*x-w*t+k*(y**2+a**2)/(2*x))*sc.cos(k*a*y/x)
#    except ValueError:
#        return 2

r = sc.linspace(-80, 80, 501)
d = 80
t = 0

fig = plt.figure(figsize=(10,10))
##plt.plot(r, wave(d, r, 1, t)+wave(d, r, -1, t))
##plt.plot(r, wave_approx(d, r, 1, t))
#plt.contourf(r, r, f(r, r))#, sc.linspace(-1, 1, 10), cmap='viridis')
#plt.grid(True)
#plt.show()

m = sc.linspace(d, d+80, 501)

x, y = sc.meshgrid(m, r)
z = wave(x, y, 1) + wave(x, y, -1)
#z = wave_approx(x, y, 1)
plt.contourf(x, y, z, sc.linspace(-2, 2, 10), cmap='viridis') #plt.imshow() !!!
