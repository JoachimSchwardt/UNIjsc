"""
Visualisation of the magnetic field of a circular loop.
"""

import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

def Hz_integrand(t=0, r=0, z=0, R=2):
    k = np.sqrt(4*r*R/((r+R)**2+z**2))
    return (R+r*np.cos(2*t))/(1-k**2*np.sin(t)**2)**(3/2)

def Hz(r=0, z=0, R=2):
    c = R/np.pi * 1/((r+R)**2+z**2)**(3/2)
    return quad(Hz_integrand, 0, np.pi/2, args=(r,z,R))[0]*c

t = np.linspace(0, np.pi/2, 200)
a = np.linspace(0, 10, 200)

Hz = np.vectorize(Hz)

plt.figure(figsize=(14,10))

#plt.plot(a, Hz(a, 0.5))

plt.plot(t, Hz_integrand(t, 0), c='k')

plt.axhline(0, c='k', lw=0.7)           
plt.axvline(0, c='k', lw=0.7)
plt.grid(True)
plt.legend(bbox_to_anchor=(0.25, 0.95))
plt.show()