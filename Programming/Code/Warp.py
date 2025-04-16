# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:21:54 2018
@author: ron
"""
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from sympy import *
#import sympy.mpmath as mp
#from sympy.mpmath import *
import mpmath as mpmath
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-2, 2, 0.01)
Y = np.arange(-2, 2, 0.01)
 
X, Y = np.meshgrid(X, Y)
 
#neuer Versuch:
#R = 1
#rs =np.sqrt(X**2 + Y**2)
#f = (np.tanh(8*(rs+1))-np.tanh(8*(rs-1)))/(2*np.tanh(8))
#df = (-1)*4*mpmath.coth(8) * ( (mpmath.sech(8-8*rs))**2 - (mpmath.sech(8*(rs+1)))**2  )
#Z = X/rs * df
'''HEUREKA!'''
#R = (np.tanh(8*(np.sqrt(X**2+Y**2)+1))-np.tanh(8*(np.sqrt(X**2+Y**2)-1)))/(2*np.tanh(8)) [Nur die Ableitung davon wird benötigt!]
Z= X/(np.sqrt(X**2+Y**2)) * (-1*( (1/(np.cosh(8*(1-np.sqrt(X**2+Y**2))))**2) - (1/(np.cosh(8*(np.sqrt(X**2+Y**2)+1))**2)))) # [<--- main function]
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.
viridis, linewidth=0, antialiased=False)
ax.set_zlim(-2.01, 2.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
