# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:17:57 2019

@author: Joachim
"""
import sympy as sy
from sympy import integrate as itg
import numpy as np
from matplotlib import pyplot as plt

x, y, z, p = sy.symbols('x, y, z, p')
r, h, a = sy.symbols('r, h, a', positive=True)

#x_s=itg(itg(2*x, (y, 0, sy.sqrt(r**2-x**2))), (x, -r, -z))
#print(sy.simplify(x_s))
#x_s=itg(-4/3*(r**4*sy.cos(p)**4), (p, 0, sy.asin(h/(2*r))))
#print(x_s)

#z_s=itg(itg(2*z, (y, 0, sy.sqrt(r**2-x**2))), (x, -r, -z))
#print(sy.simplify(z_s))
#z_s=itg(2*z*(-r**2*sy.asin(z/r) - z*sy.sqrt(r**2 - z**2)), (z, 0, h/2))
#print(z_s)

#sy.pprint(sy.simplify(-((12*sy.asin(h/(2*r))+sy.sin(4*sy.asin(h/(2*r)))+8*sy.sin(2*sy.asin(h/(2*r))))*r**4)/24))
expr=h**3*sy.sqrt(-h**2 + 4*r**2)/48 - 5*h*r**2*sy.sqrt(-h**2 + 4*r**2)/24 - r**4*sy.asin(h/(2*r))/2+\
1/16*(8*sy.asin(h/(2*r))*r**4+sy.sqrt(4*r**2-h**2)*(2*h*r**2+h**3)+8*h**2*sy.asin(h/(2*r))*r**2)
pl=sy.simplify(expr.subs(h, a*r))
print(pl)
pl = 1/12 *a**2*sy.sqrt(-a**2 + 4) + 0.5*a*sy.asin(a/2) - 1/12*sy.sqrt(-a**2 + 4)




#expr = -(r**3*((2**(9/2)-64)*r**2+(15*2**(3/2)*sy.asinh(1)-5*2**(5/2)+140)\
#       *r+5*sy.sqrt(2)*sy.ln(sy.sqrt(2)+1)-5*sy.sqrt(2)*sy.ln(sy.sqrt(2)-1)\
#       -25*2**(3/2)*sy.asinh(1)-80))/(15*2**(3/2))

#print(expr)
#
#print(expr.subs(r, 1.5).evalf(50))

t = np.arange(0, 2, 0.01)
func_t = []
for elem in t:
    func_t.append(pl.subs(a, elem).evalf())

#print(func_t)

fig = plt.figure(figsize=(14, 14))
plt.xlim(min(t), max(t))
#plt.ylim(-max(t), max(t))
plt.plot(t, func_t)
plt.grid(True)
plt.show()