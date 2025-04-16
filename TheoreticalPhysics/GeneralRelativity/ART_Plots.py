"""
ART Plots
"""
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

# def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
#     def gcd(a, b):
#         while b:
#             a, b = b, a%b
#         return a
#     def _multiple_formatter(x, pos):
#         den = denominator
#         num = np.int(np.rint(den*x/number))
#         com = gcd(num,den)
#         (num,den) = (int(num/com),int(den/com))
#         if den==1:
#             if num==0:
#                 return r'$0$'
#             if num==1:
#                 return r'$%s$'%latex
#             elif num==-1:
#                 return r'$-%s$'%latex
#             else:
#                 return r'$%s%s$'%(num,latex)
#         else:
#             if num==1:
#                 return r'$\frac{%s}{%s}$'%(latex,den)
#             elif num==-1:
#                 return r'$\frac{-%s}{%s}$'%(latex,den)
#             else:
#                 return r'$\frac{%s%s}{%s}$'%(num,latex,den)
#     return _multiple_formatter

# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
# ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

def T_cos2(x, n=1):
    result = 1
    for k in range(1, n+1, 1):
        result += 0.5 * (-1)**(k % 2) * 4**k / factorial(2*k) * x**(2*k)
    return result

def T_sec2(x, n):
    result = 1
    Koeff = [1, 1/2, 1]
    for k in range(1, n+1, 1):
        result += Koeff[k-1] * x**(2*k)
    return result

x = np.linspace(0.44, 2.7, 300)
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.axis([-0.1, 1.1, -5, 0.1])

ax.axhline(0, color='k', lw=1.3)
ax.axvline(0, color='k', lw=1.3)

ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))

nmax = 4
for n in range(3, nmax, 1):
    ax.plot(x/np.pi, -T_sec2(x-np.pi/2, n), c=[n/nmax, 0.2, 1-n/nmax, 1], 
            label='$T_{}[csc^2](x)$'.format(n), lw=1.3)
ax.plot(x/np.pi, -1 / np.sin(x)**2, lw=1.3, c='k', label='$-csc^2(x)$')

ax.grid(True)
ax.tick_params(labelsize=16)
ax.legend(fontsize=22)
plt.show()