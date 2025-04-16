# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:19:58 2020

@author: Joachim
"""

import numpy as np
import matplotlib.pyplot as plt

def cosinesum(a=1, b=1, N=50, alpha=1):
    t = np.linspace(0, alpha*N-1, alpha*N)
    return sum(np.cos(np.pi*t*a / N)*np.cos(np.pi*t*b / N))

# print(cosinesum(3, 5, 100))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect=1.0)

interval = np.linspace(0, 10, 11)
for a in interval:
    ax.plot(a, cosinesum(a, 3, 20, 2), marker='.')


plt.show()

