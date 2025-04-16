# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:54:39 2021

@author: Joachim
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("PING.txt", skiprows=1, usecols=6, dtype=str)
data = [float(string[5:]) for string in data]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.set_ylim(0, 100)
ax.plot(np.arange(len(data)), data, c='b', label='50ms')
ax.axvline(82, c='k', lw=1)
ax.legend(fontsize=18)
plt.show()