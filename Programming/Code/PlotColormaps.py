# -*- coding: utf-8 -*-
"""
Display all colormaps contained in matplotlib.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 50

def plot_colorMaps():
    cmaps = plt.colormaps()

    for i in range(0, len(cmaps), 30):
        fig, ax = plt.subplots(10, 3, figsize=(15,10))
        ax = ax.flatten()
        for j in range(30):
            try:
                cmap = plt.get_cmap(cmaps[j+i])
                mpl.colorbar.ColorbarBase(ax[j], cmap=cmap, 
                                          orientation='horizontal')
                ax[j].set_title(cmaps[j+i])
            except IndexError:
                break
    plt.show()

if __name__ == "__main__":
    plot_colorMaps()