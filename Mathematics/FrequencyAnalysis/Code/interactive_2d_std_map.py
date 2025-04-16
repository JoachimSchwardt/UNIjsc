#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example signals for the NAFF article
"""

import functools
# import numpy as np
import matplotlib.pyplot as plt
from std_map import std_map
from naff_tools import chaos_indicator
import mpl_special


def mouse_click(event, ax, k, n, col):
    mode = event.canvas.toolbar.mode
    if event.button == 1 and mode == '' and event.inaxes == ax:
        q0 = event.xdata
        p0 = event.ydata
        q_n, p_n = std_map(q0, p0, n, k)
        
        diff = chaos_indicator(q_n, p_n, tol=1e-5, correct_offset=True)
        if diff > 1e-5:
            c = 'k'
            alpha = 0.2
        else:
            c = col.get_color()
            alpha = 0.5
            
        ax.plot(q_n, p_n, alpha=alpha, ls='', marker='o', ms=1, c=c)
        print(f"{q0 = :.4f}, {p0 = :.4f}, {diff = :.2e}, {c != 'k'}")
        event.canvas.draw()


def main():
    print(__doc__)
    k = 0.7
    n = 1024
    
    col = mpl_special.Colors()
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$")
    ax.axis([0, 1, -0.5, 0.5])
    
    mouse_click_partial = functools.partial(mouse_click, ax=ax, k=k, n=n, col=col)
    plt.connect("button_press_event", mouse_click_partial)
    mpl_special.polish(fig, ax)
    return 0


if __name__ == "__main__":
    main()
