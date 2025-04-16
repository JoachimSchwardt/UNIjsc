"""Test how to properly implement blitting, especially with respect to bugged zorder"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_special
from time import perf_counter as pc

def hamilton(x, p):
    return p**2 + x**4 - x**2 + x/7

def get_random_matrix(x, y):
    return np.random.uniform(size=x*y).reshape((x, y))

def click_function(event, ax, contour, image, x_num, p_num):
    mode = event.canvas.toolbar.mode
    if mode == '' and event.inaxes and event.button == 1:
        t1 = pc()
        for i in range(20):
            mat = get_random_matrix(x_num, p_num)
            image.set_data(mat)
            event.canvas.draw()
            event.canvas.flush_events()
        t2 = pc()
        print(f"Completed in {t2 - t1:.4f} seconds.")

def click_function_blit(event, ax, contour, image, x_num, p_num):
    mode = event.canvas.toolbar.mode
    if mode == '' and event.inaxes and event.button == 3:
        t1 = pc()
        fig = plt.gcf()
        fig.canvas.draw()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
        print(bg)
        fig.canvas.blit(fig.bbox)
        for i in range(20):
            fig.canvas.restore_region(bg)
            mat = get_random_matrix(x_num, p_num)
            image.set_data(mat)
            ax.draw_artist(image)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
        t2 = pc()
        print(f"Completed in {t2 - t1:.4f} seconds.")

def main():
    print(__doc__)
    x_max = 2
    p_max = 2

    x_num = 100
    p_num = 100

    x_1d = np.linspace(-x_max, x_max, x_num)
    p_1d = np.linspace(-p_max, p_max, p_num)

    x_2d, p_2d = np.meshgrid(x_1d, p_1d)
    figsize = mpl_special.set_figsize()
    fig, ax = plt.subplots(figsize=(figsize[1], figsize[1]))
    ax.axis([x_1d[0], x_1d[-1], p_1d[0], p_1d[-1]])
    contour = ax.contour(x_2d, p_2d, hamilton(x_2d, p_2d), zorder=100)
    mat = get_random_matrix(x_num, p_num)
    image = ax.imshow(mat, cmap="Grays", extent=ax.axis())

    fig.canvas.mpl_connect("button_press_event", lambda event: click_function(
        event, ax, contour, image, x_num, p_num)
    )

    fig.canvas.mpl_connect("button_press_event", lambda event: click_function_blit(
        event, ax, contour, image, x_num, p_num)
    )

    return 0


if __name__ == "__main__":
    main()
