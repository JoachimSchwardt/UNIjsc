"""
Interactive Fractal Visualization using PyGame and C++ (v1.1.0)

Use "+" and "-" to increase the number of iterations by a fixed amount.
You may change the rate of change with "p" and "m" (10-fold increase/decrease)
"""

import os
import sys
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import pygame
import fractal_cpp


def get_mandelbrot(shape=(200, 100), extent=(-2, 1, -1.2, 1.2), z_0=0+0j, n_iter=20):
    """
    Current time complexity is T = (0.14 + 2.6e-5 * n_iter) sec
     --> Theoretical runtime for 10000 iterations is 0.26 sec
         (matches C++, roughly 0.28 sec)
    """
    c_shape = fractal_cpp.Shape(shape[0], shape[1])     # shape[1] == width
    c_extent = fractal_cpp.Extent(extent[0], extent[1], extent[2], extent[3])
    ctr_arr = fractal_cpp.get_mandelbrot(c_shape, c_extent, [z_0.real, z_0.imag], n_iter)
    return ctr_arr


def get_julia(shape=(200, 100), extent=(-2, 1, -1.2, 1.2), c_0=0+0j, n_iter=20):
    """
    Current time complexity is T = (0.14 + 2.6e-5 * n_iter) sec
     --> Theoretical runtime for 10000 iterations is 0.26 sec
         (matches C++, roughly 0.28 sec)
    """
    c_shape = fractal_cpp.Shape(shape[0], shape[1])     # shape[1] == width
    c_extent = fractal_cpp.Extent(extent[0], extent[1], extent[2], extent[3])
    ctr_arr = fractal_cpp.get_julia(c_shape, c_extent, [c_0.real, c_0.imag], n_iter)
    return ctr_arr


class Fractal:
    """
    Class handling the PyGame-visualization of an interactive fractal.
    """

    def __init__(self,
                 get_fractal,
                 res=(2560, 1440),
                 extent=(-2, 1, -1.2, 1.2),
                 n_iter=100,
                 cmap="viridis"):
        """
        """
        self.pos = (res[1] // 2, res[0] // 2)
        self.z_0 = 0.0 + 0.0j
        self.mode = "idle"
        self.zoom_factor = 1.2

        self.get_fractal = get_fractal
        self.cmap = getattr(plt.cm, cmap)
        self.res = res
        self.width, self.height = res     # screen resolution (width by heigth)
        self.extent = extent
        self.n_iter = n_iter              # initial number of iterations
        self.n_iter_change = 10           # initial change for n_iter
        self.ctr_arr = get_fractal(res, extent, self.z_0, n_iter)
        self.cvals = np.zeros((n_iter, 3), dtype=int)
        self.rgb = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        # Initialize the window with the 'pygame' library
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(res, pygame.NOFRAME)
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        os.environ["SDL_VIDEO_CENTERED"] = "1"

        self.last_extent = self.extent
        self.last_z_0 = self.z_0


    def ctr2rgb(self):
        """
        Convert the current 'counter-array' to an array of RGB values.
        """
        t_start = perf_counter()
        if len(self.cvals) != self.n_iter+1:
            lin_vals = np.linspace(0.0, 1.0, self.n_iter+1)
            self.cvals = 255 * self.cmap(lin_vals)
            self.cvals = self.cvals.astype(np.uint32)
            self.cvals = np.sum([self.cvals[:, j] * 256**j for j in range(4)],
                                axis=0, dtype=np.uint32)
            t_end = perf_counter()
            print(f"New colormap for {self.n_iter} iterations computed "
                  f"in {1000 * (t_end - t_start):.2f} ms.")
            t_start = perf_counter()
        self.rgb = fractal_cpp.ctr2rgb(self.ctr_arr, self.cvals)
        self.rgb.dtype = "4uint8"
        t_end = perf_counter()
        print(f"Colormapping completed in {1000 * (t_end - t_start):.2f} ms.")


    def start(self):
        """
        Start the game.
        """
        self.draw()
        self.decision()


    def draw(self):
        """
        https://stackoverflow.com/questions/52389624/pygame-display-2d-numpy-array
        Visualize the current state.
        """
        self.ctr2rgb()
        t_start = perf_counter()
        pygame.surfarray.blit_array(self.screen, self.rgb[:, ::-1, :3])
        # pygame.display.update()
        pygame.display.flip()
        t_end = perf_counter()
        print(f"Display update done in {1000 * (t_end - t_start):.2f} ms.")


    def update(self):
        """
        Update the fractal plot using the current state parameters
        """
        t_start = perf_counter()
        self.ctr_arr = self.get_fractal(self.res, self.extent, self.z_0, self.n_iter)
        t_comp = perf_counter()
        size = self.width * self.height
        print(f"Computation of {size} values for {self.n_iter} iterations "
              f"completed in {1000 * (t_comp - t_start):.2f} ms!")
        self.draw()


    def keypress(self, key):
        """
        Change the current state given a certain key
        """
        if key == "p":
            self.n_iter_change *= 10
        elif key == "m":
            self.n_iter_change //= 10
            self.n_iter_change = max(self.n_iter_change, 1)
        elif key == "+":
            self.n_iter += self.n_iter_change
            self.update()
        elif key == "-":
            self.n_iter -= self.n_iter_change
            self.n_iter = max(self.n_iter, 1)
            self.update()

        elif key == "r":
            self.extent = self.last_extent
            self.z_0 = self.last_z_0
            self.update()

        elif key == "l":
            print(f"Extent: {self.extent}, z_0 = {self.z_0}, n_iter = {self.n_iter}")

        print(f"State: {self.n_iter = }, change = {self.n_iter_change}")


    def pos2coord(self, pos):
        """Convert a pixel position (origin is top-left) to coordinates"""
        xval = self.extent[0] + (self.extent[1] - self.extent[0]) * pos[0] / self.width
        yval = self.extent[3] + (self.extent[2] - self.extent[3]) * pos[1] / self.height
        return (xval, yval)


    def zoom(self, zoom_exp):
        """
        Update the extent and visualization with the current positions
        """
        xpos, ypos = self.pos2coord(self.pos)              # current mouse coordinates
        xsize = self.extent[1] - self.extent[0]            # window width in coordinates
        ysize = self.extent[3] - self.extent[2]            # window height in coordinates
        new_xsize = xsize / self.zoom_factor**zoom_exp     # new width after zoom
        new_ysize = ysize / self.zoom_factor**zoom_exp     # new height after zoom

        # Implement zoom such that the mosue position pixel stays unchanged
        xratio = self.pos[0] / self.width           # relative pixel coordinates
        yratio = 1 - self.pos[1] / self.height
        xmin = xpos - xratio * new_xsize            # new xmin in coordinates
        xmax = xpos + (1 - xratio) * new_xsize
        ymin = ypos - yratio * new_ysize            # new ymin in coordinates
        ymax = ypos + (1 - yratio) * new_ysize
        self.extent = (xmin, xmax, ymin, ymax)      # update the extent
        self.update()


    def decision(self):
        """
        Allows for switching between different modes via button pressing:
            'Q': Quit.
            'Space': Start recording a handdrawn track.
            'P': Process the 'self.track'.
            'R': Run the Animation generated by 'self.coef'.
        """
        while True:
            for e in pygame.event.get():
                pressed = pygame.mouse.get_pressed(3)
                if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    self.close()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_PLUS:
                    self.keypress(key="+")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_MINUS:
                    self.keypress(key="-")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_p:
                    self.keypress(key="p")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                    self.keypress(key="m")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    self.keypress(key="r")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_l:
                    self.keypress(key="l")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self.mode = "idle"     # abort zooming / anything else

                # if e.type == pygame.MOUSEBUTTONDOWN:
                if pressed[0]:
                    self.pos = pygame.mouse.get_pos()
                    self.z_0 = complex(*self.pos2coord(self.pos))
                    self.update()

                if e.type == pygame.MOUSEWHEEL:
                    self.pos = pygame.mouse.get_pos()
                    self.zoom(e.y)
            self.clock.tick(30)


    def close(self):
        """
        Closes the display and terminates the script with 'sys.exit'.
        """
        pygame.display.quit()
        pygame.quit()
        sys.exit()


def main():
    """
    Pretty pictures ::
    (JULIA)
        Extent: (-2.1333333333333333, 2.1333333333333333, -1.2, 1.2),
            z_0 = (0.3583333333333334+0.3533333333333333j), n_iter = 210
            z_0 = (-0.8200000000000001+0.19333333333333336j), n_iter = 100
            z_0 = (0.4033333333333333+0.23333333333333328j), n_iter = 100
            z_0 = (-0.75-0.07999999999999985j), n_iter = 360
    """
    print(__doc__)
    n_iter = 100          # number of iterations
    res = (2560, 1440)    # resolution of the created window in pixels
    # res = (1280, 720)
    extent = (-2, 1, -1.2, 1.2)   # (-0.25, 0.25, -0.25, 0.25)
    width = (extent[3] - extent[2]) * res[0] / res[1]
    extent = (-width / 2, width / 2, extent[2], extent[3])

    # fractal = Fractal(get_mandelbrot, res, extent, n_iter=n_iter)
    fractal = Fractal(get_julia, res, extent, n_iter=n_iter)
    fractal.start()


if __name__ == "__main__":
    main()
