"""
Interactive Fractal Visualization using PyGame and C++

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


class FractalSurface:
    """
    Subsurface containing a fractal.
    """

    def __init__(self,
                 get_fractal,
                 origin=(0, 0),
                 res=(1280, 1440),
                 extent=(-2, 1, -1.2, 1.2),
                 n_iter=100,
                 z_0=0+0j,
                 cmap="viridis"):
        """
        """
        xsize = (extent[3] - extent[2]) * res[0] / res[1]
        xcenter = (extent[3] + extent[2]) / 2
        self.extent = (xcenter - xsize / 2, xcenter + xsize / 2, extent[2], extent[3])

        self.origin = origin
        self.z_0 = z_0
        self.zoom_factor = 1.2

        self.get_fractal = get_fractal
        self.cmap = getattr(plt.cm, cmap)
        self.res = res
        self.width, self.height = res     # screen resolution (width by heigth)
        self.n_iter = n_iter              # initial number of iterations
        self.n_iter_change = 10           # initial change for n_iter
        self.ctr_arr = get_fractal(res, self.extent, self.z_0, n_iter)
        self.cvals = np.zeros((n_iter, 3), dtype=int)
        self.rgb = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        self.last_extent = self.extent
        self.last_z_0 = self.z_0
        self.last_n_iter = self.n_iter
        self.last_n_iter_change = self.n_iter_change

        self.surf_rect = pygame.Rect(*origin, self.width, self.height)
        self.surface = pygame.Surface((self.surf_rect.width, self.surf_rect.height))


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
            # t_end = perf_counter()
            # print(f"New colormap for {self.n_iter} iterations computed "
            #       f"in {1000 * (t_end - t_start):.2f} ms.")
            # t_start = perf_counter()
        self.rgb = fractal_cpp.ctr2rgb(self.ctr_arr, self.cvals)
        self.rgb.dtype = "4uint8"
        t_end = perf_counter()
        print(f"Colormapping completed in {1000 * (t_end - t_start):.2f} ms.")


    def draw(self):
        """
        https://stackoverflow.com/questions/52389624/pygame-display-2d-numpy-array
        Visualize the current state.
        """
        self.ctr2rgb()
        pygame.surfarray.blit_array(self.surface, self.rgb[:, ::-1, :3])


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


    def relative_pos(self, pos):
        """Relative position in the subsurface of a given pixel"""
        xratio = (pos[0] - self.origin[0]) / self.width
        yratio = (pos[1] - self.origin[1]) / self.height
        return xratio, yratio


    def pos2coord(self, pos):
        """Convert a pixel position (origin is top-left) to coordinates"""
        xratio, yratio = self.relative_pos(pos)
        xval = self.extent[0] + (self.extent[1] - self.extent[0]) * xratio
        yval = self.extent[3] + (self.extent[2] - self.extent[3]) * yratio
        return (xval, yval)


    def contains(self, pos):
        """Check wether a given pixel is inside this surface"""
        return ((self.origin[0] <= pos[0] < self.origin[0] + self.width)
                and (self.origin[1] <= pos[1] < self.origin[1] + self.height))

    def zoom(self, zoom_exp, pos):
        """
        Update the extent and visualization with the current positions
        """
        xpos, ypos = self.pos2coord(pos)                   # current mouse coordinates
        xsize = self.extent[1] - self.extent[0]            # window width in coordinates
        ysize = self.extent[3] - self.extent[2]            # window height in coordinates
        new_xsize = xsize / self.zoom_factor**zoom_exp     # new width after zoom
        new_ysize = ysize / self.zoom_factor**zoom_exp     # new height after zoom

        # Implement zoom such that the mosue position pixel stays unchanged
        xratio, yratio = self.relative_pos(pos)
        yratio = 1 - yratio                         # y-axis is inverted!
        xmin = xpos - xratio * new_xsize            # new xmin in coordinates
        xmax = xpos + (1 - xratio) * new_xsize
        ymin = ypos - yratio * new_ysize            # new ymin in coordinates
        ymax = ypos + (1 - yratio) * new_ysize
        self.extent = (xmin, xmax, ymin, ymax)      # update the extent
        # print(f"{self.extent = }, {xratio = }, {yratio = }, {xpos = }, {ypos = }")


    def keypress(self, key):
        """
        Change the current state given a certain key
        """
        update = False
        if key == "p":
            self.n_iter_change *= 10
        elif key == "m":
            self.n_iter_change //= 10
            self.n_iter_change = max(self.n_iter_change, 1)
        elif key == "+":
            self.n_iter += self.n_iter_change
            update = True
        elif key == "-":
            self.n_iter -= self.n_iter_change
            self.n_iter = max(self.n_iter, 1)
            update = True

        elif key == "r":
            self.extent = self.last_extent
            self.z_0 = self.last_z_0
            self.n_iter = self.last_n_iter
            self.n_iter_change = self.last_n_iter_change
            update = True

        elif key == "l":
            print(f"Extent: {self.extent}, z_0 = {self.z_0}, n_iter = {self.n_iter}")

        print(f"State: {self.n_iter = }, change = {self.n_iter_change}")
        return update


    def leftpress(self, pos):
        """"""
        self.z_0 = complex(*self.pos2coord(pos))


class Fractal:
    """
    Class handling the PyGame-visualization of an interactive fractal.
    """

    def __init__(self, res=(2560, 1440)):
        """
        """
        self.pos = (res[1] // 2, res[0] // 2)
        self.mode = "default"
        self.surfaces = []

        # Initialize the window with the 'pygame' library
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(res, pygame.NOFRAME)
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        os.environ["SDL_VIDEO_CENTERED"] = "1"



    def add_surface(self, surface):
        """Add a surface"""
        self.surfaces.append(surface)


    def start(self):
        """
        Start the game.
        """
        self.draw_all()
        self.decision()


    def draw_all(self):
        """"""
        for surface in self.surfaces:
            self.draw(surface)


    def draw(self, surface):
        """
        https://stackoverflow.com/questions/52389624/pygame-display-2d-numpy-array
        Visualize a given surface (should be one of 'self.surfaces').
        """
        t_start = perf_counter()
        surface.draw()
        self.screen.blit(surface.surface, surface.surf_rect.topleft)
        pygame.display.update()
        t_end = perf_counter()
        print(f"Display update done in {1000 * (t_end - t_start):.2f} ms.")


    def update(self, surface):
        """
        Update a given surface using the current state parameters
        """
        surface.update()
        self.draw(surface)


    def get_surface_from_pos(self, pos):
        """Return the subsurface containg the pixel 'pos'"""
        for surface in self.surfaces:
            if surface.contains(pos):
                return surface
        return None


    def keypress(self, key):
        """
        Change the current state given a certain key
        """
        surface = self.get_surface_from_pos(self.pos)
        update = surface.keypress(key)
        if update:
            self.update(surface)


    def zoom(self, surface, zoom_exp):
        """
        Update the extent and visualization with the current positions
        """
        surface.zoom(zoom_exp, self.pos)
        self.update(surface)


    def decision(self):
        """
        Allows for switching between different modes via button pressing:
            'Q': Quit.
        """
        while True:
            pressed = pygame.mouse.get_pressed(3)
            self.pos = pygame.mouse.get_pos()
            surface = self.get_surface_from_pos(self.pos)
            for e in pygame.event.get():
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
                    print(f"Mode: {self.mode}")
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self.mode = "idle"     # abort zooming / anything else

                if len(self.surfaces) == 2:     # Mandelbrot/Julia combination
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_LCTRL:
                        if self.mode == "mandelbrot/julia":
                            self.mode = "default"
                        else:
                            self.mode = "mandelbrot/julia"

                # if e.type == pygame.MOUSEBUTTONDOWN:
                if pressed[0]:
                    if self.mode == "default":
                        surface.leftpress(self.pos)
                        self.update(surface)
                    elif self.mode == "mandelbrot/julia":
                        if self.surfaces[0].contains(self.pos):
                            z_0_temp = self.surfaces[0].z_0
                            self.surfaces[0].leftpress(self.pos)
                            self.surfaces[1].z_0 = self.surfaces[0].z_0
                            self.surfaces[0].z_0 = z_0_temp
                            self.update(self.surfaces[1])

                if e.type == pygame.MOUSEBUTTONDOWN:
                    if pressed[2]:
                        print(self.pos, surface.pos2coord(self.pos))

                if e.type == pygame.MOUSEWHEEL:
                    self.zoom(surface, e.y)
                    self.update(surface)

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
        Extent: (-2.0, 2.0, -2, 2),
            z_0 = (-0.753125-0.046875j), n_iter = 410
        Extent: (-1.150607638888889, 1.1642071759259263, -1.1593171296296299, 1.1554976851851855),
            z_0 = (0.3563912605138307+0.3465213073391714j), n_iter = 400
        Extent: (-2.0, 2.0, -2, 2), (nipy_spectral_r)
            z_0 = (-0.7634779998434044+0.09069884163812655j), n_iter = 350
    """
    print(__doc__)
    n_iter = 100          # number of iterations
    res = (2560, 1280)    # resolution of the created window in pixels
    # res = (1280, 720)
    extent = (-2, 2, -2, 2)   # (-2, 1, -1.2, 1.2)
    # cmap = "nipy_spectral_r"
    # cmap = "viridis"
    cmap = "gist_ncar_r"
    cmap = "CMRmap_r"
    cmap = "CMRmap"

    # fractal = Fractal(get_mandelbrot, res, extent, n_iter=n_iter)
    # fractal = Fractal(get_julia, res, extent, n_iter=n_iter)
    sub_res = (res[0] // 2, res[1])
    fractal = Fractal(res)
    # fractal.add_surface(FractalSurface(get_julia, (0, 0), res, extent, n_iter))
    fractal.add_surface(FractalSurface(get_mandelbrot, (0, 0), 
                                       sub_res, extent, n_iter, cmap=cmap))
    fractal.add_surface(FractalSurface(get_julia, (sub_res[0], 0), 
                                       sub_res, extent, n_iter, cmap=cmap))
    fractal.start()


if __name__ == "__main__":
    main()
