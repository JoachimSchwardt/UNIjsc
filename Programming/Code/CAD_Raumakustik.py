"""Einfache geometrische Formen mit Python und OOP darstellen."""

import numpy as np
import matplotlib.pyplot as plt

class Line(object):
    def __init__(self, cord, length, phi, c='k', lw=0.7, ls='-'):
        self.cord = cord
        self.length = length
        self.phi = phi
        self.c = c
        self.lw = lw
        self.ls = ls
        
        self.rad = phi * np.pi / 180
        self.x_cord = [cord[0], cord[0] + self.length * np.cos(self.rad)]
        self.y_cord = [cord[1], cord[1] + self.length * np.sin(self.rad)]
        
    def plot_line(self):
        t = np.linspace(0, 1, 200)
        x_plot = self.x_cord[0] + (self.x_cord[1] - self.x_cord[0]) * t
        y_plot = self.y_cord[0] + (self.y_cord[1] - self.y_cord[0]) * t
        plt.plot(x_plot, y_plot, c=self.c, lw=self.lw, ls=self.ls)

class Circle(object):
    def __init__(self, center, r=0.45, deg=[0, 360], c='k', lw=0.7, ls='-'):
        self.center = center
        self.r = r
        self.deg = deg
        self.c = c
        self.lw = lw
        self.ls = ls
        
    def plot_circle(self):
        t = np.linspace(self.deg[0], self.deg[1], 200) * np.pi/180
        x_plot = self.r * np.cos(t) + self.center[0]
        y_plot = self.r * np.sin(t) + self.center[1]
        plt.plot( x_plot, y_plot, c=self.c, lw=self.lw, ls=self.ls)
        
def main():
    print(__doc__)
    
    step_field = 0.95
    min_field = 10
    max_field = 40
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.axis([0, 40, 0, 40])
    
    lw_field = [0.7, 0.4, 0.4, 0.4, 0.4]
    ls_field = ['-', '--', '--', '--', '--']
    for j in range(int((max_field - min_field) / step_field)):
        circ = Circle([20, 0], j + min_field, [45, 135], 
                      lw=lw_field[j % 5], ls=ls_field[j % 5])
        circ.plot_circle()
    
    for phis in [45, 135]:
        line = Line([20, 0], 40, phis)
        line.plot_line()
    
    ax.grid(True)
    plt.show()
    
    
if __name__ == "__main__":
    main()
