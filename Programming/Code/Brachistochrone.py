# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:22:24 2019

@author: Joachim
"""

""" https://www.geogebra.org/m/bHQNJvZC """

import functools
import scipy as sy
from matplotlib import pyplot as plt

def cycloid_x(t, R):
    return R*t-R*sy.sin(t)

def cycloid_y(t, R):
    """ 2R entspricht der Höhe H von Punkt A über Punkt B """
    return -(R-R*sy.cos(t))

def BRC(bx, by, iterationen=2000, R=1):
    
    t = sy.linspace(0, 2*sy.pi, 2000)
    v = sy.linspace(0, bx, 100)
    
    cyc_y = cycloid_y(t, R)
    AB = by/bx*cycloid_x(t, R)
    
    delta_y_min = min(abs(cyc_y[20:]-AB[20:]))
    
    t_min_index = sy.where(delta_y_min==abs(cyc_y-AB))[0][0]
    t_min = t[t_min_index]
    x_min, y_min = cycloid_x(t_min, R), cycloid_y(t_min, R)
    
    t_c = sy.linspace(0, t_min, 200)
    R_c = -by / (1 - sy.cos(t_min))
    
    plt.figure(figsize=(14,10))
    plt.xlim(-R, 13)
    plt.ylim(-9, R)
    
    plt.plot(cycloid_x(t, R), cyc_y, c='k', label='Projection')
    plt.plot(0, 0, marker='x', markersize=10, c='k')
    plt.text(0, 0.25*R, "Punkt A", size=16)
    plt.axhline(0, c='k', lw=0.7)
    plt.axvline(0, c='k', lw=0.7)
    plt.plot(x_min, y_min, marker='x', markersize=10, c='b')
    plt.text(x_min, y_min+0.25*R, "Punkt C", size=16)
    plt.plot(v, by/bx*v, c='b')
    plt.plot(bx, by, marker='x', markersize=10, c='k')
    plt.text(bx, by+0.25*R, "Punkt B", size=16)
    plt.plot(cycloid_x(t_c, R_c), cycloid_y(t_c, R_c), label='Brachistochrone')
    
    plt.grid(True)
    plt.legend()
    plt.show()
    
#BRC(10, -5, 2000, 1)

def BRC_var(bx, by, ax=0, ay=0, iterationen=2000, R=1):
    bx -= ax
    by -= ay
    
    t = sy.linspace(0, 2*sy.pi, 2000)
    v = sy.linspace(0, bx, 100)
    
    cyc_y = cycloid_y(t, R)
    AB = by/bx*cycloid_x(t, R)
    
    delta_y_min = min(abs(cyc_y[20:]-AB[20:]))
    
    t_min_index = sy.where(delta_y_min==abs(cyc_y-AB))[0][0]
    t_min = t[t_min_index]
    x_min, y_min = cycloid_x(t_min, R), cycloid_y(t_min, R)
    
    t_c = sy.linspace(0, t_min, 200)
    R_c = -by / (1 - sy.cos(t_min))
    
    plt.figure(figsize=(14,10))
    plt.xlim(-R+ax, 13+ax)
    plt.ylim(-9+ay, R+ay)
    
    plt.plot(cycloid_x(t, R)+ax, cyc_y+ay, c='k', label='Projection')
    plt.plot(ax, ay, marker='x', markersize=10, c='k')
    plt.text(ax, ay+0.25*R, "Punkt A", size=16)
    plt.axhline(0, c='k', lw=0.7)
    plt.axvline(0, c='k', lw=0.7)
    plt.plot(x_min+ax, y_min+ay, marker='x', markersize=10, c='b')
    plt.text(x_min+ax, y_min+ay+0.25*R, "Punkt C", size=16)
    plt.plot(v+ax, by/bx*v+ay, c='b')
    plt.plot(bx+ax, by+ay, marker='x', markersize=10, c='k')
    plt.text(bx+ax, by+ay+0.25*R, "Punkt B", size=16)
    plt.plot(cycloid_x(t_c, R_c)+ax, cycloid_y(t_c, R_c)+ay, label='Brachistochrone')
    
    plt.grid(True)
    plt.legend()
    plt.show()
    
def BRC_var2(bx, by, ax=0, ay=0, iterationen=2000, R=1):
    bx -= ax
    by -= ay
    
    t = sy.linspace(0, 2*sy.pi, 2000)
    
    cyc_y = cycloid_y(t, R)
    AB = by/bx*cycloid_x(t, R)
    
    delta_y_min = min(abs(cyc_y[20:]-AB[20:]))
    
    t_min_index = sy.where(delta_y_min==abs(cyc_y-AB))[0][0]
    t_min = t[t_min_index]
    
    t_c = sy.linspace(0, t_min, 200)
    R_c = -by / (1 - sy.cos(t_min))
    
    return cycloid_x(t_c, R_c)+ax, cycloid_y(t_c, R_c)+ay
    
    
def wenn_maus_geklickt(event, ax):
    """Zeichne Brachistochrone zur Mausposition."""
    #Test, ob Klick mit linker Maustaste und im                                        
    #Koordinatensystem erfolgt, sowie ob die Zoomfunktion 
    #des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        xpos = event.xdata
        ypos = event.ydata

        #Berechne Brachistochrone
        x, y = BRC_var2(xpos, ypos)

        ax.plot(x, y, ls='-', lw=1, c='r') #Kurve hinzufügen
        event.canvas.draw()                #Kurve plotten
        

def main():
    """Hauptprogramm"""
    print(__doc__)         #Ausgabe Programm Doc-String

    #Erstelle einen Plotbereich
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.axis([-2, 10, -10, 2])  #Achsengrenzen
    ax.set_xlabel("x")       #Beschriftung x-Achse
    ax.set_ylabel("y")       #Beschriftung y-Achse
    x, y = BRC_var2(8, -5, 0, 0)
    
    plt.plot(x, y, c='b', lw=1.5, label='Brachistochrone')
    klick_funktion = functools.partial(wenn_maus_geklickt,
                                       ax=ax)
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    plt.legend()
    plt.grid(True)
    plt.show()    

if __name__ == "__main__":
    main()
    
