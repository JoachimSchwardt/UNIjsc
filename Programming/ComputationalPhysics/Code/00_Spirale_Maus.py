"""
Weiteres Beispiel zur Interaktion mit Plotfenstern.

Beim Klicken innerhalb des Plotfensters wird eine Spirale 
um den Ursprung des Mauszeigers gezeichnet.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt

def spiral(xpos, ypos, rotations=5):
    """
    Berechne eine Spirale mit gegebener Parametrisierung mit (xpos, ypos) als Ursprung.
    """
    t = np.linspace(0.0, rotations, 300)
    phi = 2*np.pi*t
    r = 0.5**t
    x = r * np.cos(phi)  # Umrechnung in kartesische 
    y = r * np.sin(phi)  # Koordinaten
    x += xpos  # Verschieben des Ursprungs der Spirale um
    y += ypos  # die Funktionsargumente (später Mausposition)
    return x, y

def wenn_maus_geklickt(event, ax):
    """Zeichne Spirale ausgehend von Mausposition."""
    # Test, ob Klick mit linker Maustaste und im                                        
    # Koordinatensystem erfolgt, sowie ob die Zoomfunktion 
    # des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        xpos = event.xdata
        ypos = event.ydata

        # Berechne Spirale
        x, y = spiral(xpos, ypos)

        ax.plot(x, y, ls='-', lw=1, c='r') # Kurve hinzufügen
        event.canvas.draw()                # Kurve plotten
        

def main():
    """Hauptprogramm"""
    print(__doc__)         # Ausgabe Programm Doc-String

    # Erstelle einen Plotbereich
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.axis([-1, 1, -1, 1])  # Achsengrenzen [xmin, xmax, ymin, ymax]
    ax.set_xlabel("x")       # Beschriftung x-Achse
    ax.set_ylabel("y")       # Beschriftung y-Achse
    plt.plot(spiral(0, 0)[0], spiral(0, 0)[1])
    klick_funktion = functools.partial(wenn_maus_geklickt,
                                       ax=ax)
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    plt.grid(True)
    plt.show()    

if __name__ == "__main__":
    main()
