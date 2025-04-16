"""Beispiel zur Interaktion in Plotfenstern.

Beim Klicken innerhalb des Plotfensters wird eine Sinuskurve
ab der Position des Mauszeigers gezeichnet.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt


def sinus_amplitude(amplitude, xpos, ypos):
    """
    Berechne Sinuswerte (mit Amplitude) fuer eine Periode und verschiebt die
    Werte auf der x-Achse um xpos und der y-Achse um ypos.
    """
    t = np.linspace(0.0, 2*np.pi, 300)
    x = t + xpos
    y = amplitude * np.sin(t) + ypos
    return x, y


def wenn_maus_geklickt(event, ax, amplitude):
    """Zeichne Sinus-Kurve ausgehend von Mausposition."""
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        xpos = event.xdata
        ypos = event.ydata

        # Berechne Sinus
        x, y = sinus_amplitude(amplitude, xpos, ypos)

        ax.plot(x, y, ls='-', lw=1, c='r')      # Kurve hinzufuegen
        event.canvas.draw()                     # plotten


def main():
    """Hauptprogramm"""
    print(__doc__)                              # Ausgabe Programm Doc-String

    ampl = 0.8

    # Erstelle einen Plotbereich
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, autoscale_on=False)
    ax.axis([0.0, 2*np.pi, -1.0, 1.0])                  # Achsengrenzen
    ax.set_xlabel("x")                                  # Beschriftung x-Achse
    ax.set_ylabel("y")                                  # Beschriftung y-Achse

    klick_funktion = functools.partial(wenn_maus_geklickt,
                                       ax=ax,
                                       amplitude=ampl)
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    plt.show()


if __name__ == "__main__":
    main()