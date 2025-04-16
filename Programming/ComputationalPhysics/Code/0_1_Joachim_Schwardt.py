"""Zeichnet eine Spirale nach Linksklick mit Zentrum im ausgewaehlten Punkt."""

import functools
import numpy as np
import matplotlib.pyplot as plt                    # Graphikbefehle


def spirale(x0, y0, anz_windung=10, anz_punkte=100):
    """Berechnet eine Spirale mit Zentrum (x0, y0) und
       anz_windung Windungen."""
    t = np.linspace(0.0, anz_windung, anz_punkte)  # Parameterisierungsvariable
    r = 0.5**t                                     # Radius
    phi = 2.0*np.pi*t                              # Winkel

    # Umrechnung von Polarkoordinaten in kartesische Koordinaten
    x = r*np.cos(phi) + x0
    y = r*np.sin(phi) + y0

    return x, y


def neue_spirale(event, ax, anz_windung=7, anz_punkte=200):
    """Nach Linksklick, plotte Spirale mit Zentrum im ausgewaehlten Punkt."""
    # Test, ob Klick mit linker Maustaste und im Plotbereich ax
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax and mode == '':
        # Spirale um gewaehlten Mittelpunkt berechnen und plotten
        x, y = spirale(event.xdata, event.ydata,
                       anz_windung=anz_windung,
                       anz_punkte=anz_punkte)
        ax.plot(x, y)

        # Fensterbereich aktualisieren
        event.canvas.draw()


def main():
    """Hauptprogramm: Initialisierung Plotfenster + Def. Mausinteraktion."""
    anz_windung = 5                           # Anzahl der Windungen
    anz_punkte = 1000                         # Anz. Punkte der Spirale

    # Nutzerfuehrung und Parameterausgabe
    print(__doc__)
    print("Mit linker Maustaste Mittelpunkt der Spirale festlegen")
    print("Es werden {} Windungen gezeichnet.".format(anz_windung))

    # Erzeuge Plotfesnter und quadratischer Plotbereich
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)

    # Achsenbereiche setzen
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])

    # Plotbeschriftungen setzen
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Spirale")

    # Spirale um den Mittelpunkt (0.0, 0.0) berechnen und plotten
    x_pkt, y_pkt = spirale(0.0, 0.0,
                           anz_windung=anz_windung,
                           anz_punkte=anz_punkte)
    ax.plot(x_pkt, y_pkt)

    # Bei Mausklick soll die Funktion ``neue_spirale`` aufgerufen werden,
    # wobei der Plotbereich ax und die Parameter `anz_windung` und `anz_punkte`
    # beim Aufruf mit uebergeben werden.
    klick_funktion = functools.partial(neue_spirale, ax=ax,
                                       anz_windung=anz_windung,
                                       anz_punkte=anz_punkte)
    fig.canvas.mpl_connect('button_press_event', klick_funktion)

    # Endlos-Schleife, die auf Ereignisse wartet:
    plt.show()


if __name__ == "__main__":
    main()
