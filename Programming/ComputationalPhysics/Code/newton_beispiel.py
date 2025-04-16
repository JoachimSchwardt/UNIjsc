"""Kurzbeschreibung des Programms ("Titel")

An dieser Stelle soll eine Beschreibung des Programms,
der Methoden, offener Fragen etc. kommen.
"""

from math import sqrt


def newton_iteration(zahl, anz_iter=5):
    """Berechne Quadratwurzel von `zahl` mittels Newton-Iteration.

    Die Newton Iteration
        x_k =  1/2 (x_{k-1} + zahl/x_{k-1})
    (mit x_0=1) konvergiert zur gesuchten Wurzel.

    Der Parameter anz_iter (mit dem Default-Wert 5) bestimmt die
    Anzahl der durchgefuehrten Iterationen.
    """
    x = 1.0                         # Startwert der Newton-Iteration
    for ctr in range(anz_iter):     # Fuehre anz_iter Iterationen durch
        x = 0.5 * (x + zahl/x)      # Newton-Iterationsschritt
    return x


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print("Newton Iteration")
    print("sqrt(2)      : %16.14f" % (sqrt(2.0)))
    print("5 Iterationen: %16.14f" % (newton_iteration(2.0)))
    print("3 iterationen: %16.14f" % (newton_iteration(2.0, 3)))
    print("2 iterationen: %16.14f" % (newton_iteration(2.0, 2)))


if __name__ == "__main__":
    main()
