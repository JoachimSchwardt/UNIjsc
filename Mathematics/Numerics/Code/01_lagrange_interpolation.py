#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Julian Fleck       :    4759587 
Joachim Schwardt   :    4768711

Einfache Implementierung der Lagrange-Interpolation (O(n**2))

Die Funktion 'polynomInterpolation' akzeptiert sowohl skalare als auch 
    vektorwertige Inputs, weshalb es keine 'polynomInterpolationA' gibt.
"""

import numpy as np
import matplotlib.pyplot as plt   # für Visualisierung

def lagrangePolynom(t, tWerte, i):
    """
    Wertet das 'i'-te Lagrange-Polynom für die Stützstellen 'tWerte' 
    an 't' aus.
    't' kann ein Skalar oder 1d-array sein.
    """
    fractions = np.array([(t - tWerte[j]) / (tWerte[i] - tWerte[j]) 
                          for j in range(tWerte.shape[0]) if (j != i)])
    return np.prod(fractions, axis=0)    # axis=0 für array inputs von 't'

def polynomInterpolation(tWerte, fWerte, t):
    """
    Werte die Polynom-Interpolation für alle Werte 't' aus.
    't' darf dabei sowohl ein Skalar, als auch ein 1d-array sein.
    Die Stützstellen sind gegeben durch (tWerte, fWerte). 
    """
    
    # für ().shape und andere Funktionen werden np.ndarrays benötigt
    if not isinstance(tWerte, np.ndarray):
        # das kann zu einem 'deprecation warning' führen, wenn nested lists
        # ungleicher Länge übergeben werden -- aber warum sollte man ;)
        tWerte = np.array(tWerte)   
    if not isinstance(fWerte, np.ndarray):
        fWerte = np.array(fWerte)
    
    # Stützstellen müssen 1D sein
    if ((tWerte.ndim != 1) or (fWerte.ndim != 1)):
        msg = ("Array-Dimensionen der Stützstellen sollten gleich 1 sein, "
               + f"aber waren {tWerte.ndim} (tWerte) und " 
               + f"{fWerte.ndim} (fWerte)!")
        raise TypeError(msg)
    
    # Stützstellen (tWerte, fWerte) müssen 'aufgehen' (--> gleich lang)
    if (tWerte.shape[0] != fWerte.shape[0]):
        msg = ("Arrays der stützstellen müsssen die gleich Länge haben, "
               + f"aber hatten {tWerte.shape[0]} (tWerte) und "
               + f"{fWerte.shape[0]} (fWerte)!")
        raise IndexError(msg)
    
    # Interpolation nur in-bounds erlaubt (--> bestimmt durch minmax(tWerte))
    if ((np.any(t < np.min(tWerte))) or (np.any(t > np.max(tWerte)))):
        bounds = f"[{np.min(tWerte)}, {np.max(tWerte)}]"
        if isinstance(t, np.ndarray):
            msg = (f"t-values should be within {bounds}, " 
                   + f"but were within [{np.min(t)}, {np.max(t)}]")
        else:    # ... andere Fehlermeldung für einen skalaren Input
            msg = f"t-value should be within {bounds}, but was {t}"
        raise ValueError(msg)
        
    # array der 'lagrange-Komponenten jeder Stützstelle' 
    components = np.array([fWerte[i] * lagrangePolynom(t, tWerte, i) 
                           for i in range(fWerte.shape[0])])
    
    # Ergebnis ist die Summe (entlang der major-axis=0 !) aller Komponenten
    return np.sum(components, axis=0)    # axis=0 für array inputs von 't'

def plotInterpolation(t, p_t, tWerte , fWerte):
    """Plottet die Werte p(t) über t, sowie die gegebenen Stützstellen"""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(t, p_t, lw=1.5, c='b', label="Interpolation")
    ax.plot(tWerte, fWerte, ls='', marker='x', ms=7, mew=1, c='k', 
            label="Stützstellen")
    ax.legend()
    fig.tight_layout()
    plt.show()

def main(case=0):
    if case == 1:
        tWerte = np.arange(10)
        fWerte = np.array([17, 20, 22, 22, 23, 15, 17, 28, 22, 22.1])
    else:
        tWerte = np.array([0, 1, 2, 3, 4])
        fWerte = np.array([17, 20, 22, 22, 23])
    
    Npoints = 200       # Anzahl an Auswertungspunkte des Polynoms
    t = np.linspace(np.min(tWerte), np.max(tWerte), Npoints)
    p_t = polynomInterpolation(tWerte, fWerte, t)
    plotInterpolation(t, p_t, tWerte, fWerte)

if __name__ == "__main__":
    print(__doc__)
    main(case = 0)  # nur 5 Stützstellen --> 'vernünftige' Interpolation
    main(case = 1)  # 10 Stützstellen --> bereits 'sehr große' Oszillationen
    
    """
Aufgabe 3 (Test der oberen Schranke für e^x-Fehler)
for N in range(2, 16, 1):   
    x = np.linspace(0, 1, 1000)
    t = np.linspace(0, 1, N+1)
    # t = np.arange(0, N+1, 1) / N
    expval = np.exp(x)
    intval = polynomInterpolation(t, np.exp(t), x)
    exp_err = np.exp(1) / np.math.factorial(N+1) * 2**(-N-1)
    true_err = np.max(np.abs(expval - intval))
    print(f"N={N}: true error {true_err} and upper bound {exp_err}.")
    """
    
