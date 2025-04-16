"""
Generieren und visualisieren der 'Dragon Curve'.
Die Folge wird indiziert mit 0,1,2,3...
Dann gilt:
    f(n) = (-1)**m * f(m) fuer n=2m
    f(n) = f(m) fuer n=2m+1
"""

import numpy as np
import matplotlib.pyplot as plt

def dragon_curve(n):
    """Berechnet die ersten 'n' Folgenglieder der 'Dragon Curve'."""
    array = np.ones(int(n))                     
    
    for i in range(1, int(n / 2), 1):
        array[2*i] = (-1)**(i % 2) * array[i]   # Terme mit Indizes 2,4,6...
        array[2*i + 1] = array[i]               # Terme mit Indizes 3,5,7...
        
    return array.astype(int)

def dragon_curve_graph(array, n):
    """Berechnet die xy-Daten aus den gegebenen Folgengliedern."""
    x_data = np.zeros(int(n) + 1)
    y_data = np.zeros(int(n) + 1)
    
    for i in range(1, int(n / 2) + 1, 1):
        # 'x' Daten iterativ
        x_data[2*i - 1] = x_data[2*i - 2] + array[2*i - 2]
        x_data[2*i] = x_data[2*i - 1]
        
        # 'y' Daten iterativ
        y_data[2*i - 1] = y_data[2*i - 2]
        y_data[2*i] = y_data[2*i - 1] + array[2*i - 1]
    
    return x_data, y_data

def main():
    print(__doc__)
    
    n = 2**16                   # Anzahl der Folgenglieder
    size = 1.3 * np.sqrt(n)     # Skalierung des Plots
    
    # Plotdaten erzeugen
    array = dragon_curve(n)
    # array = 2 * np.random.randint(0, 2, int(n)) - 1
    # array = np.random.randint(-3, 4, int(n))
    x_data, y_data = dragon_curve_graph(array, n)
    
    # Plot erstellen
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.axis([-size, size, -size, size])
    
    ax.plot(x_data, y_data, lw=0.5)
    
    plt.show()

if __name__ == "__main__":
    main()