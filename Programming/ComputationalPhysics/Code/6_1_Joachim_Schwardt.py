"""
Visualisierung der Eigenwerte und Eigenfunktionen eines 1-D Potentials. 

Die Eigenwerte entsprechen den horizontalen Linien. Die zugehoerigen
    Eigenfunktionen sind als Auslenkung um diese Linien als jeweilige 
    'x-Achse' dargestellt. Die Skalierung in 'y-Richtung' ist so angepasst,
    dass man die Eigenfunktionen klar unterscheiden kann. 
    Die Normierung ist im Plot nicht gegeben!
    
Voreingestellte Parameter s. Ueberschrift im Plot. 
Verwendete numerische Methode:
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import hermite

def V_gen(A=0.06, mode=1):
    """
    Erzeugt ein Potential mit gegebenen Parametern.
    Auswahl zwischen voreingestellten Potentialen mit 'mode'.
    """
    def V(x):
        if mode == 0:     # harmonischer Oszillator
            return 0.5 * x**2
        elif mode == 1:   # asymmetrisches Doppelmuldenpotential
            return x**4 - x**2 - A * x
        elif mode == 2:   # Potentialkasten
            return 0*x
    return V

def EwEv_Hamilton(V, h_eff=0.07, xmin=-5, xmax=5, N=100):
    """
    Berechnet eine Naeherung fuer die Eigenwerte und Eigenvektoren eines
    Hamiltonian unter Verwendung von scipy.linalg.eigh.  
        [xmin, xmax] = Potential 'unendlich' ausserhalb des Intervalls
        N = Matrixgroesse  (bestimmt Schrittweite dx)
        A = Parameter des Potentials
        h_eff = Parameter des Hamiltonian
    Die Eigenvektoren von eigh() sind normiert auf sum(ev[:, i]**2) = 1.
    Durch die konstante Diskretisierungsschrittweite 'dx' ist das Integral
    etwa um einen Faktor 'dx' kleiner. Fuer die Normierung muessen die
    Eigenvektoren also durch sqrt(dx) geteilt werden. 
    """
    dx = (xmax - xmin) / (N + 1)   # Diskretisierungsschrittweite
    z = h_eff**2 / (2 * dx**2)     # Faktor aus Naeherung 2.Abl. von psi(x)
    x_array = np.linspace(xmin - dx, xmax + dx, N)    # Diskrete Punkte
    
    # Potential in (xmin, xmax) mit N Punkten
    V_array = V(x_array) 
    
    side_array = z * np.ones(N-1)    # Werte fuer Nebendiagonalen
    main_array = V_array + 2 * z     # Hauptdiagonale
    
    # diskrete Hamilton-NxN-Matrix
    H_disc = (np.diag(-side_array, k=1) + np.diag(-side_array, k=-1) 
              + np.diag(main_array, k=0))
    ew, ev = eigh(H_disc)
    return ew, ev / np.sqrt(dx)   # Normierung der Eigenvektoren

def hermite_norm(x, n=0):
    """
    Berechnet die normierten Eigenfunktionen des harmonischen Oszillators.
    """
    norm_factor = np.pi**(1/4) * np.sqrt(2**n * np.math.factorial(n))
    return hermite(n)(x) * np.exp(-0.5*x**2) / norm_factor

def main():
    print(__doc__)
    print(EwEv_Hamilton.__doc__)
    N = 350        # Matrixgroesse
    mode = 1       # Auswahl des Potentials
    A = 0.06       # Parameter des Potentials
    if mode == 0:   # Harmonischer Oszillator fuer Programmtest
        h_eff = 1   
        xmin, xmax = -5, 5   
        E_max = 4    
        scale_factor = 1     
        
    elif mode == 1:    # asymmetrische Doppelmulde
        h_eff = 0.07             # Parameter des Hamilton
        xmin, xmax = -1.5, 1.5   # Potential 'unendlich' ausserhalb
        E_max = 0.1              # maximaler Energieeigenwert fuer Plot
        scale_factor = 0.02      # Skalierung fuer Plot
        
    elif mode == 2:
        h_eff = 1
        xmin, xmax = 0, 2
        E_max = 6
        scale_factor = 5
    
    
    title_txt = ('Eigenfunktionen und Energieeigenwerte des Potentials: ' + 
                 r'$V(x)=x^4-x^2-Ax$')
    param_txt = ('A={}, '.format(A) + '$\hbar_{eff}$=' + '{}, N={}, '
                 .format(h_eff, N) + r'$x_{min}$=' + '{}, '.format(xmin) 
                 + r'$x_{max}$=' + '{}'.format(xmax))
    
    # Eigenwerte und Eigenvektoren bestimmen
    V = V_gen(A, mode)         # Erzeugen des Potentials
    ew, ev = EwEv_Hamilton(V=V, A=A, h_eff=h_eff, xmin=xmin, xmax=xmax, N=N,
                           mode=mode)
    
    # Anzahl an Eigenwerten mit Energie kleiner E_max
    E_max_index = 6 + 1   #max(*np.where(ew <= E_max)) + 1
    
    # Plotbereich erstellen
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.suptitle(title_txt)
    ax.set_title(param_txt, size='small')
    ax.set_xlabel('x')
    ax.set_ylabel('Energie')
    
    x_array = np.linspace(xmin, xmax, N)   # Werte fuer x-Achse
    ax.axis([xmin, xmax, -1.5 * abs(ew[0]), 1.2 * ew[E_max_index - 1]]) 
    
    # Plot des Potentials
    ax.plot(x_array, V(x_array, A, mode), c='b', lw=1, label='Potential')   
    # Plot der Eigenfunktionen verschoben um zugehoerige Energie
    for count in range(E_max_index):
        # skalierte Eigenfunktionen fuer Plot
        ev_plot = scale_factor * ev[:, count] + ew[count] 
        ax.plot(x_array, ev_plot, c='r', lw=0.8)
        # Energieeigenwerte       
        ax.axhline(ew[count], c='firebrick', lw=0.8, 
                   label='Eigenenergien' if count == 0 else '')   
        print(ew[count])
        ax.text(0.9 * xmax, ew[count] + 0.01, 
                r'$\varphi_{}(x)$'.format(count), color='red')
        
        if mode == 0:   # Analytische EF des harm. Oszi. 
            ax.plot(x_array, hermite_norm(x_array, count) + ew[count],
                    c='g', lw=0.8)
    
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
    """
a) Wahl der numerischen Parameter:
    xmin, xmax: 
        Die betrachteten Eigenfunktionen sind an den Grenzen quasi 0. 
        Groessere Werte fuehren zu 'leeren' Plotbereichen, und benoetigen 
        groessere Werte fuer N. 
    N:
        Beim Vergleich mit den analytischen Eigenwerten des harmonischen 
        Oszillators ist erst in der vierten Nachkommastelle eine Abweichung
        erkennbar (zumindest fuer die ersten vier EW). Groessere Werte
        fuehren zu deutlich hoeheren Laufzeiten.
b) Struktur fuer A = 0.06:
    Sei V0 das lokale Maximum von V(x). Es ist V0 etwa 0 fuer die gegebenen
        Parameter, Eigenwerte kleiner V0 sind also insbesondere negativ.
    Der Knotensatz ist erfuellt, bei manchen Eigenfunktionen muss man die
        Zoomfunktion verwenden, um alle Nullstellen erkenne zu koennen. Die
        Eigenfunktionen fallen an den Raendern exponentiell. Fuer steigende
        Energien groesser als V0 sehen sie den Eigenfunktionen des
        harmonischen Oszillators immer aehnlicher. 
    Fuer Eigenenergien kleiner V0 sind die Eigenfunktionen naeherungsweise
        symmetrisch um die x-Koordinate des jeweiligen Minimums von V(x) und
        sehen hier wieder aus wie beim harmonischen Oszillator. Das
        Potential hat hier auch lokal etwa die Form a*x**2. 
    Fuer kleinere h_eff (z.B. 0.4) liegen fuer Eigenenergien, die groesser
        als die beiden Minima und kleiner als das lokale Maximum des
        Potentials sind, immer zwei verschiedene Werte relativ nahe
        beieinander.
    Fuer groessere h_eff nehmen die Eigenenergien zu, es werden also bei
        festem E_max weniger Funktionen dargestellt.
c) A = 0:
    Die ersten beiden Energieeigenwerte unterscheiden sich erst in der
        vierten Nachkommastelle, was auch in etwa der Genauigkeit der
        Naeherung beim harmonischen Oszillator bei den gewaehlten Parametern
        entspricht. Sie scheinen allerdings nicht entartet zu sein, da auch
        ein Test fuer groessere N die gleichen EW ergibt, wobei die
        Genauigkeit beim harmonischen Oszillator  fuer N=1000 bereits eine
        Groessenordnung besser ist. 
    Wie fuer ein symmetrisches Potential erwartet, sind die Eigenfunktionen
        gerade oder ungerade. Sie sehen fuer Energien die kleiner als V0
        sind etwa so aus wie die 'gerade und ungerade Version' der gleichen
        Eigenfunktion.
    """