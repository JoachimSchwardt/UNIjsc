# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

"""
Implementierung einfacher Quadraturformeln und Test für die Runge-Funktion.
"""

import numpy as np


def f( x, a=5.0):
    """ Rungefunktion für beliebigen Parameter 'a'. """
    return 1.0 / (1.0 + (a * x)**2)


def F( x, a=5.0 ):
    """ Stammfunktion der Rungefunktion für exakte Integration. """
    return np.arctan(a * x) / a


def stuetzstellen(a, b, N=5):
    """ 
    Gibt 'N+1' uniform verteile Stützstellen auf dem Intervall [a, b] zurück.
    
    Anmerkung:   
        Für die Quadratur brauchen wir die letzte Stelle gar nicht, 
        bei linspace(..., N) wäre die Verteilung aber falsch.
    """
    return np.linspace(a, b, N+1)


def rechteckRegel(f, a, b, N=5, args=()):
    """ 
    Rechteckregel für die Integration einer FUnktion 'f' auf dem 
        Intervall [a,b] mit 'N'  (uniformen) Stützstellen.
    """
    x = stuetzstellen(a, b, N)[:-1]    # letzte Stützstelle ignorieren
    h = x[1] - x[0]                    # Abstand benachbarter Stützstellen
    return np.sum(f(x, args)) * h


def mittelpunktsRegel(f, a, b, N=5, args=()):
    """ 
    Mittelpunktsregel für die Integration einer FUnktion 'f' auf dem 
        Intervall [a,b] mit 'N'  (uniformen) Stützstellen.
    """
    x = stuetzstellen(a, b, N)[:-1]    # letzte Stützstelle ignorieren
    h = x[1] - x[0]                    # Abstand benachbarter Stützstellen
    return np.sum(f(x + h/2, args)) * h


def berechneKoeffizienten( f, xmin, xmax, N, args=() ):
    """ 
    Kopiert von der vorherigen Aufgabe. Eleganter wäre natürlich 
        'from ... import berechneKoeffizienten', aber wir wollen nicht, dass
        es dann Problem mit anderen Dateinamen/Ordnerstrukturen etc. gibt.
    """
    M = np.zeros( (4*N, 4*N) ) 

    # die Vektoren a,b,c,d der Länge N, sowie r,x der Länge 4N
    a = np.zeros( N )
    b = np.zeros( N )
    c = np.zeros( N )
    d = np.zeros( N )

    x = np.zeros( 4*N )     # x = [a_0, ..., a_{N-1}, ... , d_0, ..., d_{N-1}]
    r = np.zeros( 4*N )

    # Stützstellen und konstanter Abstand 'h'
    xk = stuetzstellen(xmin, xmax, N)
    h = xk[1] - xk[0]

    # Den Eintrag (i,j) belegt man mit M[i,j] = ...
    # Jede Zeile (i) entspricht einer der 4N Gleichungen 
    #     (Reihenfolge im Prinzip egal)
    # Jede Spalte entspricht dann den Koeffizienten a_0, ..., a_{N-1}, b_0, ...
    for i in range(N):
        M[i + 3*N, i + 3*N] = 1           # d_k
        r[i + 3*N] = f(xk[i], args)       # f(x_k)

        M[i, i] = h**3                    # a_k * h^3
        M[i, i + N] = h**2                # b_k * h^2
        M[i, i + 2*N] = h                 # c_k * h
        r[i] = f(xk[i+1], args) - f(xk[i], args)    # f(x_{k+1}) - f(x_k)

        if i == 0: 
            M[N, N] = 1              # b_0
            M[2*N, N-1 + N] = 1      # b_{N-1}
            M[2*N, N-1] = 3*h        # 3*a_{N-1} * h

        else:
            M[i + N, i + 2*N] = 1             # c_k
            M[i + N, i-1 + 2*N] = -1          # -c_{k-1}
            M[i + N, i-1] = -3*h**2           # -3*a_{k-1} * h^2
            M[i + N, i-1 + N] = -2*h          # -2*b_{k-1} * h

            M[i + 2*N, i + N] = 1             # b_k
            M[i + 2*N, i-1 + N] = -1          # -b_{k-1}
            M[i + 2*N, i-1] = -3*h            # -3*a_{k-1} * h

    # Löse Mx = r mit einem eingebauten Löser 
    x = np.linalg.solve( M, r )

    # extrahiere a,b,c,d                                
    a = x[0:N]
    b = x[N:2*N]
    c = x[2*N:3*N]
    d = x[3*N:4*N]

    return a, b, c, d


def splineIntegration(f, a, b, N=5, args=()):
    """ 
    Interpoliert eine Funktion 'f' mit einem kubischen Spline für die 
        Integration  auf dem Intervall [a,b] mit 'N' (uniformen) Stützstellen.
        
    Exaktes Integral eines kubischen Splines ist
        I_j = (a_j/4 * h**4 + b_j/3 * h**3 + c_j/2 * h**2 + d_j * h)
        I = sum_{j=0}^{N-1} I_j
    """
    x = stuetzstellen(a, b, N)    
    h = x[1] - x[0]                    # Abstand benachbarter Stützstellen
    
    # Stützstellen werden nochmal erstellt, aber da das alles ohnehin nicht 
    #  besonders performant ist, lassen wir das einfach mal so :)
    aj, bj, cj, dj = berechneKoeffizienten(f, a, b, N, args=args)
    # int_j = (a/4 * (x[1:]**4 - x[:-1]**4) + b/3 * (x[1:]**3 - x[:-1]**3)
    #          + c/2 * (x[1:]**2 - x[:-1]**2) + d * h)
    int_j = (aj/4 * h**4 + bj/3 * h**3 + cj/2 * h**2 + dj * h)
    return np.sum(int_j)


def printResults(xmin, xmax, N, a=5.0):
    xj = np.linspace(xmin, xmax, N)   # Stützstellen
    h = xj[1] - xj[0]
    
    # exakter Integralwert
    int_exakt = F(xmax, a) - F(xmin, a)               
    
    # Rechteck-Quadratur
    int_rechteck = rechteckRegel(f, xmin, xmax, N, args=(a))    
    
    # Mittelpunkts-Quadratur
    int_mp = mittelpunktsRegel(f, xmin, xmax, N, args=(a))  
    
    # Spline-Quadratur
    int_spline = splineIntegration(f, xmin, xmax, N, args=(a))  
    
    # absoluter Fehler zum exakten Integralwert
    diff = np.abs(int_exakt - np.array([int_rechteck, int_mp, int_spline]))
    print(f"{N:<4} | {h:.{4 - (h>=10.0)}f} | {int_rechteck:.5f} {diff[0]:.4e} " 
          + f"| {int_mp:.5f} {diff[1]:.4e} | {int_spline:.5f} {diff[2]:.4e}")
    

def main():
    a = 5.0                           # Parameter für die Runge-Funktion
    xmin = -5.0                       # minimaler x-Wert
    xmax = 5.0                        # maximaler x-Wert
    Narr = 2**np.arange(1, 11, 1)     # Array mit Stützstellen-Anzahlen
    
    print(f"Exakter Integralwert ist {2 * F(xmax, a)}\n")
    
    print("N    | h      | (a)     Fehler     " 
          + "| (b)     Fehler     | (c)     Fehler    ")
    for N in Narr:
        printResults(xmin, xmax, N, a)
    
    return 0


if __name__ =="__main__":
    print(__doc__)
    main()
    
    """
Test, ob die Spline-Interpolation wirklich funktioniert. Irgendwie gibt es da
    kleine Abweichungen zur scipy-Variante (verschwinden für 'große' N...).
    Maximale Differenz für N=30 ist 2e-06, L1-Different ist 5e-06.
    
    Bei N=4 ist die Differenz sogar ohne Zoom deutlich zu erkennen.
    Da der Code trotz Änderung aber immer noch dassselbe Ergebnis wie beim
    letzten mal liefert, nimmt scipy wohl irgendwelche anderen RB an :)
    
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

xmin, xmax = -5, 5
x = np.linspace(xmin, xmax, 200)
N = 3
a,b,c,d = berechneKoeffizienten(f, xmin, xmax, N, args=(5.0))
xs = stuetzstellen(xmin, xmax, N)
s3 = interp1d(xs, f(xs), kind='cubic')
plt.plot(x, f(x), c='b', label='f(x)')
plt.plot(x, spline3(a,b,c,d,x), c='r', ls='dashed', label='spline(x)')
plt.plot(x, s3(x), c='g', ls='dotted', label='scipy(x)')
plt.legend()
    """
