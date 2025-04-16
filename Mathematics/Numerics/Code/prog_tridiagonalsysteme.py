# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np

# zum Ermitteln der Laufzeit
import time

# für die pow - Potenzfunktion
import math


# Erstelle die n x n - Tridiagonalmatrix aus der Aufgabe
def tridiagonalMatrix( n ):

    # Sonst ist der Datentyp 'int64', und interessanterweise funktioniert
    # Python in folgendem Beispiel 'unerwartet':
    #     a = np.array([5,3,1,3])
    #     a[1] = 0.5
    #     ---> a = array([5, 0, 1, 3])
    # das ist natürlich blöd, weil normalerweise Python so funktioniert:
    #     a = np.array([5,3,1,3])
    #     a = a/2
    #     ---> a = array([2.5, 1.5, 0.5, 1.5])
    # Aus irgendwelchen Gründen ist der Detentyp also bei Zuweisungen fest ...
    
    
    S = np.diag(np.full(n, 2, dtype=np.float64))             # Hauptdiagonale
    S += np.diag(np.full(n-1, -1, dtype=np.float64), k=1)    # rechts davon
    S += np.diag(np.full(n-1, -1, dtype=np.float64), k=-1)   # links davon
    return S



# Löse Tx = b sodass x = tridiagSolver( T, b )
def tridiagSolver( T, b ):
    """ Löst ein tridiagonales Gleichungssystem ohne Pivotisierung """
    n = b.shape[0]    
    for col in range(n-1):
        # Verhältnis für Gauss-Eliminierung der Spalte 'col'
        ratio = -T[col+1, col] / T[col, col]
        
        # die Spalte kann jetzt gleich 0 gesetzt werden 
        # (tridiagonal --> nur ein Element)
        T[col+1, col] = 0
        
        # 'T' und 'b' der Zeile 'col+1' werden entsprechend manipuliert
        T[col+1, col+1] += ratio * T[col, col+1]
        b[col+1] += ratio * b[col]
    
    # Rückwärtssubstitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n-1, -1, -1):
        rest = 0.0    # für die letzte Zeile gibt es nur einen Term
        if i < n-1:   # sonst müssen wir zu b[i] noch etwas addieren
            rest = -x[i+1] * T[i, i+1]
            
        # x_i ist einfach b_i / T_ii, modulo Korrekturterm von der oberen
        # Nebendiagonalen, den wir noch zu b_i addieren müssen
        x[i] = (b[i] + rest) / T[i, i]
    
    return x
    

####### START #######

# Die Laufzeit des vorigen Durchlaufs speichern (am Anfang noch leer)
tTridiagVorher = 0.0
tNpSolVorher = 0.0

# geometrisches Mittel der Faktoren
faktorTridiagSove = 1.0
faktorLinalgSolve = 1.0

print("  i      n  tridiag    Faktor     linalg.solve  Faktor         ")
print("---------------------------------------------------------------")

# falls der Arbeitsspeicher und Ihre Lebenszeit mehr hergeben, 
# können Sie gerne mehr als 14 Durchläufe machen :)
# selbst mit 32Gb gibt der Arbeitsspeicher nicht viel mehr als i=15 her...
#     RAM = 8Byte * (2**i)**2 = 2**(2i + 3) Byte --> i=16: RAM = 34GB
# ...und das bei einer 'echten' Datenemenge von 1.6MB :)
iMax = 14
for i in range(iMax):

    n = 2**i

    # erzeuge die Matrix und eine rechte Seite ( b=1 )
    T = tridiagonalMatrix( n )
    b = np.zeros( n )
    b[:] = 1.

    # starte die Stoppuhr (perf_counter hat eine genauere Auflösung)
    t = time.perf_counter()
    # berechne die Lösung
    x1 = tridiagSolver ( T, b )
    # ermittle die Laufzeit
    tTridiag = time.perf_counter() - t


    # zur Sicherheit: erzeuge die Matrix und die rechte Seite neu 
    # (womöglich wurden sie überschrieben)
    T = tridiagonalMatrix( n )
    b = np.zeros( n )
    b[:] = 1.

    # stoppe wieder die Zeit
    t = time.perf_counter()
    # löse mit dem eingebauen numpy Löser
    x2 = np.linalg.solve(T,b)
    tNpSol = time.perf_counter() - t

    # bestimme den maximalen Fehler zum Referenzalgorithmus
    # np.inf = Infimumsnorm
    fehler = np.linalg.norm( x1 - x2, np.inf )

    # kleine Rundungsfehler (Stichwort Auslöschung) werden toleriert
    if fehler > 1e-6 :
        print("Fehler! Ihre Lösung weicht um", fehler, 
              "von der echten Lösung ab!")

    
    if i < 5: # Faktoren für kleine Probleme nicht aussagekräftig!
        print('{:3}'.format(i), '{:5}'.format(n), ' {:.3e}'.format(tTridiag),
              '   --    ', '{:.3e}'.format(tNpSol), '   --    ', sep="  ")
    else:
        faktorTridiagSove *= tTridiag/tTridiagVorher
        faktorLinalgSolve *= tNpSol/tTridiagVorher
        
        print('{:3}'.format(i), '{:5}'.format(n), '{:.3e}'.format(tTridiag),
              '{:.3e}'.format(tTridiag/tTridiagVorher),'{:.3e}'.format(tNpSol),
              '{:.3e}'.format(tNpSol/tNpSolVorher), sep="  ")

    # merke die aktuelle Zeit
    tNpSolVorher = tNpSol
    tTridiagVorher = tTridiag
    
    
print( "\ngeometrisches Mittel Zeitfaktor für tridiagSolver:", 
      math.pow( faktorTridiagSove, 1.0/(iMax-5)))
print( "geometrisches Mittel Zeitfaktor für np.linalg.solve:", 
      math.pow( faktorLinalgSolve, 1.0/(iMax-5)))
    

#### AUFGABE d) ####
def scipyTest(n = 10, names = ['csc', 'csr']):
    
    from scipy.sparse import dia_matrix
    import scipy.sparse.linalg as splinalg
    
    class Benchmark:
        def __init__(self, result):
            self.times = []
            self.result = result
            self.error = []
        
        def timedCall(self, func, *args):
            t = time.perf_counter()
            x = func(*args)
            delta_t = time.perf_counter() - t
            self.times.append(delta_t)
            try:
                self.error.append( np.linalg.norm(self.result - x, np.inf) )
            except TypeError:
                self.error.append(None)
            return x
    
    def tridiagonalSparseMatrix(n):
        ones = np.ones(n, dtype=np.float64)
        data = np.array([-ones, 2 * ones, -ones])
        offsets = np.array([-1, 0, 1])
        return dia_matrix((data, offsets), shape=(n, n))
    
    def tridiagSparseGauss( T, b ):
        """ Löst ein tridiagonales Gleichungssystem ohne Pivotisierung """
        n = b.shape[0]    
        for col in range(n-1):
            # Verhältnis für Gauss-Eliminierung der Spalte 'col'
            ratio = -T.data[0, col] / T.data[1, col]
            
            # die Spalte kann jetzt gleich 0 gesetzt werden 
            # (tridiagonal --> nur ein Element)
            T.data[0, col] = 0
            
            # 'T' und 'b' der Zeile 'col+1' werden entsprechend manipuliert
            T.data[1, col+1] += ratio * T.data[2, col]
            b[col+1] += ratio * b[col]
        
    def tridiagSparseSolver(T, b):
        tridiagSparseGauss(T, b)
        x = np.zeros(n, dtype=np.float64)
        for i in range(n-1, -1, -1):
            rest = 0.0    # für die letzte Zeile gibt es nur einen Term
            if i < n-1:   # sonst müssen wir zu b[i] noch etwas addieren
                rest = -x[i+1] * T.data[2, i]
                
            # x_i ist einfach b_i / T_ii, modulo Korrekturterm von der oberen
            # Nebendiagonalen, den wir noch zu b_i addieren müssen
            x[i] = (b[i] + rest) / T.data[1, i]
        return x
    
    def tridiagSparseScipySolver(T, b):
        tridiagSparseGauss(T, b)
        # umständlich transponieren, da der 'triangular'-solver gerne eine 
        # untere statt obere Dreickmatrix hätte...
        T.data[0] = T.data[2]
        T.data[2] = 0
        x = splinalg.spsolve_triangular(T.tocsr(), b)
        return x
    
    def solver(benchmark, name='csc'):
        b = np.ones(n, dtype=np.float64)
        S = tridiagonalSparseMatrix(n)
        if name == 'scipy-csc':
            x = benchmark.timedCall(splinalg.spsolve, S.tocsc(), b)
        elif name == 'scipy-csr':
            x = bench.timedCall(splinalg.spsolve, S.tocsr(), b)
        elif name == 'gauss-elim':
            x = bench.timedCall(tridiagSparseGauss, S, b)
        elif name == 'gauss-solve':
            x = bench.timedCall(tridiagSparseSolver, S, b)
        elif name == 'gauss-scipy':
            x = bench.timedCall(tridiagSparseScipySolver, S, b)
        else:
            raise NotImplementedError(f"Method {name} not found!")
        return x
    
    b = np.ones(n, dtype=np.float64)
    S = tridiagonalSparseMatrix(n)
    result = splinalg.spsolve(S.tocsr(), b)
    bench = Benchmark(result)
    for name in names:
        solver(bench, name)
        
    return bench
    
def printBenchmarks(imin=1, imax=10):
    names = np.array(['scipy-csc', 'scipy-csr', 'gauss-elim', 
                      'gauss-solve', 'gauss-scipy'], dtype=str)
    print("n      | " + " | ".join(f"{name :<11}" for name in names))
    print("-" * 77)
    
    fmin = 10                        # timing nur ab n >= 2**fmin 
    fctr = np.zeros(len(names))      # Anzahl timings pro Methode
    factors = np.ones(len(names))    # geom. Mittel aller Timings pro Methode
    
    old_times = np.ones(len(names))
    for i in range(imin, imax, 1):
        n = 2**i
        bench = scipyTest(n, names)
        
        
        print(f"{n:<6} | " + " | ".join(f"{t:.3e}  " for t in bench.times))
        if i >= fmin and i > imin:
            ratios = np.array(bench.times) / old_times
            factors *= ratios
            fctr += 1
            print("Faktor | " 
                  + " | ".join(f"{ratios[col]:.3e}  " 
                               for col in np.arange(len(names))) )
        old_times = np.array(bench.times)    
        print("-" * 77)
        
    factors = factors**(1.0 / fctr)
    print("GeoAvg | " + " | ".join(f"{f:.3e}  " for f in factors))
    
"""
Die Sparse-Methoden:
    Idee: 
        Wir speichern nur die nicht-verschwindenden Einträge
         --> man kann also viel größere Systeme berechnen
         --> Rechnung dauert genau so lang (oder länger), aber der 
             Speicherbedarf ist nur noch O(n)
         --> inklusive Aufstellen der Matrix sind alle Methoden trotzdem
             weitaus schneller, als in b)
    CSC: 
        compressed sparse column Matrix-Format
        scipy-solver für lineare Gleichungssysteme mit 'sparse-Einträgen'
    CSR: 
        compressed sparse row Matrix-Format
        identisch zu 'CSC', allerdings etwas schneller für kleine Matrizen
         --> scheinbar vom RAM-Layout besser, Löser greift wohl auf Zeilen zu
         
    Gauss-Elimination:
        löst NICHT das Gleichungssystem, sondern bringt es nur in Dreiecksform
    Gauss-Solve:
        Eigene Implementierung analog zu b), verwendet zuvor die Gauss-Elim.
        Bringt keinen laufzeit-Gewinn, da der Rechenaufwand gleich bleibt
    Gauss-Scipy:
        Scipy-sparse-solver für sprase Dreieckssysteme, zuvor auch Gauss-Elim.
"""  
printBenchmarks(5, 17)
