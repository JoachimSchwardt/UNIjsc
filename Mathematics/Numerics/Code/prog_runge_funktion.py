
# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

# "numpy" ist ein praktisches Numerikpaket für Matrix- und Vektorrechnung
#  -> um nicht immer "numpy" schreiben zu müssen, kürzen wir es als "np" ab:
import numpy as np

# matplotlib ist eine Sammlung von Funktionen, um Graphen zu plotten
# diese importieren wir als "plt"
import matplotlib.pyplot as plt


# unsere Rungefunktion
def f( x ):
    return 1.0 / (1.0 + 25.0*x*x)




# Aufgabe (a)  "schriftlich"
    
# 1a)  d_k = f(x_k)
# 1b)  a_k * h^3 + b_k * h^2 + c_k * h = f(x_{k+1}) - f(x_k) 
#  2)  c_k - c_{k-1} - 3*a_{k-1} * h^2 - 2*b_{k-1} * h = 0
#  3)  b_k - b_{k-1} - 3*a_{k-1} * h = 0
# 4a)  b_0 = 0
# 4b)  b_{N-1} + 3*a_{N-1} * h = 0
    


# Aufgabe (b)
    
def berechneKoeffizienten( N ):
    
    # wir legen eine leere numpy-Matrix der Groesse 4N x 4N an, gefüllt mit Nullen
    # die doppelten Klammern sind wichtig!
    M = np.zeros( (4*N, 4*N) ) 
    
    # die Vektoren a,b,c,d der Laenge N, sowie r,x der Leange 4N:
    # hier nur einfache Klammern!
    a = np.zeros( N )
    b = np.zeros( N )
    c = np.zeros( N )
    d = np.zeros( N )
    
    x = np.zeros( 4*N )     # x = [a_0, ..., a_{N-1}, ... , d_0, ..., d_{N-1}]
    r = np.zeros( 4*N )

    # unser h aus der Aufgabe
    h = 10.0 / N
    
    # Den Eintrag (i,j) belegt man mit M[i,j] = ...
    # Jede Zeile (i) entspricht einer der 4N Gleichungen (Reihenfolge im Prinzip egal)
    # Jede Spalte entspricht dann den Koeffizienten a_0, ..., a_{N-1}, b_0, ...
    for i in range(N):
        M[i + 3*N, i + 3*N] = 1           # d_k
        r[i + 3*N] = f(10 * i / N - 5)    # f(x_k)
        
        M[i, i] = h**3                    # a_k * h^3
        M[i, i + N] = h**2                # b_k * h^2
        M[i, i + 2*N] = h                 # c_k * h
        r[i] = f(10 * (i+1) / N - 5) - f(10 * i / N - 5)    # f(x_{k+1}) - f(x_k)
        
        if i == 0: 
            M[N, N] = 1              # b_0
            M[2*N, N-1 + N] = 1      # b_{N-1}
            M[2*N, N-1] = 3*h        # 3*a_{N-1} * h
            
        else:
            M[i + N, i + 2*N] = 1             # c_k
            M[i + N, i-1 + 2*N] = -1          # -c_{k-1}
            M[i + N, i-1] = -3*h**2           # -3*a_{k-1} * h^2
            M[i + N, i-1 + N] = -2*h           # -2*b_{k-1} * h
            
            M[i + 2*N, i + N] = 1             # b_k
            M[i + 2*N, i-1 + N] = -1          # -b_{k-1}
            M[i + 2*N, i-1] = -3*h             # -3*a_{k-1} * h
    
    # Löse Mx = r mit einem eingebauten Löser (wirft einen Fehler, wenn M nicht regulär ist!)
    x = np.linalg.solve( M, r )
    
    # extrahiere a,b,c,d:
    # Zugriff auf Vektoreintraege: Z.B. x[0:N] ist der Sub-Vektor mit den Indizes 0,...,N-1                                  
    a = x[0:N]
    b = x[N:2*N]
    c = x[2*N:3*N]
    d = x[3*N:4*N]
        
    # gib alle 4 Vektoren zurück!
    return a,b,c,d


# Aufgabe (c)

def kubischerSpline( a, b, c, d, x ):
    if x < -5.0 or x > 5.0:
        raise ValueError(f"x muss im Intervall [-5, 5] liegen, aber war x = {x}.")
    
    i = int((x + 5) * N / 10)       # x=-5 --> i=0 und x=5 --> i=N-1
    if i == N:   # sonst IndexError für x == 5.0 möglich 
        i = N - 1
    xi = 10 * i / N - 5             # nächste Stützstelle mit x_i < x
    hval = x - xi                   # h-Wert für das gegebene x
    result = a[i] * hval**3 + b[i] * hval**2 + c[i] * hval + d[i]
    
    return result

####################################### START ###############################

# zuerst grobe Einteilung  

# print("Funktion, Spline und Differenz fuer N=2:")    

# N = 3

# a,b,c,d = berechneKoeffizienten( N )

# # Erstelle Diagramme mit den Graphen von f und s:
# # Lege dazu leere Listen an:
    
# xWerte = []
# fWerte = []
# sWerte = []

# # Differenzen
# dWerte = []

# schritte = 1000 # 1000 Messerte werden ins Diagramm eingetragen

# for i in range(schritte + 1): # aufpassen: 0, ..., schritte
#     x = i * 10.0 / schritte - 5
#     fx = f(x)
#     sx = kubischerSpline( a, b, c, d, x )
    
#     xWerte.append(     x ) # Liste verlangern
#     fWerte.append(    fx )
#     sWerte.append(    sx )
#     dWerte.append( fx-sx )
    
# plt.plot(xWerte,fWerte)
# plt.plot(xWerte,sWerte)
# plt.plot(xWerte,dWerte)
# plt.show()



print("Funktion, Spline und Differenz fuer N=200:")    

N = 200

a,b,c,d = berechneKoeffizienten( N )

# Erstelle Diagramme mit den Graphen von f und s:
# Lege dazu leere Listen an:
    
xWerte = []
fWerte = []
sWerte = []

# Differenzen
dWerte = []

schritte = 1000 # 1000 Messerte werden ins Diagramm eingetragen

for i in range(schritte + 1): # aufpassen: 0, ..., schritte
    x = i*10.0/schritte - 5
    fx = f(x)
    sx = kubischerSpline( a, b, c, d, x )
    
    xWerte.append(     x ) # Liste verlangern
    fWerte.append(    fx )
    sWerte.append(    sx )
    dWerte.append( fx-sx )
    
fig, ax = plt.subplots(figsize=(16, 9))
ax.axis([-5, 5, -0.05, 1.1])
ax.plot(xWerte, fWerte, label="f(x)")
# plt.show()
ax.plot(xWerte, sWerte, ls='--', label="Spline")
# plt.show()
ax.plot(xWerte, dWerte, label="Fehler")
ax.legend()
fig.tight_layout()
plt.show()

    
"""
These prints are for debugging the solver in an iPython console ...
for i in range(N):
    print(np.abs(a[i]*h**3 + b[i]*h**2 + c[i]*h - r[i]) < 1e-7)
    if i >= 1:
        print(np.abs(c[i] - c[i-1] - 3*a[i-1]*h**2 - 2*b[i-1]*h) < 1e-7)
        print(np.abs(b[1] - b[0] - 3*a[0]*h) < 1e-7)
    else:
        print(np.abs(b[0]) < 1e-7)
        print(np.abs(b[N-1] + 3*a[N-1]*h) < 1e-7)
    print(np.abs(d[i] - r[i + 3*N]) < 1e-7)
    print()
"""
