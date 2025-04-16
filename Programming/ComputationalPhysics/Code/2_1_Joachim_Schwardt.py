"""Numerische Differentiation:
   Stellt für eine beliebige differenzierbare Funktion den relativen Fehler
   der Naeherungsverfahren Vorwaertsdifferentiation, Zentraldifferentiation
   und Extrapolierte Differentiation zur analytischen Ableitung in
   Abhängigkeit vom Paramter h der Naeherungen logarithmisch dar.
   Voreingestellt ist f = Arctan(x**2)."""

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from textwrap import wrap

def function(x):
    """Funktion, für die die Näherungsverfahren getestet werden sollen."""
    return np.arctan(x**2)

def sp_function(x):
    """Sympy Aequivalent von 'function' um die analytische Ableitung
    bestimmen zu können.
    Anmerkung: Ginge das hier auch einfacher? Bei Funktionen, die man
    explizit (also z.B. x**2+3) angeben kann, reicht es 'function' zu
    verwenden. Wie könnte man das mit (z.B.) Arctan machen?"""
    return sp.atan(x**2)  
        
def vorwaerts_diff(f=np.arctan, x0=1/3, h=1e-6):
    """Berechnet die Vorwaertsdifferenz der Funktion 'f' an der Stelle
    x0."""
    return (f(x0+h) - f(x0)) / h

def zentral_diff(f=np.arctan, x0=1/3, h=1e-6):
    """Berechnet die Zentraldifferenz der Funktion 'f' an der Stelle x0."""
    return (f(x0+h/2) - f(x0-h/2)) / h

def extrapol_diff(f=np.arctan, x0=1/3, h=1e-6):
    """Berechnet die Extrapolierte Differenz der Funktion 'f' an der Stelle
    x0."""
    return (8*(f(x0+h/4) - f(x0-h/4)) - (f(x0+h/2) - f(x0-h/2))) / (3*h)

def analytic_diff(f_sp=sp.atan, x0=1/3):
    """Berechnet den analytischen Wert der Ableitung f'(x0)."""
    x = sp.Symbol('x')
    return sp.diff(f_sp(x), x, 1).subs(x, x0)

def rel_error(mode=0, f=np.arctan, f_sp=sp.atan, x0=1/3, h=1e-6):
    """Berechnet den Betrag der Abweichung zum analytischen Wert der
    Ableitung von Arctan(x0) für die untersuchten Näherung."""
    ana_diff = analytic_diff(f_sp=f_sp, x0=x0)
    if mode == 0:      # Vorwaertsdifferenz
        return abs((vorwaerts_diff(f, x0, h) - ana_diff) / ana_diff)
    elif mode == 1:    # Zentraldifferenz
        return abs((zentral_diff(f, x0, h) - ana_diff) / ana_diff)
    elif mode == 2:    # Extrapolierte Differenz
        return abs((extrapol_diff(f, x0, h) - ana_diff) / ana_diff)
    else:
        print("Wrong mode! Choose mode from [0, 1, 2].")
        
def error_expectancy(f_sp=sp.atan, mode=0, x0=1/3):
    """Berechnet analytisch das erwartete Verhalten des relativen Fehlers
    für die drei Näherungen. Dazu wird der Vorfaktor der fuehrenden Ordnung
    in h bestimmt. Die Koeffizienten (2, 24, 7680) kommen aus den Taylor-
    Entwicklungen der einzelnen Verfahren. Division durch ana_diff gibt dann
    den erwarteten relativen Fehler (bzw. den Vorfaktor zur h**(alpha)
    Abhängigkeit)."""
    x = sp.Symbol('x')
    ana_diff = analytic_diff(f_sp=f_sp, x0=x0)
    if mode == 0:       # Vorwaertsdifferenz
        return sp.diff(f_sp(x), x, 2).subs(x, x0) / (2*ana_diff)    
    elif mode == 1:     # Zentraldifferenz
        return sp.diff(f_sp(x), x, 3).subs(x, x0) / (24*ana_diff)    
    elif mode == 2:     # Extrapolierte Differenz
        return sp.diff(f_sp(x), x, 5).subs(x, x0) / (7680*ana_diff)  
    else:
        print("Wrong mode! Choose mode from [0, 1, 2].")
        

def main():
    print(__doc__)
    x0 = 1/3
    f = function
    f_sp = sp_function
    
    h = np.linspace(-10, 0, 200)
    h_log = 10**h    # Intervall [1e-10, 1] mit logarithmierten Abständen
    
    mode_dict = {0:'Vorwaertsdifferenz', 1:'Zentraldifferenz', 
    2:'Extrapolierte Differenz'}
    # RGB Farbcodes für die 3 modes
    rgb_color_array = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    # erwarteter Exponent h**(alpha) für das jeweilige Näherungsverfahren 
    expected_exponent = [1, 2, 4] 
       
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, xscale='log', yscale='log')
    
    ax.axis([0.5*min(h_log), 2*max(h_log), 1e-16, 10])  # Achsengrenzen
    ax.set_xlabel("h")                           # Beschriftung x,y-Achse
    ax.set_ylabel(r"$\Delta f'(x_0, h)$")        
    plt.suptitle("Numerische Differentiation")   # Ueberschrift
    plt.title("\n".join(wrap("Betrag des relativen Fehlers zum \
                             analytischen Wert der Ableitung von f({})",
                             50)).format(round(x0, 3)), size='small')
    # Etwas umstaendliche Konstruktion um die 79 Zeichen pro Zeile
    # einzuhalten ohne einige Leerzeichen im String zu haben. 
    # Ginge das auch etwas einfacher? 
    
    for mode in range(3):
        # Plot der relativen Fehler mit Beschriftung 
        ax.plot(h_log, rel_error(mode=mode, f=f, f_sp=f_sp, x0=x0, h=h_log),
                lw=0, marker='o', ms=2, mew=0, c=rgb_color_array[mode])
        # Plot der erwarteten relativen Fehler
        ax.plot(h_log, abs(error_expectancy(f_sp=f_sp, mode=mode, x0=x0)) *
                h_log**expected_exponent[mode], c=rgb_color_array[mode],
                alpha=0.6, lw=1, label=mode_dict[mode])
                # Label erst hier, weil die Farbe der kleinen 'marker' in 
                # der Legende schwer zu erkennen ist.
    
    ax.grid(True)
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
    """
    zu (a):
        Vorwärtsdifferenz A_V(x):
        Sei eps=Rundungsfehler (etwa 1e-16, s.Skript) und h der
        Diskretisierungsfehler. Dann muss bei der Differenz im Zähler ein
        Rundungsfehler der Ordnung O(eps) berücksichtigt werden: 
            A_V(x) = (f(x+h)-f(x) + O(eps))/h = f'(x)+O(h) + O(eps/h)
        Die Näherung hat also einen Term der Ordnung 1/h.
        Man findet zudem die Abschätzung für die Groessenordnung des
        optimalen Wertes für h aus:
            h > eps/h , also h > sqrt(eps) = 1e-8 
        Dieser Wert stimmt gut mit dem beobachteten Verhalten ueberein.
    zu (b):
                      Vorwaertsdiff ___ Zentraldiff ___ Extrapolierte Diff
        optimales h:     1e-8       ___    2e-5     ___       5*1e-3
        rel. Fehler:    3*1e-9      ___   5*1e-12   ___       1e-14
        
        Der optimale Wert von h ist auch von x0 und insbesondere natürlich
        von der Funktion abhängig! Die angegebenen Werte sind nur am Plot 
        für x0=1/3 abgelesen."""
"""
Man kann den Knick auch analytisch abschätzen.
Der sog. "Maschinenfehler" ist eps ~ 1e-16.

Vorwärts: 
f(x+h) - f(x) ~ hf' + O(h^2) + eps
--> v-diff = f' + O(h) + eps/h
--> h_crit ~ eps^{1/2} ~ 1e-8

Zentral:
f(x+h/2) - f(x-h/2) ~ hf' + O(h^3) + eps
--> z-diff = f' + O(h^2) + eps/h
--> h_crit ~ eps^{1/3} ~ 5e-6

(für genauere Abschätzungen müsste man den Vorfaktor des 'O(h...)'-Terms kennen)
"""
