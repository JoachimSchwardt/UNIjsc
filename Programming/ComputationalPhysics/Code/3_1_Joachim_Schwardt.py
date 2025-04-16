"""Numerische Integration:
   Stellt fuer eine beliebige integrierbare Funktion den relativen Fehler
   der Naeherungsverfahren Mittelpunktregel, Trapezregel und Simpsonregel
   zum analytischen Integral in Abhaengigkeit des Parameters h. 
   Der analytisch erwartete relative Fehler ist jeweils farblich passend als
   gestrichelte Linie dargestellt.
   Voreingestellt ist f = cosh(2*x), a = -pi/2 und b = pi/4."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.misc import derivative

def integrand_gen(preset=0):
    """Erzeugt die Funktion über die integriert werden soll. 
    Abhängig vom Parameter 'preset = 0, 1, 2' sind die Integranden von
    Aufgabenblatt 3 voreingestellt."""
    def integrand(x):
        if preset == 0:      # cosh(2*x)
            return np.cosh(2*x)
        elif preset == 1:    # exp(-100*x**2)
            return np.exp(-100*x**2)   
        elif preset == 2:    # Heaviside(x)
            return 0.5 * (1 + np.sign(x))
        else:                # beliebige Funktion
            return 1 + 0*x
    return integrand

def integral(f, a=-np.pi/2, b=np.pi/4, N=[999], mode=0):
    """Verwendet je nach mode ein anderes numerisches Verfahren zur
    Bestimmung des Integrals einer Funktion von a bis b:
        mode=0: Mittelpunktregel unter Verwendung von N Stützstellen in der
                jeweiligen Intervallmitte.
        mode=1: Trapezregel unter Verwendung von N+1 Stützstellen am
                jeweiligen Intervallanfang.
        mode=2: Simpsonregel unter Verwendung von 2N+1 Stützstellen."""
    h = (b-a) / N
    erg = np.zeros(len(N))
    if mode == 0:      # Mittelpunktregel
        for count in range(len(N)):
            N_elem = N[count]
            h_elem = h[count]
            i = np.linspace(a+h_elem/2, b-h_elem/2, N_elem)
            erg[count] = h_elem * np.sum(f(i))
        return erg
    elif mode == 1:    # Trapezregel
        for count in range(len(N)):
            N_elem = N[count]
            h_elem = h[count]
            i = np.linspace(a, b, N_elem+1)
            erg[count] = h_elem * (np.sum(f(i)) - (f(a)+f(b)) / 2)
        return erg
    elif mode == 2:    # Simpsonregel
        for count in range(len(N)):
            N_elem = N[count]
            h_elem = h[count]
            # N+1 ganzzahlige Stuetzstellen
            i = np.linspace(a, b, N_elem+1)     
            # N halbzahlige Stuetzstellen
            k = np.linspace(a+h_elem/2, b-h_elem/2, N_elem) 
            erg[count] = h_elem/3 * (np.sum(f(i)) + 2*np.sum(f(k)) 
                                  - (f(a)+f(b)) / 2)
        return erg
    else:
        print("Wrong mode! Choose mode from [0, 1, 2].")
        
def analytisch(f, a=-np.pi/2, b=np.pi/4, preset=0):
    """Analytische Ergebnisse der Integrale und scipy.integrate.quad() als
    'Richtwert' für das Integral eines beliebigen Integranden."""
    if preset == 0:      # cosh(2*x)
        return 0.5 * (np.sinh(2*b) - np.sinh(2*a))
    elif preset == 1:    # exp(-100*x**2)
        return np.sqrt(np.pi/100)   # Ergebnis fuer a, b = +/-_infty!
    elif preset == 2:    # Heaviside(x)
        return b - a * 0.5 * (1 + np.sign(a))  # Fuer a>0 ist I = b-a 
    else:
        return quad(f, a, b)[0]

def rel_error(f, a=-np.pi/2, b=np.pi/4, N=[999], mode=0, preset=0):
    """Berechnet den Betrag der relativen Abweichung des jeweiligen
    Naeherungsvarfahrens (mode = 0, 1, 2) zum analytischen Ergebnis des
    Integrals der Funktion f von a bis b. 
    Parameter ist die Anzahl der Stützstellen N."""
    ana_int = analytisch(f=f, a=a, b=b, preset=preset)
    return abs((integral(f, a, b, N, mode) - ana_int) / ana_int)

def error_expectancy(f, a=-np.pi/2, b=np.pi/4, N=[999], mode=0, preset=0):
    """Berechnet den Vorfaktor des analytisch erwarteten Fehlers der
    Naeherungsverfahren. Division durch ana_int gibt dann
    den erwarteten relativen Fehler (bzw. den Vorfaktor zur h**(alpha)
    Abhaengigkeit)."""
    ana_int = analytisch(f=f, a=a, b=b, preset=preset)
    if mode == 0:    # Mittelpunktregel
        return (derivative(f, b, n=1, dx=1e-5, order=3) - 
                derivative(f, a, n=1, dx=1e-5, order=3)) / (24*ana_int)
    elif mode == 1:  # Trapezregel
        return (derivative(f, b, n=1, dx=1e-5, order=3) - 
                derivative(f, a, n=1, dx=1e-5, order=3)) / (12*ana_int)
    elif mode == 2:  # Simpsonregel
        return (derivative(f, b, n=3, dx=1e-5, order=5) - 
                derivative(f, a, n=3, dx=1e-5, order=5)) / (2880*ana_int)
    else:
        print("Wrong mode! Choose mode from [0, 1, 2].")

def main():
    print(__doc__)
    
    preset = 0   # 0, 1, 2 fuer die Funktionen aus a, b, c
    f = integrand_gen(preset)
    
    a, b = -np.pi/2, np.pi/4
    
    N = np.linspace(0, 5, 300)
    N_log = np.int32(10**N)   # logarithmiertes Intervall [1, 1e5]
    h_log = (b-a) / N_log
    
    mode_dict = {0:'Mittelpunktregel', 1:'Trapezregel', 2:'Simpsonregel'}
    # RGB Farbcodes fuer die 3 modes
    rgb_color_array = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    # erwarteter Exponent h**(alpha) fuer das jeweilige Naeherungsverfahren 
    expected_exponent = [2, 2, 4]
       
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, xscale='log', yscale='log')
    
    ax.axis([0.5*min(h_log), 2*max(h_log), 1e-20, 10])  # Achsengrenzen
    ax.set_xlabel("h")                       # Beschriftung x,y-Achse
    ax.set_ylabel(r"$\Delta\, I(a, b, h)$")        
    plt.suptitle("Numerische Integration")   # Ueberschrift
    txt = "Betrag des relativen Fehlers zum analytischen Wert des Integrals"
    txt2 = " von f von a={} bis b={}".format(round(a, 3), round(b, 3))
    plt.title(txt + txt2, size='small')
              
    for mode in range(3):
        # Plot der relativen Fehler mit Beschriftung 
        rel_err_plot = rel_error(f, a, b, N_log, mode, preset)
        ax.plot(h_log, rel_err_plot, lw=0, marker='o', ms=2, mew=0,
                c=rgb_color_array[mode], label=mode_dict[mode])
        
        # Plot der erwarteten relativen Fehler
        expect_plot = abs(error_expectancy(f, a, b, N_log, mode, preset)) \
            * h_log**expected_exponent[mode]
        ax.plot(h_log, expect_plot, c=rgb_color_array[mode], alpha=0.6,
                lw=1, ls='dashed')
    
    ax.grid(True)
    ax.legend(numpoints=4)
    plt.show()
    
if __name__ == "__main__":
    main()
    """
a) cosh(2*x):
    Mittelpunktregel:
        'Nette' Funktion -> Uebereinstimmung mit dem erwarteten
        Skalierungsverhalten, insbesondere auch im Bezug auf den jeweiligen
        Vorfaktor.
    Trapezregel:
        Gleiches Skalierungsverhalten wie bei der Mittelpunktregel,
        allerdings bei gleichem Aufwand um einen Faktor 5-10 ungenauer
        (y-Shift im log-Plot).
    Simpsonregel:
        Auch hier alles wie erwartet, zumindest bis der relative Fehler in
        die Groessenordnung der Rechengenauigkeit von eps=1e-16 kommt. 
        Das Problem tritt also erst in der Differenzbildung in der Funktion
        'rel_error' auf. Im Gegensatz zur Differentiation zeigt sich hier
        allerdings kein Skalierungverhalten, weil nicht durch 'h' 
        (bzw. eine kleine Groesse) geteilt wird.
b) exp(-100*x**2):
    Die drei Verfahren zeigen ein nahezu identsches, im log-log-Plot nicht- 
    lineares Verhalten, das nicht mit der analytischen Erwartung
    übereinstimmt (Der Plot einer verschobenen Gauss-Funktion sieht
    verdaechtig aehnlich aus). Bei etwa h=5*1e-2 wird eine relative
    Genauigkeit von etwa 1e-16 erreicht, fuer kleinere 'h' bleibt dieser
    Wert relativ konstant. 
    Die Funktion ist nur in einem sehr kleinen Intervall (etwa [-0.1, 0.1])
    wesentlich von 0 vreschieden. Beim Integral über dieses Intervall ergibt
    sich auch das erwartete Verhalten, aehnlich dem von cosh(2*x). Durch die
    Integration über ein viel groesseres Intervall landen allerdings zu
    wenige Stuetzstellen in dem kleinen Bereich um 0, der zum Integral
    beitraegt (zumindest fuer relativ grosse 'h').
c) Heaviside:
    Auch hier zeigen alle Verfahren das gleiche, in diesem Fall aber wie
    'h**1' skalierende Verhalten, wobei die Mittelpunktregel etwa um einen
    Faktor 5 ungenauer ist. Analytisch betrachtet sollte der Fehler immer
    gleich dem kleinsten Abstand einer Stuetzstelle zur 0+ sein (d.h. eine
    Stelle auf der positiven Seite), also proportional zum Abstand 'h'
    zweier Stuetzstellen sein.
    Es gibt allerdings auch diskrete Werte für 'h', bei denen das Ergebnis
    die relative Genauigkeit von 1e-16 erreicht. Diese sind stark von den
    Intervallgrenze abhaengig. Rundet man 'b' auf 10 Stellen steigt der
    relative Fehler dieser Werte bereits auf 1e-12. Andere ganzzahlige
    Vielfache von 'b=pi/4' fuehren wieder auf ein aehnliches Verhalten.
    Hierbei ist wohl nichts besonders an pi; Es scheint nur wichtig zu sein,
    dass das Verhaeltnis von 'a' zu 'b' rational mit einem relativ kleinen
    Nenner ist. Vermutlich kommen einzelne Stuetzstellen dann auch fuer
    groessere 'h' nahe an 0+ heran. 
    """