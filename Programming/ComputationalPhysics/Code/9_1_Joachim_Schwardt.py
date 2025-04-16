"""
Simulation der Langevin-Gleichung als Beispiel einer gerichteten Diffusion
 mit absorbierendem Rand und Vergleich zwischen dem theoretischen und
 stochastischen Verlauf (Plot 1). 
Als Norm wird hier das Verhaeltnis R(t) / R bezeichnet, also der Anteil der
 Teilchen, der zur Zeit t noch nicht absorbiert wurde. In den anderen drei
 Plotbereichen werden Norm, Mittelwert und Varianz der Simulation dynamisch
 dargestellt und mit der theoretischen Erwartung fuer den Fall ohne Rand
 verglichen. Die Parameter bezeichnen:
    D       =  Diffusionskonstante
    v_drift =  Driftgeschwindigkeit
    x_0     =  raeumlicher Startwert der Simulation
    xabs    =  Ort des absorbierenden Randes
    R       =  Anzahl der Realisierungen (also Teilchen)
    S       =  Anzahl der Rechenschritte pro dargestelltem Zeitintervall
    Nt      =  Anzahl dargestellter Zeitintervalle
"""

import numpy as np
import matplotlib.pyplot as plt
import functools

def gauss(x, mu=0, var=1):
    """
    Berechnet die Gauss-Kurve fuer ein Array an mu-Werten. 
    Das Ergebnis hat die Form 'shape == (len(x), len(mu))'.
    Es sind:
        mu == Mittelwerte
        var == Varianzen
    Beachte len(mu) == len(var). 
    Falls ein var[i] == 0 ist wird stattdessen 1e-6 verwendet.
    """
    x_tens = np.outer(x, np.ones_like(mu))      # anpassen von x und mu fuer
    mu_tens = np.outer(np.ones_like(x), mu)     # Differenzbildung
    
    var[var == 0] = 1e-6                        # geteilt durch 0 vermeiden
    norm_fak = np.sqrt(2*np.pi*var)             # Normierung
    return np.exp(-(x_tens - mu_tens)**2/(2*var)) / norm_fak

def langevin_theo(x, t, x0, xabs, v_drift, D, Absorb=False):
    """
    Bestimmt den theoretischen Zeitverlauf der Wahrscheinlichkeitsdichte
    einer Langevin-Gleichung mit Diffusion, Drift und absorbierendem Rand.
    Der Koeffizient des 'Korrektur'-Faktors, der den absorbierenden Rand
    darstellt, kann hier vereinfacht werden:
    G(xabs, x0 + vt, 2Dt) / G(xabs, 2xabs - x0 + vt, 2Dt) = 
        = exp(((xabs - x0 + vt)**2 - (xabs - x0 - vt)**2) / (4Dt)) = 
        = exp((4(xabs - x0) * vt) / (4Dt)) = 
        = exp((xabs - x0) * v / D)
    """
    mu_t = x0 + v_drift*t
    var_t = 2*D*t
    gauss_x = gauss(x, mu_t, var_t)
    if Absorb == True:
        mu_abs_t = 2*xabs - x0 + v_drift*t
        gauss_xabs = gauss(x, mu_abs_t, var_t)
        fak = np.exp((xabs - x0) * v_drift / D)   # e fuer die Parameter :)
        # konstant 0 fuer x > xabs
        cutoff = np.outer((np.sign(xabs - x) + 1) / 2, np.ones(len(t)))      
        return (gauss_x - fak * gauss_xabs) * cutoff
    else:
        return gauss_x

def langevin(x0, xabs, S, Nt, v_drift, D, R):  
    """
    Berechnet iterativ die Langevin-Gleichung fuer 'R' Realisierungen mit
    Drift und absorbierendem Rand. 
    Es ist 'dt = 1 / S', der Grenzwert S -> infty entspricht also dt -> 0.
        S = Anzahl Schritte pro Zeitintervall
        Nt = Anzahl Zeitintervalle
        x0 = Startposition
        xabs = Position des absorbierenden Randes
        v_drift = Driftgeschwindigkeit
        D = Diffusionskonstante
    """
    x = [np.zeros(R) + x0]
    for i in range(0, Nt * S, 1):
        # Anzahl noch nicht absorbierter Realisierungen
        R_i = len(x[i])
        # Iterationsvorschrift 
        x_i = x[i] + v_drift / S + np.sqrt(2*D / S) * np.random.randn(R_i)
        # absorbierender Rand
        x_i_abs = x_i[np.where(x_i < xabs)]
        x.append(x_i_abs)
    return x

def p_dynamic(event, axis, x, t, x0, xabs, v_drift, D, R, S, Nt, Nbins):
    """
    Bei einem Klick mit der linken Maustaste in einem der Plotfenster wird
    die Langevin-Gleichung mit absorbierendem Rand simuliert, sowie der
    theoretische Verlauf mit bzw. ohne Rand (Plot 1). In den anderen drei
    Plots sind Norm, Mittelwert und Varianz dynamisch dargestellt, wieder im
    Vergleich mit dem theoretischen Verlauf ohne Rand. Mittelwert und
    Varianz werden dabei direkt aus den Orten der Teilchen berechnet:
        mu = 1/N * sum(x_i)_i=1 bis N
        var = 1/(N-1) * sum((x_i - mu)**2)_i=1 bis N
    Ein weiterer Klick leert alle Plots und startet eine neue Simulation
    (Zufallszahlen -> kein identischer Verlauf).
    """    
    # Test, ob Klick im Plotfenster mit linker Maustate und Zoom deaktiviert
    tool_mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and tool_mode == '':
        for ax in axis:
            ax.lines = []        # Plotbereiche leeren
        
        p_theo_abs = langevin_theo(x, t, x0, xabs, v_drift, D, True)
        p_theo = langevin_theo(x, t, x0, xabs, v_drift, D, False)
        x_t = langevin(x0, xabs, S, Nt, v_drift, D, R)
        
        # Erster Plotbereich
        # ax.bar erzeugt bei jedem Mausklick ein weiteres label -> ax.plot
        axis[0].plot(0, 2, c='r', ls='', marker='s', label='Simulation')
        # Theoretischer Verlauf ohne absorbierenden Rand
        p_theo_plot = axis[0].plot(x, p_theo[:, 0], c='k', lw=0.8,
                                   label='Theorie ohne Rand')
        # ... und mit absorbierendem Rand
        p_theo_abs_plot = axis[0].plot(x, p_theo_abs[:, 0], c='b', lw=1,
                                       label='Theorie mit Rand')
        
        # Zweiter Plotbereich
        dx = x[1] - x[0]        # fuer die Norm des theoretischen Verlaufs
        Norm_tn_plot = axis[1].plot([0, 1], [1, 1], c='firebrick', lw=1,
                                    label='Norm')
        Norm_theo_plot = axis[1].plot([0, 1], [1, 1], c='k', lw=1,
                                      label='Theorie ohne Rand')
        
        # Dritter Plotbereich
        M_tn_plot = axis[2].plot([0, 0], [0, 1], c='g', lw=1,
                                 label='Mittelwert')
        M_theo_plot = axis[2].plot([0, 0], [0, 1], c='k', lw=1,
                                   label='Theorie ohne Rand')
        
        # Vierter Plotbereich
        V_tn_plot = axis[3].plot([0, 0], [0, 1], c='orange', lw=1,
                                 label='Varianz')
        V_theo_plot = axis[3].plot([0, 0], [0, 1], c='k', lw=1,
                                   label='Theorie ohne Rand')
        
        for ax in axis:
            ax.legend()
        
        # Dynamik
        for i in range(Nt + 1):
            axis[0].patches = []
            # Zeitschritte fuer Simulation auswaehlen
            x_t_i = x_t[i * S]
            
            # Histogram
            hist_i, bins_i = np.histogram(x_t_i, bins=Nbins)
            width_i = bins_i[1] - bins_i[0] 
            # Normierung auf 1 (bzw. kleiner 1, da R(t) <= R)
            hist_i = hist_i / (R * width_i)         
            
            # Norm des Histograms
            Norm_tn = sum(hist_i) * width_i
            Norm_tn_plot[0].set_ydata(Norm_tn)
            # nur Werte mit x <= xabs
            Norm_theo = dx * sum(p_theo[x <= xabs, i])
            Norm_theo_plot[0].set_ydata(Norm_theo)
            
            # Erwartungswert des Histograms
            M_tn = sum(x_t_i) / len(x_t_i)
            M_tn_plot[0].set_xdata(M_tn)
            M_theo = x0 + v_drift * t[i]
            M_theo_plot[0].set_xdata(M_theo)
            
            # Varianz des Histograms
            V_tn = sum((x_t_i - M_tn)**2) / (len(x_t_i) - 1)
            V_tn_plot[0].set_xdata(V_tn)
            V_theo = 2*D * t[i]
            V_theo_plot[0].set_xdata(V_theo)
            
            # Theoretische Verlaeufe auf 1 normiert plotten
            if i == 0:    # sonst geteilt durch 0 moeglich
                Norm_theo_abs = 1
            else:
                Norm_theo_abs = dx * sum(p_theo_abs[:, i])
            p_theo_abs_norm = p_theo_abs[:, i] / Norm_theo_abs
            p_theo_plot[0].set_ydata(p_theo[:, i])
            p_theo_abs_plot[0].set_ydata(p_theo_abs_norm)
            
            # Histogram auf 1 normiert plotten
            hist_i_norm = hist_i / Norm_tn
            axis[0].bar(bins_i[:-1], hist_i_norm, align='edge',
                        width=width_i, fc='r')
            
            
            event.canvas.flush_events()
            event.canvas.draw()

def main():
    print(__doc__)
    
    x0 = 0                              # Startwert
    xabs = 15                           # absorbierender Rand
    xmax = xabs + 5                     # Plotbereich der Simulation
    xmin = -xmax
    Nx = 200                            # Anzahl x-Werte fuer Theorie-Plot
    Nbins = 50                          # Anzahl bins fuer Histogram
    
    v_drift = 0.1                       # Driftgeschwindigkeit
    D = 1.5                             # Diffusionskonstante
    
    R = 10000                           # Anzahl Realisierungen
    S = 100                             # Anzahl Schritte pro Zeitintervall
    tmin, tmax = 0, 40                  # zeitlicher Bereich der Simulation
    Nt = tmax - tmin                    # Anzahl Zeitintervalle
    
    t = np.linspace(tmin, tmax, Nt + 1) 
    x = np.linspace(xmin, xmax, Nx)
    
    suptitle_txt = (r'Simulation der Langevin-Gleichung mit: $v_{drift}$=' + 
                    '{}, D={}, $x_0$={}, R={}, S={}, '
                    .format(v_drift, D, x0, R, S) + '$N_t$=' + 
                    '{}, '.format(Nt) + '$x_{abs}$=' + '{}'.format(xabs))
    
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axis = axis.reshape(4)     # Indizierung mit 0123 statt 00, 01, 10, 11
    plt.suptitle(suptitle_txt)
    
    Axis_dict = {0: {'title': 'Simulation mit $R$ Realsierungen', 
                     'axis': [xmin, xmax, 0, 0.1], 
                     'xlabel': 'x', 
                     'ylabel': 'Wahrscheinlichkeitsdichten $P(x, t)$'},
                 1: {'title': 'Norm: R(t) / R', 
                     'axis': [0, 1, 0, 1], 
                     'xlabel': '',
                     'ylabel': 'Anteil Teilchen mit $x < x_{abs}$'},
                 2: {'title': 'Mittelwert: $\mu$(t)', 
                     'axis': [xmin, 1.1 * (x0 + v_drift*tmax), 0, 1], 
                     'xlabel': 'x', 
                     'ylabel': ''},
                 3: {'title': 'Varianz: $v(t)$', 
                     'axis': [0, 1.1 * 2*D*tmax, 0, 1], 
                     'xlabel': 'x', 
                     'ylabel': ''}}
    
    for i in range(len(axis)):
        axis[i].set_title(Axis_dict[i]['title'], size='small')
        axis[i].axis(Axis_dict[i]['axis'])
        axis[i].set_xlabel(Axis_dict[i]['xlabel'], size='small')
        axis[i].set_ylabel(Axis_dict[i]['ylabel'], size='small')
    axis[1].tick_params(axis='x', which='both', bottom=False,
                        labelbottom=False)
    axis[2].tick_params(axis='y', which='both', left=False, labelleft=False)
    axis[3].tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Implementierung der Mausinteraktion
    klick_funktion = functools.partial(p_dynamic, axis=axis, x=x, t=t,
                                       x0=x0, xabs=xabs, v_drift=v_drift,
                                       D=D, R=R, S=S, Nt=Nt, Nbins=Nbins)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    
    plt.show()

if __name__ == "__main__":
    main()
    """
a) S = 100:
    Der Einfluss des Randes wird etwa bei t = 10 erkennbar, also gerade
        dann, wenn die theoretische Kurve an der Stelle 'x = xabs' merklich
        groesser 0 wird. Die Norm faellt dann schneller und die Bewegung des
        Mittelwertes in positive x_richtung kehrt sich sogar um. Die Varianz
        steigt ab t = 10 deutlich langsamer.
    
b) S = 1 (--> dt = 1):
    In der Naehe des Randes waechst die Verteilung erkennbar ueber die
        theoretissche Vorhersage hinaus. Betrachte zunaechst den Fall 
        S = 100, allerdings finde die Kontrolle 'x < xabs' nur alle 100
        Schritte statt. Dann ist das Verhalten gut verstaendlich, denn wenn
        bei einem Teilschritt 'x >= xabs' ist, so kann das Teilchen bis die
        Kontrolle stattfindet auch wieder zurueck kommen, was zu einer
        hoeheren Dichte in der Naehe des Randes fuehrt. 
    Der Fall S = 1 mit einer Kontrolle bei jedem Schritt ist dann fast
        aequivalent, nur dass man eine Zufallszahl anstatt einer Summe uber
        S Zahlen geteilt durch sqrt(S) verwendet.
    
c) v_drift = 0.5:
    Der Mittelwert ist jetzt deutlich im positiven Bereich und der Drift ist
        klar erkennbar. Zudem sinken Varianz und Norm deutlich. Letzteres
        liegt daran, dass durch den staerkeren Drift in Richtung Rand
        natuerlich auch mehr Teilchen absorbiert werden.
    """


