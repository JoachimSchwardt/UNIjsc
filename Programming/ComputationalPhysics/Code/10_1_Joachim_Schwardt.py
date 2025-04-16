"""
Visualisierung des zweidimensionalen Ising-Modells durch die dynamische
    Darstellung eines n*n-Gitter Spinzustandes (mit periodischen
    Randbedingungen) bei einer Temperatur 'tau' (n=50).
    
Klick im ersten Plotbereich:
    Fuehrt fuer den gerade vorhandenen Zustand 10 Monte-Carlo Schritte
    durch, die dynamisch dargestellt werden. Im zweiten Plotbereich wird
    zudem jeweils der aktuelle Wert der mittleren Magnetisierung 'm' bei der
    aktuellen Temperatur 'tau' dargestellt.
    
Klick im zweiten Plotbereich:
    Erstellt fuer den ausgewaehlten Wert von 'm' eine zufaellige Verteilung,
    deren mittlere Magnetisierung moeglichst nahe an 'm' herankommt. 
    Zudem wird die Temperatur aktualisiert. 
    
Im zweiten Plotbereich ist zudem die analytische Vorhersage fuer 'm' im
     unendlich grossen System als Funktion von der (einheitenlosen)
     Temperatur dargestellt.

Farbcode fuer den Wert eines Spins:
    Gelb == +1
    Blau == -1
"""

import numpy as np
import matplotlib.pyplot as plt
import functools

class IsingZustand(object):
    """Klasse zur Speicherung von Zustand, Temperatur und Magnetisierung."""
    def __init__(self, spins, tau, m):
        """
        Initialisierung: 
            Spins == 'spins'
            Temperatur == 'tau' 
            mittlere Magnetisierung == 'm'
        """
        self.spins = spins
        self.tau = tau
        self.m = m
    
    def mean(self):
        """Berechnet die mittlere Magnetisierung des Zustandes."""
        return np.mean(self.spins)
        
def m_infty(tau_array):
    """
    Berechnet die theoretische Vorhersage fuer die mittlere Magnetisierung
    pro Spin im unendlichen System.
    """
    tau_crit = 2 / np.arcsinh(1)     # kritische Temperatur
    m_inf = np.zeros(len(tau_array))
    
    # sinh(x)**4 proportional zu e**(4x) --> x sollte kleiner 20 sein
    tau_copy = np.copy(tau_array)    # Kopie um Original nicht zu veraendern
    tau_copy[tau_copy < 0.1] = 0.1
    
    # m_inf = 0 fuer tau >= tau_crit
    inds = np.where(tau_copy < tau_crit)
    m_inf[inds] = (1 - 1 / np.sinh(2 / tau_copy[inds])**4)**(1/8)
    
    return m_inf

def m_zustand_gen(m, n):
    """
    Erzeugt einen zufaelligen Zustand der Groesse 'n * n' mit einer
    mittleren Magnetisierung von 'm' und unabhaenigen Spins.
    Sei 'k' die Anzahl der Spins mit Wert -1 fuer ein vorgegebenes 'm':
        m = sum(x_i) / N  
        ->  m*N = k*(-1) + (N-k)*(+1) = N - 2*k
        ->  k = N * (1-m) / 2
    """
    if m >= 1:         # alle Spins gleich +1
        m_zustand = np.ones(n * n)
    elif m <= -1:      # alle Spins gleich -1
        m_zustand = -np.ones(n * n)
    else:              # gemischter Zustand
        m_zustand = np.ones(n * n)
        m_zustand[0:int(n*n * (1-m) / 2)] = -1     # siehe docstring
        np.random.shuffle(m_zustand)
    
    return m_zustand.reshape((n, n))

def mc_schritt2(zustand, n):
    """
    Berechnet die Matrix-Elemente H_ij = S_ij * sum(naechste 4 Nachbarn)
    indem vier Kopien von 'S' addiert werden, die jeweils in eine der vier
    moeglichen Richtungen verschoben wurden. 'np.roll' realisiert dabei
    automatisch die periodischen Randbedingungen.
    Die Wahrscheinlichkeit dafuer, dass sich der Spin S_ij umdreht, ist
    durch exp(-2*H_ij / tau) gegeben. Es wird also fuer jede solche
    Wahrscheinlichkeit ein Vergleich mit einem zufaelligen Element aus 
    [0, 1) durchgefuehrt. Ist der zufaellige Wert kleiner, so dreht sich der
    Spin um.
    """
    S = zustand.spins    # 'S' ist uebersichtlicher als 'zustand.spins'
    
    # Berechnung von 'H' in zwei Schritten, siehe docstring
    H = S * (np.roll(S, 1, axis=0) + np.roll(S, -1, axis=0) + 
             np.roll(S, 1, axis=1) + np.roll(S, -1, axis=1))
    H = np.exp(-2 * H / zustand.tau)
    
    # Umsetzung eines probabilistischen Spinflip
    random_array = np.random.uniform(0, 1, np.shape(H))
    S[H > random_array] *= -1          
    return S

def mc_schritt(zustand, n):
    """
    Die Wahrscheinlichkeit dafuer, dass sich der Spin 'S_ij' umdreht, ist
    durch exp(-2*H_ij / tau) gegeben. Fuer n**2 zufaellig gewaehlte Spins
    wird 'H' als die Summe der vier naechsten Nachbarn mal den gewaehlten
    Spin berechnet. Die periodischen Randbedingungen werden durch 
    Modulo-Arithmetik beim Indizieren des Spin-Arrays realisiert.
    Das Erstellen eines 'Sammelarrays' fuer Indizes bzw. Zufallszahlen
    verringert die Anzahl an Funktionsaufrufen in der Schleife.
    """
    S = zustand.spins    # 'S' ist uebersichtlicher als 'zustand.spins'
    
    # Arrays mit Indizes und Zufallszahlen 
    index_array = np.random.randint(0, n, size=(2, n*n))
    random_array = np.random.uniform(0, 1, size=n*n)
    # n**2 zufaellige Eintraege auf Spinflip pruefen
    for count in range(n*n):
        i, j = index_array[:, count]
        # Berechnung von 'H' fuer den ausgewaehlten Spin
        H = S[i, j] * (S[(i+1) % n, j] + S[i, (j+1) % n ] + 
                       S[(i-1) % n, j] + S[i, (j-1) % n])
        # Wahrscheinlichkeit fuer Spinflip
        W = np.exp(-2 * H / zustand.tau)  
        
        # Spinflip mit 'Wuerfel'
        if W > random_array[count]:
            S[i, j] *= -1 
                
    return S
        
def mausklick(event, axis, Nt, n, zustand, spins_img, tau_array,
              m_infty_plot, plot_dict):
    """
    Mausklick muss im 'default'-Modus (kein Zoom etc.) erfolgen.
    Klick im ersten Plotbereich:
        Dynamische Darstellung von 10 Monte-Carlo Schritte, ausgehend vom
        aktuellen Zustand. Im zweiten Plotbereich wird bei jedem Schritt der
        Mittelwert gezeichnet.
    Klick im zweiten Plotbereich:
        Auswahl eines neuen Wertepaares (m, tau). Im ersten Plotbereich wird
        ein zufaelliger Zustand erstellt, der dem gewaehlten 'm' entspricht.
    """
    tool_mode = event.canvas.toolbar.mode
    # Mausklick im ersten Plotbereich
    if event.button == 1 and event.inaxes == axis[0] and tool_mode == '':
        marker = axis[1].plot(zustand.tau, zustand.m, c='r', ls='',
                              marker='o', mew=0, ms=3)
        
        # Dynamik
        for i in range(Nt):
            # Monte-Carlo Schritt ausfuehren
            zustand.spins = mc_schritt(zustand, n)
            # Neuen Zustand im ersten Plot darstellen
            spins_img.set_data(zustand.spins)
            
            # Mittelwert berechnen und im zweiten Plot darstellen
            zustand.m = zustand.mean()
            marker[0].set_ydata(zustand.m)
            # Titel aktualisieren
            axis[1].set_title(plot_dict[1]['title'] + '{}'
                              .format(round(zustand.m, 4)), size='small')
            
            event.canvas.flush_events()
            event.canvas.draw()
    
    # Mausklick im zweiten Plotbereich
    if event.button == 1 and event.inaxes == axis[1] and tool_mode == '':
        # Plotbereich leeren und Theorie erneut zeichnen
        axis[1].lines = []
        axis[1].plot(tau_array, m_infty_plot, c='b', lw=1)
        axis[1].plot(tau_array, -m_infty_plot, c='b', lw=1)
        
        # Mauskoordinaten als 'tau' und 'm'
        tau0 = event.xdata
        m0 = event.ydata
        
        # Zufaelligen Zustand mit Mittelwert von etwa m0 erstellen
        zustand.spins = m_zustand_gen(m0, n)
        zustand.tau = tau0
        # Tatsaechlichen Mittelwert berechnen
        zustand.m = zustand.mean()
        
        # Zustand im ersten Plot zeigen
        spins_img.set_data(zustand.spins)
        # Tatsaechliches 'm' als Marker im zweiten Plot
        axis[1].plot(tau0, zustand.m, c='k', ls='', marker='o', mew=0, ms=3) 
        # Titel aktualisieren
        axis[0].set_title(plot_dict[0]['title'] + '{}'
                          .format(round(zustand.tau, 3)), size='small')
        axis[1].set_title(plot_dict[1]['title'] + '{}'
                          .format(round(zustand.m, 4)), size='small')
        
        event.canvas.flush_events()
        event.canvas.draw()
    
def main():
    print(__doc__)
    
    m0 = 0.4            # Startwert Magnetisierung
    tau = 1.0           # Startwert Temperatur
    tau_max = 4.0       # Maximale Temperatur im Plot
    Ntau = 200          # Anzahl Temperaturschritte
    Nt = 10             # Anzahl Monte-Carlo Schritte pro Mausklick
    n = 50              # Spinmatrix-Groesse ist (n, n)
    
    
    # Strings fuer Plotueberschriften
    txt1 = (r'Spinzustand eines {}$\times${} Gitters, $\tau$='.format(n, n)) 
    txt2 = (r'Mittlere Magnetisierung pro Spin $m$=')
    
    plot_dict = {0: {'xlabel': 'x', 'ylabel': 'y', 'title': txt1}, 
                 1: {'xlabel': r'$\tau$', 'ylabel': r'$m$', 'title': txt2}}
    
    # Zustand praeparieren
    spins_matrix = m_zustand_gen(m0, n)
    ising_zustand = IsingZustand(spins_matrix, tau, m0)
    ising_zustand.m = ising_zustand.mean()
    
    # mittlere Magnetisierung fuer unendliches System berechnen
    tau_array = np.linspace(0, tau_max, Ntau)
    m_infty_plot = m_infty(tau_array)
    
    # Plotbereiche erstellen
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    plt.suptitle('Visualisierung des 2D Ising-Modells')
    for i in range(2):
        axis[i].set_xlabel(plot_dict[i]['xlabel'])
        axis[i].set_ylabel(plot_dict[i]['ylabel'])
    axis[0].set_title(plot_dict[0]['title'] + '{}'.format(tau),
                      size='small')
    axis[1].set_title(plot_dict[1]['title'] + '{}'.format(m0), size='small')
    axis[1].axis([0, tau_max, -1.05, 1.05])
    axis[1].yaxis.set_label_coords(-0.15, 0.5)  # label sonst im ersten Plot
    
    
    # Visualisierung der Spinmatrix
    spins_img = axis[0].imshow(ising_zustand.spins, cmap='plasma')
    
    # Plot von Startwert (m0, tau) und analytischem m_infty
    axis[1].plot(tau, ising_zustand.m, c='k', ls='', marker='o', mew=0,
                 ms=3, label=r'Startwert $(m,\tau)$')
    axis[1].plot(tau_array, m_infty_plot, c='b', lw=1, label='$m_\infty$')
    axis[1].plot(tau_array, -m_infty_plot, c='b', lw=1)
    axis[1].grid(True)
    
    # Mausinteraktion
    klick_funktion = functools.partial(mausklick, axis=axis, Nt=Nt, n=n,
                                       zustand=ising_zustand,
                                       spins_img=spins_img,
                                       tau_array=tau_array,
                                       m_infty_plot=m_infty_plot,
                                       plot_dict=plot_dict)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
    """
tau = 0.1:
    Um m=0 (+- 0.1) konvergiert das Verfahren sehr langsam, es bilden sich
        relativ stabile Muster/Flecken aus. Das Vorzeichen des Startwertes
        stimmt nicht immer mit dem des Endzustandes ueberein. 
        Die sequentielle Methode ist hier deutlich schneller, wird
        allerdings in der Naehe von tau_c recht instabil. 
    Bei groesseren |m| konvergiert das Verfahren sehr schnell gegen die
        theoretische Vorhersage.

tau = 3.5:
    Jeder Startwert von m (auch 'reine' Zustaende m=+-1) oszillieren nach
        wenigen MC-Schritten um eine mittlere Magnetisierung von 0. 
        Trotzdem ist die Dynamik ist chaotisch, jeder MC-Schritt fuehrt zu
        einem stark veraenderten Zustand.
    """
