"""Unter Verwendung das einfachen SIR-Modells wird der Verlauf einer
Pandemie simuliert. Dargestellt werden die zeitlichen Verlaeufe von S(t),
I(t) und R(t), sowie der Einfluss einer Lockdown Periode und das Maximum
von I(t), in Abhaengigkeit von beta im Intervall [beta_min, beta_max].
Voreingestellt sind:
    beta=0.5, gamma=0.1, N=8*1e7, T=150, beta_min=0.11, beta_max=0.5
Durch das Druecken von 'l' in einem Plotbereich wird die y-Achse logarithmiert."""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_abl(y, t, beta_0=0.5, gamma=0.1, t_mod=np.array([0]),
            mod_list=[1]):
    """
    Es bezeichenen:
        S = ansteckbare Personen ("susceptible")
        I = infizierte Personen ("infected")
        R = genesene und gestorbene Personen ("recovered" oder "removed")
        beta = Wahrscheinlichkeit einer Uebertragung (von S -> I)
        gamma = Anteil der Infizierten, die taeglich 'genesen' (von I -> R)
    Das einfache SIR-Modell wird dann durch folgende DGL beschrieben:
        S_dot = -beta * S * I
        I_dot = beta * S * I - gamma * I
        R_dot = gamma * I
    Es gilt S + I + R = 1.
    """
    S, I, R = y
    beta_t = beta(t, t_mod, mod_list, beta_0)   # beta fuer Lockdown o.a.
    S_dot = -beta_t * S * I
    I_dot = beta_t * S * I - gamma * I
    R_dot = gamma * I
    return np.array([S_dot, I_dot, R_dot])

def beta(t=0, t_mod=np.array([0]), mod_list=[1], beta_0=0.5):
    """
    Implementierung eines diskret zeitabhaengigen Wertes fuer 'beta':
        Ordnet 't' in ein gegebenes Array an aufsteigenden Zeitpunkten ein
        und bestimmt den zugehoerigen Index. Fuer eine gegebene Liste an
        Modifikatoren von 'beta' wird entsprechend der Modifikator dieses
        Index mit beta_0 multipliziert.
    """
    t_mod = np.sort(np.append(t_mod, t))
    t_index = max(*np.where(t_mod <= t))
    return beta_0 * mod_list[t_index - 1]
    
def main():
    print(__doc__)
    print(SIR_abl.__doc__)
    
    # Parameter des einfachen SIR-Modells
    beta_0 = 0.5    # Wahrscheinlichkeit einer Uebertragung
    gamma = 0.1     # Anteil Personen, die taeglich genesen
    population = 8*1e7   # Gesamtbevoelkerung
    sys_limit = 0.1      # Belastbarkeitsgrenze des Gesundheitssystems
    
    # Anzahl Personen in S, I, R bei t0
    I_t0_abs = 1
    R_t0_abs = 0
    S_t0_abs = population - I_t0_abs - R_t0_abs
    
    # Normierung auf S+I+R = 1
    S_t0 = S_t0_abs / population
    I_t0 = I_t0_abs / population
    R_t0 = R_t0_abs / population
    y0 = [S_t0, I_t0, R_t0]
    
    # Parameter der Simulation
    t0 = 0            # Anfangs- und Endzeitpunkt 
    t_end = 150 
    T = t_end - t0    # Anzahl Tage
    m = 200        # Anzahl Zeitpunkte pro Tag
    m_beta = 250   # Anzahl Punkte 'beta' fuer R_0-Plot
    beta_min, beta_max = 0.11, 0.5    # beta-Intervallgrenzen fuer R_0-Plot
    
    # Zusaetzliche Parameter der Einschraenkungen
    t_hammer = 20   # Beginn Lockdown
    t_dance = 60    # Ende Lockdown
    hammer_mod = 0.3    # Modifikatoren fuer beta
    dance_mod = 0.5
    t_mod = np.array([t0, t_hammer, t_dance])
    mod_list = [1, hammer_mod, dance_mod]
    
    param_txt = (r"Parameter: T={}, $\beta_0$={}, $\gamma$={} und N="
                .format(int(T), round(beta_0, 3), round(gamma, 3)) + 
                f"{'{:,}'.format(int(population))}")
    
    SIR_dict = {0: {'color': [0, 0, 1, 1], 'label': 'Susceptible'}, 
                1: {'color': [1, 0, 0, 1], 'label': 'Infected'}, 
                2: {'color': [0, 1, 0, 1], 'label': 'Removed'}}
    
    plot_dict = {0: {'title': param_txt, 'axis': [t0, t_end, 1e-5, 1], 
                     'xlable': '    t / Tagen', 
                     'ylable': 'Anteil der Bevölkerung'}, 
                 1: {'title': ' ', 'axis': [beta_min, beta_max, 1e-5, 1],
                     'xlable': r'      $\beta$', 
                     'ylable': 'Infektionsmaximum'}}
    
    # Loesen der ODE mit 'odeint':
    zeiten = np.linspace(t0, t_end, m)  
    y_t = odeint(SIR_abl, y0, zeiten, args=(beta_0, gamma))
    
    # Loesen der ODE fuer Einschraenkungen:
    y_t_mod = odeint(SIR_abl, y0, zeiten, 
                     args=(beta_0, gamma, t_mod, mod_list))
    
    SIR_data = {0: {'normal': y_t[:, 0], 'modified': y_t_mod[:, 0]}, 
                1: {'normal': y_t[:, 1], 'modified': y_t_mod[:, 1]}, 
                2: {'normal': y_t[:, 2], 'modified': y_t_mod[:, 2]}}
        
    # Plotbereiche erstellen
    fig, axis = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    plt.suptitle("Simulation des einfachen SIR-Modells")
    
    # Plot SIR-Modell mit und ohne Einschraenkungen
    for elem in SIR_dict:
        axis[0].plot(zeiten, SIR_data[elem]['normal'], lw=1,
                      c=SIR_dict[elem]['color'], 
                      label=SIR_dict[elem]['label'])
        axis[0].plot(zeiten, SIR_data[elem]['modified'], lw=1,
                     ls='dashed', c=SIR_dict[elem]['color'])
    # Markierung des Lockdown-Zeitraums
    axis[0].axvline(t_hammer, c='k', lw=0.5, ls='dashed', 
                    label=r'Beginn Lockdown {}$\cdot\beta_0$'
                    .format(round(hammer_mod, 3)))
    axis[0].axvline(t_dance, c='k', lw=0.5, ls='dotted', 
                    label=r'Ende Lockdown {}$\cdot\beta_0$'
                    .format(round(dance_mod, 3)))
    
        
    # Plot Belastbarkeitsgrenze
    beta_array = np.linspace(beta_min, beta_max, m_beta)
    # Peak muss im Intervall liegen -> groesseres Zeitintervall
    zeiten_extend = np.linspace(t0, 4 * t_end, 2 * m) 
    
    I_max = np.zeros_like(beta_array)
    for count in range(len(beta_array)):
        # Loesung ODE fuer alle beta
        y_beta_t = odeint(SIR_abl, y0, zeiten_extend,
                          args=(beta_array[count], gamma))
        I_max[count] = max(y_beta_t[:, 1])   # Auswahl Maximum von I(t)
    axis[1].axhline(sys_limit, lw=1, c='k', label="Belastbarkeitsgrenze")
    axis[1].plot(beta_array, I_max, lw=1, c='g', label=r"max $I(t, R_0)$")
    
    # Ueberschriften und Legende
    for count in range(len(axis)):
        axis[count].axis(plot_dict[count]['axis'])
        axis[count].set_title(plot_dict[count]['title'], size='small')
        axis[count].text(plot_dict[count]['axis'][1], -0.12,
                          plot_dict[count]['xlable'])
        axis[count].set_ylabel(plot_dict[count]['ylable'])
        axis[count].legend(loc=2, prop={'size': 7})
    plt.show()
    
if __name__ == "__main__":
    main()
    """
a)  Fuer steigende 'beta' steigt der Wert des Maximums etwa linear an. 
    Die Lage des Maximums ist etwa proportional zu 1 / (beta - gamma). Sie
    aendert sich fuer groessere Werte von 'beta' also immer weniger.
    
b)  Bei beta = 0.17, also R0 = beta / gamma = 1.7 wird das
    Gesundheitssystem gerade noch nicht ueberlastet. 
    Anmerkung:
        Verwendet man fuer den Plot der Belastbarkeitsgrenze nur 150 Tage,
        so wird max I(t, R_0) fuer beta von 0 bis etwa 0.25 verzerrt. Hier
        wuerde man dann 0.2 ablesen, weil der tatsaechliche Peak von I(t)
        gar nicht innerhalb der ersten 150 Tage liegt.
    
c)  I(t) steigt waehrend des Lockdowns im logarithmischen Plot linear an,
    das exponentielle Wachstum wird also naeherungsweise unterdrueckt
    (wartet man lange genug, ergibt sich aber auch bei unbegrenztem Lockdown
    wieder das uebliche exponentielle Verhalten).
    
    Nach der Lockerung steigt I(t) nach etwa 90 Tagen wieder sichtbar
    exponentiell. Das Maximum wird etwa um einen Faktor 2 kleiner und von
    etwa 50 auf 120 Tage zeitlich nach hinten verschoben. Der Anteil von
    R(t) erreicht etwa 90 statt quasi 100 Prozent.
    
    Werden die Massnahmen vollstaendig aufgehoben, so verschiebt sich der
    gesamte Verlauf einfach zu spaeteren Zeiten. Beim 'zweiten Beginn' der
    Pandemie ist die Zahl der Infizierten groesser als 1, weshalb das
    Maximum des 'zweiten Berges' nach nur etwa 30 statt 50 Tagen erreicht
    ist. Die Form, und damit der Verlauf, ist allerdings nicht zu
    unterscheiden.
    """


