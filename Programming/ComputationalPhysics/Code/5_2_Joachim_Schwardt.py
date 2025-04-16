"""Dargestellt sind die genesenen und gestorbenen Personen, sowie die aktiven Faelle. Diese ergeben sich aus der Summe der Infizierten mit oder ohne Symptome und den isolierten Personen. 
Logarithmische Skalierung durch Druecken von 'l'."""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_abl(y, t, alpha=0.05, beta_0=0.5, delta=0.7, gamma=0.1, rho_0=0.9,
            k=0.5, t_lock=28, mod_lock=0.3):
    """
Es bezeichenen:
    S = ansteckbare Personen ("susceptible")
    E = infizierte Personen ohne Symptome ("exposed")
    I = infizierte Personen mit Symptomen ("infected")
    R = genesene Personen ("recovered")
    L = isolierte Personen ("lockdowned")
    D = gestorbene Personen ("dead")
    alpha = Mortalitaet (von I -> D)
    beta = Wahrscheinlichkeit einer Uebertragung (von S -> E)
    delta = Wahrscheinlichkeit Symptome auszubilden (von E -> I)
    gamma = Anteil der Infizierten, die taeglich genesen (von I -> R)
            (vereinfacht: Uebergaenge I -> R und I -> D gleiche Zeitdauer)
    rho = Anteil infizierter Personen, die isoliert werden (von I -> L)
    
Das SEILRD-Modell wird dann durch folgende DGL beschrieben:
    S_dot = -beta * S * E
    E_dot = beta * S * E  - delta * E
    I_dot = delta * E - gamma * I - rho * I
    L_dot = rho * I - gamma * L
    R_dot = gamma * (1-alpha) * (I + L)
    D_dot = gamma * alpha * (I + L)
Es gilt S + E + I + L + R + D = 1.
    """
    S, E, I, L, R, D = y
    # beta Lockdown
    t_index = beta(t, t_lock, mod_lock, beta_0)[1]
    beta_t = beta_logistic(t, t_index, t_lock, mod_lock, k, beta_0)  
    
    rho = max(rho_0 * (1 - k * I / S), 0)
    S_dot = -beta_t * S * E
    E_dot = beta_t * S * E  - gamma * E
    I_dot = delta * gamma * E - gamma * I - rho * I
    L_dot = rho * I - gamma * L
    R_dot = gamma * (1 - alpha) * (I + L) + (1-delta) * gamma * E
    D_dot = gamma * alpha * (I + L)
    return np.array([S_dot, E_dot, I_dot, L_dot, R_dot, D_dot])

def beta(t=0, t_lock=np.array([0]), mod_lock=[1], beta_0=0.5):
    """
    Implementierung eines diskret zeitabhaengigen Wertes fuer 'beta':
        Ordnet 't' in ein gegebenes Array an aufsteigenden Zeitpunkten ein
        und bestimmt den zugehoerigen Index. Fuer eine gegebene Liste an
        Modifikatoren von 'beta' wird entsprechend der Modifikator dieses
        Index mit beta_0 multipliziert.
    """
    t_lock = np.sort(np.append(t_lock, t))
    t_index = max(*np.where(t_lock <= t)) - 1
    return beta_0 * mod_lock[t_index], t_index

def beta_logistic(t=0, t_index=0, t_lock=[0, 28], mod_lock=[1, 0.3], 
                  k=0.5, beta_0=0.5):
    try:
        mod_old = mod_lock[t_index - 1]
    except IndexError:
        mod_old = 1
    logistic_mod = ((mod_old - mod_lock[t_index]) / 
                    (1 + np.exp(k * (t - t_lock[t_index]))) 
                    + mod_lock[t_index])
    return beta_0 * logistic_mod
    
def main():
    print(__doc__)
    
    # Parameter des SEILRD-Modells
    N = 8*1e7         # Gesamtbevoelkerung
    alpha = 0.03      # Mortalitaet
    gamma = 1 / 6     # 6 Tage durchschnittliche Dauer der Erkrankung
    beta_0 = 2.8 * gamma    # beta_0 = R_0 * gamma
    delta = 0.57     # Wahrscheinlichkeit fuer Symptome
    rho_0 = 0.90     # maximale Effizienz der Isolation von Infizierten
    k = 0.5           # Parameter fuer rho (beta sinkt in etwa einer Woche)

    # 0.05, 1/6, 2.8*gamma, 0.57, 0.95, 10
    
    # Anzahl Personen in S, E, I, L, R, D bei t0
    E_t0_abs = 10
    I_t0_abs = 1
    L_t0_abs = 0
    R_t0_abs = 0
    D_t0_abs = 0
    S_t0_abs = (N - E_t0_abs - I_t0_abs - L_t0_abs - R_t0_abs 
                - D_t0_abs)
    
    # Normierung auf S+I+R = 1
    S_t0 = S_t0_abs / N
    E_t0 = E_t0_abs / N
    I_t0 = I_t0_abs / N
    L_t0 = L_t0_abs / N
    R_t0 = R_t0_abs / N
    D_t0 = D_t0_abs / N
    y0 = [S_t0, E_t0, I_t0, L_t0, R_t0, D_t0]
    
    # Parameter der Simulation
    t0 = 0            # Anfangs- und Endzeitpunkt 
    t_end = 350 
    T = t_end - t0    # Anzahl Tage
    m = 300        # Anzahl Zeitpunkte pro Tag
    
    # Zusaetzliche Parameter der Einschraenkungen
    t_lock = [0, 28, 180, 240]    # Beginn Lockdown nach etwa 4 Wochen
    # Drueckt R_0 auf 0.9, dann 1.25:
    mod_lock = np.array([2.8, 0.9, 1.25, 1.05]) * gamma / beta_0 
    
    param_txt = (r"Parameter: T={}, $\alpha$={}, $\beta_0$={}, $\delta$={}"
                 .format(int(T), round(alpha, 3), round(beta_0, 3),
                        round(delta, 3))
                 + r", $\gamma$={}, $\rho_0$={} und N="
                .format(round(gamma, 3), round(rho_0, 3)) + 
                f"{'{:,}'.format(int(N))}")
    
    SIR_dict = {0: {'color': [0, 0, 1, 1], 'label': 'Susceptible'},
                1: {'color': [1, 0, 1, 1], 'label': 'Exposed'},
                2: {'color': [1, 0, 0, 1], 'label': 'Infected'},
                3: {'color': [0, 1, 1, 1], 'label': 'Lockdowned'},
                4: {'color': [0, 1, 0, 1], 'label': 'Recovered'},
                5: {'color': [1, 0, 0, 1], 'label': 'Active cases'},
                6: {'color': [0, 0, 0, 1], 'label': 'Dead'}}
    
    plot_dict = {'title': param_txt, 'xlable': '    t / Tagen',
                 'axis': [t0, t_end, 1, 1.5*1e6], 
                 'ylable': 'Anteil der Bevölkerung'}
    
    # Loesen der ODE fuer Einschraenkungen:
    zeiten = np.linspace(t0, t_end, m)  
    y_t = odeint(SIR_abl, y0, zeiten, args=(alpha, beta_0, delta, gamma,
                                            rho_0, k, t_lock, mod_lock))
    
    SIR_data = {0: y_t[:, 0], 1: y_t[:, 1], 2: y_t[:, 2], 3: y_t[:, 3], 
                4: y_t[:, 4], 5: y_t[:, 1] + y_t[:, 2] + y_t[:, 3], 
                6: y_t[:, 5]}
        
    # Plotbereich erstellen
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("Simulation des SEILRD-Modells\n")
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot SIR-Modell mit und ohne Einschraenkungen
    for elem in [4, 5, 6]:
        ax.plot(zeiten, N * SIR_data[elem], lw=1,
                      ls='-', c=SIR_dict[elem]['color'], 
                      label=SIR_dict[elem]['label'])
    
    # # plot beta(t)
    # beta_plot = []
    # for t in zeiten:
    #     t_index = beta(t, t_lock, mod_lock, beta_0)[1]
    #     beta_plot.append(beta_logistic(t, t_index, t_lock, mod_lock, 
    #                                    k, beta_0))
    # ax.plot(zeiten, beta_plot)
    
    # Markierung des Lockdown-Zeitraums
    for i in range(len(t_lock)):
        ax.axvline(t_lock[i], c='k', lw=0.5, ls='--', label=r'$R_0\ =\ {}$'
                   .format(round(mod_lock[i] * beta_0 / gamma, 3)))
    
    # Ueberschriften und Legende
    ax.axis(plot_dict['axis'])
    ax.set_title(plot_dict['title'], size='small')
    ax.text(plot_dict['axis'][1], -0.12,
                      plot_dict['xlable'])
    ax.set_ylabel(plot_dict['ylable'])
    ax.legend(loc=2, prop={'size': 7})
    ax.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
    """
Zusaetzliche untersuchte Gruppen / Parameter:
    D = Gestorbene Personen (Unterteilung von R_alt in R und D)
    alpha = Mortalitaet
    
    E = Uebertraeger ohne Symptome
    delta = Wahrscheinlichkeit Symptome auszubilden
    
    L = isolierbare Personen (nur fuer infizierte Personen mit Symptomen)
    rho = Effizienz der Isolation infizierter Personen
        (Idee: System ist nur bis zu einer bestimmten Anzahl an Infizierten
        dazu in der Lage einzelene Personen gezielt zu isolieren 
        -> sonst Lockdown effizienter)
        
Das Modell reagiert sehr empfindlich auf kleine Aenderungen der Parameter 'beta_0', 'delta' und 'gamma'. Unguenstigerweise sind das auch die Parameter, die nicht besonders genau bekannt sind. 
Man kann durch das Einstellen 'vernuenftiger' Werte im Rahmen der Schaetzungen das Verhalten allerdings gut nachbilden, vgl. dazu bspw. die realen Fallzahlen von https://de.wikipedia.org/wiki/COVID-19-Pandemie_in_Deutschland#Zahl_der_aktiven_F%C3%A4lle 
Wert fuer 'delta' von 'Diamond Princess'-Studie (etwa 0.43 ohne Symptome)
Werte fuer 'R_0' und 'gamma' vom RKI.
    """


