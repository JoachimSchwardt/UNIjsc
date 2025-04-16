"""
Python-Skript zur Auswertung der Langzeitmessung zur Bestimmung der
Lebensdauer von Myonen im Versuch LM. 
Alle Zeitangaben sind in 'mu s'.
Es werden drei verschiedene Plotfenster geoeffnet, in denen die Daten ueber
den jeweiligen Kanaelen aufgetragen sind. Fuer die Details der
Ausgleichskurven sei auf das Protokoll verwiesen.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.integrate import quad

print(__doc__)

ch_min = 46 - 1                 # first channel number - 1 (for list index)
ch_max = 145 - 1                # last channel numbber - 1 (for list index)

# ch_min = 40 - 1                 # first channel number - 1 (for list index)
# ch_max = 147 - 1                # last channel numbber - 1 (for list index)
K = ch_max - ch_min + 1         # total number of chosen channels

t1 = 10 / 256                   # Breite eines Kanals
T1 = ch_min * t1 + t1/2         # Effektiver Beginn der Messung in 1
T2 = ch_max * t1 + t1/2         # Effektives Ende der Messung in 1
tau_theory = 2.197              # theoretischer Wert der Lebensdauer

print("Anzahl verwendeter Kanäle: {} (ch{} bis ch{})"
      .format(K, ch_min+1, ch_max+1))


def decay(x, tau=tau_theory, T1=T1, T2=T2):
    x = t1 * x
    return N * np.exp(-x/tau) / (np.exp(-T1/tau) - np.exp(-T2/tau)) * t1/tau 

# data = np.loadtxt("LM1_2017_12_18.txt", skiprows=85, unpack=False)[:, 1]
data = np.loadtxt("LM1_2020_11_20.txt", skiprows=85, unpack=False)[:, 1]

N = np.sum(data[ch_min:ch_max])     # Gesamte counts in gewaehlten Kanaelen
# data = decay(np.arange(0, 255, 1) + 0.5)
Nk_sum = 0
Nk2_sum = 0
for k in range(ch_min, ch_max, 1):
    Nk_sum += (k + 1) * data[k]         # Summe N_k * k aus Formel berechnen
    Nk2_sum += (k + 1)**2 * data[k]     # Summe N_k * k**2 fuer Delta tau
    
tau_error = t1 / N * np.sqrt(Nk2_sum)   # Delta tau berechnen
    
C = t1 * Nk_sum / N + t1 / 2            # Konstante aus Formel berechnen

def func(tau, T1=T1, T2=T2, C=C):
    """Die Lebensdauer tau entspricht der Nullstelle dieser Funktion. """
    return (C - tau - (T1*np.exp(-T1/tau) - T2*np.exp(-T2/tau)) / 
            (np.exp(-T1/tau) - np.exp(-T2/tau)))

data_plot = data[ch_min:ch_max]


x = np.arange(ch_min, ch_max, 1) + 0.5    # Array mit Kanal-Mitten fuer plot

# # Plotbereich fuer die grafische Darstellung des gesamten Datensatzes
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# ax.errorbar(np.arange(1, len(data)+1, 1), data, yerr=np.sqrt(data),
#             fmt='x', c='k', ecolor='k', ls='', ms=4, mew=0.5, 
#             elinewidth=0.5, capsize=2, label='data')

# # Achsenbeschriftungen, Gitter und Legende hinzufuegen
# ax.set_xlabel("channel number", fontsize=14)
# ax.set_ylabel("channel counts", fontsize=14)
# ax.grid(True)
# ax.legend(fontsize=12)
# ax.tick_params(labelsize=12)

# plt.show()

tau_initial_guess = 2                       # Schaetzwert fuer NST-Suche
tau_solution = fsolve(func, tau_initial_guess)    # NST-Suche mit .fsolve()

# tau mit .curve_fit() finden
Amp_theory = decay(ch_min+0.5) * np.exp((ch_min+0.5) * t1/tau_theory)
p0_theory = [tau_initial_guess]
p0 = [tau_initial_guess, 1000]

def decay_fit(x, tau, A=Amp_theory):
    x = t1 * x
    return A * np.exp(-x/tau)

params, covparams = curve_fit(decay_fit, x, data_plot, p0=p0)
param_theory, covparam_theory = curve_fit(decay_fit, x, data_plot,
                                          p0=p0_theory)

tau_fit, Amp_fit = params
cov_tau_fit, cov_amp = covparams[0][0], covparams[1][1]

tau_fit_theory = param_theory
cov_tau_fit_theory = covparam_theory[0][0]

# Plotbereich erstellen
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot von: Daten, Zerfallsgesetz aus Likelihood, aus Theorie und aus Fit
ax.errorbar(x, data_plot, yerr=np.sqrt(data_plot), fmt='x', c='k', ecolor='k',
            ls='', ms=4, mew=0.5, elinewidth=0.5, capsize=2, label='data')
ax.plot(x, decay(x), c='g', lw=0.8, 
        label=r'Theory, $\tau={}\,\mu$s'.format(tau_theory))
# ax.plot(x, decay_fit(x, tau_fit_theory, Amp_theory), c='orange', lw=0.8, 
#         label=r'Curve fit, fixed Amp., $\tau={}\pm {}\,\mu$s'
#         .format(round(tau_fit_theory[0], 3),
#                 round(np.sqrt(cov_tau_fit_theory), 4)))
ax.plot(x, decay_fit(x, tau_fit, Amp_fit), c='b', lw=0.8, 
        label=r'Curve fit, $\tau={}\pm {}\,\mu$s'
        .format(round(tau_fit, 3), round(np.sqrt(cov_tau_fit), 4)))
ax.plot(x, decay(x, tau_solution), lw=0.8, c='red', 
        label=r'Likelihood, $\tau={}\pm {}\,\mu$s'
        .format(round(tau_solution[0], 3), round(tau_error, 4)))
ax.plot(x, decay(x, tau_solution + tau_error), lw=0.8, c='purple', ls='--', 
        label=r'Likelihood+, $\tau_+={}\,\mu$s'
        .format(round(tau_solution[0] + tau_error, 3)))
ax.plot(x, decay(x, tau_solution - tau_error), lw=0.8, c='purple', ls='--', 
        label=r'Likelihood-, $\tau_-={}\,\mu$s'
        .format(round(tau_solution[0] - tau_error, 3)))

# Achsenbeschriftungen, Gitter und Legende hinzufuegen
ax.set_title("Anwendung der Likelihood-Methode auf das Zerfallsgesetz", 
             fontsize=14)
ax.set_xlabel("channel number", fontsize=14)
ax.set_ylabel("channel counts", fontsize=14)
ax.grid(True)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

plt.show()

# print(tau_fit, tau_fit_theory)

# Normierung und numerisch gefundene Nullstelle von func() ueberpruefen
print('Integrated Theory: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5)[0] / N))
print('Integrated Likelihood: {}'.format(quad(decay, ch_min+0.5, ch_max+0.5,
                                              args=(tau_solution))[0] / N))
print('Integrated Curve fit: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, args=(tau_fit))[0] / N))
print("Check numerical solution: func(tau_solution) = {}"
      .format(func(tau_solution)))
print("Decay: {} +- {}".format(tau_solution, tau_error))

##########################################
# Poisson-Teil

def N0(tau, N=N, t1=t1, k_min=ch_min, k_max=ch_max):
    return N / (np.exp(-k_min * t1/tau) - np.exp(-(k_max + 1) * t1/tau))

def f(k, tau, N=N, t1=t1, k_min=ch_min, k_max=ch_max):
    N0_val = N0(tau, N=N, t1=t1, k_min=ch_min, k_max=ch_max)
    return N0_val * np.exp(-k * t1/tau) * (1 - np.exp(-t1/tau))

def logL(tau, data, N=N, t1=t1, k_min=ch_min, k_max=ch_max):
    res = np.zeros_like(tau)
    for k in range(k_min, k_max + 1, 1):
        Nk = data[k-1]
        res += Nk * np.log(f(k, tau, N=N, t1=t1, k_min=ch_min, k_max=ch_max))
    return res

tau = np.linspace(2.0, 2.4, 1000)     # moegliche tau fuer Maximum-Suche
x = np.arange(ch_min, ch_max, 1)

tau_solution = 2
logL_data = logL(tau, data)
logL_plus = logL(tau, data + np.sqrt(data))
logL_minus = logL(tau, data - np.sqrt(data))

# Plotbereich erstellen
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot von: Daten, Zerfallsgesetz mit Poisson aus Likelihood und Theorie

tau_solution = tau[logL_data == max(logL_data)]
tau_plus = tau[logL_plus == max(logL_plus)] - tau_solution
tau_minus = tau_solution - tau[logL_minus == max(logL_minus)]
# print(tau_solution, tau_plus, tau_minus)

# ax.plot(tau, logL_data, c='b', label='logL')
ax.errorbar(x, data_plot, yerr=np.sqrt(data_plot), fmt='x', c='k', ecolor='k',
            ls='', ms=4, mew=0.5, elinewidth=0.5, capsize=2, label='data')
ax.plot(x, decay(x), c='g', lw=0.8, 
        label=r'Theory, $\tau={}\,\mu$s'.format(tau_theory))
ax.plot(x, decay(x, tau_solution), lw=0.8, c='red', 
        label=r'Likelihood, $\tau={}+{}-{}\,\mu$s'
        .format(round(tau_solution[0], 3), round(tau_plus[0], 4),
                round(tau_minus[0], 4)))
ax.plot(x, decay(x, tau_solution + tau_plus), lw=0.8, c='purple', ls='--', 
        label=r'Likelihood+, $\tau_+={}\,\mu$s'
        .format(round(tau_solution[0] + tau_plus[0], 3)))
ax.plot(x, decay(x, tau_solution - tau_minus), lw=0.8, c='purple', ls='--', 
        label=r'Likelihood-, $\tau_-={}\,\mu$s'
        .format(round(tau_solution[0] - tau_minus[0], 3)))

# Achsenbeschriftungen, Gitter und Legende hinzufuegen
ax.set_title("Anwendung der Likelihood-Methode auf eine Poisson-Verteilung", 
             fontsize=14)
ax.set_xlabel("channel number", fontsize=14)
ax.set_ylabel("channel counts", fontsize=14)
ax.grid(True)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

plt.show()

# Normierung der einzelnen Zerfallsplots ueberpruefen
print('Integrated Likelihood: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, 
                   args=(tau_solution))[0] / N))
print('Integrated Likelihood+: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, 
                   args=(tau_solution + tau_plus))[0] / N))
print('Integrated Likelihood-: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, 
                   args=(tau_solution - tau_minus))[0] / N))
print("Poisson: {} + {} - {}".format(tau_solution, tau_plus, tau_minus))
    
##########################################
# Gauss-Teil

def logLgauss(tau, data, N=N, t1=t1, k_min=ch_min, k_max=ch_max):
    res = np.zeros_like(tau)
    for k in range(k_min, k_max + 1, 1):
        Nk, fk = data[k-1], f(k, tau, N=N, t1=t1, k_min=ch_min, k_max=ch_max)
        res -= (Nk - fk)**2 / fk + np.log(fk)
    return res

x = np.arange(ch_min, ch_max, 1)

tau_solution = 2
logL_data = logLgauss(tau, data)
logL_plus = logLgauss(tau, data + np.sqrt(data))
logL_minus = logLgauss(tau, data - np.sqrt(data))

# Plotbereich erstellen
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot von: Daten, Zerfallsgesetz mit Poisson aus Likelihood und Theorie

tau_solution = tau[logL_data == max(logL_data)]
tau_plus = tau[logL_plus == max(logL_plus)] - tau_solution
tau_minus = tau_solution - tau[logL_minus == max(logL_minus)]

# ax.plot(tau, logL_data, c='b', label='logL')
ax.errorbar(x, data_plot, yerr=np.sqrt(data_plot), fmt='x', c='k', ecolor='k',
            ls='', ms=4, mew=0.5, elinewidth=0.5, capsize=2, label='data')
ax.plot(x, decay(x), c='g', lw=0.8, 
        label=r'Theory, $\tau={}\,\mu$s'.format(tau_theory))
ax.plot(x, decay(x, tau_solution), lw=0.8, c='red', 
        label=r'Likelihood, $\tau={}+{}-{}\,\mu$s'
        .format(round(tau_solution[0], 3), round(tau_plus[0], 4),
                round(tau_minus[0], 4)))
ax.plot(x, decay(x, tau_solution + tau_plus), lw=0.8, c='purple', ls='--', 
        label=r'Likelihood+, $\tau_+={}\,\mu$s'
        .format(round(tau_solution[0] + tau_plus[0], 3)))
ax.plot(x, decay(x, tau_solution - tau_minus), lw=0.8, c='purple', ls='--', 
        label=r'Likelihood-, $\tau_-={}\,\mu$s'
        .format(round(tau_solution[0] - tau_minus[0], 3)))

# Achsenbeschriftungen, Gitter und Legende hinzufuegen
ax.set_title("Anwendung der Likelihood-Methode auf eine Gauss-Verteilung", 
             fontsize=14)
ax.set_xlabel("channel number", fontsize=14)
ax.set_ylabel("channel counts", fontsize=14)
ax.grid(True)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

plt.show()
    
# Normierung der einzelnen Zerfallsplots ueberpruefen
print('Integrated Likelihood: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, 
                   args=(tau_solution))[0] / N))
print('Integrated Likelihood+: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, 
                   args=(tau_solution + tau_plus))[0] / N))
print('Integrated Likelihood-: {}'
      .format(quad(decay, ch_min+0.5, ch_max+0.5, 
                   args=(tau_solution - tau_minus))[0] / N))
print("Gauss: {} + {} - {}".format(tau_solution, tau_plus, tau_minus))
    
    
# for k in range(ch_min, ch_max + 1, 1):
#     print(f(k, 2.197), data[k-1])
    
    
    
    
    
    
    
    
    