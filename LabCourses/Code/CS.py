"""
@author: Joachim Schwardt, Auswertung des Versuchs ComptonStreuung.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.ndimage.filters import uniform_filter1d
from matplotlib import rcParams
rcParams["figure.dpi"] = 50

NA = 6.02214076 * 10**(23)          # 1/mol
c = 299762458                       # m/s
u = 1.66 * 10**(-27)                # kg
e = 1.602176634 * 10**(-19)         # As
eps0 = 8.8541878128 * 10**(-12)     # As/Vm

h = 6.62607015 * 10**(-34)          # J s
hbar = 1.054571817 * 10**(-34)      # J s

me = 9.1093837015 * 10**(-31)       # kg
meev = me * c**2 / e                # ev
alpha = e**2/(4*np.pi*eps0*hbar*c)  # Feinstrukturkonstante

def plot_params(ax, xlabel='x', ylabel='y', title='title', vline=None,
                hline=None, gridlines=True, Adjust=0, xlog=0, ylog=0):
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')    
    if vline != None:
        ax.axvline(vline, lw=0.8, c='k')
    if hline != None:
        ax.axhline(hline, lw=0.8, c='k')
    ax.grid(gridlines)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    if Adjust:
        plt.subplots_adjust(bottom=0.15)
    plt.show()
    pass

def cse(theta=np.pi/2, E=59540):
    return E / (1 + E / meev * (1 - np.cos(theta)))
def cse_mu(mu=0, E=59540):
    return E / (1 + E / meev * (1 - mu))

def sigma_kn(theta=np.pi/2, E=59540, norm=1):
    Pval = 1 / (1 + E / meev * (1 - np.cos(theta)))
    return 0.5 * Pval**2 * (Pval + 1/Pval - np.sin(theta)**2) * norm
def sigma_kn_mu(mu=0, E=59540, norm=1):
    Pval = 1 / (1 + E / meev * (1 - mu))
    return 0.5 * Pval**2 * (Pval + 1/Pval + mu**2 - 1) * norm

def gauss(x, avg=0, cov=1e3, norm=1):
    return np.exp(-(x - avg)**2 / (2*cov)) * norm

def k_auto_detect(x, y, size=15, order=15, PlotSmooth=0):
    """
    'https://stackoverflow.com/questions/13728392/
    moving-average-or-running-mean' used for noice reduction of data.
    'https://stackoverflow.com/questions/48023982/
    pandas-finding-local-max-and-min' used for finding local maxima of data.
    """
    # smooth out signal for peak detection
    y = uniform_filter1d(y, size=size)  
        
    # detect peak indizes
    maxima = argrelextrema(y, np.greater, order=order)[0] 
    minima = argrelextrema(y, np.less, order=order)[0]
    if minima[0] >= maxima[0]:
        minima = np.append([1], minima)
    minima = np.append(minima, len(y) - 1)
    
    if PlotSmooth:
        plt.plot(x, y)
        print(minima, "\n", maxima)
        
    k_val = []
    for i in range(len(maxima)):
        # consider channel numbers between two following peaks
        try:
            min_k_width = min(maxima[i] - minima[i], minima[i+1] - maxima[i])
        except IndexError:
            print("Warning, peak detection failed atleast partly!")
            return k_auto_detect(x, y, size=size, order=order, 
                                 PlotSmooth=PlotSmooth)
        if min_k_width < 0:
            print("Warning, peak detection may be incorrect!")
            return k_auto_detect(x, y, size=size, order=order, 
                                 PlotSmooth=PlotSmooth)
        k_val.append([maxima[i] + j*min_k_width for j in [-1, 1]])
    return k_val

def gauss_param(xdata, ydata, PeakPlot=0, lw=1):
    # estimated channel number average at maximum of data
    avg0 = xdata[ydata == max(ydata)][0]  
    # estimate standard deviation by half-width half maximum
    xdata_sig = xdata[ydata > 0.5 * max(ydata)]
    cov0 = (max(xdata_sig) - min(xdata_sig))**2 / 4
    # estimate norm of data as maximum of data
    norm0 = max(ydata)
    
    # initial values for 'curve_fit'
    p0 = [avg0, cov0, norm0]
    
    try:
        param, cov = curve_fit(gauss, xdata, ydata, p0=p0)
    except RuntimeError:
        print("Fit did not converge for some peak!")
        return
    if PeakPlot:   # plot zoom in of the detected peak
        title_str =("Zoom von Fit mit $k={}\pm {}$"
                    .format(round(param[0], 2), round(np.sqrt(cov[0][0]), 3)))
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.plot(xdata, ydata, lw=lw, c='b', label='data')
        ax.plot(xdata, gauss(xdata, *param), lw=lw, c='r', label='fit')
        plot_params(ax, "Kanalnummer", "counts", title_str)
    return param, cov, p0

def gauss_plot_peak(x, y, k_peak=None, TestPlot=0, PeakPlot=0, lw=1, 
                    filter_ampl=0.3, FiltDataPlot=0):
    title_str = "Impulshöhenspektrum mit Gaussfit"
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(x, y, lw=1, c='b', label='Messwerte')
    
    c = ['r', 'lime', [0.7, 0.1, 1, 1], 'orange', [0, 1, 1, 1]]
    params, covs = [], []
    for i, k_val in enumerate(k_peak):
        [k_min, k_max] = k_val
        
        y_val = y[k_min:k_max]
        x_val = x[k_min:k_max]
        
        indizes = (y_val > filter_ampl * max(y_val))
        x_val, y_val = x_val[indizes], y_val[indizes]
        
        param, cov, p0 = gauss_param(x_val, y_val, PeakPlot)
        label =("Fit mit $k={}\pm {}$"
                .format(round(param[0], 2), round(np.sqrt(cov[0][0]), 3)))
        ax.plot(x_val, gauss(x_val, *param), lw=lw, 
                c=c[i % len(c)], label=label)
        if FiltDataPlot:
            ax.plot(x_val, y_val, lw=1, c='k')
        
        params.append(param)
        covs.append(cov)
        if TestPlot:
            label = "Guess mit $k={}$".format(p0[0])
            ax.plot(x_val, gauss(x_val, *p0), lw=lw, ls='--', 
                    c=c[i % len(c)], label=label)
    
    plot_params(ax, "Kanalnummer", "counts", title_str, hline=0)
    return params, covs

def c_elem(nf):
    return [[i/nf, 0.3 * i/nf, 1 - i/nf, 1] for i in range(nf)]

def avg_energy():
    ECS = np.array([31.817, 32.194, 36.304, 36.378, 37.255]) 
    PCS = np.array([1.99, 3.64, 0.348, 0.672, 0.213])
    EBA = np.array([30.625, 30.973, 34.92, 34.987, 35.818] )  
    PBA = np.array([33.9, 62.2, 5.88, 11.4, 3.51])
    EBA2 = np.array([79.614, 80.998])
    PBA2 = np.array([2.65, 32.9])
    EEU = np.array([39.522, 40.118, 45.293, 45.414, 46.578] )  
    PEU = np.array([21.0, 37.7, 3.75, 7.26, 2.40] )
    EEU2 = np.array([5.64] )  
    PEU2 = np.array([14.0] )
    EAM2 = np.array([59.541])
    PAM2 = np.array([35.9])
    # EEU = np.array([42.309, 42.996, 48.551, 48.695, 49.959] )  
    # PEU = np.array([0.248, 0.443, 0.0448, 0.0867, 0.029] )
    Eval = [ECS, EBA, EBA2, EEU2, EEU, EAM2]
    Pval = [PCS, PBA, PBA2, PEU2, PEU, PAM2]
    return [sum(E*P) / sum(P) for [E, P] in zip(Eval, Pval)]

def time_optimum(ng, n0, t=1200, alpha=0.05):
    t_opt = t * (ng + n0) / (alpha * (ng - n0))**2
    delta_t_opt = t * (np.sqrt(ng * (ng + 3*n0)**2 + n0 * (3*ng + n0)**2) / 
                       (alpha**2 * (ng - n0)**3))
    return t_opt, delta_t_opt


def E(k, alpha, E0):
    return alpha * k + E0
    

# x = np.arange(1, 1025, 1)
# y = gauss(x, 400, 5000, 1e-3) * np.random.uniform(0.8, 1.2, len(x))
# # y += gauss(x, 500, 200, 0.5*1e-3) * np.random.uniform(0.9, 1.9, len(x))
# # y += gauss(x, 100, 100, 0.3*1e-3) * np.random.uniform(0.85, 1.15, len(x))
# # y += gauss(x, 250, 150, 0.2*1e-3) * np.random.uniform(0.85, 1.15, len(x))
# # y += gauss(x, 650, 100, 1*1e-3) * np.random.uniform(0.85, 1.15, len(x))
# # y += gauss(x, 870, 50, 0.6*1e-3) * np.random.uniform(0.85, 1.15, len(x))
# # y += gauss(x, 900, 200, 0.8*1e-3) * np.random.uniform(0.75, 1.25, len(x))


# data = np.loadtxt("Cs.SPC.dat", skiprows=1, unpack=True)
# x, y = data[0, :], data[1, :]
# k_peak = [[310, 400]]
# CSpara, CScov = gauss_plot_peak(x, y, k_peak=k_peak, filter_ampl=-1, 
#                                 TestPlot=0, PeakPlot=0, lw=2)

# data = np.loadtxt("Ba.SPC.dat", skiprows=1, unpack=True)
# x, y = data[0, :], data[1, :]
# # k_peak = k_auto_detect(x, y, size=20, order=15, PlotSmooth=0)
# k_peak = [[300, 390], [820, 960]]
# BApara, BAcov = gauss_plot_peak(x, y, k_peak=k_peak, filter_ampl=0, 
#                                 TestPlot=0, PeakPlot=0, lw=2)

# data = np.loadtxt("Eu.SPC.dat", skiprows=1, unpack=True)
# x, y = data[0, :], data[1, :]
# k_peak = [[110, 150], [370, 500]]
# EUpara, EUcov = gauss_plot_peak(x, y, k_peak=k_peak, filter_ampl=-1, 
#                                 TestPlot=0, PeakPlot=0, lw=2)

# data = np.loadtxt("Am.SPC.dat", skiprows=1, unpack=True)
# x, y = data[0, :], data[1, :]
# # k_peak = k_auto_detect(x, y, size=20, order=15, PlotSmooth=0)
# k_peak = [[600, 700]]
# AMpara, AMcov = gauss_plot_peak(x, y, k_peak=k_peak, filter_ampl=0, 
#                                 TestPlot=0, PeakPlot=0, lw=2)

# xdata = np.array([y[0] for x in [CSpara, BApara, EUpara, AMpara] 
#                   for y in x])
# xerror = np.array([np.sqrt(y[0][0]) for x in [CScov, BAcov, EUcov, AMcov]
#                    for y in x])

ydata = avg_energy()
xdata = np.array([355.13385831, 348.798238, 888.56000149, 
                  131.82406667, 442.96658445, 655.2609371 ])
xerror = np.array([1.43665192, 0.22605779, 0.19445538,
                    0.50495716, 0.61847482, 0.11869378])
# print(xdata, xerror, ydata)
p0 = [0.097, -3.71]
Eparam, Ecov = curve_fit(E, xdata, ydata, p0)

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# x = np.arange(1, 1025, 1)
# ax.errorbar(xdata, ydata, xerr=xerror, ls='', marker='x', ms=8, mew=1, 
#             c='b', label='Daten', capsize=5)
# ax.plot(x, E(x, *Eparam), lw=1, c='r', 
#         label=r'$E(k)\ =\ ({}\cdot k {})\,keV$'
#         .format(round(Eparam[0], 3), round(Eparam[1], 3)))
# plot_params(ax, 'Kanalnummer', r'$E\ /\ keV$', 'Kalibrierungskurve')

# [xN0, yN0] = np.loadtxt("Am20min_ohneStab.SPC.dat", skiprows=1, unpack=True)
# [xNg, yNg] = np.loadtxt("Am20min_mitStab.SPC.dat", skiprows=1, unpack=True)

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# ax.plot(xN0, yN0, lw=1, c='k', label='Untergrund')
# ax.plot(xNg, yNg, lw=1, c='b', label='mit Quelle')
# kmin, kmax = 500, 700
# param, cov, p0 = gauss_param(xNg[kmin:kmax], yNg[kmin:kmax])
# kmin = int(param[0] - 2*np.sqrt(param[1]))
# kmax = int(param[0] + 2*np.sqrt(param[1]))
           
# ax.plot(xNg[kmin:kmax], gauss(xNg[kmin:kmax], *param), lw=2, c='r',
#         label="Fit mit $k={}\pm {}$"
#         .format(round(param[0], 2), round(np.sqrt(cov[0][0]), 3)))
# plot_params(ax, 'Kanalnummer', 'counts', 'Zeitoptimierung')

# print(param[0], np.sqrt(param[1]), kmin, kmax)
# n0 = sum(yN0[kmin:kmax])
# ng = sum(yNg[kmin:kmax])
# print(time_optimum(ng, n0, alpha=0.05), ng, n0)


####### integrated counts for angle dependance

# fig, ax = plt.subplots(2, 4, figsize=(15, 10))
# i = 0
# ax = ax.flatten()
# nval = []
# ngval, n0val = [], []
# for filename in [["30grad.SPC.dat", "30grad0.SPC.dat"], 
#                   ["55grad.SPC.dat", "55grad0.SPC.dat"], 
#                   ["70grad.SPC.dat", "70grad0.SPC.dat"], 
#                   ["Am20min_mitStab.SPC.dat", "Am20min_ohneStab.SPC.dat"], 
#                   ["100grad.SPC.dat", "100grad0.SPC.dat"], 
#                   ["115grad.SPC.dat", "115grad0.SPC.dat"], 
#                   ["130grad.SPC.dat", "130grad0.SPC.dat"]]:
#     data = np.loadtxt(filename[0], skiprows=1, unpack=True)
#     data0 = np.loadtxt(filename[1], skiprows=1, unpack=True)
#     x, yg, y0 = data[0, :], data[1, :], data0[1, :]
#     y = yg - y0
#     k_peak = [[400, 700]]
#     # k30para, e30cov = gauss_plot_peak(x, y, k_peak=k_peak, lw=2)
#     # kval, errorval = k30para[0][0], np.sqrt(e30cov[0][0][0])
#     param, cov, p0 = gauss_param(x, y)
#     kval, errorval = param[0], np.sqrt(param[1])
#     kmin, kmax = int(kval - 2*errorval), int(kval + 2*errorval)
#     # print(kval, errorval)
#     ngval.append(sum(yg[kmin:kmax]))
#     n0val.append(sum(y0[kmin:kmax]))
#     nval.append(sum(y[kmin:kmax]))
#     ax[i].plot(x[kmin:kmax], y[kmin:kmax], lw=1)
#     ax[i].plot(x[kmin:kmax], gauss(x[kmin:kmax], *param), lw=1)
#     print(i, filename[0], nval[i], ngval[i], n0val[i])
#     i += 1
# plt.show()

[a, E0] = Eparam
delta_a, delta_E0 = np.sqrt(Ecov[0][0]), np.sqrt(Ecov[1][1])
# print(a, delta_a, E0, delta_E0)

# print(nval, ngval, n0val)

# nval = np.array([2654.0, 1908.0, 1535.0, 5086.0, 1232.0, 1207.0, 1351.0] )
# ngval = np.array([5843.0, 2774.0, 2267.0, 8270.0, 1598.0, 1533.0, 1658.0] )
# n0val = np.array([3189.0, 866.0, 732.0, 3184.0, 366.0, 326.0, 307.0])
# tval = np.array([270, 270, 270, 1200, 270, 270, 270])
# for array in [nval, ngval, n0val]:
#     array /= tval

# # print(nval, ngval, n0val)

# kval = np.array([629.5641550551994, 607.9196371286355, 594.233325901187,
#                   579.3785409491516, 565.3630132274805, 551.9420390596441,
#                   538.7626201071939])
# errorval = np.array([2.156731385615297, 2.168077970545, 3.145044184125872,
#                       1.3998573900880342, 2.227852790555779, 
#                       2.667027838039188, 2.1938136143463436])

# thetaval = np.array([30, 55, 70, 90, 100, 115, 130])
# muval = np.cos(thetaval * np.pi/180)
# mu = np.linspace(np.cos(140 * np.pi/180), np.cos(20 * np.pi/180), 300)

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# xerror = 2 * np.pi/180 * np.sin(thetaval * np.pi/180)
# yerror = np.sqrt((a * errorval)**2 + (kval * delta_a + delta_E0)**2)
# ax.errorbar(muval, E(kval, *Eparam), xerr=xerror, yerr=yerror, 
#             ls='', c='b', marker='x', mew=1, 
#             ms=8, label='Messwerte')
# ax.plot(mu, cse_mu(mu) * 1e-3, lw=1, c='r', label='Theorie')
# plot_params(ax, r'$\mu = \cos\theta$', r'$Energie\,/\,keV$', 
#             'Winkelabhängigkeit der Energie')

# def sigma_kn_mu_fixE(mu, norm=1):
#     return sigma_kn_mu(mu, E=59540, norm=norm)
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# Nerror = np.sqrt(ngval / tval + n0val / tval)
# ax.errorbar(muval, nval, xerr=xerror, yerr=Nerror, 
#             ls='', c='b', marker='x', mew=1, 
#             ms=8, label='Messwerte')
# norm0 = nval[3] / sigma_kn_mu()
# norm, norm_cov = curve_fit(sigma_kn_mu_fixE, muval, nval, p0=norm0)
# ax.plot(mu, sigma_kn_mu(mu, norm=norm[0]), lw=1, c='r', label='Theorie')
# plot_params(ax, r'$\mu = \cos\theta$', r'$\dot{N} = \dot{N}_g - \dot{N}_0$', 
#             'Winkelabhängigkeit des Wirkungsquerschnitts')

# for i in range(len(nval)):
#     print(thetaval[i], "&", round(kval[i], 2), "&", round(errorval[i], 3),
#           "&", round(E(kval[i], *Eparam), 2), "&", round(yerror[i], 3), "&",
#           round(nval[i], 2), "&", round(Nerror[i], 3))

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# x0, y0 = np.loadtxt("55grad0.SPC.dat", skiprows=1, unpack=True)
# filenames = [["30grad.SPC.dat", "30grad0.SPC.dat"], 
#              ["55grad.SPC.dat", "55grad0.SPC.dat"], 
#              ["70grad.SPC.dat", "70grad0.SPC.dat"], 
#              ["Am20min_mitStab.SPC.dat", "Am20min_ohneStab.SPC.dat"], 
#              ["100grad.SPC.dat", "100grad0.SPC.dat"], 
#              ["115grad.SPC.dat", "115grad0.SPC.dat"], 
#              ["130grad.SPC.dat", "130grad0.SPC.dat"]]
# for i, filename in enumerate(filenames):
#     data = np.loadtxt(filename[0], skiprows=1, unpack=True)
#     data0 = np.loadtxt(filename[1], skiprows=1, unpack=True)
#     x, yg, y0 = data[0, :], data[1, :], data0[1, :]
#     y = yg - y0
#     k_peak = [[400, 700]]
#     param, cov, p0 = gauss_param(x, y)
#     kval, errorval = param[0], np.sqrt(param[1])
#     kmin, kmax = int(kval - 2*errorval), int(kval + 2*errorval)
#     ax.plot(x[kmin:kmax], gauss(x[kmin:kmax], *param) / tval[i],
#             c=c_elem(len(filenames))[i], lw=2,
#             label=r'$\theta\ =\ {}\,\degree$'.format(thetaval[i]))
# plot_params(ax, 'Kanalnummer', r'$\dot{N} = \dot{N}_g - \dot{N}_0$', 
#             'Abhängigkeit vom Streuwinkel', hline=0)

############# Durchmesser der Streukoerper
# fig, ax = plt.subplots(2, 4, figsize=(15, 10))
# i = 0
# ax = ax.flatten()
# nval, ngval, n0val, kvals, kerror = [], [], [], [], []
# x0, data0 = np.loadtxt("55grad0.SPC.dat", skiprows=1, unpack=True)
# for filename in ["55grad2mm.SPC.dat", "55grad4mm.SPC.dat",
#                  "55grad.SPC.dat", "55grad10mm.SPC.dat",
#                  "55grad15mm.SPC.dat"]:
#     x, data = np.loadtxt(filename, skiprows=1, unpack=True)
#     y = data - data0
#     param, cov, p0 = gauss_param(x, y)
#     kval, errorval = param[0], np.sqrt(param[1])
#     kmin, kmax = int(kval - 2*errorval), int(kval + 2*errorval)
#     ngval.append(sum(data[kmin:kmax]))
#     n0val.append(sum(data0[kmin:kmax]))
#     nval.append(sum(y[kmin:kmax]))
#     kvals.append(kval)
#     kerror.append(np.sqrt(cov[0][0]))  
#     ax[i].plot(x[kmin:kmax], y[kmin:kmax], lw=1)
#     ax[i].plot(x[kmin:kmax], gauss(x[kmin:kmax], *param), lw=1)
#     i += 1
#     # gauss_plot_peak(x, y, [[400, 800]], filter_ampl=-1000.0, 
#     #                 FiltDataPlot=1)
# plt.show()
    
# print(nval, ngval, n0val, kvals, kerror)

###### Plots

# nval = np.array([315.0, 1062.0, 1908.0, 2858.0, 2768.0] )
# ngval = np.array([962.0, 1958.0, 2774.0, 3762.0, 3658.0] )
# n0val = np.array([647.0, 896.0, 866.0, 904.0, 890.0] )
# kval = np.array([613.9232795130902, 606.9769608086041, 608.8826274181589,
#                   604.5298901302477, 603.2899265303] )
# errorval = np.array([2.16462067846562, 1.313478352671718, 0.749917547227687,
#                       0.637769622665455, 0.611986607456738])

# for array in [nval, ngval, n0val]:
#     array /= 270
    
# def counts_per_diameter(dia, lamda=1, N0=0):
#     return lamda * dia + N0

# dval = np.array([2, 4, 6, 10, 15])
# dval2, nval2 = dval[:-1], nval[:-1]
# d = np.linspace(1, 16, 300)
# dparam, dcov = curve_fit(counts_per_diameter, dval2, nval2, p0=[10, 1])

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# Nerror = np.sqrt((ngval + n0val) / 270)
# ax.errorbar(dval, nval, yerr=Nerror, 
#             ls='', c='b', marker='x', mew=1, 
#             ms=8, label='Messwerte')
# ax.plot(d, counts_per_diameter(d, *dparam), 
#         lw=1, c='r', label=r'$\dot{N}=$' + r'$({}\cdot $'
#         .format(round(dparam[0], 2)) + r'$\frac{d}{\mathrm{mm}}$' 
#         + r' ${})\,$'.format(round(dparam[1], 3)) + r'$s^{-1}$')
# print(dparam[0], np.sqrt(dcov[0][0]), dparam[1], np.sqrt(dcov[1][1]))
# plot_params(ax, r'$d\ /\ mm$', r'$\dot{N} = \dot{N}_g - \dot{N}_0$', 
#             'Abhängigkeit der Zählrate vom Durchmesser des Streukörpers')

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# x0, y0 = np.loadtxt("55grad0.SPC.dat", skiprows=1, unpack=True)
# filenames = ["55grad2mm.SPC.dat", "55grad4mm.SPC.dat",
#               "55grad.SPC.dat", "55grad10mm.SPC.dat",
#               "55grad15mm.SPC.dat"]
# for i, filename in enumerate(filenames):
#     x, y = np.loadtxt(filename, skiprows=1, unpack=True)
#     param, cov, p0 = gauss_param(x, y - y0)
#     kval, errorval = param[0], np.sqrt(param[1])
#     kmin, kmax = int(kval - 2*errorval), int(kval + 2*errorval)
#     ax.plot(x, y - y0, lw=0.7, c=[0, 0, 0, 0.25])
#     ax.plot(x[kmin:kmax], gauss(x[kmin:kmax], *param),
#             c=c_elem(len(filenames))[i], lw=2,
#             label=r'$d\ =\ {}\,mm$'.format(dval[i]))
# plot_params(ax, 'Kanalnummer', r'$\dot{N} = \dot{N}_g - \dot{N}_0$', 
#             'Abhängigkeit vom Durchmesser')

# for i in range(len(nval)):
#     print(dval[i], "&", round(kval[i], 2), "&", round(errorval[i], 3), "&",
#           round(nval[i], 2), "&", round(Nerror[i], 3), "\\\ \hline")

########## Abhaengigkeit vom Abstand zur Quelle
filenames = [["55grad.SPC.dat", "55grad0.SPC.dat"], 
             ["55grad5cm.SPC.dat", "55grad5cm0.SPC.dat"],
             ["55grad7cm.SPC.dat", "55grad7cm0.SPC.dat"], 
             ["55grad9cm.SPC.dat", "55grad9cm0.SPC.dat"],
             ["55grad11cm.SPC.dat", "55grad11cm0.SPC.dat"]]
# fig, ax = plt.subplots(2, 4, figsize=(15, 10))
# i = 0
# ax = ax.flatten()
# nval, ngval, n0val, kvals, kerror = [], [], [], [], []
# for filename in filenames:
#     x, data = np.loadtxt(filename[0], skiprows=1, unpack=True)
#     x0, data0 = np.loadtxt(filename[1], skiprows=1, unpack=True)
#     y = data - data0
#     param, cov, p0 = gauss_param(x, y)
#     kval, errorval = param[0], np.sqrt(param[1])
#     kmin, kmax = int(kval - 2*errorval), int(kval + 2*errorval)
#     ngval.append(sum(data[kmin:kmax]))
#     n0val.append(sum(data0[kmin:kmax]))
#     nval.append(sum(y[kmin:kmax]))
#     kvals.append(kval)
#     kerror.append(np.sqrt(cov[0][0]))  
#     ax[i].plot(x[kmin:kmax], y[kmin:kmax], lw=1)
#     ax[i].plot(x[kmin:kmax], gauss(x[kmin:kmax], *param), lw=1)
#     i += 1
#     # gauss_plot_peak(x, y, [[400, 800]], filter_ampl=-1000.0, 
#     #                 FiltDataPlot=1)
# plt.show()
    
# print(nval, ngval, n0val, kvals, kerror)

# ##### Plots

# nval = np.array([1908.0, 1186.0, 763.0, 716.0, 506.0])
# ngval = np.array([2774.0, 1932.0, 1312.0, 1229.0, 971.0])
# n0val = np.array([866.0, 746.0, 549.0, 513.0, 465.0])
# kval = np.array([608.8826274181589, 604.8681627440993, 600.0475861363893,
#                   602.2486555033256, 604.1792803359614])
# errorval = np.array([0.7499175472276868, 1.2263496478145635,
#                       1.5556109984784103, 1.8377614538244191,
#                       2.3404841779346452])

# for array in [nval, ngval, n0val]:
#     array /= 270
    
# def counts_per_distance(r, gamma, r0):
#     return gamma / r**2 + r0

# rval = np.array([3, 5, 7, 9, 11])
# r = np.linspace(3, 14, 300)
# rparam, rcov = curve_fit(counts_per_distance, rval, nval, p0=[10, 1])

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# Nerror = np.sqrt((ngval + n0val) / 270)
# ax.errorbar(1 / rval**2, nval, yerr=Nerror, 
#             ls='', c='b', marker='x', mew=1, 
#             ms=8, label='Messwerte')
# ax.plot(1 / r**2, counts_per_distance(r, *rparam), c='r', 
#         lw=1, label=r'$\dot{N}=\left(\frac{gamma}{rsq}+{f0}\right)$'
#         .format(N="{N}", gamma="{47.8}", 
#                 rsq="{r^2\,/\,cm^2}", f0=round(rparam[1], 3)) 
#         + r'$\,\mathrm{s}^{-1}$')
# print(rparam[0], np.sqrt(rcov[0][0]), rparam[1], np.sqrt(rcov[1][1]))
# plot_params(ax, r'$1/r^2\ /\ cm^{-2}$', 
#             r'$\dot{N}\ = \dot{N}_g - \dot{N}_0 $', 
#             'Einfluss des Abstands der Quelle zum Streukörpers')

# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# for i, filename in enumerate(filenames):
#     x, y = np.loadtxt(filename[0], skiprows=1, unpack=True)
#     x0, y0 = np.loadtxt(filename[1], skiprows=1, unpack=True)
#     param, cov, p0 = gauss_param(x, y - y0)
#     kval, errorval = param[0], np.sqrt(param[1])
#     kmin, kmax = int(kval - 2*errorval), int(kval + 2*errorval)
#     if i == 0:
#         y_gauss = gauss(x[kmin:kmax], *param)
#         y_max = max(y_gauss)
#         scale_factor = 1
#     else:
#         y_gauss = gauss(x[kmin:kmax], *param)
#         scale_factor = y_max / max(y_gauss)
#     ax[0].plot(x, y - y0, lw=0.7, c=[0, 0, 0, 0.25])
#     ax[0].plot(x[kmin:kmax], gauss(x[kmin:kmax], *param), 
#                 c=c_elem(len(filenames))[i], lw=2,
#                 label=r'$r\ =\ {}\,cm$'.format(rval[i]))
#     ax[1].plot(x[kmin:kmax], scale_factor * y_gauss,
#                 c=c_elem(len(filenames))[i], lw=2,
#                 label=r'$r\ =\ {}\,cm$'.format(rval[i]))
# plot_params(ax[0], 'Kanalnummer', 'Zählrate in bel. Einheiten', 
#             'Einfluss des Abstands der Quelle zum Streukörpers')
# plot_params(ax[1], 'Kanalnummer', 'Zählrate in bel. Einheiten', 
#             'Einfluss des Abstands der Quelle zum Streukörpers')


# for i in range(len(nval)):
#     print(rval[i], "&", round(kval[i], 2), "&", round(errorval[i], 3), "&",
#           round(nval[i], 2), "&", round(Nerror[i], 3), "\\\ \hline")

########## Theorie-Plots

print(cse_mu(E=26340))

# param = [6*1e3, 6*1e4, 6*1e5, 6*1e6]
# mu = np.linspace(-1, 1, 300)    
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# for i, p in enumerate(param):
#     ax.plot(mu, sigma_kn_mu(mu, p), lw=1, c=c_elem(len(param))[i])
# plot_params(ax, r'$\mu$', r'$\sigma$')


# param = np.array([0, 10, 20, 30, 45, 60, 90, 180]) * np.pi/180
# E = 10**np.linspace(4, 8, 300)    
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# for i, p in enumerate(param):
#     ax.plot(E, cse(p, E), lw=1, c=c_elem(len(param))[i])
# plot_params(ax, r'$\mu$', r'$\sigma$', xlog=True, ylog=True)

