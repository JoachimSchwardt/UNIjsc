"""
Auswertung zum Versuch SZ am 04.12.2020.
Fuer den Ablauf des Programmes siehe "Hinweise zur Auswertung - Versuch 
Solarzelle".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit


########################################################
#Entpackung der Dateien

#Aufgabenteil A
a_d_an_u, a_d_an_i, temp = np.loadtxt("a_d_an.dat", skiprows=11, 
                                      max_rows=97, unpack=True)
a_d_or_u, a_d_or_i, temp = np.loadtxt("a_d_or.dat", skiprows=11, 
                                      max_rows=101, unpack=True)

a_h_an_u, a_h_an_i, temp = np.loadtxt("a_h_an.dat", skiprows=11, 
                                      max_rows=101, unpack=True)
a_h_or_u, a_h_or_i, temp = np.loadtxt("a_h_or.dat", skiprows=11, 
                                      max_rows=101, unpack=True)


#Aufgabenteil B
b_5_u, b_5_i, temp     = np.loadtxt("b_5.dat", skiprows=11, 
                                    max_rows=96, unpack=True)
b_10_u, b_10_i, temp   = np.loadtxt("b_10.dat", skiprows=11, 
                                    max_rows=96, unpack=True)
b_50_u, b_50_i, temp   = np.loadtxt("b_50.dat", skiprows=11, 
                                    max_rows=96, unpack=True)
b_75_u, b_75_i, temp   = np.loadtxt("b_75.dat", skiprows=11, 
                                    max_rows=96, unpack=True)
b_100_u, b_100_i, temp = np.loadtxt("b_100.dat", skiprows=11, 
                                    max_rows=97, unpack=True)


#Aufgabenteil C
c1_d_u, c1_d_i, temp = np.loadtxt("c1_d.dat", skiprows=11, 
                                  max_rows=52, unpack=True)
c1_h_u, c1_h_i, temp = np.loadtxt("c1_h.dat", skiprows=11, 
                                  max_rows=94, unpack=True)

c2_h_p1_u, c2_h_p1_i, temp   = np.loadtxt("c2_h_p1.dat", skiprows=11, 
                                          max_rows=68, unpack=True)
c2_h_p2_u, c2_h_p2_i, temp   = np.loadtxt("c2_h_p2.dat", skiprows=11, 
                                          max_rows=68, unpack=True)
c2_h_p3_u, c2_h_p3_i, temp   = np.loadtxt("c2_h_p3.dat", skiprows=11, 
                                          max_rows=68, unpack=True)
c2_h_ps1_u, c2_h_ps1_i, temp = np.loadtxt("c2_h_ps1.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c2_h_ps2_u, c2_h_ps2_i, temp = np.loadtxt("c2_h_ps2.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c2_h_s1_u, c2_h_s1_i, temp   = np.loadtxt("c2_h_s1.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c2_h_s2_u, c2_h_s2_i, temp   = np.loadtxt("c2_h_s2.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
                                        
c3_h_v1_u, c3_h_v1_i, temp   = np.loadtxt("c3_h_v1.dat", skiprows=11, 
                                          max_rows=98, unpack=True)
# c3_h_v2_u, c3_h_v2_i, temp   = np.loadtxt("c3_h_v2.dat", skiprows=11, 
#                                          max_rows=96, unpack=True)
c3_h_v2b_u, c3_h_v2b_i, temp = np.loadtxt("c3_h_v2b.dat", skiprows=11, 
                                          max_rows=94, unpack=True)
# c3_h_v3_u, c3_h_v3_i, temp   = np.loadtxt("c3_h_v3.dat", skiprows=11, 
#                                          max_rows=96, unpack=True)
c3_h_v3b_u, c3_h_v3b_i, temp = np.loadtxt("c3_h_v3b.dat", skiprows=11, 
                                          max_rows=93, unpack=True)

c4_d1_u, c4_d1_i, temp       = np.loadtxt("c4_d1.dat", skiprows=11, 
                                          max_rows=100, unpack=True)            
c4_d2_u, c4_d2_i, temp       = np.loadtxt("c4_d2.dat", skiprows=11, 
                                          max_rows=100, unpack=True)
c4_d3_u, c4_d3_i, temp       = np.loadtxt("c4_d3.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c4_d_v1_u, c4_d_v1_i, temp   = np.loadtxt("c4_d_v1.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c4_d_v2_u, c4_d_v2_i, temp   = np.loadtxt("c4_d_v2.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c4_d_v3_u, c4_d_v3_i, temp   = np.loadtxt("c4_d_v3.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c4_d_v3m_u, c4_d_v3m_i, temp = np.loadtxt("c4_d_v3m.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c4_h_u, c4_h_i, temp         = np.loadtxt("c4_h.dat", skiprows=11, 
                                          max_rows=101, unpack=True)
c4_h_v1_u, c4_h_v1_i, temp   = np.loadtxt("c4_h_v1.dat", skiprows=11, 
                                          max_rows=101, unpack=True)                 
c4_h_v2_u, c4_h_v2_i, temp   = np.loadtxt("c4_h_v2.dat", skiprows=11, 
                                          max_rows=101, unpack=True)


#Aufgabenteil D
d_t = [30,  35,  40,  45,  50,  55,  60,  65]
d_u = [570, 565, 555, 550, 546, 541, 533, 523]


#Aufgabenteil E
e_0_an_i   = np.loadtxt("e_0_an.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_0_or1_i  = np.loadtxt("e_0_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
# e_0_or2_i  = np.loadtxt("e_0_or2.dat", skiprows=11,    #Ausreißer
#                         max_rows=1, unpack=False)[1]

e_10_an_i  = np.loadtxt("e_10_an.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_10_or1_i = np.loadtxt("e_10_or1.dat", skiprows=11,
                        max_rows=1, unpack=False)[1]
e_10_or2_i = np.loadtxt("e_10_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]

e_20_an1_i = np.loadtxt("e_20_an1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_20_an2_i = np.loadtxt("e_20_an2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_20_or1_i = np.loadtxt("e_20_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_20_or2_i = np.loadtxt("e_20_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]

e_30_an1_i = np.loadtxt("e_30_an1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_30_an2_i = np.loadtxt("e_30_an2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_30_or1_i = np.loadtxt("e_30_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_30_or2_i = np.loadtxt("e_30_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]

e_40_an1_i = np.loadtxt("e_40_an1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_40_an2_i = np.loadtxt("e_40_an2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_40_or1_i = np.loadtxt("e_40_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_40_or2_i = np.loadtxt("e_40_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]

e_50_an1_i = np.loadtxt("e_50_an1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_50_an2_i = np.loadtxt("e_50_an2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_50_or1_i = np.loadtxt("e_50_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_50_or2_i = np.loadtxt("e_50_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]

e_60_an1_i = np.loadtxt("e_60_an1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_60_an2_i = np.loadtxt("e_60_an2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_60_or1_i = np.loadtxt("e_60_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_60_or2_i = np.loadtxt("e_60_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]

e_70_an1_i = np.loadtxt("e_70_an1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_70_an2_i = np.loadtxt("e_70_an2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_70_or1_i = np.loadtxt("e_70_or1.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]
e_70_or2_i = np.loadtxt("e_70_or2.dat", skiprows=11, 
                        max_rows=1, unpack=False)[1]



#Boltzmannkonstante ([kb] = J/K)
kb = 1.380649e-23
#Zellenflächen ([A] = m^2)
A_an = 0.0026
A_or = 0.000006
A_an, A_or, Sol = 26, 0.06, 0.1    # cm**2, cm**2, W/cm**2



#################################################
#Hauptprogramm

#Haeufig verwendete Funktionen
def j_k(u, i, A):
    u_min = 2
    for x in range(len(u)):
        if abs(u[x]) < u_min:
            u_min = abs(u[x])
            y = x
    return(i[y] / A)

def u_l(u, i):
    i_min = 1
    for x in range(len(u)):
        if abs(i[x]) < i_min:
            i_min = abs(i[x])
            y = x
    return(u[y])

def mlp(u, i):
    p_max = 0
    for x in range(len(u)):
        if i[x]<=0 and u[x]>=0 and abs(i[x] * u[x]) > p_max:
            p_max = abs(i[x] * u[x])
            y = x
    return([u[y], i[y], p_max])

def ff(u_oc, i_sc, p_mlp, p):
    return(abs(p_mlp / (u_oc * i_sc)))

def eta(p_mlp, p):
    return(abs(p_mlp / p))


#######################################    
#Aufgabenteil A
    
# def linfit(u, r_s):
#     return(u/r_s)

# def logfit(i, n, i_s):
#     return((kb*294.15/np.exp(1)) * n * np.log(abs(i) / i_s)) #T=21°C
        
# rs_an, covrs_an = curve_fit(linfit, a_d_an_u[:60], a_d_an_i[:60])

# logf, covlogf   = curve_fit(f = logfit, xdata = a_d_an_i[90:], 
#                             ydata = a_d_an_u[90:] - rs_an * a_d_an_i[90:])

# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("U' laut A1")
# ax.plot(np.log(abs(a_d_an_i)), a_d_an_u - a_d_an_i * rs_an)
# ax.plot(np.log(abs(a_d_an_i)), logfit(a_d_an_i, 1, 1))
# ax.grid(True)
# ax.set_xlabel("log(I)", fontsize=14)
# ax.set_ylabel("U' / V", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# plt.show()

print("Zu A1: Siehe Zeilen 217...237. Ich bin maximal verwirrt, warum ",
      "funktioniert das so nicht? Ich bekomme für n und i_s immer nur 1.\n",
      "(diesen Kommentar bitte löschen :D)")



# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("Dunkelkennlinie der anorganischen Zelle")
# ax.plot(a_d_an_u[40:68], a_d_an_i[40:68])
# ax.grid(True)
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("I / A", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("Hellkennlinie der anorganischen Zelle")
# ax.plot(a_h_an_u[45:90], a_h_an_i[45:90])
# ax.grid(True)
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("I / A", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("Dunkelkennlinie der organischen Zelle")
# ax.plot(a_d_or_u[30:74], a_d_or_i[30:74])
# ax.grid(True)
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("I / A", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("Hellkennlinie der organischen Zelle")
# ax.plot(a_h_or_u[30:79], a_h_or_i[30:79])
# ax.grid(True)
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("I / A", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# plt.show()


# #A2
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("Kennlinienvergleich der anorganischen Zelle")
# ax.plot(a_h_an_u, a_h_an_i / A_an, "^-", ms=8, markevery=200, 
#         label="Hellkennlinie")
# ax.plot(a_d_an_u, a_d_an_i / A_an, "v-", ms=8, markevery=200, 
#         label="Dunkelkennlinie")
# ax.grid(True)
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.legend(loc="upper left")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# fig.suptitle("Hellkennlinienvergleich")
# ax.plot(a_h_an_u, a_h_an_i / A_an, "^-", ms=8, markevery=200, 
#         label="anorganisch")
# ax.plot(a_h_or_u, a_h_or_i / A_or, "v-", ms=8, markevery=200, 
#         label="organisch")
# ax.grid(True)
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel(r"$j / A/m^2$", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.legend(loc="upper left")
# plt.show()

# print("Aufgabe A: Intensität = 1 Sonne\n\n")
# p = 1000 #W/m**2
# p = 1 * Sol    # W/cm**2
# j = j_k(a_h_an_u, a_h_an_i, A_an)
# u = u_l(a_h_an_u, a_h_an_i)
# u_m, i_m, p_m = mlp(a_h_an_u, a_h_an_i)
# print("Anorganisch, hell: \nj_sc = {} A/m^2, \nu_oc = {} V,\nMLP: ({} V, {} A/m^2)\nFF = {}, \nEta = {}\n"
#       .format(j, u, u_m, i_m / A_an, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(a_d_an_u, a_d_an_i, A_an)
# u = u_l(a_d_an_u, a_d_an_i)
# u_m, i_m, p_m = mlp(a_d_an_u, a_d_an_i)
# print("Anorganisch, dunkel: \nj_sc = {} A/m^2, \nu_oc = {} V,\nMLP: ({} V, {} A/m^2)\nFF = {}, \nEta = {}\n"
#       .format(j, u, u_m, i_m / A_an, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(a_h_or_u, a_h_or_i, A_or)
# u = u_l(a_h_or_u, a_h_or_i)
# u_m, i_m, p_m = mlp(a_h_or_u, a_h_or_i)
# print("Organisch, hell: \nj_sc = {} A/m^2, \nu_oc = {} V,\nMLP: ({} V, {} A/m^2)\nFF = {}, \nEta = {}\n"
#       .format(j, u, u_m, i_m / A_or, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(a_d_or_u, a_d_or_i, A_or)
# u = u_l(a_d_or_u, a_d_or_i)
# u_m, i_m, p_m = mlp(a_d_or_u, a_d_or_i)
# print("Organisch, dunkel: \nj_sc = {} A/m^2, \nu_oc = {} V,\nMLP: ({} V, {} A/m^2)\nFF = {}, \nEta = {}\n"
#       .format(j, u, u_m, i_m / A_or, ff(u, j, p_m, p), eta(p_m, p)))


# #Aufgabenteil B
# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("j(U) bei verschiedenen Intensitäten, 100% ≙ 1 Sonne", 
#              fontsize=16)
# ax.plot(b_5_u[45:84],   b_5_i[45:84]   / 0.0026, "o-", markevery=200, ms=8, 
#         label="5%")
# ax.plot(b_10_u[45:84],  b_10_i[45:84]  / 0.0026, "^-", markevery=200, ms=8, 
#         label="10%")
# ax.plot(b_50_u[45:84],  b_50_i[45:84]  / 0.0026, "v-", markevery=200, ms=8, 
#         label="50%")
# ax.plot(b_75_u[45:84],  b_75_i[45:84]  / 0.0026, "<-", markevery=200, ms=8, 
#         label="75%")
# ax.plot(b_100_u[45:84], b_100_i[45:84] / 0.0026, ">-", markevery=200, ms=8, 
#         label="100%")

# ax.legend(loc="upper left")
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.grid(True)
# plt.show()

# b_j = [j_k(b_5_u,   b_5_i,   A_an),
#        j_k(b_10_u,  b_10_i,  A_an),
#        j_k(b_50_u,  b_50_i,  A_an),
#        j_k(b_75_u,  b_75_i,  A_an),
#        j_k(b_100_u, b_100_i, A_an),]
# b_u = [u_l(b_5_u,   b_5_i),
#        u_l(b_10_u,  b_10_i),
#        u_l(b_50_u,  b_50_i),
#        u_l(b_75_u,  b_75_i),
#        u_l(b_100_u, b_100_i),]
# b_int = np.array([1.7, 3.2, 15.8, 23.6, 31.5]) * 1000 / 32.2

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Abhängigkeit des Kurzschlussstroms von der Intensität",
#              fontsize=16)
# ax.set_xlabel("I / W/m^2", fontsize=14)
# ax.set_ylabel("j_sc / A/m^2", fontsize=14)
# ax.plot(b_int, b_j, "x")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Abhängigkeit der Leerlaufspannung von der Intensität",
#              fontsize=16)
# ax.set_xlabel("I / W/m^2", fontsize=14)
# ax.set_ylabel("U_oc / V", fontsize=14)
# ax.plot(b_int, b_u, "x")
# plt.show()



# #Aufgabenteil C

# #Intensität = 1/3 Sonne:
# p = (10.3/31.5) * 1000


# #C1, C2
# c2_p_u  = (c2_h_p1_u + c2_h_p2_u + c2_h_p3_u) / 3
# c2_s_u  = (c2_h_s1_u + c2_h_s2_u) / 2
# c2_ps_u = (c2_h_ps1_u + c2_h_ps2_u) / 2

# c2_p_i  = (c2_h_p1_i + c2_h_p2_i + c2_h_p3_i) / 3
# c2_s_i  = (c2_h_s1_i + c2_h_s2_i) / 2
# c2_ps_i = (c2_h_ps1_i + c2_h_ps2_i) / 2

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Kennlinien des Moduls bei verschiedenen Schaltungen", 
#              fontsize=16)
# ax.plot(c1_h_u[7:75],  c1_h_i[7:75]  / (A_an*6), "o-", markevery=93, ms=8, 
#         label="Gesamt")
# ax.plot(c2_p_u[5:55],  c2_p_i[5:55]  / (A_an*6), "^-", markevery=67, ms=8, 
#         label="Parallel")
# ax.plot(c2_s_u[5:60],  c2_s_i[5:60]  / (A_an*6), ">-", markevery=200, ms=8, 
#         label="Serie")
# ax.plot(c2_ps_u[5:60], c2_ps_i[5:60] / (A_an*6), "<-", markevery=200, ms=8, 
#         label="Parallel und Serie")
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.grid(True)
# ax.legend(loc="upper left")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Kennlinien des Moduls bei verschiedenen Schaltungen, gesamter Messbereich", 
#              fontsize=16)
# ax.plot(c1_h_u,  c1_h_i  / (A_an*6), "o-", markevery=93, ms=8, 
#         label="Gesamt")
# ax.plot(c2_p_u,  c2_p_i  / (A_an*6), "^-", markevery=67, ms=8, 
#         label="Parallel")
# ax.plot(c2_s_u,  c2_s_i  / (A_an*6), ">-", markevery=200, ms=8, 
#         label="Serie")
# ax.plot(c2_ps_u, c2_ps_i / (A_an*6), "<-", markevery=200, ms=8, 
#         label="Parallel und Serie")
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.grid(True)
# ax.legend(loc="upper left")
# plt.show()

# print("Aufgabe C: Intensität = 1/3 Sonne\n\n")
# j = j_k(c1_h_u, c1_h_i, A_an*6)
# u = u_l(c1_h_u, c1_h_i)
# u_m, i_m, p_m = mlp(c1_h_u, c1_h_i)
# print("Modul gesamt: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}, \nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(c2_p_u, c2_p_i, A_an*6)
# u = u_l(c2_p_u, c2_p_i)
# u_m, i_m, p_m = mlp(c2_p_u, c2_p_i)
# print("Modul, Parallelschaltung: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}, \nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(c2_s_u, c2_s_i, A_an*6)
# u = u_l(c2_s_u, c2_s_i)
# u_m, i_m, p_m = mlp(c2_s_u, c2_s_i)
# print("Modul, Serienschaltung: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}, \nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(c2_ps_u, c2_ps_i, A_an*6)
# u = u_l(c2_ps_u, c2_ps_i)
# u_m, i_m, p_m = mlp(c2_ps_u, c2_ps_i)
# print("Modul, Parallel- und Serienschaltung: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}\nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))


# #C3
# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Teilverschattungen", fontsize=16)
# ax.plot(c1_h_u[15:75], c1_h_i[15:75] / (A_an*6), "o-", markevery=200, 
#         ms=8, label="Keine Verschattung")
# ax.plot(c3_h_v1_u[15:75], c3_h_v1_i[15:75] / (A_an*5), "^-", markevery=200, 
#         ms=8, label="Version 1")
# ax.plot(c3_h_v2b_u[15:74], c3_h_v2b_i[15:74] / (A_an*5), "<-", markevery=58, 
#         ms=8, label="Version 2")
# ax.plot(c3_h_v3b_u[15:72], c3_h_v3b_i[15:72] / (A_an*3), ">-", markevery=56, 
#         ms=8, label="Version 3")
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.grid(True)
# ax.legend(loc="upper left")
# plt.show()

# j = j_k(c3_h_v1_u, c3_h_v1_i, A_an*5)
# u = u_l(c3_h_v1_u, c3_h_v1_i)
# u_m, i_m, p_m = mlp(c3_h_v1_u, c3_h_v1_i)
# print("Modul, Version 1: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}\nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(c3_h_v2b_u, c3_h_v2b_i, A_an*5)
# u = u_l(c3_h_v2b_u, c3_h_v2b_i)
# u_m, i_m, p_m = mlp(c3_h_v2b_u, c3_h_v2b_i)
# print("Modul, Version 2: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}\nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))

# j = j_k(c3_h_v3b_u, c3_h_v3b_i, A_an*3)
# u = u_l(c3_h_v3b_u, c3_h_v3b_i)
# u_m, i_m, p_m = mlp(c3_h_v3b_u, c3_h_v3b_i)
# print("Modul, Version 3: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}\nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))


# #C4
# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Großes Modul mit und ohne Verbraucher, Hellkennlinien", 
#              fontsize=16)
# ax.plot(c4_h_u, c4_h_i / (A_an*12))
# ax.plot(c4_h_v2_u, c4_h_v2_i / (A_an*12))
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.grid(True)
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# fig.suptitle("Großes Modul mit und ohne Verbraucher, Dunkelkennlinien", 
#              fontsize=16)
# ax.plot(c4_d3_u, c4_d3_i / (A_an*12))
# ax.plot(c4_d_v3_u, c4_d_v3_i / (A_an*12))
# ax.set_xlabel("U / V", fontsize=14)
# ax.set_ylabel("j / A/m^2", fontsize=14)
# ax.axvline(c="k")
# ax.axhline(c="k")
# ax.grid(True)
# plt.show()

# j = j_k(c4_h_u, c4_h_i, A_an*12)
# u = u_l(c4_h_u, c4_h_i)
# u_m, i_m, p_m = mlp(c4_h_u, c4_h_i)
# u_v = c4_h_v2_u[43]
# i_v = c4_h_v2_i[43]
# p_v = u_v * abs(i_v)
# print("Großes Modul, hell: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}\nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))
# print("Verbraucherarbeitspunkt, hell: ({} V, {} A/m^2) \n≙ P = {} W \n= {} * P_max\n"
#       .format(u_v, i_v, p_v, p_v / p_m))

# j = j_k(c4_d3_u, c4_d3_i, A_an*12)
# u = u_l(c4_d3_u, c4_d3_i)
# u_m, i_m, p_m = mlp(c4_d3_u, c4_d3_i)
# u_v = c4_d_v3_u[60]
# i_v = c4_d_v3_i[60]
# p_v = u_v * abs(i_v)
# print("Großes Modul, dunkel: \nj_sc = {} A/m^2, \nu_oc = {} V,\nFF = {}\nEta = {}\n"
#       .format(j, u, ff(u, j, p_m, p), eta(p_m, p)))
# print("Verbraucherarbeitspunkt, dunkel: ({} V, {} A/m^2) \n≙ P = {} W \n= {} * P_max\n"
#       .format(u_v, i_v, p_v, p_v / p_m))



#Aufgabenteil D
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
fig.suptitle("Temperaturabhängigkeit der Leerlaufspannung", fontsize=22)
ax.plot(d_t, d_u, ls='', marker='x', mew=1, ms=8, c='b')
ax.set_xlabel(r"$T\ /\ \degree C$", fontsize=22)
ax.set_ylabel(r"$U_L\ /\ mV$", fontsize=22)
ax.tick_params(labelsize=16)
ax.grid(True)
plt.show()



#Aufgabenteil E
e_j_an = abs(np.array([(e_0_an_i)                / (1 * A_an),
                       (e_10_an_i)               / (1 * A_an),
                       (e_20_an1_i + e_20_an2_i) / (2 * A_an),
                       (e_30_an1_i + e_30_an2_i) / (2 * A_an),
                       (e_40_an1_i + e_40_an2_i) / (2 * A_an),
                       (e_50_an1_i + e_50_an2_i) / (2 * A_an),
                       (e_60_an1_i + e_60_an2_i) / (2 * A_an),
                       (e_70_an1_i + e_70_an2_i) / (2 * A_an)]))

e_j_or = abs(np.array([(e_0_or1_i)               / (1 * A_or),
                       (e_10_or1_i + e_10_or2_i) / (2 * A_or),
                       (e_20_or1_i + e_20_or2_i) / (2 * A_or),
                       (e_30_or1_i + e_30_or2_i) / (2 * A_or),
                       (e_40_or1_i + e_40_or2_i) / (2 * A_or),
                       (e_50_or1_i + e_50_or2_i) / (2 * A_or),
                       (e_60_or1_i + e_60_or2_i) / (2 * A_or),
                       (e_70_or1_i + e_70_or2_i) / (2 * A_or)]))

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
fig.suptitle("Neigungsabhängigkeit des normierten Stroms", fontsize=22)
ax.plot(np.arange(0,80,10), e_j_an / e_j_an[0], 
        ls='', marker='x', mew=1, ms=8, c='b', label='anorganisch')
ax.plot(np.arange(0,80,10), e_j_or / e_j_or[0], 
        ls='', marker='+', mew=1, ms=10, c='r', label='organisch')
ax.set_xlabel(r"$\theta\ /\ \degree$", fontsize=22)
ax.set_ylabel(r"$I_K(\theta)\ /\ I_K(\theta=0\degree)$", fontsize=22)
ax.tick_params(labelsize=16)
ax.grid(True)
ax.legend(fontsize=20)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
fig.suptitle("Normierter Strom pro projezierter Fläche", fontsize=22)
theta = np.arange(0,80,10)
ax.plot(theta, e_j_an / e_j_an[0] * 1 / np.cos(theta * np.pi/180), 
        ls='', marker='x', mew=1, ms=8, c='b', label='anorganisch')
ax.plot(theta, e_j_or / e_j_or[0] * 1 / np.cos(theta * np.pi/180), 
        ls='', marker='+', mew=1, ms=10, c='r', label='organisch')
ax.set_xlabel(r"$\theta\ /\ \degree$", fontsize=22)
ax.set_ylabel(r"$I_K(\theta)\ /\ (I_K(\theta=0\degree)\cdot\cos(\theta))$",
              fontsize=22)
ax.tick_params(labelsize=16)
ax.grid(True)
ax.legend(fontsize=20)
plt.show()


##################################################
#Schlussanmerkungen

print("Der Stromeinbruch der unverschatteten Messreihe um 2.5 V kann dadurch",
      "verursacht worden sein, dass bei der Temperaturkontrolle ein Ärmel",
      "einen Teil des Moduls verschattet hat.\n")
print("Verbraucherarbeitspunkt dunkel: ???\n")