"""
Created on Fri Jul  5 19:44:28 2019

@author: Joachim
"""

import scipy as sc
from scipy.integrate import quad
from matplotlib import pyplot as plt

def F_integrand(x, k=0):
    return 1/sc.sqrt(1-k**2*sc.sin(x)**2)

def E_integrand(x, k=0):
    return sc.sqrt(1-k**2*sc.sin(x)**2)

def Pi_integrand(x, n=0, k=0):
    return 1/((1-n**2*sc.sin(x)**2)*sc.sqrt(1-k**2*sc.sin(x)**2))

def F(k=0, phi=sc.pi/2):
    return quad(F_integrand, 0, phi, args=(k))[0]

def K(k=0):
    return F(k, sc.pi/2)

def E(k=0, phi=sc.pi/2):
    return quad(E_integrand, 0, phi, args=(k))[0]

def Pi(k=0, n=0, phi=sc.pi/2):
    return quad(Pi_integrand, 0, phi, args=(k, n))[0]

def K_approx(k):
    c1 = 53/393
    c2 = -17/128
    c3 = 35/216
    c4 = -31/204
    return sc.pi * (0.5+(c1*k**2 + c2*k**4 + c3*k**6 + c4*k**8)/(1 - k**2))
    
#def K_plot(eps=1e-3, f=K):
#    t = sc.linspace(0, 1-eps, 1000)
#    K = sc.vectorize(f)
#
#    plt.figure(figsize=(14,10))   
#    plt.plot(t, K(t), c='k', label='Complete K (first order)') 
#    plt.axhline(0, c='k', lw=0.7)           
#    plt.axvline(0, c='k', lw=0.7)
#    plt.grid(True)
#    plt.legend(bbox_to_anchor=(0.25, 0.95))
#    plt.show()


eps = 5e-3
R=2
z=2
r=0

#def A_integrand(phi, r, R, z):
#    return 1/sc.sqrt(r**2+R**2+z**2-2*r*R*sc.cos(phi))

#def A(r=1, z=0.5, R=2):
#    return 4*K(sc.sqrt(4*r*R/((r+R)**2+z**2)))/sc.sqrt((r+R)**2+z**2)
#
def A_2(r=1, theta=sc.pi/4, R=2):
    k = sc.sqrt((4*r*R*sc.sin(theta))/(R**2+r**2+2*r*R*sc.sin(theta)))
    return R/sc.sqrt(R**2+r**2+2*r*R*sc.sin(theta))*(((2-k**2)*K(k)-2*E(k))/(k**2))

def Bz(r=1, z=0, R=2):
    alpha = sc.sqrt(R**2+r**2+z**2-2*r*R)
    beta = sc.sqrt(R**2+r**2+z**2+2*r*R)
    k = sc.sqrt(1-alpha**2/beta**2)
    return 1/(2*sc.pi*alpha**2*beta)*((R**2-r**2-z**2)*E(k)+alpha**2*K(k))#-0.25*R**2/(R**2+z**2)**(3/2)

def Bz_approx(r=1, z=0, R=2):
#    alpha = sc.sqrt(R**2+r**2+z**2-2*r*R)
#    beta = sc.sqrt(R**2+r**2+z**2+2*r*R)
#    k = sc.sqrt((4*r*R)/((r+R)**2+z**2))
    return R**2/2*(R**2-r**2-z**2)/((r**2-R**2)**2+(2*r*z)**2-z**4)*1/sc.sqrt(R**2+z**2+r**2)
#    return R**2/2*(R**2-r**2-z**2+r*z/2)/((r**2-R**2)**2+(2*r*z)**2-z**4)*1/(R**2+r**2+z**2)**(1/2)
    
def E1(r=1, z=0, R=2):
    alpha = sc.sqrt(R**2+r**2+z**2-2*r*R)
    beta = sc.sqrt(R**2+r**2+z**2+2*r*R)
    k = sc.sqrt((4*r*R)/((r+R)**2))
#    return 0.5/((r+R)**2+z**2)**(3/2)*(Pi(k,k)*(R**2-r**2-z**2)+((r+R)**2+z**2)*K(k))
    return 1/(2*beta)*(E(k)*(R**2-r**2-z**2)/alpha**2+K(k))

def K1(r=1, z=0, R=2):
#    return 1/abs(r**2-R**2)*(1-sc.exp(-1.5*sc.sqrt((r-R)**2)))
#    return 1/(R**2+r**2/(1+1/(1+z)))
    return 1/(R**2+r**2)

#print("Simplification: {}".format(4*K(sc.sqrt(4*r*R/((r+R)**2+z**2)))/sc.sqrt((r+R)**2+z**2)))
#print("Direct Integration: {}".format(quad(A_integrand, 0, 2*sc.pi, args=(r, R, z))[0]))
#print("A_2 result is: {}".format(4*A_2(2, 0+eps)))

t = sc.linspace(0, 1-eps, 1000)
K = sc.vectorize(K)
E = sc.vectorize(E)
Pi = sc.vectorize(Pi)
#K_approx = sc.vectorize(K_approx)
#A = sc.vectorize(A)
A_2 = sc.vectorize(A_2)
Bz = sc.vectorize(Bz)
a = sc.linspace(0, 10, 200)
#b = sc.concatenate([sc.linspace(0, 1.9, 50),sc.linspace(2.1, 8, 200)])
c = sc.linspace(0, 10, 200)
plt.figure(figsize=(14,10))   
#plt.ylim(-10, 10)
#plt.plot(t, K(t), c='k', label='Complete K') 
#plt.plot(t, E(t), c='k', label='Complete E') 
#plt.plot(t, Pi(t,t), c='k', label='Pi(x,x)')
#plt.plot(t, K_approx(t), c='b', label='K Approximation')

#plt.plot(a, A(a, 0), c='#00FF00', label='A(z=0)')
#plt.plot(a, A(a), c='#00FF8F', label='A(z=0.5)')
#plt.plot(a, A(a, 1), c='#00FFDF', label='A(z=1)')
#plt.plot(a, A(a, 2), c='#015FFF', label='A(z=2)')
#
#plt.plot(a, A_2(a, sc.pi/24), c='#00FF00', label='A_2(z=0)')
#plt.plot(a, A_2(a, sc.pi/6), c='#00FF8F', label='A_2(z=0.5)')
#plt.plot(a, A_2(a), c='#00FFDF', label='A_2(z=1)')
#plt.plot(a, A_2(a, sc.pi/3), c='#015FFF', label='A_2(z=2)')

z = 1.8
R = 2

#plt.plot(a, Bz(a, 0.2, R), c='#00FF00', label='Bz(z=0.2)')
#plt.plot(a, Bz(a, 0.5, R), c='#00FF8F', label='Bz(z=0.5)')
plt.plot(a, Bz(a, z, R), c='#00FFDF', label='Bz(z=1)')
#plt.plot(a, Bz(a, 2), c='#015FFF', label='Bz(z=2)')

#plt.plot(a, Bz_approx(a, 0.2, R), c='k', label='Bz_approx(z=0.2)')
#plt.plot(a, Bz_approx(a, 0.5, R), c='k', label='Bz_approx(z=0.5)')
plt.plot(a, Bz_approx(a, z, R), c='k', label='Bz_approx(z=1)')

plt.plot(a, Bz(a, z, R)-Bz_approx(a, z, R))
#plt.plot(a, Bz(a, 0.2)-Bz_approx(a, 0.2))
#plt.plot(a, Bz(a, 0.5)-Bz_approx(a, 0.5))
#plt.plot(a, Bz(a, 1)-Bz_approx(a, 1))

K1 = sc.vectorize(K1)
E1 = sc.vectorize(E1)
#plt.plot(a, K1(a, z, R), c='k')
#plt.plot(a, K1(a, 0.2), c='k')
#plt.plot(c, E1(c, 0.2))


plt.axhline(0, c='k', lw=0.7)           
plt.axvline(0, c='k', lw=0.7)
plt.grid(True)
plt.legend(bbox_to_anchor=(0.25, 0.95))
plt.show()

