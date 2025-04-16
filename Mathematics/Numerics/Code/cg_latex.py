#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:18:26 2022

@author: joachim
"""
from fraction import FractionArray


def cg_solve(A, b, x0=0, N=3, Print=False):
    r0 = b - A @ x0
    d = r0
    r = r0
    for i in range(N):
        Ad = A @ d
        alpha = (r @ r) / (d @ Ad)
        x0 = x0 + alpha * d
        r = r0 - alpha * Ad
        beta = (r @ r) / (r0 @ r0)
        d = r + beta * d
                
        if Print:
            print(r0, Ad, alpha, x0, r, beta, d, sep='\n', end='\n\n')
        r0 = r
    return x0


def cg_solve_latex(A, b, x0, N=3):
    begin, end = "\n\\begin{align}\n", "\n\\end{align}"
    print(fr"CG-Verfahren für {begin} A &= {A} &&\mathrm{{und}} & {b} {end} "
          + fr"mit dem Startwert {begin} x_0 &= {x0}. {end}")
    r0 = b - A @ x0
    d = r0
    r = r0
    print("Residuum und Suchrichtung sind zunächst "
          + fr"{begin} r_0 &= d_0 = {r0}. {end}")
    for i in range(N):
        Ad = A @ d
        print("Wir speichern das Matrix-Vektor Produkt "
              + fr"{begin} Ad_{i+1} &= {Ad}. {end}")
        alpha = (r @ r) / (d @ Ad)
        print("Die optimale Schrittweite ist "
              + fr"$\alpha_{i} = \frac{{r_{i}^\top r_{i}}}{{d_{i}^\top Ad_{i}}} = {alpha}$. ")
        x0 = x0 + alpha * d
        print("Der nächste Iterationsschritt liefert dann "
              + fr"{begin} x_{i+1} &= x_{i} + \alpha_{i}d_{i} = {x0}. {end}")
        r = r0 - alpha * Ad
        print("Das neue Residuum ist dann "
              + fr"{begin} r_{i+1} &= r_{i} - \alpha_{i}Ad_{i} = {r}. {end}")
        beta = (r @ r) / (r0 @ r0)
        print("Der Parameter für die Gram-Schmidt-Orthogonalisierung ist "
              + fr"$\beta_{i+1} = \frac{{r_{i+1}^\top r_{i+1}}}{{r_{i}^\top r_{i}}} = {beta}$. ")
        d = r + beta * d
        print("Die neue Suchrichtung wird dann zu "
              + fr"{begin} d_{i+1} &= r_{i+1} + \beta_{i+1}d_{i} = {d}. {end}")
        r0 = r
    return x0


def main():
    num = [[1, 2, 1], [2, 5, 2], [1, 2, 2]]
    den = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    A = FractionArray(num, den)
    b = FractionArray([1, 1, 1], [1, 1, 1])
    x0 = FractionArray([1, 0, 0], [1, 1, 1])
    x = cg_solve_latex(A, b, x0)
    return 0

if __name__ == "__main__":
    main()
    