# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np
import matplotlib.pyplot as plt


def function(t, x):
    return -2 * t * x**2


def exact_solution(t):
    return 1 / (t**2 + 1)


def eulerExplizit(func, n):

    tk = 0
    x0 = 1
    tau = 1 / n

    xn = np.zeros(n+1)
    xn[0] = x0

    xk = x0

    for i in range(n):
        xk = eulerStep(func, tau, xk, tk)
        tk = tk + tau

        xn[i+1] = xk

    return xn


def eulerStep(func, tau, xk, tk):

    return xk + tau * func(tk, xk)


def runge(func, n):

    tk = 0
    x0 = 1
    tau = 1/n

    xn = np.zeros(n+1)
    xn[0] = x0

    xk = x0

    for i in range(n):
        xk = rungeStep(func, tau, xk, tk)
        tk = tk + tau

        xn[i+1] = xk

    return xn


def rungeStep(func, tau, xk, tk):

    return xk + tau * func(tk + tau/2, xk + tau/2 * func(tk, xk))


def main():

    n = 100
    t = np.linspace(0, 1, n+1)
    xn_euler = eulerExplizit(function, n)
    xn_runge = runge(function, n)
    xn_exact = exact_solution(t)

    # 2**2 = 4, ..., 2**6 = 64 (insgesamt 5 Werte)
    n_all = np.logspace(2, 6, 5, base=2)

    euler_error_list = np.zeros(len(n_all))
    runge_error_list = np.zeros(len(n_all))

    for i, n in enumerate(n_all):

        n = int(n)
        t = np.linspace(0,1, n+1)
        xn_exact = exact_solution(t)

        xn_euler = eulerExplizit(function, n)
        xn_runge = runge(function, n)

        euler_error = np.linalg.norm(xn_euler - xn_exact, ord=np.inf)
        runge_error = np.linalg.norm(xn_runge - xn_exact, ord=np.inf)
        euler_error_list[i] = euler_error
        runge_error_list[i] = runge_error
        print(f'Max Euler Error for n = {n} : ', euler_error)
        print(f'Max Runge Error for n = {n} : ', runge_error)

    _, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$\max_k\ |x_k - x(t_k)|$')
    ax.plot(n_all, euler_error_list, label='Euler')
    ax.plot(n_all, runge_error_list, label='Runge')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()

