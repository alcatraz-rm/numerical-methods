from pprint import pprint

import numpy as np


def f(t, x):
    return 1


def mu(x):
    return 1


def mu_1(t):
    return 1


def mu_2(t):
    return 1


def solve():
    t_max = 1.0
    tau = 10**(-4)

    x_max = 1.0
    h = 10**(-4)
    a = 0.027

    d = a / (h * h)

    x_grid = np.array([i * tau for i in range(int(x_max / h) + 1)])
    t_grid = np.array([i * h for i in range(int(t_max / tau) + 1)])

    u = np.zeros((int(x_max / h) + 1, int(t_max / tau) + 1))

    for i in range(int(t_max / tau) + 1):
        u[0][i] = mu_1(t_grid[i])

    for i in range(int(t_max / tau) + 1):
        u[int(x_max / h)][i] = mu_2(t_grid[i])

    for i in range(int(x_max / h) + 1):
        u[i][0] = mu(x_grid[i])

    pprint(u)

    for j in range(int(t_max / tau) - 1):
        matrix = np.zeros((int(t_max / tau), int(t_max / tau)))
        b = np.zeros(int(t_max / tau))

        matrix[0][0] = -d
        b[0] = -u[0][j+1](1/tau + 2*a/(h*h)) + u[0][0]/tau + f(0, 1)

        matrix[1][0] = (1/tau) + 2*a/(h*h)
        matrix[1][1] = -d
        b[1] = u[1][0]/tau + f(1, 1) + d * u[0][1]

        for k in range(2, int(t_max / tau)):
            matrix[k][k-1] = (1/tau) + 2*a/(h*h)
            matrix[k][k] = -d

solve()
