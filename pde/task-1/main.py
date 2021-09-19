import time

import numpy as np
from scipy import linalg


def f(x, t):
    return x + 2 * t - np.exp(x) - 0.027 * (-12 * (x ** 2) - t * np.exp(x))


def mu(x):
    return -x ** 4 + x


def mu_1(t):
    return t ** 2 - t


def mu_2(t):
    return t ** 2 + t - np.e * t


def u_exact(t, x):
    return -x ** 4 + x + t * x + t ** 2 - t * np.exp(x)


def u_table_exact(t_grid, x_grid):
    result = np.zeros((len(t_grid), len(x_grid)))

    for i, t in enumerate(t_grid):
        for j, x in enumerate(x_grid):
            result[i][j] = u_exact(t, x)

    return result


def find_max_dev(u, u_ex):
    arr = np.subtract(u, u_ex)
    return max(abs(np.min(arr)), abs(np.max(arr))), np.average(arr)


def solve(t_max, tau, x_max, h, a):
    # t_max = 1.0
    # tau = 10**(-4)
    #
    # x_max = 1.0
    # h = 10**(-4)
    # a = 0.027

    d = a / (h * h)
    p = 1 / tau

    M = int(t_max / tau)
    N = int(x_max / h)

    x_grid = np.array([i * tau for i in range(N)])
    t_grid = np.array([i * h for i in range(M)])

    u_ex = u_table_exact(t_grid, x_grid)

    u = np.zeros((M, N))  # X ->;

    for i in range(N):
        u[0][i] = mu(x_grid[i])

    for i in range(M):
        u[i][0] = mu_1(t_grid[i])

    for i in range(M):
        u[i][N - 1] = mu_2(t_grid[i])

    matrix = np.zeros((M - 2, N - 2))
    b = np.zeros(M - 2)

    matrix[0][0] = p + 2*d
    matrix[0][1] = -d

    for i in range(1, N - 3):
        matrix[i][i - 1] = -d
        matrix[i][i] = p + 2*d
        matrix[i][i + 1] = -d

    matrix[N - 3][N - 4] = -d
    matrix[N - 3][N - 3] = p + 2 * d

    matrix_prep = np.zeros((3, N - 2))

    for i in range(M - 2):
        for j in range(N - 2):
            index = 1 + i - j
            if 0 <= index < 3:
                matrix_prep[index][j] = matrix[i][j]

    for s in range(M - 1):
        print(s)

        b[0] = u[s][1] * p + d * u[s+1][0] + f(x_grid[1], t_grid[s+1])

        for i in range(2, N - 2):
            b[i - 1] = u[s][i] * p + f(x_grid[i], t_grid[s + 1])

        # FIXME: if N != M?

        b[N - 3] = u[s][N-1] * p + d * u[s+1][N] + f(x_grid[N - 1], t_grid[s + 1])

        vec = linalg.solve_banded((1, 1), matrix_prep, b)

        for k in range(1, N-1):
            u[s + 1][k] = vec[k - 1]

    print(find_max_dev(u, u_ex))


start_time = time.time()
solve(1, 1 / 10000, 1, 1 / 10000, 0.027)
end_time = time.time()

print((end_time - start_time))
