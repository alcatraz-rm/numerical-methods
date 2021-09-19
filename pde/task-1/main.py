import time

import numpy as np
from progress.bar import Bar
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
    result = []
    with Bar('Calculating exact function on grid:', max=(len(t_grid))) as bar:
        for i in range(len(t_grid)):
            result.append(np.array([u_exact(t_grid[i], x) for x in x_grid]))
            bar.next()

    return result


def find_max_dev(u, u_ex):
    arr = np.subtract(u, u_ex)
    return max(abs(np.min(arr)), abs(np.max(arr)))


def solve(t_max, tau, x_max, h, a):
    d = a / (h * h)
    p = 1 / tau

    M = int(t_max / tau)
    N = int(x_max / h)

    x_grid = np.array([i * tau for i in range(N)])
    t_grid = np.array([i * h for i in range(M)])

    u_ex = u_table_exact(t_grid, x_grid)

    u = np.zeros((M, N))

    with Bar('Setting initial values:', max=N + 2 * M) as bar:
        for i in range(N):
            u[0][i] = mu(x_grid[i])
            bar.next()

        for i in range(M):
            u[i][0] = mu_1(t_grid[i])
            bar.next()

        for i in range(M):
            u[i][N - 1] = mu_2(t_grid[i])
            bar.next()

    matrix = np.zeros((M - 2, N - 2))
    b = np.zeros(M - 2)

    matrix[0][0] = p + 2 * d
    matrix[0][1] = -d

    for i in range(1, N - 3):
        matrix[i][i - 1] = -d
        matrix[i][i] = p + 2 * d
        matrix[i][i + 1] = -d

    matrix[N - 3][N - 4] = -d
    matrix[N - 3][N - 3] = p + 2 * d

    matrix_prep = np.zeros((3, N - 2))

    for i in range(M - 2):
        for j in range(N - 2):
            index = 1 + i - j
            if 0 <= index < 3:
                matrix_prep[index][j] = matrix[i][j]

    with Bar('Solving (time layer):', max=M-1) as bar:
        for s in range(M - 1):
            b[0] = u[s][1] * p + d * u[s + 1][0] + f(x_grid[1], t_grid[s + 1])

            for i in range(1, N - 3):
                b[i] = u[s][i + 1] * p + f(x_grid[i + 1], t_grid[s + 1])

            b[N - 3] = u[s][N - 2] * p + d * u[s + 1][N - 1] + f(x_grid[N - 2], t_grid[s + 1])

            vec = linalg.solve_banded((1, 1), matrix_prep, b)

            for k in range(1, N - 1):
                u[s + 1][k] = vec[k - 1]

            bar.next()

    print("\nMax deviation:", find_max_dev(u, u_ex))


start_time = time.time()
solve(1, 1 / 50000, 1, 1 / 50000, 0.027)
end_time = time.time()

print("Elapsed seconds:", (end_time - start_time))
