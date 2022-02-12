import time
from datetime import timedelta

import numpy as np
from scipy import optimize
from numpy import linalg
from matplotlib import pyplot as plt
from progress.bar import ChargingBar

from tasks_configs import *


def find_max_dev(u, u_ex):
    arr = np.subtract(u, u_ex)
    return max(abs(np.min(arr)), abs(np.max(arr)))


# guess that x_max - x_min == y_max - y_min == 1 ([0, 1] x [0, 1])
def solve(x_min, x_max, y_min, y_max, h, a, b, f, phi_1, phi_2, phi_3, phi_4):
    N = int((x_max - x_min) / h)

    x_grid = np.arange(x_min, x_max + h, h)
    y_grid = np.arange(y_min, y_max + h, h)
    xx, yy = np.meshgrid(x_grid, y_grid, sparse=True)

    u = np.zeros((N + 1, N + 1))

    u[0] = phi_2(xx)
    u[:, 0] = phi_1(y_grid[None, :])
    u[N] = phi_4(xx)
    u[:, N] = phi_3(y_grid[None, :])

    b_vec = np.zeros((N - 1) ** 2)
    Ah = np.zeros(((N - 1) ** 2, (N - 1) ** 2))

    c_0 = 2 * (a + b) / (h * h)
    c_1 = -a / (h * h)
    c_2 = -b * (h * h)

    for j in range(2, N - 1):
        # i = 1
        row = (j - 1) * (N - 1)
        Ah[row][(j - 1) * (N - 1)] = c_0
        Ah[row][(j - 1) * (N - 1) + 1] = c_1
        Ah[row][(j - 2) * (N - 1)] = c_2
        Ah[row][j * (N - 1)] = c_2

        b_vec[row] = f(h, j * h) - c_1 * u[j][0]

    for j in range(2, N - 1):
        # i = N-1
        row = (j - 1) * (N - 1) + N - 2

        Ah[row][(j - 1) * (N - 1) + N - 3] = c_1
        Ah[row][(j - 1) * (N - 1) + N - 2] = c_0
        Ah[row][(j - 2) * (N - 1) + N - 2] = c_2
        Ah[row][j * (N - 1) + N - 2] = c_2

        b_vec[row] = f((N - 1) * h, j * h) - c_1 * u[j][N]

    for i in range(2, N - 1):
        # j = 1
        row = i - 1

        Ah[row][(j - 1) * (N - 1) + i - 2] = c_1
        Ah[row][(j - 1) * (N - 1) + i - 1] = c_0
        Ah[row][(j - 1) * (N - 1) + i] = c_1
        Ah[row][j * (N - 1) + i - 1] = c_2

        b_vec[row] = f(i * h, h) - c_2 * u[0][i]

    for i in range(2, N - 1):
        # j = N-1
        row = (N - 2) * (N - 1) + i - 1

        Ah[row][(N - 2) * (N - 1) + i - 2] = c_1
        Ah[row][(N - 2) * (N - 1) + i - 1] = c_0
        Ah[row][(N - 2) * (N - 1) + i] = c_1
        Ah[row][(N - 2) * (N - 1) + i - 1] = c_2

        b_vec[row] = f(i * h, (N - 1) * h) - c_2 * u[N][i]

    # i, j = 1
    row = 0
    Ah[row][0] = c_0
    Ah[row][1] = c_1
    Ah[row][N - 1] = c_2
    b_vec[row] = f(h, h) - c_1 * u[1][0] - c_2 * u[0][1]

    # i, j = N-1
    row = (N - 2) * (N - 1) + N - 2
    Ah[row][(N - 2) * (N - 1) + N - 3] = c_1
    Ah[row][(N - 2) * (N - 1) + N - 2] = c_0
    Ah[row][(N - 3) * (N - 1) + N - 2] = c_2
    b_vec[row] = f((N - 1) * h, (N - 1) * h) + - c_1 * u[N - 1][N - 2] - c_2 * u[N - 2][N - 1]

    # i = 1, j = N-1
    row = (N - 2) * (N - 1)

    Ah[row][(N - 2) * (N - 1)] = c_0
    Ah[row][(N - 2) * (N - 1) + 1] = c_1
    Ah[row][(N - 3) * (N - 1)] = c_2
    b_vec[row] = f(h, (N - 1) * h) - c_1 * u[N - 1][0] - c_2 * u[N][1]

    # i = N-1, j = 1
    row = N - 2

    Ah[row][N - 3] = c_1
    Ah[row][N - 2] = c_0
    Ah[row][N - 1 + N - 2] = c_2
    b_vec[row] = f((N - 1) * h, h) - c_1 * u[1][N] - c_2 * u[0][N - 1]

    for j in range(2, N - 1):
        for i in range(2, N - 1):
            row = (j - 1) * (N - 1) + i - 1

            Ah[row][(j - 1) * (N - 1) + i - 2] = c_1
            Ah[row][(j - 1) * (N - 1) + i - 1] = c_0
            Ah[row][(j - 1) * (N - 1) + i] = c_1
            Ah[row][(j - 2) * (N - 1) + i - 1] = c_2
            Ah[row][j * (N - 1) + i - 1] = c_2
            b_vec[row] = f(i * h, j * h)

    # def func(x):
    #     return Ah.dot(np.asarray(x)) - b_vec

    # u_vec = optimize.fmin_cg(func, np.zeros((N - 1)**2))

    # TODO: replace this to gradient descent
    u_vec = linalg.linalg.solve(Ah, b_vec)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            u[j][i] = u_vec[(j - 1)*(N - 1) + i - 1]

    # with ChargingBar('Solving (time layer):', max=M - 1) as bar:
    #     for n in range(0, M - 1):
    #         u[n + 1] = np.array(
    #             [u[n + 1][0]] + [tau * f(x_grid[j], t_grid[n]) + u[n][j] - d * (u[n][j] - u[n][j - 1]) for j in
    #                              range(1, N)])
    #         bar.next()

    return u


CASE_TO_SOLVE = CASE_5
CASE_TO_SOLVE['kwargs']['h'] = float(input('h (use decimal repr): '))

assert CASE_TO_SOLVE['kwargs']['h'] <= 1 / 3, 'h must be less than 1/3'

u_exact = CASE_TO_SOLVE['u_exact']
h = CASE_TO_SOLVE['kwargs']['h']
x_min = CASE_TO_SOLVE['kwargs']['x_min']
x_max = CASE_TO_SOLVE['kwargs']['x_max']
y_min = CASE_TO_SOLVE['kwargs']['y_min']
y_max = CASE_TO_SOLVE['kwargs']['y_max']

start_time = time.time()

u = solve(**CASE_TO_SOLVE['kwargs'])

end_time = time.time()

xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h), np.arange(y_min, y_max + h, h), sparse=True)
u_ex = u_exact(xx, yy)

err = find_max_dev(u, u_ex)
print("\nMax error:", err)
# print(f"tau + h = {tau + h}")
print("Elapsed time:", timedelta(seconds=end_time - start_time))

# graph logic
# x = np.linspace(x_min, x_max, int((x_max - x_min) / h))
# x_ex = np.linspace(x_min, x_max, 10000)

# t = np.linspace(t_min, t_max, int((t_max - t_min) / tau))
# t_ex = np.linspace(t_min, t_max, 10000)

# X, T = np.meshgrid(x, t)
# X_ex, T_ex = np.meshgrid(x_ex, t_ex)

# U = u_exact(X_ex, T_ex)

# font = {'size': 20}

# plt.rc('font', **font)

# fig = plt.figure(figsize=(18, 12), dpi=200)
# fig1 = plt.figure(figsize=(18, 12), dpi=200)
# fig2 = plt.figure(figsize=(18, 12), dpi=200)

# ax_ex = fig.add_subplot(projection='3d')
# ax_c = fig1.add_subplot(projection='3d')
# ax_all = fig2.add_subplot(projection='3d')

# ax_ex.azim = 200
# ax_c.azim = 200
# ax_all.azim = 200
# ax_ex.elev = 7

# ax_ex.set_title('Exact solution')
# ax_ex.set_xlabel('x')
# ax_ex.set_ylabel('t')
# ax_ex.set_zlabel('u')
# ax_ex.plot_surface(X_ex, T_ex, U, cmap='gnuplot')

# ax_c.set_title('Calculated solution')
# ax_c.set_xlabel('x')
# ax_c.set_ylabel('t')
# ax_c.set_zlabel('u')
# ax_c.plot_wireframe(X, T, u, cmap='binary')

# ax_all.set_title('Both solutions')
# ax_all.set_xlabel('x')
# ax_all.set_ylabel('t')
# ax_all.set_zlabel('u')
# ax_all.plot_surface(X_ex, T_ex, U, cmap='gnuplot')
# ax_all.plot_wireframe(X, T, u, cmap='binary')

# plt.xticks([0, 0.5, 1])
# plt.yticks(np.linspace(0, 1, 11))

# plt.show()
