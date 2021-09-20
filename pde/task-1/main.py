import time

import numpy as np
from matplotlib import pyplot as plt
from progress.bar import Bar
from scipy import linalg

from tasks_configs import *


def find_max_dev(u, u_ex):
    arr = np.subtract(u, u_ex)
    return max(abs(np.min(arr)), abs(np.max(arr)))


def solve(t_min, t_max, tau, x_min, x_max, h, a, f, mu, mu_1, mu_2):
    d = a / (h * h)
    p = 1 / tau

    M = int((t_max - t_min) / tau)
    N = int((x_max - x_min) / h)

    x_grid = np.arange(x_min, x_max, h)
    t_grid = np.arange(t_min, t_max, tau)
    xx, tt = np.meshgrid(x_grid, t_grid, sparse=True)

    u = np.zeros((M, N))

    u[0] = mu(xx)
    u[:, 0] = mu_1(t_grid[None, :])
    u[:, N - 1] = mu_2(t_grid[None, :])

    b = np.zeros(M - 2)
    matrix_prep = np.zeros((3, N - 2))

    # TODO: remove loops here
    for i in range(M - 2):
        for j in range(N - 2):
            if abs(i - j) == 1:
                matrix_prep[1 + i - j][j] = -d
            elif j == i:
                matrix_prep[1 + i - j][j] = p + 2 * d

    with Bar('Solving (time layer):', max=M - 1) as bar:
        for s in range(M - 1):
            b[0] = u[s][1] * p + d * u[s + 1][0] + f(x_grid[1], t_grid[s + 1])
            for i in range(1, N - 3):
                b[i] = u[s][i + 1] * p + f(x_grid[i + 1], t_grid[s + 1])
            b[N - 3] = u[s][N - 2] * p + d * u[s + 1][N - 1] + f(x_grid[N - 2], t_grid[s + 1])

            u[s + 1][1:N - 1] = linalg.solve_banded((1, 1), matrix_prep, b)

            bar.next()

    return u


CASE_TO_SOLVE = CASE_10

u_exact = CASE_TO_SOLVE['u_exact']
tau = CASE_TO_SOLVE['kwargs']['tau']
h = CASE_TO_SOLVE['kwargs']['h']
x_min = CASE_TO_SOLVE['kwargs']['x_min']
x_max = CASE_TO_SOLVE['kwargs']['x_max']
t_min = CASE_TO_SOLVE['kwargs']['t_min']
t_max = CASE_TO_SOLVE['kwargs']['t_max']

start_time = time.time()

u = solve(**CASE_TO_SOLVE['kwargs'])

end_time = time.time()

xx, tt = np.meshgrid(np.arange(x_min, x_max, h), np.arange(t_min, t_max, tau), sparse=True)
u_ex = u_exact(xx, tt)
print("\nMax deviation:", find_max_dev(u, u_ex))

print("Elapsed seconds:", (end_time - start_time))

# graph logic
x = np.linspace(x_min, x_max, int((x_max - x_min) / h))
t = np.linspace(t_min, t_max, int((t_max - t_min) / tau))

X, T = np.meshgrid(x, t)
U = u_exact(X, T)

fig = plt.figure()
fig1 = plt.figure()
fig2 = plt.figure()

ax_ex = fig.add_subplot(projection='3d')
ax_c = fig1.add_subplot(projection='3d')
ax_all = fig2.add_subplot(projection='3d')

ax_ex.set_title('Exact solution')
ax_ex.set_xlabel('x')
ax_ex.set_ylabel('t')
ax_ex.set_zlabel('u')
ax_ex.plot_surface(X, T, U, cmap='viridis')

ax_c.set_title('Calculated solution')
ax_c.set_xlabel('x')
ax_c.set_ylabel('t')
ax_c.set_zlabel('u')
ax_c.plot_surface(X, T, u, cmap='plasma')

ax_all.set_title('Both solutions')
ax_all.set_xlabel('x')
ax_all.set_ylabel('t')
ax_all.set_zlabel('u')
ax_all.plot_surface(X, T, U, cmap='viridis')
ax_all.plot_surface(X, T, u, cmap='plasma')

plt.show()
