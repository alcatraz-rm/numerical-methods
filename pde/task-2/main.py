import time
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt
from progress.bar import ChargingBar

from tasks_configs import *


def find_max_dev(u, u_ex):
    arr = np.subtract(u, u_ex)
    return max(abs(np.min(arr)), abs(np.max(arr)))


def solve(t_min, t_max, tau, x_min, x_max, h, a, f, phi, g):
    d = a * tau / h

    M = int((t_max - t_min) / tau)
    N = int((x_max - x_min) / h)

    x_grid = np.arange(x_min, x_max, h)
    t_grid = np.arange(t_min, t_max, tau)
    xx, tt = np.meshgrid(x_grid, t_grid, sparse=True)

    u = np.zeros((M, N))

    u[0] = phi(xx)
    u[:, 0] = g(t_grid[None, :])

    with ChargingBar('Solving (time layer):', max=M - 1) as bar:
        for n in range(0, M - 1):
            u[n + 1] = np.array(
                [u[n + 1][0]] + [tau * f(x_grid[j], t_grid[n]) + u[n][j] - d * (u[n][j] - u[n][j - 1]) for j in
                                 range(1, N)])
            bar.next()

    return u


CASE_TO_SOLVE = CASE_3
CASE_TO_SOLVE['kwargs']['tau'] = float(input('tau (use decimal repr): '))
CASE_TO_SOLVE['kwargs']['h'] = float(input('h (use decimal repr): '))

assert CASE_TO_SOLVE['kwargs']['a'] * CASE_TO_SOLVE['kwargs']['tau'] / CASE_TO_SOLVE['kwargs'][
    'h'] <= 1, "method doesn't converge"

assert CASE_TO_SOLVE['kwargs']['tau'] <= 1 / 3, 'tau must be less than 1/3'
assert CASE_TO_SOLVE['kwargs']['h'] <= 1 / 3, 'h must be less than 1/3'

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

err = find_max_dev(u, u_ex)
print("\nMax error:", err)
print(f"tau + h = {tau + h}")
print("Elapsed time:", timedelta(seconds=end_time - start_time))

# graph logic
x = np.linspace(x_min, x_max, int((x_max - x_min) / h))
x_ex = np.linspace(x_min, x_max, 10000)

t = np.linspace(t_min, t_max, int((t_max - t_min) / tau))
t_ex = np.linspace(t_min, t_max, 10000)

X, T = np.meshgrid(x, t)
X_ex, T_ex = np.meshgrid(x_ex, t_ex)

U = u_exact(X_ex, T_ex)

font = {'size': 20}

plt.rc('font', **font)

fig = plt.figure(figsize=(18, 12), dpi=200)
fig1 = plt.figure(figsize=(18, 12), dpi=200)
fig2 = plt.figure(figsize=(18, 12), dpi=200)

ax_ex = fig.add_subplot(projection='3d')
ax_c = fig1.add_subplot(projection='3d')
ax_all = fig2.add_subplot(projection='3d')

ax_ex.azim = 200
ax_c.azim = 200
ax_all.azim = 200
# ax_ex.elev = 7

ax_ex.set_title('Exact solution')
ax_ex.set_xlabel('x')
ax_ex.set_ylabel('t')
ax_ex.set_zlabel('u')
ax_ex.plot_surface(X_ex, T_ex, U, cmap='gnuplot')

ax_c.set_title('Calculated solution')
ax_c.set_xlabel('x')
ax_c.set_ylabel('t')
ax_c.set_zlabel('u')
ax_c.plot_wireframe(X, T, u, cmap='binary')

ax_all.set_title('Both solutions')
ax_all.set_xlabel('x')
ax_all.set_ylabel('t')
ax_all.set_zlabel('u')
ax_all.plot_surface(X_ex, T_ex, U, cmap='gnuplot')
ax_all.plot_wireframe(X, T, u, cmap='binary')

# plt.xticks([0, 0.5, 1])
# plt.yticks(np.linspace(0, 1, 11))

plt.show()
