import sys

import numpy as np


def eigenvalue(A, v):
    return v.dot(A.dot(v))


def power_iteration(A, num_simulations: int, eps=1e-10):
    print(f"Finding eigenvalue and eigenvector using power iteration with {num_simulations} simulations: ")

    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)
    cnt = 0

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < eps or cnt >= num_simulations:
            break

        if cnt % 10 == 0:
            sys.stdout.write(f"\rIterations: {cnt}, abs: {np.abs(ev - ev_new)}")
            sys.stdout.flush()

        v = v_new
        ev = ev_new
        cnt += 1

    return v_new, ev_new
