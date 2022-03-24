import sys
import numpy as np


def descent(A, main_diag, b, eps=1e-3):
    print("Solving Ax=b with gradient descent:")

    x = np.array([b[i]/main_diag[i] for i in range(len(b))])
    cnt = 0

    r = b - A.dot(x)
    while np.linalg.norm(r) >= eps:
        Ar = A.dot(r)
        mu = np.dot(r, r) / np.dot(r, Ar)
        x += mu * r
        r -= mu * Ar
        cnt += 1

        if cnt % 100 == 0:
            sys.stdout.write(f"\rIterations: {cnt}, ||r||: {np.linalg.norm(r)}")
            sys.stdout.flush()

    return x
