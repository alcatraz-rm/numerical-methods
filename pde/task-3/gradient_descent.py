import sys

import numpy as np


def descent(A, b, eps=1e-5):
    print("Solving Ax=b with gradient descent:")

    x_0 = np.array([b[i]/A[i][i] for i in range(len(b))])
    r = b - A.dot(x_0)

    x = x_0
    cnt = 0

    while np.linalg.norm(r) >= eps:
        r = b - A.dot(x)
        mu = np.dot(r, r) / np.dot(r, A.dot(r))

        x += mu * r
        cnt += 1

        if cnt % 10 == 0:
            sys.stdout.write(f"\rIteration: {cnt}, ||r||: {np.linalg.norm(r)}")
            sys.stdout.flush()

    return x
