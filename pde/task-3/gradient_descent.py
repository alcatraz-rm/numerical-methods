import numpy as np


def descent(A, b, eps=1e-5):
    x_0 = np.array([b[i]/A[i][i] for i in range(len(b))])
    r = b - A.dot(x_0)

    x = x_0
    cnt = 0

    while np.linalg.norm(r) >= eps:
        r = b - A.dot(x)
        mu = np.dot(r, r) / np.dot(r, A.dot(r))

        x += mu * r
        cnt += 1
        print(cnt, np.linalg.norm(r))

    return x
