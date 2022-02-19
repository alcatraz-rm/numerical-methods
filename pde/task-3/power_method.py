import sys

import numpy as np


# CODE FROM WIKIPEDIA

def find_approx_eigvalue(A, x):
    Ax = np.dot(A, x)

    for i in range(len(x)):
        if x[i] != 0:
            eig_val = Ax[i] / x[i]
            return eig_val

    return np.inf


def power_iteration(A, num_simulations: int):
    print(f"Finding eigenvalue and eigenvector using power iteration with {num_simulations} simulations: ")

    b_k = np.random.rand(A.shape[1])

    for i in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k = b_k1 / np.linalg.norm(b_k1)

        if i % 10 == 0:
            sys.stdout.write(f"\rIterations: {i}/{num_simulations}")
            sys.stdout.flush()

    Ab_k = A.dot(b_k)
    eig_val = 0

    for i in range(len(b_k)):
        if b_k[i] != 0:
            eig_val = Ab_k[i] / b_k[i]
            return b_k, eig_val

    return b_k, eig_val