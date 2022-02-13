import numpy as np


# CODE FROM WIKIPEDIA


def power_iteration(A, num_simulations: int):
    b_k = np.random.rand(A.shape[1])

    for i in range(num_simulations):
        print(i)
        b_k1 = np.dot(A, b_k)
        b_k = b_k1 / np.linalg.norm(b_k1)

    Ab_k = A.dot(b_k)
    eig_val = 0

    for i in range(len(b_k)):
        if b_k[i] != 0:
            eig_val = Ab_k[i] / b_k[i]

    return b_k, eig_val
