import math
from copy import deepcopy
import numpy as np


def derivative(func, x, epsilon):
    return (func(x + epsilon) - func(x)) / epsilon


# ∂f/∂x_i
def partial_derivative(func, x_0, i, epsilon):
    dx_0 = deepcopy(x_0)
    dx_0[i] = dx_0[i] + epsilon

    return (func(dx_0) - func(x_0)) / epsilon


def jacoby_matrix(functions, x_0, epsilon):
    return np.array([[partial_derivative(func, x_0, i, epsilon) for i in range(len(functions))]
                     for func in functions])


def newton(functions, x_0, epsilon):
    cnt = 0

    x = x_0

    while True:
        # x = x + step_matrix * F(x)
        x_prev = deepcopy(x)
        jacoby_matrix_ = jacoby_matrix(functions, x, 10 ** (-9))
        if np.linalg.det(jacoby_matrix_) == 0:
            raise ValueError("Incorrect x_0")

        step_matrix = np.linalg.inv(jacoby_matrix_)

        x = list(np.add(np.array(x), -step_matrix.dot(np.array([func(x) for func in functions]))))

        # print(np.add(np.array(x), -np.array(x_prev)))
        if np.linalg.norm(np.add(np.array(x), -np.array(x_prev))) < epsilon:
            break

        print(np.linalg.norm(np.add(np.array(x), -np.array(x_prev))))
        cnt += 1

    print(x)
    print(cnt)


def fixed_point_iteration_system(functions, x_0, epsilon):
    jacoby_matrix_ = jacoby_matrix(functions, x_0, 10 ** (-9))
    cnt = 0

    if np.linalg.det(jacoby_matrix_) == 0:
        raise ValueError("Incorrect x_0")

    step_matrix = np.linalg.inv(jacoby_matrix_)

    x = x_0

    while True:
        # x = x + step_matrix * F(x)
        x_prev = deepcopy(x)
        x = list(np.add(np.array(x), -step_matrix.dot(np.array([func(x) for func in functions]))))

        # print(np.add(np.array(x), -np.array(x_prev)))
        if np.linalg.norm(np.add(np.array(x), -np.array(x_prev))) < epsilon:
            break

        print(np.linalg.norm(np.add(np.array(x), -np.array(x_prev))))
        cnt += 1

    print(x)
    print(cnt)


try:
    functions = [lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 12,
                 lambda x: math.log(x[0]) + x[1]**3 + x[2]**2 - 29,
                 lambda x: 2**x[0] + x[1] - x[2]**4 - 1]
    x_0 = [1, 4, 2]
    newton(functions, x_0, 10 ** (-12))
except RuntimeWarning:
    print("Method doesn't converge")
except ValueError:
    print("Bad x_0")

