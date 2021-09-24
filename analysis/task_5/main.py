import math
from typing import Tuple


def euler(func, y_0: float, grid: Tuple[float]) -> list:
    result = [y_0]

    for i in range(1, len(grid)):
        result.append(result[i - 1] + (grid[i] - grid[i - 1]) * func(grid[i - 1], result[i - 1]))

    return result


def runge_kutta(func, y_0: float, grid: Tuple[float]) -> tuple:
    h = grid[1] - grid[0]

    result = [y_0]

    for i in range(1, len(grid)):
        k_1 = func(grid[i - 1], result[i - 1])
        k_2 = func(grid[i - 1] + h / 2, result[i - 1] + h / 2 * k_1)
        k_3 = func(grid[i - 1] + h / 2, result[i - 1] + h / 2 * k_2)
        k_4 = func(grid[i - 1] + h, result[i - 1] + h * k_3)

        result.append(result[i - 1] + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4))

    return result


def linear_multistep(func, y_0: float, grid: Tuple[float]) -> tuple:
    result = []
    h = grid[1] - grid[0]
    tmp_grid = grid[:4]

    result.extend(runge_kutta(func, y_0, tmp_grid))

    for i in range(4, len(grid)):
        result.append(
            result[i - 1] + h * (55 / 24 * func(grid[i - 1], result[i - 1]) - 59 / 24 * func(grid[i - 2], result[i - 2])
                                 + 37 / 24 * func(grid[i - 3], result[i - 3]) - 3 / 8 * func(grid[i - 4],
                                                                                             result[i - 4])))

    return result


def f(x, y) -> float:
    return 3 * x * x * y + x * x * math.exp(x ** 3)


accuracy = 6
grid = tuple([round(i * 0.001, accuracy) for i in range(1001)])
print(grid)
x_0 = 0
y_0 = 0
b = 1

result = [round(a, accuracy) for a in euler(f, y_0, grid)]
result_1 = [round(a, accuracy) for a in runge_kutta(f, y_0, grid)]
result_2 = [round(a, accuracy) for a in linear_multistep(f, y_0, grid)]

for x, y, z, w in zip(grid, result, result_1, result_2):
    print(x, y, z, w, sep='\t\t')
