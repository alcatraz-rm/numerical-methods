import math


def left_rectangles(func, a, b):
    return func(a)*(b - a)


def central_rectangles(func, a, b):
    return func((a + b) / 2) * (b - a)


def rectangles(func, a, b, epsilon, n_0=2, central=False, left=False):
    if not central and not left:
        raise EnvironmentError()

    omega = 1/3
    result_prev = 0

    while True:
        step = (b - a) / n_0
        result = 0

        for i in range(n_0):
            start = a + i * step
            end = a + (i + 1) * step

            if left:
                result += left_rectangles(func, start, end)
            else:
                result += central_rectangles(func, start, end)

        if omega * abs(result_prev - result) < epsilon:
            return result, n_0
        else:
            # print(omega * abs(result_prev - result))
            result_prev = result
            n_0 *= 2


def _trapezoid(func, a, b):
    return (func(a) + func(b)) * (b - a) / 2


def trapezoid(func, a, b, epsilon, n_0=2):
    omega = 1/3
    result_prev = 0

    while True:
        step = (b - a) / n_0
        result = 0

        for i in range(n_0):
            start = a + i * step
            end = a + (i + 1) * step

            result += _trapezoid(func, start, end)

        if omega * abs(result_prev - result) < epsilon:
            return result, n_0
        else:
            # print(omega * abs(result_prev - result))
            result_prev = result
            n_0 *= 2


def _simpson(func, a, b):
    return ((b - a) / 6) * (func(a) + 4 * func((a + b) / 2) + func(b))


def simpson(func, a, b, epsilon, n_0=2):
    omega = 1/15
    result_prev = 0

    while True:
        step = (b - a) / n_0
        result = 0

        for i in range(n_0):
            start = a + i * step
            end = a + (i + 1) * step

            result += _simpson(func, start, end)

        if omega * abs(result_prev - result) < epsilon:
            return result, n_0
        else:
            # print(omega * abs(result_prev - result))
            result_prev = result
            n_0 *= 2


print(rectangles(lambda x: math.log(x), 2, 5, 10**(-9), n_0=2, central=True))
print(trapezoid(lambda x: math.log(x), 2, 5, 10**(-9), n_0=2))
print(simpson(lambda x: math.log(x), 2, 5, 10**(-9), n_0=2))
