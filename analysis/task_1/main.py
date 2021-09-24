import math
import builtins

golden = (1 + 5 ** 0.5) / 2


def bisections(func, a, b, epsilon):
    cnt = 0

    while b - a > epsilon:
        cnt += 1

        c = (a + b) / 2

        if func(c) * func(a) < 0:
            b = c
        elif func(c) * func(b) < 0:
            a = c
        elif func(c) == 0:
            return c

    return f"\nSolution: {(a + b) / 2}\nIterations: {cnt}"


def derivative(func, x, epsilon):
    return (func(x + epsilon) - func(x)) / epsilon


def find_extr(func, a, b, epsilon, max=False, min=False):
    a_0 = a
    b_0 = b
    while b - a > epsilon:
        x_1 = b - (b - a) / golden
        x_2 = a + (b - a) / golden
        if max:
            if func(x_1) <= func(x_2):
                a = x_1
            else:
                b = x_2
        if min:
            if func(x_1) >= func(x_2):
                a = x_1
            else:
                b = x_2
    return builtins.max(func(a_0), func(b_0), func((a + b) / 2)) if max else builtins.min(func(a_0), func(b_0), func((a + b) / 2))


def sor(func, x_0, a, b, epsilon):
    cnt = 0
    cnt_max = 10 ** 9
    # g(x) = x - sgn * func(x); x_{i+1} = g(x_{i})
    try:
        df = lambda x: derivative(func, x, epsilon=10 ** (-9))
        maximum = find_extr(df, a, b, epsilon=10**(-9), max=True)
        minimum = find_extr(df, a, b, epsilon=10**(-9), min=True)
        tau = 2 / (maximum + minimum)
        g = lambda x: x - tau*func(x)
    except OverflowError:
        return "Method doesn't converge (overflow error)"

    x_curr = g(x_0)

    try:
        while abs(x_curr - g(x_curr)) > epsilon and cnt < cnt_max:
            x_curr = g(x_curr)
            cnt += 1
            print(x_curr)
    except OverflowError:
        return "Method doesn't converge (overflow error)"

    if cnt >= cnt_max:
        return "Method doesn't converge"

    return f"\nSolution: {x_curr}\nIterations: {cnt}"


def fixed_point_iteration(func, x_0, epsilon, newton=False):
    sgn = 1 if derivative(func, x_0, 10**(-9)) > 0 else -1
    cnt = 0
    cnt_max = 10**9

    # g(x) = x - sgn * func(x); x_{i+1} = g(x_{i})
    try:
        if newton:
            g = lambda x: x - (1 / derivative(func, x, 10**(-9))) * func(x)
        else:
            g = lambda x: x - func(x)
    except OverflowError:
        return "Method doesn't converge (overflow error)"

    x_curr = g(x_0)

    try:
        while abs(x_curr - g(x_curr)) > epsilon and cnt < cnt_max:
            x_curr = g(x_curr)
            cnt += 1
            print(x_curr)
    except OverflowError:
        return "Method doesn't converge (overflow error)"

    if cnt >= cnt_max:
        return "Method doesn't converge"

    return f"\nSolution: {x_curr}\nIterations: {cnt}"


# bisections
# print(bisections(lambda x: math.log(x), 1/2, 10, epsilon=10 ** (-9)))

# fixed point iteration
# print(fixed_point_iteration(lambda x: x**3, 1.3, epsilon=10**(-9)))

# newton method
# print(fixed_point_iteration(lambda x: math.cos(x) - x**3, 1/2,  epsilon=10**(-9), newton=True))

# sor
# print(sor(lambda x: x**3 - 8 + math.sin(x), 10, -10, 5, epsilon=10**(-9)))

# extr check
# print(find_extr(lambda x: x**2 - 3*x**4, -1, 0, epsilon=10**(-9), max=True))
