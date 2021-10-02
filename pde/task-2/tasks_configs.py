from numpy import pi, cos, sin

CASE_3 = {
    'u_exact': lambda x, t: cos(pi * x) - sin(2 * pi * t) / 2 + 2 * pi * x - 3.5 * t,
    'kwargs': {
        'a': 0.41,
        't_min': 0,
        't_max': 1,
        'x_min': 0,
        'x_max': 1,
        'f': lambda x, t: -cos(2 * pi * t) * pi - 3.5 + 0.41 * (-sin(pi * x) * pi + 2 * pi),
        'phi': lambda x: cos(pi * x) + 2 * pi * x,
        'g': lambda t: 1 - sin(2 * pi * t) / 2 - 3.5 * t,
    }
}
