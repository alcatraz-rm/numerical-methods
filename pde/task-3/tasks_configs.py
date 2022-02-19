from numpy import pi, cos, sin

CASE_5 = {
    'u_exact': lambda x, y: cos(2 * x) * sin(y) + y ** 3,
    'kwargs': {
        'a': 0.9,
        'b': 1.2,
        'x_min': 0,
        'x_max': 1,
        'y_min': 0,
        'y_max': 1,
        'f': lambda x, y: 4.8 * cos(2 * x) * sin(y) - 7.2 * y,
        'phi_1': lambda y: sin(y) + y ** 3,
        'phi_2': lambda x: 0,
        'phi_3': lambda y: cos(2) * sin(y) + y ** 3,
        'phi_4': lambda x: sin(1) * cos(2 * x) + 1
    }
}

CASE_11 = {
    'u_exact': lambda x, y: cos(2 * x) * sin(y) + y ** 3,
    'kwargs': {
        'a': 1.3,
        'b': 0.9,
        'x_min': 0,
        'x_max': 1,
        'y_min': 0,
        'y_max': 1,
        'f': lambda x, y: (3.6 * y * y + 5.2) * cos(2 * x - y * y) - 1.8 * sin(2 * x - y * y),
        'phi_1': lambda y: cos(y * y),
        'phi_2': lambda x: cos(2 * x),
        'phi_3': lambda y: cos(2 - y * y) + y,
        'phi_4': lambda x: cos(2 * x - 1) + x
    }
}
