import numpy as np

CASE_12 = {
    'u_exact': lambda x, t: -x ** 4 + x + t * x + t ** 2 - t * np.exp(x),
    'kwargs': {
        'a': 0.027,
        'tau': 1 / 1000,
        'h': 1 / 1000,
        't_min': 0,
        't_max': 1,
        'x_min': 0,
        'x_max': 1,
        'f': lambda x, t: x + 2 * t - np.exp(x) - 0.027 * (-12 * (x ** 2) - t * np.exp(x)),
        'mu': lambda x: -x ** 4 + x,
        'mu_1': lambda t: t ** 2 - t,
        'mu_2': lambda t: t ** 2 + t - np.e * t
    }
}

CASE_11 = {
    'u_exact': lambda x, t: -0.5 * x ** 4 + x ** 2 - x + t * x + 2 * t ** 2 - t * np.exp(x),
    'kwargs': {
        'a': 0.032,
        'tau': 1 / 10000,
        'h': 1 / 10000,
        't_min': 0,
        't_max': 1,
        'x_min': 0,
        'x_max': 1,
        'f': lambda x, t: x + 4 * t - np.exp(x) - 0.032 * (-6 * (x ** 2) - t * np.exp(x) + 2),
        'mu': lambda x: -0.5 * x ** 4 + x ** 2 - x,
        'mu_1': lambda t: 2 * t ** 2 - t,
        'mu_2': lambda t: 2 * t ** 2 + t - np.e * t - 0.5
    }
}

CASE_10 = {
    'u_exact': lambda x, t: x ** 4 + t * x + t ** 2 - t * np.exp(x),
    'kwargs': {
        'a': 0.026,
        'tau': 1 / 1000,
        'h': 1 / 1000,
        't_min': 0,
        't_max': 1,
        'x_min': 0,
        'x_max': 1,
        'f': lambda x, t: x + 2 * t - np.exp(x) - 0.026 * (12 * (x ** 2) - t * np.exp(x)),
        'mu': lambda x: x ** 4,
        'mu_1': lambda t: t ** 2 - t,
        'mu_2': lambda t: 1 + t ** 2 + t - np.e * t
    }
}
