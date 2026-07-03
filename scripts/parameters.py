"""
SMA tuning parameters.

epsilon_target(m): target accuracy epsilon as a function of m.
J_I_from_epsilon(m, eps): iteration count J and grid size I implied by (m, eps).
"""

import math
import numpy as np

def epsilon_target(m):
    return 0.2 * m**2 / (1 + m**2)


def J_I_from_epsilon(m: float, eps: float) -> tuple:
    G = 4 * m**2
    C = 3 * m**3 + 4 * m**2
    J_float = 2 * (G**2) * np.log(1.0 + C / (eps/2)) / ((eps/2)**2)
    I_float = 1 + C / (eps/2)
    J = int(math.ceil(J_float))
    I = int(math.ceil(I_float))
    return J, I
