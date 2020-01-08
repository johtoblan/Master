import numpy as np
import numba

class HelicopterTriggerIndex(object):
    """docstring for HelicopterTriggerIndex."""

    def __init__(self, functions, neighbourhood_size = 1):
        self.functions = functions
        self.N = len(functions)
        self.nSize = neighbourhood_size

    def __call__(self, arrays):
        """
        arrays is a list of arrays, self.N long, which are either 1x1 or neighbourhood_size X neighbourhood_size big

        return HelicopterTriggerIndex with values in [0,1]
        """
        HTI = 0

        for subfunction, i in enumerate(self.functions):
            HTI += subfunction(arrays[i])

        # Average subIndexes to last HTI
        return HTI/self.N


def temperature_max_band_from_b_to_c(b,c):
    """
    Gives high risk (1 or 100%) for temperatures in [b,c] gives [0,1] for [a,b]
    and [1,0] for [c,d] (linear mapping)
    """
    # Define a and d for linear mapping
    a = b + 1
    d = c - 1
    def f(T):
        # Risk is 0 if not inside band or outliers
        temperature = 0
        # In higher band, increasing risk
        if T > b and T < a:
            # Already checked if valued between a and b, won't get problems from abs()
            temperature = abs(abs(T) - abs(a))
        # In main band of temperatures
        elif b <= T and c >= T:
            temperature = 1
        # In lower band, decreasing risk
        elif T < c and T > c:
            temperature = abs(abs(T) - abs(c))
        return temperature
    return np.vectorize(f)
