import numpy as np
class HelicopterTriggerIndex(object):
    """docstring for HelicopterTriggerIndex."""

    def __init__(self, functions, neighbourhood_size = 1):
        self.functions = functions
        self.N = len(functions)
        self.nSize = neighbourhood_size

    def __call__(self, arrays):
        """
        arrays is a list of arrays, self.N long, which are either 1x1 or neighbourhood_size X neighbourhood_size big

        return HTI valued [0,1]
        """
        HTI = 0

        for subfunction, i in enumerate(self.functions):
            HTI += subfunction(arrays[i])

        # Average subIndexes to last HTI
        return HTI/self.N


def temperature_linear_from_a_to_b(a,b,c):
    def f(T):
        if T < a:
            return 0
        elif T <= b:
            # Returns 0 for T = a, 1 for T = b
            return (T-a) / (b - a)
        elif T > b and T < c:
            # Returns 1 for T = b, 0 for T = c
            return 1 - (T-b)/(c-b)
        else:
            return 0
    return np.vectorize(f)


# Define test-case (As forecasted today)
