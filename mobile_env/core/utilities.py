from typing import Tuple

import numpy as np


class BoundedLogUtility:
    def __init__(self, lower: float, upper: float, coeffs: Tuple[float, float, float]):
        self.lower = lower
        self.upper = upper
        self.coeffs = coeffs

    def utility(self, datarate):
        w1, w2, w3 = self.coeffs
        if datarate <= 0.0:
            return self.lower

        utility = np.clip(w1 * np.log(w2 + datarate) /
                          np.log(w3), self.lower, self.upper)
        return utility

    def scale(self, utility):
        # scale utility to range [-1, 1]
        return 2 * (utility - self.lower) / (self.upper - self.lower) - 1

    def unscale(self, utility):
        # invert scaling of utility
        return (utility + 1) / 2 * (self.upper - self.lower) + self.lower
