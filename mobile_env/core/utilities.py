from typing import Tuple

import numpy as np


class BoundedLogUtility:
    def __init__(self, lower: float, upper: float, coeffs: Tuple[float, float, float]):
        self.lower = lower
        self.upper = upper
        self.coeffs = coeffs

    def utility(self, ue_datarate):
        w1, w2, w3 = self.coeffs
        utility = np.clip(w1 * np.log(w2 + ue_datarate) /
                          np.log(w3), self.lower, self.upper)

        # normalize utility to bounds (-1, 1)
        return (utility - self.lower) / (self.upper - self.lower)
