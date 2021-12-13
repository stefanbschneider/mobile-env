import matplotlib.pyplot as plt
import numpy as np


MIN_UTILITY = -20
MAX_UTILITY = 20


def log_utility(curr_dr):
    """
    More data rate increases the utility following a log function: High initial increase, then flattens.
    :param curr_dr: Current data rate
    :param factor: Factor to multiply the log function with
    :param add: Add to current data rate before passing to log function
    :return: Utility
    """
    # 4*log(0.1+x) looks good: around -10 for no dr; 0 for 0.9 dr; slightly positive for more
    # 10*log10(0.1+x) is even better because it's steeper, is exactly -10 for dr=0, and flatter for larger dr
    # with many UEs where each UE only gets around 0.1 data rate, 100*log(0.9+x) looks good (eg, 50 UEs on medium env)

    # better: 10*log10(x) --> clip to [-20, 20]; -20 for <= 0.01 dr; +20 for >= 100 dr
    # ensure min/max utility are set correctly for this utility function
    assert MIN_UTILITY == -20 and MAX_UTILITY == 20, "The chosen log utility requires min/max utility to be -20/+20"
    if curr_dr == 0:
        return MIN_UTILITY
    return np.clip(10 * np.log10(curr_dr), MIN_UTILITY, MAX_UTILITY)


dr = [i for i in range(100)]
util = [log_utility(dr) for dr in dr]
plt.plot(dr, util)
plt.xlabel("Data Rate [Mbit/s]")
plt.ylabel("Utility (QoE)")
plt.title("User utility based on data rate")
plt.show()
