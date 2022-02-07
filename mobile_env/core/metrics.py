import numpy as np


def number_connections(sim):
    """Calculates the total number of connections."""
    return sum([len(con) for con in sim.connections.values()])


def number_connected(sim):
    """Calculates the number of UEs that are connected."""
    return len(set.union(set(), *sim.connections.values()))


def mean_datarate(sim):
    """Calculates the average data rate of UEs."""
    if not sim.macro:
        return 0.0

    return np.mean(list(sim.macro.values()))


def mean_utility(sim):
    """Calculates the average utility of UEs."""
    if not sim.utilities:
        return sim.utility.lower

    return np.mean(list(sim.utilities.values()))
