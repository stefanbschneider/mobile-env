from abc import abstractmethod
from typing import List

from mobile_env.core.entities import BaseStation
import numpy as np


class Scheduler:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        pass


class ResourceFair(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        return [rate / len(rates) for rate in rates]


class RateFair(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        if all(rate == 0 for rate in rates):
            return [0.0] * len(rates)  # Avoid division by zero

        total_inv_rate = sum(1.0 / rate if rate > 0 else 0 for rate in rates)
        return [(1.0 / rate if rate > 0 else 0) / total_inv_rate for rate in rates]
    
class ProportionalFair(Scheduler):
    def __init__(self):
        self.average_rates = {}

    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        if not rates:
            return []

        # Update the average rates
        for ue, rate in zip(bs.connected_ues, rates):
            if ue not in self.average_rates:
                self.average_rates[ue] = rate
            else:
                self.average_rates[ue] = 0.9 * self.average_rates[ue] + 0.1 * rate

        # Calculate the proportional fairness metric
        pf_metric = [rate / self.average_rates[ue] for ue, rate in zip(bs.connected_ues, rates)]
        total_pf_metric = sum(pf_metric)

        # Allocate resources proportionally
        return [(metric / total_pf_metric) * bs.total_resources for metric in pf_metric]

class RoundRobin(Scheduler):
    def __init__(self):
        self.last_served_index = -1

    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        if not rates:
            return []

        num_ues = len(rates)
        allocation = [0] * num_ues
        resources_per_ue = bs.total_resources / num_ues

        for i in range(num_ues):
            self.last_served_index = (self.last_served_index + 1) % num_ues
            allocation[self.last_served_index] = min(resources_per_ue, rates[self.last_served_index])

        return allocation


