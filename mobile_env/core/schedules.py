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
    def share(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
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
    def __init__(self, quantum: float = 1.0, **kwargs):
        super().__init__()
        self.last_served_index = {}
        self.quantum = quantum

    def reset(self):
        self.last_served_index.clear()

    def share(self, bs: BaseStation, rates: List[float], total_resources: float) -> List[float]:
        if not rates:
            return []

        num_ues = len(rates)
        if bs.bs_id not in self.last_served_index:
            self.last_served_index[bs.bs_id] = -1

        allocation = [0] * num_ues
        rem_rates = rates[:]
        #total_resources = bs.bw  # Assuming 'bandwidth' represents the total resources
        t = 0  # Current time for resource allocation

        while True:
            done = True

            for i in range(num_ues):
                if rem_rates[i] > 0:
                    done = False  # There is a pending process
                    if rem_rates[i] > self.quantum:
                        t += self.quantum
                        allocation[i] += self.quantum
                        rem_rates[i] -= self.quantum
                    else:
                        t += rem_rates[i]
                        allocation[i] += rem_rates[i]
                        rem_rates[i] = 0

            if done:
                break

        # Normalize the allocation based on the total resources available
        total_allocated = sum(allocation)
        if total_allocated > total_resources:
            allocation = [alloc * total_resources / total_allocated for alloc in allocation]

        return allocation
    
    def share_ue(self, bs: BaseStation, rates: List[float], ue_bandwidth: float) -> List[float]:
        return self.share(bs, rates, ue_bandwidth)

    def share_sensor(self, bs: BaseStation, rates: List[float], sensor_bandwidth: float) -> List[float]:
        return self.share(bs, rates, sensor_bandwidth)