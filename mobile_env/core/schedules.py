from abc import abstractmethod
from typing import List

from mobile_env.core.entities import BaseStation


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
