from typing import List
from abc import abstractmethod

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
        total_inv_rate = sum([1 / rate for rate in rates])
        return 1 / total_inv_rate
