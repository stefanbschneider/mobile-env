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
        total_inv_rate = sum([1 / rate for rate in rates])
        return 1 / total_inv_rate


class RoundRobin(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        candidates = np.nonzero(rates)
        scheduled = np.random.choice(candidates)
        