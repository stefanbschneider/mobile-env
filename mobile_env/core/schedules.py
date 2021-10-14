from typing import List

from mobile_env.core.entities import BaseStation


class Schedule:
    pass


class ResourceFair:
    @classmethod
    def share(cls, bs: BaseStation, rates: List[float]):
        return [rate / len(rates) for rate in rates]


class RateFair:
    @classmethod
    def share(cls, bs: BaseStation, rates: List[float]):
        total_inv_rate = sum([1 / rate for rate in rates])
        return 1 / total_inv_rate
