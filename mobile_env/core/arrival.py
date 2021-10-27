from abc import abstractmethod

import numpy as np

from mobile_env.core.entities import UserEquipment


class Arrival:
    def __init__(self, ep_time: int, seed: int, reset_rng_episode: bool):
        self.ep_time = ep_time
        self.seed = seed
        self.reset_rng_episode = reset_rng_episode
        # RNG for arrival and departure times of UEs
        self.rng = None

    def reset(self) -> None:
        # case: movement patterns remain unchanged between episodes
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def arrival(self, ue: UserEquipment) -> int:
        pass

    @abstractmethod
    def departure(self, ue: UserEquipment) -> int:
        pass


class NoDeparture(Arrival):
    """Alle UEs immediately request service and do not depart thereafter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def arrival(self, ue: UserEquipment) -> int:
        return 0

    def departure(self, ue: UserEquipment) -> int:
        return self.ep_time
