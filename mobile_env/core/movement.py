from typing import Dict, Tuple
from abc import abstractmethod

import numpy as np

from mobile_env.core.entities import UserEquipment


class Movement:
    def __init__(self, width: float, height: float,
                 seed: int, reset_rng_episode: str):

        self.width, self.height = width, height
        self.reset_rng_episode = reset_rng_episode

        # RNG for movement and initial positions of UEs
        self.seed = seed
        self.rng = None

    def reset(self) -> None:
        """Reset state of movement object after episode ends."""
        # case: movement patterns remain unchanged between episodes
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE at each time step."""
        pass

    @abstractmethod
    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Reset position of UE e.g. after episode ends."""
        pass


class RandomWaypointMovement(Movement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # track waypoints and initial positions per UE
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = None
        self.initial: Dict[UserEquipment, Tuple[float, float]] = None

    def reset(self) -> None:
        super().reset()
        # NOTE: if RNG is not resetted after episode ends,
        # initial positions will differ between episodes
        self.waypoints = {}
        self.initial = {}

    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE a step towards the random waypoint."""
        # generate random waypoint if UE has none so far
        if ue not in self.waypoints:
            wx = self.rng.uniform(0, self.width)
            wy = self.rng.uniform(0, self.height)
            self.waypoints[ue] = (wx, wy)

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.waypoints[ue])

        # if already close enough to waypoint, move directly onto waypoint
        if np.linalg.norm(position - waypoint) <= ue.velocity:
            # remove waypoint from dict after it has been reached
            waypoint = self.waypoints.pop(ue)
            return waypoint

        # else move by self.velocity towards waypoint
        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)

        return tuple(position)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Return initial position of UE at the beginning of the episode."""
        if ue not in self.initial:
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            self.initial[ue] = (x, y)

        x, y = self.initial[ue]
        return x, y
