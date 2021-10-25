from typing import Dict, Tuple
from abc import abstractmethod

import numpy as np

from mobile_env.core.entities import UserEquipment


class Movement:
    def reset(self) -> None:
        pass

    @abstractmethod
    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        pass

    @abstractmethod
    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        pass


class RandomWaypointMovement(Movement):
    def __init__(self, width: float, height: float, seed: int,
                 reset_seed_on: str, **kwargs):
        self.width, self.height = width, height
        self.seed = seed
        self.reset_seed_on = reset_seed_on

        # track waypoints and initial positions per UE
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = None
        self.initial: Dict[UserEquipment, Tuple[float, float]] = None
        # seed random movement and random initial positions
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset state of movement object after episode ends."""
        # case: movement patterns remain unchanged between episodes
        if self.reset_seed_on == "episode_end":
            self.rng = np.random.default_rng(self.seed)

        # reset UE waypoints and initial positions
        self.waypoints = {}
        self.initial = {}

    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE a step prop. to its velocity towards its random waypoint."""
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
