import random
from typing import Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np
from shapely.geometry import Point

from mobile_env.core.entities import UserEquipment


class Movement(ABC):
    @staticmethod
    @abstractmethod
    def move(ue):
        pass


class RandomWaypointMovement():
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = {}

        # TODO: seed appropriately!!!
        self.rng = random.Random()

    def move(self, ue):
        # generate random waypoint if UE has none so far
        if ue not in self.waypoints:
            wx = self.rng.uniform(0, self.width)
            wy = self.rng.uniform(0, self.height)
            self.waypoints[ue] = (wx, wy)

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.waypoints[ue])

        # if already close enough to waypoint, move directly onto waypoint (not past it)
        if np.linalg.norm(position - waypoint) <= ue.velocity:
            # remove waypoint from dict after it has been reached
            waypoint = self.waypoints.pop(ue)
            return waypoint

        # else move by self.velocity towards waypoint: https://math.stackexchange.com/a/175906/234077
        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)

        return tuple(position)
