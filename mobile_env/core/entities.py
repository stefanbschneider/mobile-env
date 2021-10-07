from typing import Tuple

from shapely.geometry import Point


class BaseStation:
    def __init__(self, bs_id: int, pos: Tuple[float, float], bw: float, freq: float, tx: float, height: float):
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw   # in Hz
        self.frequency = freq    # in MHz
        self.tx_power = tx  # in dBm
        self.height = height    # in m

        # visualization purpose:
        # TODO: compute connection range
        # TODO: compute 1MB range

    def __str__(self):
        return str(self.bs_id)

    @property
    def point(self):
        return Point(int(self.x), int(self.y))


class UserEquipment:
    def __init__(self, ue_id: int, init_pos: Tuple[float, float], stime: int, extime: int, velocity: float, snr_tr: float, noise: float, height: float):
        self.ue_id = ue_id
        self.init_pos: Tuple[float, float] = init_pos
        self.x, self.y = self.init_pos
        self.stime: int = stime
        self.extime: int = extime
        self.velocity: float = velocity

        self.snr_threshold = snr_tr
        self.noise = noise
        self.height = height

    def __lt__(self, other):
        return self.extime < other.extime

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def reset(self):
        self.x, self.y = self.init_pos

    def __str__(self):
        return str(self.ue_id)
