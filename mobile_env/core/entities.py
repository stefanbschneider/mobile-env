from typing import Tuple

from shapely.geometry import Point


class BaseStation:
    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bw: float,
        freq: float,
        tx: float,
        height: float,
    ):
        # BS ID is final, i.e., cannot be set to another value & must be unique
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        self.height = height  # in m

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"BS: {self.bs_id}"


class UserEquipment:
    def __init__(
        self,
        ue_id: int,
        init_pos: Tuple[float, float],
        stime: int,
        extime: int,
        velocity: float,
        snr_tr: float,
        noise: float,
        height: float,
    ):
        # UE ID is final, i.e., cannot be set to another value & must be unique
        self.ue_id = ue_id
        self.init_pos: Tuple[float, float] = init_pos
        self.x, self.y = self.init_pos
        self.stime: int = stime
        self.extime: int = extime
        self.velocity: float = velocity

        self.snr_threshold = snr_tr
        self.noise = noise
        self.height = height

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"UE: {self.ue_id}"
