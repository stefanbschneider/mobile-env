from typing import Tuple, List

from shapely.geometry import Point
import random


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
        # BS ID should be final, i.e., BS ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
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
        velocity: float,
        snr_tr: float,
        noise: float,
        height: float,
    ):
        # UE ID should be final, i.e., UE ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.ue_id = ue_id
        self.velocity: float = velocity
        self.snr_threshold = snr_tr
        self.noise = noise
        self.height = height

        self.x: float = None
        self.y: float = None
        self.stime: int = None
        self.extime: int = None

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"UE: {self.ue_id}"


class EdgeServer:
    def __init__(
        self, es_id: int, inp_id: int, bs_id: int, loc_x: float, loc_y: float
    ) -> None:
        self.es_id = es_id
        self.inp_id = inp_id
        self.bs_id = bs_id
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.bundle = None

    def offer_bundle(self):
        self.bundle = {
            "storage": random.randint(1, 1000),  # in GB
            "cpu": random.randint(1, 230),  # in vCPU
        }
        return self.bundle

    def choose_bid_winner(self, bids: List[Tuple[int, int]]):
        # Pay attention to the case of equal bids
        return max(bids)[1]

    def __str__(self) -> str:
        return f"ES: {self.es_id}"


class Task:
    def __init__(
        self,
        ue_id: int,
        computing_req: int,
        data_req: int,
        latency_req: float,
    ):
        self.ue_id = ue_id
        self.computing_req = computing_req
        self.data_req = data_req
        self.latency_req = latency_req

    def __str__(self):
        return f"Task: (ue: {self.ue_id}, req: ({self.computing_req, self.data_req, self.latency_req}))"
