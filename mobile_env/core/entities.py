from typing import Tuple
from shapely.geometry import Point
from queues import Buffer


class BaseStation:
    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bw: float,
        freq: float,
        tx: float,
        height: float,
        size = 10,
    ):
        # BS ID should be final, i.e., BS ID must be unique
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        self.height = height  # in m
        self.data_buffer_uplink = Buffer(size) #adding buffer for uplink communicaiton 

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
        size = 10,
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
        self.data_buffer_uplink = Buffer(size) 
    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"UE: {self.ue_id}"

class Sensor:
    def __init__(
            self,
            sensor_id: int,
            pos: Tuple[float, float],
            height: float,
            snr_tr: float,
            range: float,
            velocity: float,
            radius: float,
            logs: dict[int, int],
            size = 10,
    ):
        self.sensor_id = sensor_id
        self.x, self.y = pos
        self.height = height
        self.snr_threshold = snr_tr
        self.range = range
        self. velocity = velocity
        self.radius = radius
        self.logs = logs
        self.data_buffer_1 = Buffer(size)

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def is_within_range(self, ue_point):
        """Check if a UE is within the sensor's range."""
        return self.point.distance(ue_point) <= self.radius
    
    def __str__(self):
        return f"Sensor: {self.sensor_id}"