from typing import Tuple, Dict, Optional
from shapely.geometry import Point
from mobile_env.core.buffers import JobQueue


class BaseStation:
    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bw: float,
        freq: float,
        tx: float,
        height: float,
        computational_power: float,
    ):
        # BS ID should be final, i.e., BS ID must be unique
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        self.height = height  # in m
        self.computational_power = computational_power  # units

        # Initialize data buffers for UEs and Sensors
        self.data_buffer_uplink_ue = self._init_job_queue()
        self.data_buffer_uplink_sensor = self._init_job_queue()
        self.data_buffer_downlink_ue = self._init_job_queue()
        self.data_buffer_downlink_sensor = self._init_job_queue()

        # Existing buffers (if still needed)
        self.transferred_jobs_ue = self._init_job_queue()
        self.transferred_jobs_sensor = self._init_job_queue()
        self.accomplished_jobs_ue = self._init_job_queue()
        self.accomplished_jobs_sensor = self._init_job_queue()

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"BS: {self.bs_id}"
        
    def _init_job_queue(self) -> JobQueue:
        return JobQueue(size=1000)


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
        self.data_buffer_uplink = self._init_job_queue()

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"UE: {self.ue_id}"
    
    def _init_job_queue(self) -> JobQueue:
        return JobQueue(size=1000)


class Sensor:
    def __init__(
            self,
            sensor_id: int,
            pos: Tuple[float, float],
            height: float,
            snr_tr: float,
            noise: float,
            velocity: float,
            radius: float,
            logs: Optional[Dict[int, int]] = None,
    ):
        self.sensor_id = sensor_id
        self.x, self.y = pos
        self.height = height
        self.snr_threshold = snr_tr
        self.noise = noise
        self.velocity = velocity
        self.radius = radius
        self.logs = logs if logs else {}
        self.connected_base_station: Optional[BaseStation] = None
        self.data_buffer_uplink = self._init_job_queue()

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"Sensor: {self.sensor_id}"
    
    def _init_job_queue(self) -> JobQueue:
        return JobQueue(size=1000)

    def _initialize_sensor_logs(self, sensors: list["Sensor"]):
        for sensor in sensors:
            if not isinstance(sensor.logs, dict):
                sensor.logs = {}

    def _is_within_range(self, ue_point):
        """Check if a UE is within the sensor's range."""
        return self.point.distance(ue_point) <= self.radius