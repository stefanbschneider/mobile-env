from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.util import deep_dict_merge
import random

class MComSmartCity(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        station_pos = [(100, 100)]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(station_pos)
        ]
        num_ues = 3
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]
        width = 150
        height = 150
        num_sensors = 5
        
        sensor_pos = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_sensors)]
  
        sensors = [
            Sensor(sensor_id, position, **config["sensor"]) 
            for sensor_id, position in enumerate(sensor_pos)]


        super().__init__(stations, ues, sensors, config, render_mode)
        self.time_step = 1