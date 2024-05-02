from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.util import deep_dict_merge

import random


class MComSmartCity(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        random.seed(10)
        station_pos = [(random.randint(50,100),random.randint(50,100)) for _ in range(0,5)]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(station_pos)
        ]
        num_ues = 10
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]
        
        bs_range = 
        sensor_pos = [for _ ]
        sensors = [
            Sensor(sensor_id, position, **config["sensor"]) 
            for sensor_id, position in enumerate(sensor_pos)]
        

        super().__init__(stations, ues, sensors, config, render_mode)