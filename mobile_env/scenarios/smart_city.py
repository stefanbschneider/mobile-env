from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.util import deep_dict_merge


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

        sensor_pos = [
            (50, 50), (80, 50), (110, 50), (140, 50), (170, 50),
            (50, 80), (80, 80), (110, 80), (140, 80), (170, 80),
            (50, 110), (80, 110), (110, 110), (140, 110), (170, 110),
            (50, 140), (80, 140), (110, 140), (140, 140), (170, 140),
            (50, 170), (80, 170), (110, 170), (140, 170), (170, 170)]
        
        sensors = [
            Sensor(sensor_id, position, **config["sensor"]) 
            for sensor_id, position in enumerate(sensor_pos)]


        super().__init__(stations, ues, sensors, config, render_mode)
        self.time_step = 1