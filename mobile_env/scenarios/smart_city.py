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
        num_ues = 5
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]
        num_sensors = 5

        max_distance = 70  # Maximum distance from the base station
        min_distance = 30   # Minimum distance between sensors

        
        def is_within_max_distance(x, y, center_x, center_y, max_dist):
            return (x - center_x) ** 2 + (y - center_y) ** 2 <= max_dist ** 2

        def is_far_enough_from_others(x, y, other_positions, min_dist):
            for (ox, oy) in other_positions:
                if (x - ox) ** 2 + (y - oy) ** 2 < min_dist ** 2:
                    return False
            return True

        sensor_pos = []
        for _ in range(num_sensors):
            while True:
                x = random.randint(0, 200)
                y = random.randint(0, 200)
                if is_within_max_distance(x, y, station_pos[0][0], station_pos[0][1], max_distance) and \
                   is_far_enough_from_others(x, y, sensor_pos, min_distance):
                    sensor_pos.append((x, y))
                    break
  
        sensors = [
            Sensor(sensor_id, position, **config["sensor"]) 
            for sensor_id, position in enumerate(sensor_pos)]


        super().__init__(stations, ues, sensors, config, render_mode)
        self.time_step = 1