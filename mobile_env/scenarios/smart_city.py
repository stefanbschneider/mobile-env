from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.util import deep_dict_merge
import random

class MComSmartCity(MComCore):
    def __init__(self, config={}, render_mode=None):
        # Set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)
        
        # Set the seed for random number generation
        self.seed = config["seed"]
        if self.seed is not None:
            random.seed(self.seed)

        # Initialize BSs
        station_positions = [(100, 100)]
        stations = self.create_stations(station_positions, config["bs"])

        # Initialize UEs
        num_ues = 5
        ues = self.create_user_equipments(num_ues, config["ue"])

        # Initialize sensors
        num_sensors = 10
        max_distance = 90               # Maximum distance from any base station
        min_distance = 20                # Minimum distance between sensors and between sensors and base stations
        sensor_positions = self.place_sensors(num_sensors, station_positions, max_distance, min_distance)

        sensors = self.create_sensors(num_sensors, sensor_positions, config["sensor"])

        super().__init__(stations, ues, sensors, config, render_mode)

    def create_stations(self, station_positions, bs_config):
        """Create base stations given their positions and configuration."""
        return [
            BaseStation(bs_id, pos, **bs_config)
            for bs_id, pos in enumerate(station_positions)
        ]

    def create_user_equipments(self, num_ues, ue_config):
        """Create user equipment objects based on their configuration."""
        return [
            UserEquipment(ue_id, **ue_config) 
            for ue_id in range(num_ues)
        ]

    def create_sensors(self, num_sensors, sensor_positions, sensor_config):
        """Create sensors based on their positions and configuration."""
        return [
            Sensor(sensor_id, position, **sensor_config) 
            for sensor_id, position in enumerate(sensor_positions)
        ]

    def place_sensors(self, num_sensors, station_positions, max_distance, min_distance):
        """Place sensors far from each other and away from multiple base stations."""
        sensor_positions = []
        for _ in range(num_sensors):
            while True:
                x, y = self.generate_random_position()
                if self.is_valid_position(x, y, station_positions, sensor_positions, max_distance, min_distance):
                    sensor_positions.append((x, y))
                    break
        return sensor_positions

    def generate_random_position(self):
        """Generate a random (x, y) position within the grid."""
        return random.randint(0, 200), random.randint(0, 200)

    def is_valid_position(self, x, y, station_positions, existing_sensor_positions, max_distance, min_distance):
        """Check if the position is within the max distance from any BS and far enough from all other sensors."""
        return (self.is_within_max_distance_from_stations(x, y, station_positions, max_distance) and
                self.is_far_enough_from_others(x, y, existing_sensor_positions, min_distance) and
                self.is_far_enough_from_stations(x, y, station_positions, min_distance))

    def is_within_max_distance_from_stations(self, x, y, station_positions, max_dist):
        """Check if a sensor is within the maximum distance from any of the BSs."""
        return any((x - sx) ** 2 + (y - sy) ** 2 <= max_dist ** 2 for sx, sy in station_positions)

    def is_far_enough_from_others(self, x, y, existing_sensor_positions, min_dist):
        """Check if a sensor point is far enough from all other sensor points."""
        return all((x - ox) ** 2 + (y - oy) ** 2 >= min_dist ** 2 for (ox, oy) in existing_sensor_positions)

    def is_far_enough_from_stations(self, x, y, station_positions, min_dist):
        """Check if a sensor point is far enough from all BSs."""
        return all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist ** 2 for sx, sy in station_positions)