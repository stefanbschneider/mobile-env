from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.util import deep_dict_merge


class MComSmartCity(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        station_pos = [(75, 50), (125, 50), (50, 100),
                       (150, 100), (75, 150), (125, 150)]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(station_pos)
        ]
        num_ues = 10
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]

        sensor_pos = [
            (25.0, 25.0), (62.5, 25.0), (100.0, 25.0), (137.5, 25.0), (175.0, 25.0),
            (25.0, 62.5), (62.5, 62.5), (100.0, 62.5), (137.5, 62.5), (175.0, 62.5),
            (25.0, 100.0), (62.5, 100.0), (100.0, 100.0), (137.5, 100.0), (175.0, 100.0),
            (25.0, 137.5), (62.5, 137.5), (100.0, 137.5), (137.5, 137.5), (175.0, 137.5),
            (25.0, 175.0), (62.5, 175.0), (100.0, 175.0), (137.5, 175.0), (175.0, 175.0)]
        sensors = [
            Sensor(sensor_id, position, **config["sensor"]) 
            for sensor_id, position in enumerate(sensor_pos)]

        super().__init__(stations, ues, sensors, config, render_mode)
