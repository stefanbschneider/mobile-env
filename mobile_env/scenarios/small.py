from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.util import deep_dict_merge


class MComSmall(MComCore):
    def __init__(self, config={}):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        station_pos = [(110, 130), (65, 80), (120, 30)]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(station_pos)
        ]
        num_ues = 5
        ues = [
            UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)
        ]

        super().__init__(stations, ues, config)
