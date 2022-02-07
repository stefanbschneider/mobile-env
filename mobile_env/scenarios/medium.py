from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.util import deep_dict_merge


class MComMedium(MComCore):
    def __init__(self, config={}):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)
        config["width"], config["height"] = 200, 300

        stations = [
            (95, 240),
            (100, 140),
            (105, 60),
            (35, 200),
            (40, 95),
            (165, 220),
            (160, 110),
        ]
        stations = [(x, y) for x, y in stations]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(stations)
        ]

        num_ues = 15
        ues = [
            UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)
        ]

        super().__init__(stations, ues, config)
