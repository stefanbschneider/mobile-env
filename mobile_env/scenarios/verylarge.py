from mobile_env.core.base import MComCore
from mobile_env.core.entities import EdgeServer, UserEquipment
from mobile_env.core.util import deep_dict_merge

import pandas

class MComVeryLarge(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        df = pandas.read_csv('./very_large/site.csv').iloc[1:, :3]

        print(df)
        return
        edge_servers = [
            (50, 100),
        ]

        stations = [(x, y) for x, y in stations]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(stations)
        ]

        num_ues = 30
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(num_ues)]

        super().__init__(stations, ues, config, render_mode)

vl = MComVeryLarge()
