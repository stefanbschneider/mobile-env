from mobile_env.core.base import MComCore
from mobile_env.core.entities import EdgeServer, UserEquipment, BaseStation
from mobile_env.core.util import deep_dict_merge

import pandas


class MComVeryLarge(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        df = pandas.read_csv("./very_large/site.csv").iloc[1:, :3]
        edge_servers = [
            (df.at[_, 0], 0, 0, df.at[_, 1], df.at[_, 2]) for _ in range(len(df))
        ]

        df = pandas.read_csv("./very_large/users-aus.csv").iloc[1:, 1:3]
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(len(df))]

        # the following code for station is temperory
        stations = [(x, y) for x, y in stations]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(stations)
        ]

        super().__init__(stations, ues, config, render_mode)
