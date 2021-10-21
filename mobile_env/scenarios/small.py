from typing import Dict

import gym

from mobile_env.core.base import MComCore
from mobile_env.core.channels import OkumuraHata
from mobile_env.core.schedules import ResourceFair
from mobile_env.core.entities import BaseStation, UserEquipment


class MComSmall(MComCore):
    def __init__(self, config={}):
        # set unspecified parameters to default configuration
        config = {**self.default_config(), **config}

        stations = [(110, 130), (65, 80), (120, 30)]
        stations = [(x, y) for x, y in stations]
        stations = [BaseStation(bs_id, pos, **config['bs'])
                    for bs_id, pos in enumerate(stations)]

        ues = [(0, 5), (10, 20), (50, 20), (50, 70), (60, 30)]
        ues = [UserEquipment(ue_id, pos, **config['ue'])
               for ue_id, pos in enumerate(ues)]

        super().__init__(stations, ues, config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({'bs': {'bw': 9e6, 'freq': 2500,
                              'tx': 30, 'height': 50}})
        config.update({'ue': {'stime': 0, 'extime': config['EP_MAX_TIME'], 'velocity': 1.5,
                              'snr_tr': 2e-8, 'noise': 1e-9, 'height': 1.5}})
        return config
