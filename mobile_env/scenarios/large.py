from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment


class MComLarge(MComCore):
    def __init__(self, config={}):
        # set unspecified parameters to default configuration
        config = {**self.default_config(), **config}

        config.update({"width": 300, "height": 300})
        stations = [
            (50, 100),
            (60, 210),
            (90, 60),
            (120, 130),
            (130, 215),
            (140, 190),
            (160, 70),
            (200, 250),
            (210, 135),
            (230, 70),
            (250, 240),
            (255, 170),
            (265, 50),
        ]
        stations = [(x, y) for x, y in stations]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(stations)
        ]

        num_ues = 30
        ues = [
            UserEquipment(ue_id, **config["ue"])
            for ue_id in range(num_ues)
        ]

        super().__init__(stations, ues, config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
                "bs": {"bw": 9e6, "freq": 2500, "tx": 30, "height": 50}
                })
        config.update(
            {
                "ue": {
                    "velocity": 1.5,
                    "snr_tr": 2e-8,
                    "noise": 1e-9,
                    "height": 1.5,
                }
            }
        )
        return config
