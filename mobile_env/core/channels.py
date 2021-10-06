from abc import abstractmethod

import numpy as np


EPSILON = 1e-16


class Channel:
    @classmethod
    @abstractmethod
    def power_loss(cls, bs, ue):
        pass

    @classmethod
    def snr(cls, bs, ue):
        loss = cls.power_loss(bs, ue)
        power = 10**((bs.tx_power - loss) / 10)
        return power / ue.noise


class PathLoss(Channel):
    @classmethod
    def power_loss(cls, bs, ue):
        pass


class OkumuraHata(Channel):
    @classmethod
    def power_loss(cls, bs, ue):
        distance = bs.point.distance(ue.point)

        ch = 0.8 + (1.1 * np.log10(bs.frequency) - 0.7) * \
            ue.height - 1.56 * np.log10(bs.frequency)
        tmp_1 = 69.55 + 26.16 * \
            np.log10(bs.frequency) - 13.82 * np.log10(bs.height) - ch
        tmp_2 = 44.9 - 6.55 * np.log10(bs.height)

        # add small epsilon to avoid log(0) if distance = 0
        return tmp_1 + tmp_2 * np.log10(distance + EPSILON)
