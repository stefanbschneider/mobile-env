from abc import abstractmethod

import numpy as np

from mobile_env.core.entities import UserEquipment


EPSILON = 1e-16


class Channel:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def power_loss(cls, bs, ue):
        """Calculate power loss for transmission between BS and UE."""
        pass

    @classmethod
    def snr(cls, bs, ue):
        """Calculate SNR for transmission between BS and UE."""
        loss = cls.power_loss(bs, ue)
        power = 10 ** ((bs.tx_power - loss) / 10)
        return power / ue.noise

    @classmethod
    def datarate(cls, bs, ue, snr):
        """Calculate max. data rate for transmission between BS and UE."""
        if snr > ue.snr_threshold:
            return bs.bw * np.log2(1 + snr)

        return 0.0

    @classmethod
    def isoline(cls, bs, ue_config, map_bounds, dthresh, num=32):
        """Isoline where UEs receive at least `dthres` max. data."""
        width, height = map_bounds

        dummy = UserEquipment(0, (0.0, 0.0), **ue_config)

        isoline = []

        for theta in np.linspace(EPSILON, 2 * np.pi, num=num):
            # calculate collision point with map boundary
            x1, y1 = cls.boundary_collison(theta, bs.x, bs.y, width, height)

            # points on line between BS and collision with map
            slope = (y1 - bs.y) / (x1 - bs.x)
            xs = np.linspace(bs.x, x1, num=100)
            ys = slope * (xs - bs.x) + bs.y

            # compute data rate for each point
            def drate(point):
                dummy.x, dummy.y = point
                snr = cls.snr(bs, dummy)

                return cls.datarate(bs, dummy, snr)

            points = zip(xs.tolist(), ys.tolist())
            datarates = np.asarray(list(map(drate, points)))

            # find largest / smallest x coordinate where drate is exceeded
            (idx,) = np.where(datarates > dthresh)
            idx = np.max(idx)

            isoline.append((xs[idx], ys[idx]))

        xs, ys = zip(*isoline)
        return xs, ys

    @classmethod
    def boundary_collison(cls, theta, x0, y0, width, height):
        """Find point on map boundaries with angle theta to BS."""
        # collision with right boundary of map rectangle
        rgt_x1, rgt_y1 = width, np.tan(theta) * (width - x0) + y0
        # collision with upper boundary of map rectangle
        upr_x1, upr_y1 = (-1) * np.tan(theta - 1 / 2 * np.pi) * (
            height - y0
        ) + x0, height
        # collision with left boundary of map rectangle
        lft_x1, lft_y1 = 0.0, np.tan(theta) * (0.0 - x0) + y0
        # collision with lower boundary of map rectangle
        lwr_x1, lwr_y1 = np.tan(theta - 1 / 2 * np.pi) * (y0 - 0.0) + x0, 0.0

        if theta == 0.0:
            return width, y0

        elif theta > 0.0 and theta < 1 / 2 * np.pi:
            x1 = np.min((rgt_x1, upr_x1, width))
            y1 = np.min((rgt_y1, upr_y1, height))
            return x1, y1

        elif theta == 1 / 2 * np.pi:
            return x0, height

        elif theta > 1 / 2 * np.pi and theta < np.pi:
            x1 = np.max((lft_x1, upr_x1, 0.0))
            y1 = np.min((lft_y1, upr_y1, height))
            return x1, y1

        elif theta == np.pi:
            return 0.0, y0

        elif theta > np.pi and theta < 3 / 2 * np.pi:
            return np.max((lft_x1, lwr_x1, 0.0)), np.max((lft_y1, lwr_y1, 0.0))

        elif theta == 3 / 2 * np.pi:
            return x0, 0.0

        else:
            x1 = np.min((rgt_x1, lwr_x1, width))
            y1 = np.max((rgt_y1, lwr_y1, 0.0))
            return x1, y1


class OkumuraHata(Channel):
    @classmethod
    def power_loss(cls, bs, ue):
        distance = bs.point.distance(ue.point)

        ch = (
            0.8
            + (1.1 * np.log10(bs.frequency) - 0.7) * ue.height
            - 1.56 * np.log10(bs.frequency)
        )
        tmp_1 = (
            69.55 - ch + 26.16 * np.log10(bs.frequency)
            - 13.82 * np.log10(bs.height)
        )
        tmp_2 = 44.9 - 6.55 * np.log10(bs.height)

        # add small epsilon to avoid log(0) if distance = 0
        return tmp_1 + tmp_2 * np.log10(distance + EPSILON)
