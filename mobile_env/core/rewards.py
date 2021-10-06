from typing import Dict, Tuple

import numpy as np

from mobile_env.core.entities import BaseStation, UserEquipment


class MultiUserReward:

    def rewards(self, utilities, connections, connectable):
        """
        Define each UEs' reward as its own utility aggregated with 
        the average utility of nearby basestations.  
        """

        # compute average utility of UEs for each BS
        # set default to 0.0 if no UEs are connected
        total_bs_utilities = {
            bs: sum(utilities[ue] for ue in ues) for bs, ues in connections.items()}

        rewards = {}
        for ue in utilities:
            # utilities are broadcasted, i.e., aggregate utilities of BSs in range
            total_ngbr_utility = sum(
                total_bs_utilities[bs] for bs in connectable[ue])

            # calculate rewards as average weighted by the number of each BSs' connections
            total_ngbr_connections = sum(
                len(connections[bs]) for bs in connectable[ue])
            reward = (total_ngbr_utility +
                      utilities[ue]) / (total_ngbr_connections + 1)

            rewards[ue.ue_id] = reward

        return reward
