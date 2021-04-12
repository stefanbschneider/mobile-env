"""
Heuristic algorithms to use as baseline. Only work as multi-agent, not central (would be the same anyways).
"""
import numpy as np

from deepcomp.agent.base import MultiAgent


class Heuristic3GPP(MultiAgent):
    """
    Agent that is always connected to at most one BS. Greedily chooses the BS with highest achievable data rate.
    This is comparable to 3GPP LTE cell selection based on highest SINR (with a hysteresis threshold of 0)
    """
    def __init__(self):
        super().__init__()

    def compute_action(self, obs, policy_id):
        """
        Compute an action for one UE by connecting to the BS with highest data rate (if not connected yet).
        Gets called for all UEs by simulator.

        :param obs: Observation of a UE
        :param policy_id: Ignored since the heuristic behaves identically for all UEs; just based on obs.
        :return: Selected action: 0 = noop. 1-n = index of BS +1 to connect/disconnect
        """
        # identify BS with highest data rate; in case of a tie, take the first one
        best_bs = np.argmax(obs['dr'])
        # if already connected to this BS, stay connected = do nothing
        if obs['connected'][best_bs]:
            return 0
        # if connected to other BS, disconnect first
        if sum(obs['connected']) > 0:
            conn_bs = obs['connected'].index(1)
            return conn_bs + 1
        # else: not connected yet --> connect to best BS
        return best_bs + 1


class FullCoMP(MultiAgent):
    """Agent that always greedily connects to all BS. I refer to this agent as 'FullCoMP' in the paper."""
    def __init__(self):
        super().__init__()

    def compute_action(self, obs, policy_id):
        """
        Compute action for a UE. Try to connect to all BS. Prioritize BS with higher data rate.

        :param obs: Observations of the UE
        :param policy_id: Ignored
        :return: Action for the UE
        """
        # identify BS that are not yet connected
        disconn_bs = [idx for idx, conn in enumerate(obs['connected']) if not conn]
        # if connected to all BS already, do nothing
        if len(disconn_bs) == 0:
            return 0
        # else connect to the BS with the highest data rate
        best_bs = disconn_bs[0]
        best_dr = obs['dr'][best_bs]
        for bs in disconn_bs:
            if obs['dr'][bs] > best_dr:
                best_bs = bs
                best_dr = obs['dr'][bs]
        # 0 = noop --> select BS with BS index + 1
        return best_bs + 1


class DynamicSelection(MultiAgent):
    """
    Heuristic that dynamically selects cells per UE depending on the SINR.
    It always selects the strongest cell with SINR-1st and all cells that are within epsilon * SINR-1st.

    Based on the following paper: 'Multi-point fairness in resource allocation for C-RAN downlink CoMP transmission'
    https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-015-0501-4
    """
    def __init__(self, epsilon):
        """
        :param epsilon: Scaling factor
        """
        super().__init__()
        self.epsilon = epsilon

    def compute_action(self, obs, policy_id):
        """Select strongest BS and all that are within epsilon * SINR of that BS"""
        # get set of selected cells
        best_snr = max(obs['dr'])
        threshold = best_snr * self.epsilon
        selected_bs = [idx for idx, snr in enumerate(obs['dr']) if snr >= threshold]

        connected_bs = [idx for idx, conn in enumerate(obs['connected']) if conn]
        # disconnect from any BS not in the set of selected BS
        for bs in connected_bs:
            if bs not in selected_bs:
                # 0 = noop --> select BS with BS index + 1
                return bs + 1

        # then connect to BS inside set, starting with the strongest --> sort with decreasing SINR
        selected_bs_sorted = sorted(selected_bs, key=lambda idx: obs['dr'][idx], reverse=True)
        for bs in selected_bs_sorted:
            if not obs['connected'][bs]:
                return bs + 1

        # else do nothing
        return 0
