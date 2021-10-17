from typing import Dict

import gym
import numpy as np


class MultiAgentMComWrapper(gym.ActionWrapper, gym.ObservationWrapper, gym.RewardWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

        # define dictionary action space for multi-agent setting
        self.action_space = gym.spaces.Dict({ue.ue_id: gym.spaces.Discrete(
            self.NUM_STATIONS + 1) for ue in self.users.values()})

        # define what features are observed in MA setting per UE
        self.ue_obs_features = ['connections', 'snrs',
                                'utility', 'bcast', 'stations_connected']

        # define dictionary observation spaec for multi-agent setting
        self.UE_OBS_SIZE = sum(
            self.feature_sizes[ftr] for ftr in self.ue_obs_features)
        self.observation_space = gym.spaces.Dict({ue.ue_id: gym.spaces.Box(
            low=-1, high=1, shape=(self.UE_OBS_SIZE,), dtype=np.float32) for ue in self.users.values()})

    def step(self, action: Dict[int, int]):
        """Must be overwritten to allow multiple inheritance of Gym wrappers."""
        action = self.action(action)
        obs, rew, info, done = self.env.step(action)

        return self.observation(), self.reward(), info, done

    def reward(self):
        """Define each UEs' reward as its own utility aggregated with the average utility of nearby stations."""
        # check what BS-UE connections are possible
        connectable = self.available_connections()

        # compute average utility of UEs for each BS; set to lower bound if no UEs are connected
        bs_utilities = self.station_utilities()

        def ue_utility(ue):
            """UE's utility incorporates the UE's own utility and utilities of UEs connected to nearby stations."""
            # utilities are broadcasted, i.e., aggregate utilities of BSs in range
            ngbr_utility = sum(bs_utilities[bs] for bs in connectable[ue])

            # calculate rewards as average weighted by the number of each BSs' connections
            ngbr_counts = sum(len(self.connections[bs])
                              for bs in connectable[ue])

            return (ngbr_utility + self.utilities[ue]) / (ngbr_counts + 1)

        rewards = {ue.ue_id: ue_utility(ue) for ue in self.active}
        return rewards

    def observation(self) -> Dict[int, np.ndarray]:
        """Select features for multi-agent setting anf flatten each UE's features to vector shape."""
        # select observations for multi-agent setting from base feature set
        obs = {ue_id: [obs_dict[key] for key in self.ue_obs_features]
               for ue_id, obs_dict in self.features().items()}

        # flatten each UE's Dict observation to vector representation
        return {ue_id: np.concatenate([o for o in ue_obs]) for ue_id, ue_obs in obs.items()}

    def action(self, action: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        return action

    def reset(self):
        """Overwrites reset() to allow application of observation wrapper."""
        obs = self.env.reset()
        return self.observation()
