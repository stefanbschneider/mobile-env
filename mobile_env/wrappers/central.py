from typing import Dict, Tuple

import gym
import numpy as np


class CentralMComWrapper(gym.ActionWrapper, gym.ObservationWrapper, gym.RewardWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

        assert [ue.stime <= 0.0 and ue.extime >= self.EP_MAX_TIME for ue in self.users.values(
        )], 'Central environment cannot handle a changing number of UEs.'

        # define multi-discrete action space for central setting
        # each element of a multi-discrete action denotes the decision for one UE
        self.action_space = gym.spaces.MultiDiscrete(
            [self.NUM_STATIONS + 1 for ue_id in self.users])

        # define what features are observed in the central setting per UE
        self.ue_obs_features = ['connections', 'snrs', 'utility']

        # observation is a single vector of concatenated UE representations
        self.UE_OBS_SIZE = sum(
            self.feature_sizes[ftr] for ftr in self.ue_obs_features)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(self.users) * self.UE_OBS_SIZE,))

    def action(self, actions: Tuple[int]) -> Dict[int, int]:
        """Transform flattend action vector to Dict representation expected for base environment."""
        assert len(actions) == len(
            self.env.users), 'Number of actions must equal overall UEs.'

        users = sorted(self.env.users)
        return {ue_id: action for ue_id, action in zip(users, actions)}

    def observation(self) -> np.ndarray:
        """Select from & flatten observations from MA setting."""
        # select observations considered in the central setting
        obs = {ue_id: [obs_dict[key] for key in self.ue_obs_features]
               for ue_id, obs_dict in self.features().items()}

        # flatten observation to single vector
        return np.concatenate([o for ue_obs in obs.values() for o in ue_obs])

    def reward(self):
        """The central agent receives the average UE utility as reward."""
        utilities = np.asarray(
            [utility for utility in self.env.utilities.values()])
        # assert that rewards are in range [-1, +1]
        bounded = np.logical_and(utilities >= -1, utilities <= 1).all()
        assert bounded, 'Utilities must be in range [-1, +1]'
        
        # return average utility of UEs to central agent as reward
        return np.mean(utilities)

    def step(self, action):
        """Overwrites step() to allow multiple inheritance for Gym wrappers."""
        action = self.action(action)
        obs, rew, info, done = self.env.step(action)

        return self.observation(), self.reward(), info, done

    def reset(self):
        """Overwrites reset() to allow application of observation wrapper."""
        obs = self.env.reset()
        return self.observation()
