from typing import Tuple

import gym
import numpy as np


class CentralMComWrapper(gym.ObservationWrapper, gym.ActionWrapper, gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert [ue.stime <= 0.0 and ue.extime >= env.EP_MAX_TIME for ue in env.users.values(
        )], 'Central environment cannot handle a changing number of UEs.'

        # multi-discrete action space where each element denotes the decision for one UE
        self.action_space = gym.spaces.MultiDiscrete(
            [self.NUM_STATIONS + 1 for ue_id in self.users])

        # observations are single vectors of concatenated UE representations
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(env.users) * (2 * env.NUM_STATIONS + 1),))

    def action(self, actions: Tuple):
        """Transform flattend action vector to Dict representation expected for MA setting."""

        return {ue_id: action for ue_id, action in zip(self.env.users, actions)}

    def observation(self, obs) -> np.ndarray:
        """Select from & flatten observations from MA setting."""
        observed = ['connections', 'snrs', 'utility']
        obs = self.env._observation()
        # select observations considered in the central setting from MA setting
        obs = {ue_id: [obs_dict[key] for key in observed]
               for ue_id, obs_dict in obs.items()}
        # flatten observation vector to single vector
        obs = np.concatenate([o for ue_obs in obs.values() for o in ue_obs])

        return obs

    def reward(self, reward):
        """The central agent receives the average UE utility as reward."""
        utilities = np.asarray(
            [utility for utility in self.env.utilities.values()])
        # assert that rewards are in range [-1, +1]
        bounded = np.logical_and(utilities >= -1, utilities <= 1).all()
        assert bounded, 'Utilities must be in range [-1, +1]'

        # return average utility of UEs to central agent as reward
        return np.mean(utilities)

    def step(self, action):
        """Must be overwritten to allow multiple inheritance of Gym wrappers."""
        action = self.action(action)
        obs, rew, info, done = self.env.step(action)

        return self.observation(obs), self.reward(rew), info, done
