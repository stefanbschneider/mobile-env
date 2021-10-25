from typing import Dict, Tuple

import gym
import numpy as np
from gym import spaces


class MComCentralHandler:
    ftrs = ["connections", "snrs", "utility"]

    @classmethod
    def ue_obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.ftrs)

    @classmethod
    def action_space(cls, env) -> spaces.Dict:
        # define multi-discrete action space for central setting
        # each element of a multi-discrete action denotes one UE's decision
        return spaces.MultiDiscrete([env.NUM_STATIONS + 1 for _ in env.users])

    @classmethod
    def observation_space(cls, env) -> spaces.Dict:
        # observation is a single vector of concatenated UE representations
        size = cls.ue_obs_size(env)
        return gym.spaces.Box(low=-1, high=1, shape=(env.NUM_USERS * size,))

    @classmethod
    def action(cls, env, actions: Tuple[int]) -> Dict[int, int]:
        """Transform flattend actions to expected shape of core environment."""
        assert len(actions) == len(
            env.users
        ), "Number of actions must equal overall UEs."

        users = sorted(env.users)
        return {ue_id: action for ue_id, action in zip(users, actions)}

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Select from & flatten observations from MA setting."""
        # select observations considered in the central setting
        obs = {
            ue_id: [obs_dict[key] for key in cls.ftrs]
            for ue_id, obs_dict in env.features().items()
        }

        # flatten observation to single vector
        return np.concatenate([o for ue_obs in obs.values() for o in ue_obs])

    @classmethod
    def reward(cls, env):
        """The central agent receives the average UE utility as reward."""
        utilities = np.asarray([utility for utility in env.utilities.values()])
        # assert that rewards are in range [-1, +1]
        bounded = np.logical_and(utilities >= -1, utilities <= 1).all()
        assert bounded, "Utilities must be in range [-1, +1]"

        # return average utility of UEs to central agent as reward
        return np.mean(utilities)

    @classmethod
    def check(cls, env):
        assert [
            ue.stime <= 0.0 and ue.extime >= env.EP_MAX_TIME
            for ue in env.users.values()
        ], "Central environment cannot handle a changing number of UEs."

    @classmethod
    def info(cls, env):
        return {}
