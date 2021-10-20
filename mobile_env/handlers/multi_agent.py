from typing import Dict

import gym
import numpy as np
from gym import spaces


class MComMAHandler:
    ftrs = ['connections', 'snrs', 'utility', 'bcast', 'stations_connected']

    @classmethod
    def ue_obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.ftrs)

    @classmethod
    def action_space(cls, env) -> spaces.Dict:
        return spaces.Dict({ue.ue_id: gym.spaces.Discrete(env.NUM_STATIONS + 1) for ue in env.users.values()})

    @classmethod
    def observation_space(cls, env) -> spaces.Dict:
        space = {ue_id: spaces.Box(
            low=-1, high=1, shape=(cls.ue_obs_size,), dtype=np.float32) for ue_id in env.users}
        # print("MCOM-MA-HANDLER: ", space.keys())
        
        return spaces.Dict(space)

    @classmethod
    def reward(cls, env):
        """Define each UEs' reward as its own utility aggregated with the average utility of nearby stations."""
        # check what BS-UE connections are possible
        connectable = env.available_connections()

        # compute average utility of UEs for each BS; set to lower bound if no UEs are connected
        bs_utilities = env.station_utilities()

        def ue_utility(ue):
            """UE's utility incorporates the UE's own utility and utilities of UEs connected to nearby stations."""
            # utilities are broadcasted, i.e., aggregate utilities of BSs in range
            ngbr_utility = sum(bs_utilities[bs] for bs in connectable[ue])

            # calculate rewards as average weighted by the number of each BSs' connections
            ngbr_counts = sum(len(env.connections[bs])
                              for bs in connectable[ue])

            return (ngbr_utility + env.utilities[ue]) / (ngbr_counts + 1)

        rewards = {ue.ue_id: ue_utility(ue) for ue in env.active}
        return rewards

    @classmethod
    def observation(cls, env) -> Dict[int, np.ndarray]:
        """Select features for multi-agent setting anf flatten each UE's features to vector shape."""

        # get features for currently active UEs
        active = set([ue.ue_id for ue in env.active if not env.done])
        features = {ue_id: obs for ue_id, obs in env.features().items() if ue_id in active}

        # select observations for multi-agent setting from base feature set
        obs = {ue_id: [obs_dict[key] for key in cls.ftrs] for ue_id, obs_dict in features.items()}

        # flatten each UE's Dict observation to vector representation
        obs = {ue_id: np.concatenate([o for o in ue_obs])
               for ue_id, ue_obs in obs.items()}
        return obs

    @classmethod
    def action(cls, env, action: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        return action

    @classmethod
    def info(cls, env):
        return {}

    @classmethod
    def check(cls, env):
        pass

