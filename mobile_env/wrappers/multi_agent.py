from typing import Tuple

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class RLlibMAWrapper(MultiAgentEnv):
    def __init__(self, env):
        # class wrapps environment object
        self.env = env

        # set number of overall controllable actors
        self.num_agents = len(self.env.users)
        # set max. number of steps for RLlib trainer
        self.max_episode_steps = self.env.EP_MAX_TIME

        # override action and observation space defined for wrapped environment
        # RLlib expects the action and observation space
        # to be defined per actor, i.e, per UE
        self.action_space = gym.spaces.Discrete(self.env.NUM_STATIONS + 1)
        size = self.env.handler.ue_obs_size(self.env)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(size,), dtype=np.float32
        )

        # track UE IDs of last observation's dictionary, i.e.,
        # what UEs were active in the previous step
        self.prev_step_ues = None

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset()
        self.prev_step_ues = set(obs.keys())
        return obs

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rews, done, infos = self.env.step(action_dict)

        # UEs that are not active after `step()` are done
        # NOTE: `dones` keys are keys of previous observation dictionary
        dones = self.prev_step_ues - set([ue.ue_id for ue in self.env.active])
        dones = {
            ue_id: True if ue_id in dones else False
            for ue_id in self.prev_step_ues
        }
        dones["__all__"] = done

        # update keys of previous observation dictionary
        self.prev_step_ues = set(obs.keys())

        return obs, rews, dones, infos


class PettingZooWrapper:
    pass
