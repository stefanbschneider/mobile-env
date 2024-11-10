from typing import Optional, Tuple

import gymnasium
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from mobile_env.core.base import MComCore


class RLlibMAWrapper(MultiAgentEnv):
    def __init__(self, env: MComCore):
        super().__init__()

        # class wrapps environment object
        self.env = env

        # set max. number of steps for RLlib trainer
        self.max_episode_steps = self.env.EP_MAX_TIME

        # override action and observation space defined for wrapped environment
        # RLlib expects the action and observation space
        # to be defined per actor, i.e, per UE
        self.action_space = gymnasium.spaces.Discrete(self.env.NUM_STATIONS + 1)
        size = self.env.handler.ue_obs_size(self.env)
        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(size,), dtype=np.float32
        )

        # track UE IDs of last observation's dictionary, i.e.,
        # what UEs were active in the previous step
        self.prev_step_ues: Optional[set[int]] = None

    def reset(self, *, seed=None, options=None) -> MultiAgentDict:
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_step_ues = set(obs.keys())
        return obs, info

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rews, terminated, truncated, infos = self.env.step(action_dict)

        # UEs that are not active after `step()` are done (here: truncated)
        # NOTE: `truncateds` keys are keys of previous observation dictionary
        assert self.prev_step_ues is not None
        inactive_ues = self.prev_step_ues - set([ue.ue_id for ue in self.env.active])
        truncateds: MultiAgentDict = {
            ue_id: True if ue_id in inactive_ues else False
            for ue_id in self.prev_step_ues
        }
        truncateds["__all__"] = truncated
        # Terminated is always False since there is no particular terminal end state.
        assert (
            not terminated
        ), "There is no natural episode termination. terminated should be False."
        terminateds: MultiAgentDict = {ue_id: False for ue_id in self.prev_step_ues}
        terminateds["__all__"] = False

        # update keys of previous observation dictionary
        self.prev_step_ues = set(obs.keys())

        # RLlib expects the keys of infos to be a subset of obs + __common__
        # Put all infos under __common__
        infos = {"__common__": infos}

        return obs, rews, terminateds, truncateds, infos

    def render(self) -> None:
        return self.env.render()


class PettingZooWrapper:
    pass
