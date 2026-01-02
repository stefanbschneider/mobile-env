from typing import Optional, Tuple

import gymnasium
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from mobile_env.core.base import MComCore


class RLlibMAWrapper(MultiAgentEnv):
    def __init__(self, env: gymnasium.Env):
        # Store env reference early so it's available for possible_agents property
        # Keep a reference to the mobile-env base environment, which is wrapped by this class.
        # Remove any gymnasium wrappers first if needed.
        if isinstance(env, MComCore):
            self.env: MComCore = env
        else:
            assert isinstance(env.unwrapped, MComCore), "The unwrapped env should be a mobile-env."
            self.env = env.unwrapped
        
        # Track possible agents (will be populated after first reset)
        # Initialize this BEFORE calling super().__init__() because it may be accessed
        self._possible_agents = None
        
        super().__init__()

        # set max. number of steps for RLlib trainer
        self.max_episode_steps = self.env.EP_MAX_TIME

        # Define action and observation space per agent for the new API stack
        # RLlib expects the action and observation space to be defined per actor, i.e, per UE
        self._agent_action_space = gymnasium.spaces.Discrete(self.env.NUM_STATIONS + 1)
        size = self.env.handler.ue_obs_size(self.env)
        self._agent_observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(size,), dtype=np.float32
        )

        # For the new API stack, set observation_spaces and action_spaces as None
        # This forces Ray to use get_observation_space and get_action_space methods
        self.observation_spaces = None
        self.action_spaces = None
        
        # Set observation_space and action_space to None to avoid iteration errors
        self.observation_space = None
        self.action_space = None

        # track UE IDs of last observation's dictionary, i.e.,
        # what UEs were active in the previous step
        self.prev_step_ues: Optional[set[int]] = None

    def reset(self, *, seed=None, options=None) -> MultiAgentDict:
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_step_ues = set(obs.keys())
        # Update possible agents after reset
        self._possible_agents = list(obs.keys())
        return obs, info
    
    @property
    def possible_agents(self):
        """Return the list of possible agent IDs (required for new API stack)."""
        if self._possible_agents is None:
            # Return all user IDs from the environment
            return list(self.env.users.keys())
        return self._possible_agents

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

    def get_observation_space(self, agent_id):
        """Return the observation space for a given agent (required for new API stack)."""
        return self._agent_observation_space

    def get_action_space(self, agent_id):
        """Return the action space for a given agent (required for new API stack)."""
        return self._agent_action_space

    def render(self) -> None:
        return self.env.render()


class PettingZooWrapper:
    pass
