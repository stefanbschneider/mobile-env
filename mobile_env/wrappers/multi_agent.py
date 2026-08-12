from typing import Optional, Tuple

import gymnasium
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from mobile_env.core.base import MComCore


class RLlibMAWrapper(MultiAgentEnv):
    def __init__(self, env: gymnasium.Env):
        # Keep a reference to the mobile-env base environment, which is wrapped by this class.
        # Remove any gymnasium wrappers first if needed.
        if isinstance(env, MComCore):
            self.env: MComCore = env
        else:
            assert isinstance(env.unwrapped, MComCore), "The unwrapped env should be a mobile-env."
            self.env = env.unwrapped

        # set max. number of steps for RLlib trainer
        self.max_episode_steps = self.env.EP_MAX_TIME

        # RLlib's MultiAgentEnv expects per-agent (i.e., per-UE) action/observation spaces,
        # keyed by agent ID. `MComMAHandler` already exposes exactly that (a `gymnasium.spaces.Dict`
        # keyed by `ue_id`), so reuse it as-is instead of re-deriving per-UE spaces here.
        self.action_spaces = dict(self.env.action_space.spaces)
        self.observation_spaces = dict(self.env.observation_space.spaces)

        # all UEs that may ever appear in the environment (fixed for the lifetime of the env,
        # even though not all of them are necessarily active at any given time)
        self.possible_agents = list(self.env.users.keys())
        self.agents = self.possible_agents.copy()

        super().__init__()

        # track UE IDs of last observation's dictionary, i.e.,
        # what UEs were active in the previous step
        self.prev_step_ues: Optional[set[int]] = None

    def reset(self, *, seed=None, options=None) -> MultiAgentDict:
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_step_ues = set(obs.keys())
        self.agents = list(obs.keys())
        return obs, info

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rews, terminated, truncated, infos = self.env.step(action_dict)

        # UEs that are not active after `step()` are done (here: truncated). When the whole
        # episode ends (`truncated`), every UE that was active going into this step is
        # considered done too, even if `self.env.active` (still) lists it -- there is no
        # further step for it to act in.
        # NOTE: `truncateds` keys are keys of previous observation dictionary
        assert self.prev_step_ues is not None
        active_ue_ids = set(ue.ue_id for ue in self.env.active)
        inactive_ues = set(self.prev_step_ues) if truncated else self.prev_step_ues - active_ue_ids
        truncateds: MultiAgentDict = {
            ue_id: True if ue_id in inactive_ues else False for ue_id in self.prev_step_ues
        }
        truncateds["__all__"] = truncated
        # Terminated is always False since there is no particular terminal end state.
        assert not terminated, (
            "There is no natural episode termination. terminated should be False."
        )
        terminateds: MultiAgentDict = {ue_id: False for ue_id in self.prev_step_ues}
        terminateds["__all__"] = False

        # RLlib requires a final ("truncation") observation and reward for any UE that
        # acted this step and is now truncated (e.g., for value-function bootstrapping).
        # `MComMAHandler.observation()`/`reward()` only report values for UEs that are
        # still active *and* the episode isn't over, so both UEs departing this step and
        # (on the last step) every other acting UE need synthetic final values here.
        for ue_id in set(action_dict) - set(obs.keys()):
            obs[ue_id] = np.zeros(
                self.observation_spaces[ue_id].shape,
                dtype=self.observation_spaces[ue_id].dtype,
            )
            rews.setdefault(ue_id, 0.0)

        # update the set of UEs considered active as of this step
        self.prev_step_ues = active_ue_ids
        self.agents = list(active_ue_ids)

        # RLlib expects the keys of infos to be a subset of obs + __common__
        # Put all infos under __common__
        infos = {"__common__": infos}

        return obs, rews, terminateds, truncateds, infos

    def render(self) -> None:
        return self.env.render()


class PettingZooWrapper:
    pass
