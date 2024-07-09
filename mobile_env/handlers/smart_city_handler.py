from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["connections", "snrs", "utility"]

    @classmethod
    def ue_obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.features)

    @classmethod
    def action_space(cls, env) -> spaces.Box:
        # Define continuous action space for bandwidth and computational power allocation
        return spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @classmethod
    def observation_space(cls, env) -> spaces.Box:
        # Define observation space
        size = cls.ue_obs_size(env)
        return spaces.Box(low=-1.0, high=1.0, shape=(env.NUM_USERS * size,))

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Dict[str, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth and computational power allocations."

        # Clip actions to ensure they are within the valid range
        bandwidth_allocation = np.clip(actions[0], 0.0, 1.0)
        computational_allocation = np.clip(actions[1], 0.0, 1.0)

        return {"bandwidth_allocation": bandwidth_allocation, "computational_allocation": computational_allocation}

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute observations for agent."""
        obs = {
            ue_id: [obs_dict[key] for key in cls.features]
            for ue_id, obs_dict in env.features().items()
        }
        return np.concatenate([o for ue_obs in obs.values() for o in ue_obs])

    @classmethod
    def reward(cls, env) -> float:
        """The central agent receives the average UE utility as reward."""
        utilities = np.asarray([utility for utility in env.utilities.values()])
        bounded = np.logical_and(utilities >= -1, utilities <= 1).all()
        assert bounded, "Utilities must be in range [-1, +1]"
        return np.mean(utilities)

    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        assert all(
            ue.stime <= 0.0 and ue.extime >= env.EP_MAX_TIME
            for ue in env.users.values()
        ), "Central environment cannot handle a changing number of UEs."

    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        return {}
