from typing import Dict
from abc import abstractclassmethod

from gym.spaces.space import Space


class Handler:
    """Defines Gym interface methods called by core simulation."""

    @abstractclassmethod
    def action_space(cls, env) -> Space:
        """Defines action space for passed environment."""
        pass

    @abstractclassmethod
    def observation_space(cls, env) -> Space:
        """Defines observation space for passed environment."""
        pass

    @abstractclassmethod
    def action(cls, env, action) -> Dict[int, int]:
        """Transform passed action(s) to dict shape expected by simulation."""
        pass

    @abstractclassmethod
    def observation(cls, env):
        """Computes observations for agent."""
        pass

    @abstractclassmethod
    def reward(cls, env):
        """Computes rewards for agent."""
        pass

    @classmethod
    def check(cls, env):
        """Check if handler is applicable to simulation configuration."""
        pass

    @classmethod
    def info(cls, env):
        """Compute information for feedback loop."""
        return {}
