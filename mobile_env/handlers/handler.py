import abc
from typing import Dict

from gym.spaces.space import Space


class Handler:
    """Defines Gym interface methods called by core simulation."""

    @classmethod
    @abc.abstractmethod
    def action_space(cls, env) -> Space:
        """Defines action space for passed environment."""
        pass

    @classmethod
    @abc.abstractmethod
    def observation_space(cls, env) -> Space:
        """Defines observation space for passed environment."""
        pass

    @classmethod
    @abc.abstractmethod
    def action(cls, env, action) -> Dict[int, int]:
        """Transform passed action(s) to dict shape expected by simulation."""
        pass

    @classmethod
    @abc.abstractmethod
    def observation(cls, env):
        """Computes observations for agent."""
        pass

    @classmethod
    @abc.abstractmethod
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
