import numpy as np

from deepcomp.agent.base import CentralAgent


class RandomAgent(CentralAgent):
    """Agent that always selects a random action. Following the stable_baselines API."""
    def __init__(self, action_space, num_vec_envs=None, seed=None):
        super().__init__()
        self.action_space = action_space
        self.action_space.seed(seed)
        # number of envs inside the VecEnv determines the number of actions to make in each step; or None if no VecEnv
        self.num_vec_envs = num_vec_envs

    def compute_action(self, observation):
        """Choose a random action independent of the observation and other args"""
        # num_vec_envs=None means we don't use a VecEnv --> return action directly (not in array)
        if self.num_vec_envs is None:
            return self.action_space.sample()
        else:
            return [self.action_space.sample() for _ in range(self.num_vec_envs)]


class FixedAgent(CentralAgent):
    """Agent that always selects a the same fixed action. Following the stable_baselines API."""
    def __init__(self, action, noop_interval=0, num_vec_envs=None):
        super().__init__()
        self.action = action
        # number of no op actions (action 0) between repeating actions
        self.noop_interval = noop_interval
        self.noop_counter = noop_interval
        # number of envs inside the VecEnv determines the number of actions to make in each step; or None if no VecEnv
        self.num_vec_envs = num_vec_envs

    def compute_action(self, observation):
        """
        Choose a same fixed action independent of the observation and other args.
        In between the same action, choose no operation (action 0) for the configured interval.
        """
        # no op during the interval
        if self.noop_counter < self.noop_interval:
            action = np.zeros(len(self.action))
            self.noop_counter += 1
        else:
            action = self.action
            self.noop_counter = 0

        # return action
        if self.num_vec_envs is None:
            return action
        else:
            return [action for _ in range(self.num_vec_envs)]
