import logging

import numpy as np
from joblib import Parallel, delayed

from deepcomp.util.logs import config_logging
from deepcomp.agent.base import CentralAgent


class BruteForceAgent(CentralAgent):
    """
    Brute force approach, testing all possible actions and choosing the best one.
    Finds the optimal action per step but requires access to the env to test and evaluate each action.
    Optimal in terms of the reward function of the central agent, eg, sum of UE utilities per step.
    """
    def __init__(self, num_workers=1):
        """
        :param num_workers: Number of jobs to run in parallel (should be < num cores).
        Also >1 only makes sense for 3+ UEs and BS, otherwise overhead is higher than gain.
        """
        super().__init__()
        self.num_workers = num_workers
        self.env = None

    @staticmethod
    def number_to_base(n, b, num_digits=None):
        """
        Convert any decimal integer to a new number with any base.
        Adjusted from: https://stackoverflow.com/a/28666223/2745116

        :param n: Decimal integer
        :param b: Base
        :param num_digits: Number of digits to return
        :return: List representing the new number. One list element per digit.
        """
        # special case n=0
        if n == 0:
            if num_digits is None:
                return [0]
            else:
                return [0 for _ in range(num_digits)]

        # actual conversion
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        result = digits[::-1]

        if num_digits is None:
            return result

        # pad with zeros to get the desired number of digits
        assert num_digits >= len(result), "Num digits too small to represent converted number."
        missing_digits = num_digits - len(result)
        result = [0 for _ in range(missing_digits)] + result
        assert len(result) == num_digits
        return result

    def get_ith_action(self, i):
        """Get the i-th action, when walking through the entire action space."""
        # convert to number with base num_bs + 1, ie, actions selecting 0 (=noop) or one of the BS
        action_list = self.number_to_base(i, self.env.num_bs + 1, num_digits=self.env.max_ues)
        assert self.env.action_space.contains(action_list)
        return action_list

    def test_ith_action(self, i):
        """Test the i-th action and return the action and reward"""
        # configure logging each time; necessary for parallel execution with joblib
        config_logging()
        self.env.set_log_level({'deepcomp.util.simulation': logging.DEBUG})

        action_list = self.get_ith_action(i)
        # need to test the action in dict form
        action_dict = self.env.get_ue_actions(action_list)
        rewards = self.env.test_ue_actions(action_dict)
        reward = self.env.step_reward(rewards)
        return action_list, reward

    def compute_action(self, observation):
        """Test all actions and return the best one"""
        assert self.env is not None, "Set agent's env before computing actions."

        # parallelized version
        zipped_results = Parallel(n_jobs=self.num_workers)(
            delayed(self.test_ith_action)(i)
            for i in range((self.env.num_bs + 1)**self.env.max_ues)
        )
        actions, rewards = map(list, zip(*zipped_results))

        # get best action
        best_idx = np.argmax(rewards)
        best_action = actions[best_idx]

        return best_action
