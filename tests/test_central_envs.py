import gym
import unittest

from stable_baselines3.common.env_checker import check_env

import mobile_env


class TestCentralEnvs(unittest.TestCase):
    def test_central_small(self):
        check_env(gym.make('mobile-small-central-v0'))

    def test_central_medium(self):
        check_env(gym.make('mobile-medium-central-v0'))

    def test_central_large(self):
        check_env(gym.make('mobile-large-central-v0'))


if __name__ == '__main__':
    unittest.main()
