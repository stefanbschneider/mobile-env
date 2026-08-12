import unittest

import gymnasium
from stable_baselines3.common.env_checker import check_env

import mobile_env  # noqa: F401


class TestCentralEnvs(unittest.TestCase):
    def test_central_small(self):
        check_env(gymnasium.make("mobile-small-central-v0"))

    def test_central_medium(self):
        check_env(gymnasium.make("mobile-medium-central-v0"))

    def test_central_large(self):
        check_env(gymnasium.make("mobile-large-central-v0"))


if __name__ == "__main__":
    unittest.main()
