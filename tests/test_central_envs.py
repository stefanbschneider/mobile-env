import unittest

import gymnasium
import pytest

import mobile_env  # noqa: F401

# stable-baselines3 (and its dependency torch) is only installed for Python >= 3.10,
# see tests/requirements.txt
check_env = pytest.importorskip("stable_baselines3.common.env_checker").check_env


class TestCentralEnvs(unittest.TestCase):
    def test_central_small(self):
        check_env(gymnasium.make("mobile-small-central-v0"))

    def test_central_medium(self):
        check_env(gymnasium.make("mobile-medium-central-v0"))

    def test_central_large(self):
        check_env(gymnasium.make("mobile-large-central-v0"))


if __name__ == "__main__":
    unittest.main()
