"""Simple test of small env similar to test notebook."""

import gymnasium
import pytest

# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env  # noqa: F401
from mobile_env.scenarios.registry import handlers, scenarios


@pytest.mark.parametrize("scenario", list(scenarios.keys()))
@pytest.mark.parametrize("handler", list(handlers.keys()))
def test_env_stepping(scenario: str, handler: str):
    """Create a mobile-env and run with random actions until done.

    Just to ensure that it does not crash.
    """
    env_name: str = f"mobile-{scenario}-{handler}-v0"
    # create a small mobile environment for a single, centralized control agent
    env = gymnasium.make(env_name)
    obs, info = env.reset()
    done = False

    while not done:
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
