"""Simple test of small env similar to test notebook."""
import gymnasium
import pytest

# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env  # noqa: F401


@pytest.mark.parametrize(
    "env_name",
    ["mobile-small-central-v0", "mobile-medium-central-v0", "mobile-large-central-v0"],
)
def test_env_stepping(env_name):
    """Create a mobile-env and run with random actions until done.

    Just to ensure that it does not crash.
    """
    # create a small mobile environment for a single, centralized control agent
    env = gymnasium.make(env_name)
    obs, info = env.reset()
    done = False

    while not done:
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
