"""Tests for MComCore.render(), covering both the 'rgb_array' and 'human' render modes."""

import gymnasium
import numpy as np
import pytest

# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env  # noqa: F401
from mobile_env.scenarios.registry import handlers, scenarios


def test_render_fps_metadata():
    """The pygame clock in human-mode rendering is paced by metadata['render_fps']."""
    env = gymnasium.make("mobile-small-central-v0")
    render_fps = env.unwrapped.metadata["render_fps"]
    assert isinstance(render_fps, int)
    assert render_fps > 0


@pytest.mark.parametrize("scenario", list(scenarios.keys()))
@pytest.mark.parametrize("handler", list(handlers.keys()))
def test_render_rgb_array(scenario: str, handler: str):
    """render() returns a stable-shaped RGB frame at every step, before and after the first one."""
    env_name = f"mobile-{scenario}-{handler}-v0"
    env = gymnasium.make(env_name, render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    # before the first step() (env.unwrapped.time == 0), render_simulation()/render_dashboard()
    # are skipped, but render() still returns a (blank) frame of the right shape
    frame_shape = env.render().shape
    done = False
    for _ in range(5):
        if done:
            break
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = env.render()
        assert frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3
        # consecutive frames of the same episode must have the same size
        assert frame.shape == frame_shape

    env.close()


def test_render_human_mode_headless(monkeypatch):
    """human-mode rendering (incl. the render_fps-paced pygame clock) runs headlessly."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    env = gymnasium.make("mobile-small-central-v0", render_mode="human")
    obs, info = env.reset(seed=0)

    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # must not raise, e.g. from the clock.tick() call added for smoother playback
        env.render()

    env.close()
