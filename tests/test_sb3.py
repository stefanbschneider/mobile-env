"""Test that stable-baselines3 (single-agent PPO) trains and predicts on mobile-env.

Mirrors the single-agent RL flow shown in examples/demo.ipynb, at a much smaller training
budget: we only assert that training and inference run without errors, not that the policy
converges. stable-baselines3 (and torch) are only installed on Python >= 3.10, see
tests/requirements.txt, so this whole module is skipped otherwise.
"""

import gymnasium
import pytest

import mobile_env  # noqa: F401

stable_baselines3 = pytest.importorskip("stable_baselines3")


def test_sb3_ppo_train_and_predict():
    env = gymnasium.make("mobile-small-central-v0")

    model = stable_baselines3.PPO("MlpPolicy", env, verbose=0)
    # small budget: just enough to exercise the training loop, not to converge
    model.learn(total_timesteps=256)

    obs, info = env.reset()
    action, _ = model.predict(obs)
    assert env.action_space.contains(action)

    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
