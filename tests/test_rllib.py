"""Test that Ray RLlib (multi-agent PPO) trains and predicts on mobile-env.

Mirrors the multi-agent RL flow shown in examples/rllib.ipynb, at a much smaller training
budget: we only assert that training, checkpointing, reloading, and inference run without
errors, not that the policy converges.

Uses RLlib's "new API stack" (the default since Ray ~2.4x for PPO). Inference no longer goes
through `Algorithm.compute_single_action()` (removed on the new stack); instead, the trained
RLModule is fetched via `Algorithm.get_module()` and queried directly with `forward_inference()`.
"""

import gymnasium
import numpy as np
import pytest
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env

import mobile_env  # noqa: F401
from mobile_env.wrappers.multi_agent import RLlibMAWrapper

ENV_NAME = "mobile-small-ma-v0"


def _register(config):
    env = gymnasium.make(ENV_NAME)
    return RLlibMAWrapper(env)


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    register_env(ENV_NAME, _register)
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


def test_rllib_ppo_train_checkpoint_and_predict(tmp_path):
    config = (
        PPOConfig()
        .environment(env=ENV_NAME)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=0)
    )

    algo = config.build_algo()
    # a single, small training iteration: just enough to exercise the training loop
    result = algo.train()
    assert result["num_env_steps_sampled_lifetime"] > 0

    checkpoint = algo.save(str(tmp_path / "checkpoint"))
    algo.stop()

    # reload the trained policy from the checkpoint, like the notebook does
    checkpoint_path = getattr(checkpoint, "checkpoint", checkpoint)
    reloaded = Algorithm.from_checkpoint(checkpoint_path)
    module = reloaded.get_module("shared_policy")
    action_dist_cls = module.get_inference_action_dist_cls()

    test_env = RLlibMAWrapper(gymnasium.make(ENV_NAME))
    obs, info = test_env.reset()
    for agent_id, agent_obs in obs.items():
        input_dict = {Columns.OBS: torch.from_numpy(np.array([agent_obs], dtype=np.float32))}
        action_dist_inputs = module.forward_inference(input_dict)[Columns.ACTION_DIST_INPUTS]
        action = int(action_dist_cls.from_logits(action_dist_inputs).sample()[0].numpy())
        assert test_env.action_spaces[agent_id].contains(action)
    reloaded.stop()
