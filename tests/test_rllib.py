"""Test that Ray RLlib (multi-agent PPO) trains and predicts on mobile-env.

Mirrors the multi-agent RL flow shown in examples/rllib.ipynb, at a much smaller training
budget: we only assert that training, checkpointing, reloading, and inference run without
errors, not that the policy converges.

NOTE: Uses RLlib's "old API stack" (`enable_rl_module_and_learner=False,
enable_env_runner_and_connector_v2=False`) deliberately. mobile-env's `RLlibMAWrapper` targets
that interface; the new API stack requires a different multi-agent env interface and, as of
Ray 2.57, breaks single-action inference for multi-agent policies (see tests/requirements.txt).
"""

import gymnasium
import pytest
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
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
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=ENV_NAME)
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=0)
    )

    algo = config.build()
    # a single, small training iteration: just enough to exercise the training loop
    result = algo.train()
    assert result["num_env_steps_sampled_lifetime"] > 0

    checkpoint = algo.save(str(tmp_path / "checkpoint"))
    algo.stop()

    # reload the trained policy from the checkpoint, like the notebook does
    checkpoint_path = getattr(checkpoint, "checkpoint", checkpoint)
    reloaded = Algorithm.from_checkpoint(checkpoint_path)

    test_env = RLlibMAWrapper(gymnasium.make(ENV_NAME))
    obs, info = test_env.reset()
    for agent_id, agent_obs in obs.items():
        action = reloaded.compute_single_action(agent_obs, policy_id="shared_policy")
        # RLlibMAWrapper defines a single per-agent action space (Discrete), shared by all UEs
        assert test_env.action_space.contains(action)
    reloaded.stop()
