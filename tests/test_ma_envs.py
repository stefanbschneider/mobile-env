import ray
import gym
import unittest
from ray.tune.registry import register_env
from ray.rllib.agents import ppo

import mobile_env


class TestMAEnvs(unittest.TestCase):
    def _test_ma_env(self, name):
        name = f"mobile-{name}-ma-v0"

        def register(config):
            import mobile_env
            from mobile_env.wrappers.multi_agent import RLlibMAWrapper
            return RLlibMAWrapper(gym.make(name))

        stop = {
            "training_iteration": 1
        }

        config = {
            # enviroment configuration:
            "env": name,
            "framework": "torch",

            # agent configuration:
            "multiagent": {
                "policies": {"shared_policy"},
                "policy_mapping_fn": (
                    lambda agent_id, **kwargs: "shared_policy"),
            },
        }

        register_env("mobile-small-ma-v0", register)

        ray.init(
            num_cpus=1,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
        )

        analysis = ray.tune.run(ppo.PPOTrainer, config=config, local_dir=None, stop=stop, checkpoint_at_end=False)
        ray.shutdown()

    def test_small_ma(self):
        self._test_ma_env('small')

    def test_medium_ma(self):
        self._test_ma_env('medium')

    def test_large_ma(self):
        self._test_ma_env('large')
