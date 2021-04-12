from typing import Dict

from ray.rllib import Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.utils.typing import PolicyID


class CustomMetricCallbacks(DefaultCallbacks):
    """
    Callbacks for including custom scalar metrics for monitoring with tensorboard
    https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
    """
    @staticmethod
    def get_info(base_env, episode):
        """Return the info dict for the given base_env and episode"""
        # different treatment for MultiAgentEnv where we need to get the info dict from a specific UE
        if hasattr(base_env, 'envs'):
            # get the info dict for the first UE (it's the same for all)
            ue_id = base_env.envs[0].ue_list[0].id
            info = episode.last_info_for(ue_id)
        else:
            info = episode.last_info_for()
        return info

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        info = self.get_info(base_env, episode)
        # add all custom scalar metrics in the info dict
        if info is not None and 'scalar_metrics' in info:
            for metric_name, metric_value in info['scalar_metrics'].items():
                episode.custom_metrics[metric_name] = metric_value

                # increment (or init) the sum over all time steps inside the episode
                eps_metric_name = f'eps_{metric_name}'
                if eps_metric_name in episode.user_data:
                    episode.user_data[eps_metric_name] += metric_value
                else:
                    episode.user_data[eps_metric_name] = metric_value

    @staticmethod
    def on_episode_end(*, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
        # log the sum of scalar metrics over an episode as metric
        for key, value in episode.user_data.items():
            episode.custom_metrics[key] = value
