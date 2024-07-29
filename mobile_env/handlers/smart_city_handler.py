from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
import logging

from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["connections", "snrs", "utility"]

    @classmethod
    def ue_obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.features)

    @classmethod
    def action_space(cls, env) -> spaces.Box:
        """Define continuous action space for bandwidth allocation and computational power allocation"""
        return spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @classmethod
    def observation_space(cls, env) -> spaces.Box:
        """Define observation space"""
        size = cls.ue_obs_size(env)
        return spaces.Box(low=-1.0, high=1.0, shape=(env.NUM_USERS * size,))

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Tuple[float, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth allocation and computational power allocation."

        bandwidth_allocation, computational_allocation = actions

        # Ensure actions are within valid range
        bandwidth_allocation = max(0.0, min(1.0, bandwidth_allocation))
        computational_allocation = max(0.0, min(1.0, computational_allocation))

        return bandwidth_allocation, computational_allocation

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute observations for agent."""
        obs = {
            ue_id: [obs_dict[key] for key in cls.features]
            for ue_id, obs_dict in env.features().items()
        }
        return np.concatenate([o for ue_obs in obs.values() for o in ue_obs])

    @classmethod
    def reward(cls, env) -> float:
        """Computes rewards for agent."""

        total_reward = 0
        R = 1
        beta = 0.7

         # Find the UE packets accomplished at the current time step
        accomplished_ue_packets = env.job_generator.packet_df_ue[env.job_generator.packet_df_ue['accomplished_computing_time'] == env.time]

        if accomplished_ue_packets.empty:
            # No UE packets accomplished at this time step
            return 0

        # Iterate over all accomplished UE packets
        for _, ue_packet in accomplished_ue_packets.iterrows():
            ue_generating_time = ue_packet['generating_time']
 
            # Find the sensor packet with the most recent accomplished computing time
            latest_sensor_packet = env.job_generator.packet_df_sensor.loc[
                env.job_generator.packet_df_sensor['accomplished_computing_time'].idxmax()
            ]
                
            sensor_generating_time = latest_sensor_packet['generating_time']

            # Compute the delay
            delay = ue_generating_time - sensor_generating_time
            logging.info(f"The delay is: {delay}")

            # Compute the reward based on the delay
            # For simplicity, let's assume a linear reward function: reward = -delay
            reward = R * (beta ** delay)

            # Add the reward for this packet to the total reward
            total_reward += reward

        return total_reward
    
    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        assert all(
            ue.stime <= 0.0 and ue.extime >= env.EP_MAX_TIME
            for ue in env.users.values()
        ), "Central environment cannot handle a changing number of UEs."

    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        return {}
