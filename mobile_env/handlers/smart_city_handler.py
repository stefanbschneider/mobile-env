from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
import pandas as pd

from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["connections", "snrs", "utility"]

    def __init__(self, env):
        self.env = env
        self.logger = env.logger  

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
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        total_reward = 0
        ue_penalty = env.default_config()["reward_calculation"]["ue_penalty"]
        sensor_penalty = env.default_config()["reward_calculation"]["sensor_penalty"]

        # List of packets that exceeded the delay constraint or were accomplished
        indices_to_remove_ue_jobs = []
        indices_to_remove_sensor_jobs = []

        for index, row in env.job_generator.packet_df_ue.iterrows():
            dt = env.time - row['creation_time']

            if dt > row['e2e_delay_threshold']:
                # Packet failed due to exceeding the delay constraint
                total_reward += ue_penalty
                env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['device_id']} failed due to delay. Penalty applied.")
                # Mark as accomplished with failure and remove from data frame
                indices_to_remove_ue_jobs.append(index)
            elif row['is_accomplished'] and row['accomplished_time'] == env.time:
                # Packet succeeded within the time threshold
                reward = cls.compute_reward(cls, env, row)
                total_reward += reward
                env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['device_id']} succeeded. Reward applied.")
                indices_to_remove_ue_jobs.append(index)

        for index, row in env.job_generator.packet_df_sensor.iterrows():
            dt = env.time - row['creation_time']

            if dt > row['e2e_delay_threshold']:
                # Packet failed due to exceeding the delay constraint
                total_reward += sensor_penalty
                env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from Sensor {row['device_id']} failed due to delay. Penalty applied.")
                # Mark as accomplished with failure and remove from data frame
                indices_to_remove_sensor_jobs.append(index)

        # Drop processed UE packets
        if indices_to_remove_ue_jobs:
            env.job_generator.packet_df_ue.drop(indices_to_remove_ue_jobs, inplace=True)

        # Drop processed Sensor packets
        if indices_to_remove_sensor_jobs:
            env.job_generator.packet_df_sensor.drop(indices_to_remove_sensor_jobs, inplace=True)

        return total_reward
    
    def compute_reward(cls, env, ue_packet: pd.Series) -> float:
        """Computes the reward based on the delay between the latest accomplished sensor packet and the UE packet."""

        # Step 1: Find all accomplished sensor packets
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ]

        if accomplished_sensor_packets.empty:
            env.logger.log_reward("No accomplished sensor packets found.")
            return 0

        # Find the highest accomplished_time
        max_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == max_accomplished_time
        ]

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[
            latest_packets['creation_time'].idxmax()
        ]

        # Step 2: Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)

        # Step 3: Calculate the reward using the delay
        reward = env.default_config()["reward_calculation"]["base_reward"] * (env.default_config()["reward_calculation"]["discount_factor"] ** delay)

        env.logger.log_reward(f"Time step: {env.time} Reward computed {reward} with delay: {delay}")

        return reward

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
