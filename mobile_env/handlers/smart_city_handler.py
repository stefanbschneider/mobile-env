from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
import logging
import pandas as pd

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
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        total_reward = 0
        penalty = -10

        # List of packets that exceeded the delay constraint or were accomplished
        indices_to_remove = []

        for index, row in env.job_generator.packet_df_ue.iterrows():
            dt = env.time - row['generating_time']

            if dt > row['e2e_delay_constraints']:
                # Packet failed due to exceeding the delay constraint
                total_reward += penalty
                logging.warning(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['user_id']} failed due to delay. Penalty applied.")
                # Mark as accomplished with failure and remove from data frame
                indices_to_remove.append(index)
            elif row['is_accomplished'] and row['accomplished_computing_time'] == env.time:
                # Packet succeeded within the time threshold
                reward = cls.compute_reward(env, row)
                total_reward += reward
                logging.info(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['user_id']} succeeded. Reward applied.")
                indices_to_remove.append(index)

        # Drop processed packets
        if indices_to_remove:
            env.job_generator.packet_df_ue.drop(indices_to_remove, inplace=True)

        return total_reward

    @classmethod
    def compute_reward(cls, env, ue_packet: pd.Series) -> float:
        """Computes the reward based on the delay between the latest accomplished sensor packet and the UE packet."""
        discount_factor = 0.9       # Discount factor for the reward calculation
        base_reward = 10            # Base reward value

        # Step 1: Find the latest accomplished sensor packet
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            env.job_generator.packet_df_sensor['accomplished_computing_time'].notnull()
        ]

        if accomplished_sensor_packets.empty:
            logging.warning("No accomplished sensor packets found.")
            return 0

        latest_sensor_packet = accomplished_sensor_packets.loc[
            accomplished_sensor_packets['accomplished_computing_time'].idxmax()
        ]

        # Step 2: Calculate the delay
        sensor_generating_time = latest_sensor_packet['generating_time']
        ue_generating_time = ue_packet['generating_time']
        delay = abs(ue_generating_time - sensor_generating_time)

        # Step 3: Calculate the reward using the delay
        reward = base_reward * (discount_factor ** delay)

        logging.info(f"Time step: {env.time} Reward computed {reward} with delay: {delay}")

        return reward

    @classmethod
    def compute_reward_with_delay_sign(cls, env, ue_packet: pd.Series) -> float:
        """Computes the reward based on the delay between the latest accomplished sensor packet and the UE packet."""
        positive_discount_factor = 0.9      # Discount factor for positive delay
        negative_discount_factor = 0.8      # Discount factor for negative delay
        base_reward = 10                    # Base reward value

        # Step 1: Find the latest accomplished sensor packet
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            env.job_generator.packet_df_sensor['accomplished_computing_time'].notnull()
        ]

        if accomplished_sensor_packets.empty:
            logging.warning("No accomplished sensor packets found.")
            return 0

        latest_sensor_packet = accomplished_sensor_packets.loc[
            accomplished_sensor_packets['accomplished_computing_time'].idxmax()
        ]

        # Step 2: Calculate the delay
        sensor_generating_time = latest_sensor_packet['generating_time']
        ue_generating_time = ue_packet['generating_time']
        delay = ue_generating_time - sensor_generating_time

        # Step 3: Calculate the reward using different discount factors for positive and negative delay
        if delay > 0:
            # Positive delay: UE packet generated after the sensor packet
            reward = base_reward * (positive_discount_factor ** delay)
        else:
            # Negative delay: UE packet generated before the sensor packet
            reward = base_reward * (negative_discount_factor ** abs(delay))

        logging.info(f"Time step: {env.time} Reward computed {reward} with delay: {delay}")

        return reward   
    
    @classmethod
    def compute_reward_for_positive_delay(cls, env, ue_packet: pd.Series) -> float:
        """Computes the reward based on the delay between the latest accomplished sensor packet and the UE packet,
        considering only positive delays."""
        positive_discount_factor = 0.9      # Discount factor for positive delay
        base_reward = 10                    # Base reward value

        # Step 1: Find sensor packets that have a positive delay relative to the UE packet
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            env.job_generator.packet_df_sensor['accomplished_computing_time'].notnull()
        ]

        if accomplished_sensor_packets.empty:
            logging.warning("No accomplished sensor packets found.")
            return 0

        # Filter sensor packets to only include those with a generating time before the UE packet's generating time
        positive_delay_sensors = accomplished_sensor_packets[
            accomplished_sensor_packets['generating_time'] <= ue_packet['generating_time']
        ]

        if positive_delay_sensors.empty:
            logging.info("No sensor packets with positive delay relative to the UE packet.")
            return 0

        # Step 2: Find the latest accomplished sensor packet among those with a positive delay
        latest_sensor_packet = positive_delay_sensors.loc[
            positive_delay_sensors['accomplished_computing_time'].idxmax()
        ]

        # Step 3: Calculate the delay (since it's guaranteed to be positive or zero)
        sensor_generating_time = latest_sensor_packet['generating_time']
        ue_generating_time = ue_packet['generating_time']
        delay = ue_generating_time - sensor_generating_time

        # Step 4: Calculate the reward using the positive discount factor
        reward = base_reward * (positive_discount_factor ** delay)

        logging.info(f"Time step: {env.time} Reward computed {reward} with delay: {delay}")

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
