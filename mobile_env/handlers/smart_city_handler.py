from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
import pandas as pd

from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["queue_lengths", "resource_utilization"]

    def __init__(self, env):
        self.env = env
        self.logger = env.logger  

    @classmethod
    def obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.features)

    @classmethod
    def action_space(cls, env) -> spaces.Box:
        """Define continuous action space for bandwidth allocation and computational power allocation"""
        return spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @classmethod
    def observation_space(cls, env) -> spaces.Box:
        """Define observation space"""
        size = cls.obs_size(env)
        env.logger.log_reward(f"Observation size is: {size}")        
        return spaces.Box(low=-1.0, high=1.0, shape=(size,), dtype=np.float32)

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
        """Compute system-wide observations for the RL agent."""
        
        # Gather the queue lengths (from base station)
        queue_lengths = np.array(env.get_queue_lengths()).ravel()
        env.logger.log_reward(f"Queue lengths: {queue_lengths}")      

        # Get resource utilization (bandwidth and CPU)
        resource_utilization = env.get_resource_utilization()
        env.logger.log_reward(f"Resource utilization: {resource_utilization}")   

        # Get the frequency of requests or updates
        #request_frequency = env.get_request_frequency()   
        #env.logger.log_reward(f"Request frequency: {request_frequency}")

        # Concatenate all observations into a single array
        observation = np.concatenate([
            queue_lengths,              # 4 values
            resource_utilization       # 2 values
        ])
        
        return observation

    @classmethod
    def reward(cls, env) -> float:
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        total_reward = 0
        config = env.default_config()["reward_calculation"]
        penalty = config["ue_penalty"]
        base_reward = config["base_reward"]
        discount_factor = config["discount_factor"]

        # List of packets that exceeded the delay constraint or were accomplished
        indices_to_remove_ue_jobs = []

        for index, row in env.job_generator.packet_df_ue.iterrows():
            # Check if packet is accomplished at this timestep
            if row['is_accomplished'] and row['accomplished_time'] == env.time:
                # Packet is accomplished in this time step
                dt = env.time - row['creation_time']
                
                if dt > row['e2e_delay_threshold']:
                    # Packet exceeds threshold, penalty applied
                    total_reward += penalty
                    env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['device_id']} failed due to delay. Penalty applied: {penalty}.")
                else:
                    # Packet succeeded within the time threshold, compute delay and reward
                    delay = cls.compute_delay(cls, env, row)
                    reward = base_reward * (discount_factor ** delay)        
                    total_reward += reward
                    env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['device_id']} succeeded within time threshold. Reward applied: {reward}.")

                # Mark as processed and remove from the data frame
                indices_to_remove_ue_jobs.append(index)

        # Drop processed UE packets
        if indices_to_remove_ue_jobs:
            env.job_generator.packet_df_ue.drop(indices_to_remove_ue_jobs, inplace=True)

        return total_reward
    
    def compute_delay(cls, env, ue_packet: pd.Series) -> float:
        """Computes the delay between the latest accomplished sensor packet and the UE packet."""

        # Find all accomplished sensor packets
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ]

        if accomplished_sensor_packets.empty:
            env.logger.log_reward("No accomplished sensor packets found.")
            return 0

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ]

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]

        # Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)

        return delay

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
