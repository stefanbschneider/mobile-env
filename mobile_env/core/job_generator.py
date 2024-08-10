import numpy as np
import pandas as pd
import random
import logging
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor

# Type alias for job data
Job = Dict[str, Optional[Union[float, int]]]

class JobGenerator:
    def __init__(self, env) -> None:
        self.env = env
        self.job_counter: int = 0
        self.packet_df_ue = pd.DataFrame(columns=[
            'user_id', 'device_type', 'packet_id', 'is_accomplished', 'generating_time', 
            'arrival_time', 'accomplished_computing_time', 'e2e_delay_constraints'
        ])
        self.packet_df_sensor = pd.DataFrame(columns=[
            'user_id', 'device_type', 'packet_id', 'is_accomplished', 'generating_time', 
            'arrival_time', 'accomplished_computing_time', 'e2e_delay_constraints'
        ])

    def _generate_index(self) -> int:
        # Generate a unique index for each job.
        self.job_counter += 1
        return self.job_counter

    @staticmethod
    def _generate_communication_request(device_type: str) -> float:
        # Generate data size for communication request based on device type
        if device_type == 'sensor':
            poisson_lambda = 2.0    # Mean size for sensors in MB
        elif device_type == 'user_device':
            poisson_lambda = 15.0   # Mean size for user devices in MB
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")

        # Generate data size using Poisson distribution
        request_size = np.random.poisson(lam=poisson_lambda)
    
        # Ensure a non-zero value for request size
        return max(request_size, 1)  # Return at least 1 MB if generated size is 0
    
    @staticmethod
    def _generate_computational_requirement(device_type: str) -> int:
        # Generate the computational requirement for a device based on its category.
        if device_type == 'sensor':
            computational_power = 10    # Computational requirement in FLOPS for sensors
        elif device_type == 'user_device':
            computational_power = 50    # Computational requirement in FLOPS for user devices
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")
        
        return computational_power

    def _generate_job(self, time: float, device_id: int, device_type: str) -> Job:
        # Generate jobs for the sensors and UEs
        job_index = self._generate_index()
        initial_size = self._generate_communication_request(device_type)
        computational_requirement = self._generate_computational_requirement(device_type)

        # Create a new job record
        job = {
            'index': job_index,
            'serving_bs': None,
            'initial_size': initial_size,
            'remaining_size': initial_size,
            'creation_time': time,
            'serving_time_start': None,
            'serving_time_end': None,
            'serving_time_total': None,
            'device_type': device_type,  # Added type to distinguish between sensor and user device jobs
            'device_id': device_id,
            'processing_time': None,
            'computational_requirement': computational_requirement,
            'target_id': None
        }

        # For reward computation, create a data frame
        packet = {
            'user_id': device_id,
            'device_type': device_type,
            'packet_id': job_index,
            'is_accomplished': False,
            'generating_time': time,
            'arrival_time': None,
            'accomplished_computing_time': None,
            'e2e_delay_constraints': 10
        }

        # Convert job to DataFrame and concatenate with existing DataFrame
        packet_df = pd.DataFrame([packet])
        if device_type == 'sensor':
            self.packet_df_sensor = pd.concat([self.packet_df_sensor, packet_df], ignore_index=True)
        elif device_type == 'user_device':
            self.packet_df_ue = pd.concat([self.packet_df_ue, packet_df], ignore_index=True)
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")

        return job
    
    def generate_job_ue(self, ue: UserEquipment) -> None:
        """Generate jobs for user equipments for device updates."""
        if random.random() < 0.5:  # Example probability for job generation
            job = self._generate_job(self.env.time, ue.ue_id, "user_device")
            if ue.data_buffer_uplink.enqueue_job(job):
                logging.info(f"Time step: {self.env.time} Job generated: {job['index']} at time: {job['creation_time']} by {job['device_type']} {job['device_id']} with initial size of {job['initial_size']} and remaining size of {job['remaining_size']}")

    def generate_job_sensor(self, sensor: Sensor) -> None:
        """Generate jobs for sensors for environmental updates."""
        job = self._generate_job(self.env.time, sensor.sensor_id, "sensor")
        if sensor.data_buffer_uplink.enqueue_job(job):
            logging.info(f"Time step: {self.env.time} Job generated: {job['index']} at time: {job['creation_time']} by {job['device_type']} {job['device_id']} with initial size of {job['initial_size']} and remaining size of {job['remaining_size']}")

    def log_packets_ue(self) -> None:
        """Log the DataFrame at the current time step"""
        logging.info(f"{self.packet_df_ue}")
    
    def log_packets_sensor(self) -> None:
        """Log the DataFrame at the current time step"""
        logging.info(f"{self.packet_df_sensor}")