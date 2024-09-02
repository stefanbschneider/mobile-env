import numpy as np
import pandas as pd
import random
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor

# Type alias for job data
Job = Dict[str, Optional[Union[float, int]]]

class JobGenerator:
    def __init__(self, env) -> None:
        self.env = env
        self.job_counter: int = 0
        self.logger = env.logger
        self.config = env.default_config()
        self.packet_df_ue = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 'is_accomplished', 'creation_time',
            'arrival_time', 'accomplished_time', 'e2e_delay_threshold'
        ])
        self.packet_df_sensor = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 'is_accomplished', 'creation_time',
            'arrival_time', 'accomplished_time', 'e2e_delay_threshold'
        ])

    def _generate_index(self) -> int:
        # Generate a unique index for each job.
        self.job_counter += 1
        return self.job_counter

    @staticmethod
    def _generate_communication_request(self, device_type: str) -> float:
        # Generate data size for communication request based on device type
        if device_type == 'sensor':
            poisson_lambda = self.config["sensor_job"]["communication_job_lambda_value"]
        elif device_type == 'user_device':
            poisson_lambda = self.config["ue_job"]["communication_job_lambda_value"]  
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")

        # Generate data size using Poisson distribution
        request_size = np.random.poisson(lam=poisson_lambda)
    
        # Ensure a non-zero value for request size
        return max(request_size, 1.0)     # Return at least 1 MB if generated size is 0
    
    @staticmethod
    def _generate_computation_request(self, device_type: str) -> int:
        # Generate the computational requirement for a device based on its category.
        if device_type == 'sensor':
            poisson_lambda = self.config["sensor_job"]["computation_job_lambda_value"]
        elif device_type == 'user_device':
            poisson_lambda = self.config["ue_job"]["computation_job_lambda_value"]
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")
        
        # Generate computation request using Poisson distribution
        computation_request = np.random.poisson(lam=poisson_lambda)
    
        return max(computation_request, 10.0)

    def _generate_job(self, time: float, device_id: int, device_type: str) -> Job:
        # Generate jobs for devices
        job_index = self._generate_index()
        communication_request_size = self._generate_communication_request(self, device_type)
        computation_request = self._generate_computation_request(self, device_type)

        # Create a new job record
        job = {
            'packet_id': job_index,
            'device_type': device_type,
            'device_id': device_id,
            'serving_bs': None,
            'creation_time': time,
            'computation_request': computation_request,
            'initial_request_size': communication_request_size,
            'remaining_request_size': communication_request_size,
            'serving_time_start': None,
            'serving_time_end': None,
            'serving_time_total': None,
            'processing_time': None,
        }

        # For reward computation, create a data frame
        packet = {
            'packet_id': job_index,
            'device_type': device_type,
            'device_id': device_id,
            'is_accomplished': False,
            'creation_time': time,
            'arrival_time': None,
            'accomplished_time': None,
            'e2e_delay_threshold': self.config["e2e_delay_threshold"]
        }

        # Convert job to data frame and concatenate with existing data frame
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
        if random.random() < self.config["ue_job"]["job_generation_probability"]:
            job = self._generate_job(self.env.time, ue.ue_id, "user_device")
            if ue.data_buffer_uplink.enqueue_job(job):
                self.logger.log_simulation(
                    f"Time step: {self.env.time} Job generated: {job['packet_id']} by {job['device_type']} {job['device_id']} " 
                    f"with initial size of {job['initial_request_size']} MB and computational request of {job['computation_request']} units"
                )

    def generate_job_sensor(self, sensor: Sensor) -> None:
        """Generate jobs for sensors for environmental updates."""
        job = self._generate_job(self.env.time, sensor.sensor_id, "sensor")
        if sensor.data_buffer_uplink.enqueue_job(job):
            self.logger.log_simulation(
                f"Time step: {self.env.time} Job generated: {job['packet_id']} by {job['device_type']} {job['device_id']} "
                f"with initial size of {job['initial_request_size']} MB and computational request of {job['computation_request']} units"
            )

    def log_df_ue(self) -> None:
        """Log the data frame of UEs at the current time step."""
        self.logger.log_reward(f"{self.packet_df_ue}")
    
    def log_df_sensor(self) -> None:
        """Log the data frame of sensors at the current time step."""
        self.logger.log_reward(f"{self.packet_df_sensor}")