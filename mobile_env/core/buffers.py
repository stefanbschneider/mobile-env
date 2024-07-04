from queue import Queue
import numpy as np
import logging
from typing import Dict, Optional

# type for the job data dictionary
Job = Dict[str, Optional[float]]

class Buffer:
    def __init__(self, size=100):
        self.size = size
        self.data_queue: Queue[Job] = Queue(maxsize=size)
    
    def add_job(self, job: Job) -> bool:
        if not self.data_queue.full():
            self.data_queue.put(job)
            logging.info(f"Job index {job['index']} added to the buffer.")
            return True
        else:
            logging.warning("Buffer is full, job dropped!")
            return False

    def remove_job(self) -> Optional[Job]:
        if not self.data_queue.empty():
            job = self.data_queue.get()
            return job
        else:
            logging.warning("Buffer is empty!")
            return None

    def get_job(self) -> Optional[Job]:
        if not self.data_queue.empty():
            return self.data_queue.queue[0]
        else:
            return None


class JobGenerator:
    def __init__(self):
        self.counter: int = 0  # Instance variable to keep track of the number of jobs

    def generate_index(self) -> int:
        self.counter += 1
        return self.counter

    @staticmethod
    def generate_data(device_type: str) -> float:
        # Generate bits for jobs following Poisson distribution based on device type
        if device_type == 'sensor':
            data = np.random.poisson(lam=2)    # 2 MB
            return data if data != 0 else 1    # Ensure non-zero value
        elif device_type == 'user_device':
            data = np.random.poisson(lam=15)   # 15 MB
            return data if data != 0 else 1    # Ensure non-zero value
        else:
            raise ValueError("Unknown device type")

    @staticmethod
    def generate_computational_requirement(device_type: str) -> int:
        # Determine the computational requirement based on device type
        if device_type == 'sensor':
            return 10  # FLOPS
        elif device_type == 'user_device':
            return 50  # FLOPS
        else:
            raise ValueError("Unknown device type")

    def create_job(self, time: float, device_id: int, device_type: str) -> Job:
        # Generate jobs for the sensors and UEs
        index = self.generate_index()
        initial_size = self.generate_data(device_type)
        computational_requirement = self.generate_computational_requirement(device_type)
        job = {
            'index': index,
            'initial_size': initial_size,
            'remaining_size': initial_size,
            'creation_time': time,
            'serving_bs': None,
            'serving_time_start': None,
            'serving_time_end': None,
            'serving_time_total': None,
            'device_type': device_type,  # Added type to distinguish between sensor and user device jobs
            'device_id': device_id,
            'processing_time': None,
            'computational_requirement': computational_requirement,
            'target_id': None
        }
        return job