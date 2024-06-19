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
    counter: int = 0 # Class variable to keep track of the number of jobs

    @classmethod
    def generate_index(cls) -> int:
        cls.counter += 1
        return cls.counter
    
    @staticmethod
    def generate_data(device_type: str) -> float:
        # Generate bits for jobs following Poisson distribution based on device type
        if device_type == 'sensor':
            return np.random.poisson(lam=5)    # MB per second
        elif device_type == 'user_device':
            return np.random.poisson(lam=15)   # MB per second
        else:
            raise ValueError("Unknown device type")

    @classmethod
    def create_job(cls, time: float, device_type: str) -> Job:
        # Generate jobs for the sensors and ues
        index = cls.generate_index()
        initial_size = cls.generate_data(device_type)
        job = {
            'index': index,
            'initial_size': initial_size,
            'remaining_size': initial_size,
            'creation_time': time,
            'serving_bs': None,
            'serving_time_start': None,
            'serving_time_end': None,
            'serving_time_total': None
        }
        return job
    
    @staticmethod
    def update_job_size(job: Job, data_rate: float) -> Job:
        job['remaining_size'] -= data_rate
        if job['remaining_size'] < 0:
            job['remaining_size'] = 0
        return job
 