from queue import Queue
import numpy as np
import logging
import pandas as pd
from typing import Dict, Optional, Union

# Type alias for the job data
Job = Dict[str, Optional[Union[float, int]]]

class JobQueue:
    def __init__(self, size=100) -> None:
        # Initialize the buffer with a fixed size and an empty job queue
        self.size = size
        self.data_queue: Queue[Job] = Queue(maxsize=size)
    
    def enqueue_job(self, job: Job) -> bool:
        # Add a job to the buffer queue if there is space available
        if not self.data_queue.full():
            self.data_queue.put(job)
            logging.info(f"Job index {job['index']} added to the buffer.")
            return True
        else:
            logging.warning("Buffer is full, job dropped!")
            return False

    def dequeue_job(self) -> Optional[Job]:
        # Remove and return the job from the front of the queue
        if not self.data_queue.empty():
            job = self.data_queue.get()
            return job
        else:
            logging.warning("Buffer is empty!")
            return None

    def peek_job(self) -> Optional[Job]:
        # Retrieve the job at the front of the queue without removing it
        if not self.data_queue.empty():
            return self.data_queue.queue[0]
        else:
            return None