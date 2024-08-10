import logging
import pandas as pd
from typing import Dict, Optional, Union

# Type alias for the job data
Job = Dict[str, Optional[Union[float, int]]]

class JobQueue:
    def __init__(self, size: int)-> None:
        # Initialize the buffer with a fixed size and an empty DataFrame for jobs
        self.size = size
        self.jobs_df = pd.DataFrame(columns=[
            'index', 'serving_bs', 'initial_size', 'remaining_size', 
            'creation_time', 'serving_time_start', 'serving_time_end', 
            'serving_time_total', 'device_type', 'device_id', 
            'processing_time', 'computational_requirement', 'target_id'
        ])
    
    def enqueue_job(self, job: Job) -> bool:
        # Add a job to the DataFrame if there is space available
        if len(self.jobs_df) < self.size:
            self.jobs_df = pd.concat([self.jobs_df, pd.DataFrame([job])], ignore_index=True)
            logging.info(f"Job index {job['index']} added to the buffer.")
            return True
        else:
            logging.warning("Buffer is full, job dropped!")
            return False

    def dequeue_job(self) -> Optional[Job]:
        # Remove and return the job from the front of the DataFrame
        if not self.jobs_df.empty:
            job = self.jobs_df.iloc[0]
            self.jobs_df = self.jobs_df.iloc[1:].reset_index(drop=True)
            return job.to_dict()
        else:
            logging.warning("Buffer is empty!")
            return None

    def peek_job(self) -> Optional[Job]:
        # Retrieve the job at the front of the DataFrame without removing it
        if not self.jobs_df.empty:
            return self.jobs_df.iloc[0].to_dict()
        else:
            return None