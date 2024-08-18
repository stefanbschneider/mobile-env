from queue import Queue
from typing import Dict, Optional, Union

# Type alias for the job data
Job = Dict[str, Optional[Union[float, int]]]

class JobQueue:
    def __init__(self, size=1000) -> None:
        # Initialize the buffer with a fixed size and an empty job queue
        self.size = size
        self.data_queue: Queue[Job] = Queue(maxsize=size)

    def enqueue_job(self, job: Job) -> bool:
        # Add a job to the queue if there is enough space
        if not self.data_queue.full():
            self.data_queue.put(job)
            return True
        else:
            return False

    def dequeue_job(self) -> Optional[Job]:
        # Remove and return the job from the front of the queue
        if not self.data_queue.empty():
            job = self.data_queue.get()
            return job
        else:
            return None

    def peek_job(self) -> Optional[Job]:
        # Retrieve the job at the front of the queue without removing it
        if not self.data_queue.empty():
            return self.data_queue.queue[0]
        else:
            return None
    
    def size(self) -> int:
        # Return the current size of the queue.
        return self.data_queue.qsize()