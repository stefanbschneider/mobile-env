from queue import Queue
import numpy as np


class Buffer:
    def __init__(self):
        self.size = 100
        self.data_queue = Queue(maxsize=self.size)

    def add(self, packet):
        try: 
            self.data_queue.put(packet)
            print(f"Added to Buffer: {packet.index}")
            return True
        except Exception as e:
            print(f"Packet dropped! Error: {e}")
            return False

    def get(self):
        try:
            return self.data_queue.get(block=False)
        except Exception as e:
            print(f"Error getting packet: {e}")
            return None


class Packet:
    counter = 0  # Class variable to keep track of the index across all instances

    def __init__(self):
        self.index = self.generate_index()
        self.initial_size: float = self.generate_data()
        self.remaining_size: float = self.initial_size
        self.creation_time = None
        self.serving_bs = None
        self.serving_time_start = None
        self.serving_time_end = None
        self.serving_time_total = None

    @classmethod
    def generate_index(cls):
        cls.counter += 1
        return cls.counter

    @staticmethod
    def generate_data():
        # Generate bits for packet size following Poisson distribution with lambda
        return np.random.poisson(lam=3)

    def update_packet_size(self, data_rate):
        # Subtract data rate from remaining_size
        self.remaining_size -= data_rate

        # Ensure remaining_size does not go below zero
        if self.remaining_size < 0:
            self.remaining_size = 0



"""
def producer(buffer):
    while True:
        packet = Packets()
        packet.queue_update(buffer)
        sleep(1)  # Sleep to simulate time between packet generation

def consumer(buffer):
    while True:
        buffer.get()
        sleep(1)  # Sleep to simulate processing time

def signal_handler(signum, frame):
    print("Signal received, terminating processes.")
    producer_process.terminate()
    consumer_process.terminate()

if __name__ == "__main__":
    # Create a multiprocessing Queue
    queue = Queue()

    # Create a buffer object with the multiprocessing Queue
    shared_buffer = Buffer(queue)

    # Create producer and consumer processes
    producer_process = Process(target=producer, args=(shared_buffer,))
    consumer_process = Process(target=consumer, args=(shared_buffer,))

    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)

    # Start the processes
    producer_process.start()
    consumer_process.start()

    # Wait for the processes to finish (they won't unless interrupted)
    producer_process.join()
    consumer_process.join()

"""
