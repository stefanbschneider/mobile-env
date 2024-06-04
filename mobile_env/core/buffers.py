from queue import Queue
import numpy as np
import logging

class Buffer:
    def __init__(self, size=100):
        self.size = size
        self.data_queue = np.zeros(size, dtype=[
            ('index', 'i4'),
            ('initial_size', 'f4'),
            ('remaining_size', 'f4'),
            ('creation_time', 'f4'),
            ('serving_bs', 'i4'),
            ('serving_time_start', 'f4'),
            ('serving_time_end', 'f4'),
            ('serving_time_total', 'f4')
        ])
        self.current_size = 0

    def add(self, packet):
        try: 
            self.data_queue.put(packet)
            print(f"Added to Buffer: {packet.index}")
            return True
        else:
            logging.warning("Buffer is full, packet dropped!")

    def remove(self):
        if self.current_size > 0:
            packet = self.data_queue[0]
            self.data_queue = np.delete(self.data_queue, 0)
            self.current_size -= 1
            return packet
        else:
            logging.warning("Buffer is empty!")
            return None

    def get(self):
        if self.current_size > 0:
            return self.data_queue[0]
        else:
            return None


class PacketGenerator:
    counter = 0 # Class variable to keep track of the number of packets

    @classmethod
    def generate_index(cls):
        cls.counter += 1
        return cls.counter
    
    @staticmethod
    def generate_data():
        # Generate bits for packets following Poisson distribution with lambda
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
