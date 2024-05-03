import queue
import threading
import random
from time import sleep

class Buffer:
    def __init__(self, size):
        self.size = size
        self.data_queue = queue.Queue(maxsize=size)

    def add(self, item):
        try:
            self.data_queue.put(item, block=True, timeout=2)  
            print(f"Added to Buffer: {item.index}")
        except queue.Full:
            print("Buffer is full, could not add the item.")

    def get(self):
        try:
            item = self.data_queue.get(block=True, timeout=2)
            print(f"Removed from Buffer: {item.index}")
            return item
        except queue.Empty:
            print("Buffer is empty, nothing to consume.")

    def is_full(self):
        return self.data_queue.full()

    def is_empty(self):
        return self.data_queue.empty()

    def current_size(self):
        return self.data_queue.qsize()


class Packets:
    counter = 0  # Class variable to keep track of the index across all instances

    def __init__(self):
        self.size = self.generate_data()
        self.index = self.generate_index()
        self.data = format(self.size, '08b')
        
    @staticmethod
    def generate_data():
        return random.randint(0, 255)

    @classmethod
    def generate_index(cls):
        cls.counter += 1
        return cls.counter
    
    def queue_update(self, buffer):
        if random.random() < 0.5:  # Let's assume a 50% chance to enqueue the packet
            buffer.add(self)
            return True
        else:
            print("didn't enter ")
            return False

def producer(buffer):
    while True:
        packet = Packets()
        packet.queue_update(buffer)
        sleep(1)  # Sleep to simulate time between packet generation

def consumer(buffer):
    while True:
        if not buffer.is_empty():
            buffer.get()
        sleep(1)  # Sleep to simulate processing time

# Create a buffer object with a size limit
buffer_size = 10  # Reduced for demonstration purposes
shared_buffer = Buffer(buffer_size)

# Create producer and consumer threads
producer_thread = threading.Thread(target=producer, args=(shared_buffer,))
consumer_thread = threading.Thread(target=consumer, args=(shared_buffer,))

# Start the threads
producer_thread.start()
consumer_thread.start()