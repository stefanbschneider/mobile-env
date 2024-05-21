import queue
import random
from time import sleep
from multiprocessing import Queue, Process
import signal

class Buffer:
    def __init__(self, queue):
        self.data_queue = queue

    def add(self, item):
        self.data_queue.put(item)
        print(f"Added to Buffer: {item.index}")

    def get(self):
        item = self.data_queue.get()
        print(f"Removed from Buffer: {item.index}")
        return item

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
            print(f"Packet {self.index} didn't enter the queue.")
            return False

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

