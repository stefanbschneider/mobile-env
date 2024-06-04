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
        if self.current_size < self.size:
            self.data_queue[self.current_size] = packet
            self.current_size += 1
            logging.info(f"Packet added to buffer: {packet['index']}")
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
    
    @classmethod
    def create_packet(cls, current_time):
        index = cls.generate_index()
        initial_size = cls.generate_data()
        packet = np.array((index, initial_size, initial_size, current_time, -1, -1, -1, -1),
                          dtype=[('index', 'i4'),
                                 ('initial_size', 'f4'),
                                 ('remaining_size', 'f4'),
                                 ('creation_time', 'f4'),
                                 ('serving_bs', 'i4'),
                                 ('serving_time_start', 'f4'),
                                 ('serving_time_end', 'f4'),
                                 ('serving_time_total', 'f4')])
        return packet

