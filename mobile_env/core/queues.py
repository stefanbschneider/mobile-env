import queue
import threading

class Buffer:
    def __init__(self, size):
        self.size = size
        self.data_queue = queue.Queue(maxsize=size)

    def add(self, item):
        """Add an item to the buffer if not full, otherwise block until a slot is available."""
        self.data_queue.put(item, block=True)

    def get(self):
        """Remove and return an item from the buffer. If empty, block until an item is available."""
        return self.data_queue.get(block=True)

    def is_full(self):
        """Check if the buffer is full."""
        return self.data_queue.full()

    def is_empty(self):
        """Check if the buffer is empty."""
        return self.data_queue.empty()

    def current_size(self):
        """Return the current number of items in the buffer."""
        return self.data_queue.qsize()
        

    def producer(buffer):
        for i in range(50):
            item = f"Item{i}"  # Create a unique item
            if not buffer.is_full():
                buffer.add(item)
                print(f"Produced: {item}")
            else:
                print("Buffer is full, stopping production.")
                break

    def consumer(buffer):
        while not buffer.is_empty:
            item = buffer.get()
            print(f"Consumed: {item}")

# Create a buffer object with a size limit
buffer_size = 100
shared_buffer = Buffer(buffer_size)

# Create producer and consumer threads
producer_thread = threading.Thread(target=Buffer.producer, args=(shared_buffer,))
consumer_thread = threading.Thread(target=Buffer.consumer, args=(shared_buffer,))

# Start the threads
producer_thread.start()
consumer_thread.start()

# Wait for both threads to complete
producer_thread.join()
consumer_thread.join()
