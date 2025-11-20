#!/usr/bin/env python3
"""
03_ray_producer_consumer.py - Basic Ray Producer-Consumer Pattern (CPU-Only)

PURPOSE:
Learn Ray's object store and actor patterns without GPU complexity.

CONFIGURATION:
Edit CONFIG dict below.
"""

import ray
from ray.util.queue import Queue
import numpy as np
import signal
import sys

# Configuration
CONFIG = {
    'num_batches': 100,
    'batch_size': 8,
    'data_shape': (512, 512),  # 2D array shape
    'num_consumers': 2,  # Number of parallel consumers
    'queue_size': 10,  # Max items in queue
}


@ray.remote
class DataProducer:
    """
    Producer actor that generates batches and pushes to queue.

    In a real pipeline, this would read from disk/network/stream.
    Here we generate random data for demonstration.
    """

    def __init__(self, batch_size, data_shape):
        self.batch_size = batch_size
        self.data_shape = data_shape
        print(f"[Producer] Initialized with batch_size={batch_size}, shape={data_shape}")

    def produce(self, queue, num_batches):
        """
        Generate batches and push to queue.

        Args:
            queue: Ray Queue to push data to
            num_batches: Number of batches to generate
        """
        print(f"[Producer] Starting production of {num_batches} batches")

        for batch_idx in range(num_batches):
            # Generate random batch (simulates reading real data)
            batch = np.random.randn(self.batch_size, *self.data_shape).astype(np.float32)

            # Push to queue
            # Ray automatically puts batch in object store and sends ObjectRef to queue
            queue.put((batch_idx, batch))

            if batch_idx % 20 == 0:
                print(f"[Producer] Produced batch {batch_idx}/{num_batches}")

        print(f"[Producer] Finished producing {num_batches} batches")


@ray.remote
class DataConsumer:
    """
    Consumer actor that processes batches from queue.

    In a real pipeline, this might do inference, transformation, etc.
    Here we do simple CPU computation for demonstration.
    """

    def __init__(self, consumer_id):
        self.consumer_id = consumer_id
        self.processed_count = 0
        print(f"[Consumer {consumer_id}] Initialized")

    def consume(self, queue, expected_batches):
        """
        Pull batches from queue and process them.

        Args:
            queue: Ray Queue to pull data from
            expected_batches: Expected number of batches to process

        Returns:
            Number of batches processed
        """
        print(f"[Consumer {self.consumer_id}] Starting consumption")

        while self.processed_count < expected_batches:
            try:
                # Get from queue (blocks if empty)
                batch_idx, batch = queue.get(timeout=5.0)

                # Process batch (simple CPU computation)
                result = self._process_batch(batch)

                self.processed_count += 1

                if self.processed_count % 20 == 0:
                    print(f"[Consumer {self.consumer_id}] Processed {self.processed_count} batches")

            except Exception as e:
                # Timeout or queue closed
                print(f"[Consumer {self.consumer_id}] Queue timeout or closed: {e}")
                break

        print(f"[Consumer {self.consumer_id}] Finished. Total processed: {self.processed_count}")
        return self.processed_count

    def _process_batch(self, batch):
        """
        Simple CPU processing (matrix operations).

        In a real pipeline, this would be your actual workload.
        """
        # Example: mean normalization + matrix operations
        normalized = (batch - np.mean(batch, axis=(1, 2), keepdims=True))
        result = np.matmul(normalized.reshape(batch.shape[0], -1),
                           normalized.reshape(batch.shape[0], -1).T)
        return result


def setup_shutdown_handler(queue):
    """Setup graceful shutdown on Ctrl+C"""
    def signal_handler(sig, frame):
        print("\n\nShutdown signal received. Cleaning up...")
        # Note: In production, you'd coordinate shutdown across actors
        # For this simple example, we just exit
        ray.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def ray_producer_consumer_pipeline(config):
    """
    Run producer-consumer pipeline with Ray.

    Architecture:
    - 1 Producer actor generating batches
    - N Consumer actors processing in parallel
    - Ray Queue for communication (backed by object store)
    """
    num_batches = config['num_batches']
    batch_size = config['batch_size']
    data_shape = config['data_shape']
    num_consumers = config['num_consumers']
    queue_size = config['queue_size']

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    print("Ray Cluster Resources:")
    resources = ray.cluster_resources()
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  GPUs: {resources.get('GPU', 0)}")
    print(f"  Object Store Memory: {resources.get('object_store_memory', 0) / 1e9:.2f} GB")
    print()

    print(f"Configuration:")
    print(f"  Number of batches: {num_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data shape: {data_shape}")
    print(f"  Number of consumers: {num_consumers}")
    print(f"  Queue size: {queue_size}")
    print()

    # Create Ray Queue
    # This queue stores ObjectRefs, enabling zero-copy data sharing
    queue = Queue(maxsize=queue_size)

    # Setup shutdown handler
    setup_shutdown_handler(queue)

    # Create producer actor
    producer = DataProducer.remote(batch_size, data_shape)

    # Create consumer actors
    consumers = [DataConsumer.remote(i) for i in range(num_consumers)]

    print(f"Created {num_consumers} consumer actors")

    # Start producer (async call)
    producer_task = producer.produce.remote(queue, num_batches)

    # Start consumers (async calls)
    # Each consumer will process approximately num_batches/num_consumers batches
    batches_per_consumer = num_batches // num_consumers
    consumer_tasks = [
        consumer.consume.remote(queue, batches_per_consumer)
        for consumer in consumers
    ]

    # Wait for producer to finish
    ray.get(producer_task)
    print("\n[Main] Producer finished")

    # Wait for all consumers to finish
    processed_counts = ray.get(consumer_tasks)
    total_processed = sum(processed_counts)

    print(f"Processed {total_processed} batches across {num_consumers} consumers: {processed_counts}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    ray_producer_consumer_pipeline(CONFIG)
