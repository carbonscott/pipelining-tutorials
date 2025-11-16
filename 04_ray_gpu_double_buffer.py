#!/usr/bin/env python3
"""
04_ray_gpu_double_buffer.py - Ray + GPU Double Buffering Integration

PURPOSE:
Combine Ray orchestration with GPU double-buffering for multi-GPU scaling.

LEARNING GOALS:
- Wrap DoubleBufferedPipeline in Ray actors
- Allocate one GPU per actor with num_gpus=1
- Use NVTX with actor_id for multi-GPU profiling
- Scale throughput across multiple GPUs
- Understand complete pipeline: Producer -> GPU Workers -> Results

WHAT TO OBSERVE:
- Run with: python 04_ray_gpu_double_buffer.py
- Profile with: nsys profile -o ray_gpu.nsys-rep --trace=cuda,nvtx python 04_ray_gpu_double_buffer.py
- In nsys timeline: See multiple GPU streams (one per actor) working in parallel
- Each GPU shows double-buffered pipeline pattern
- Ray dashboard: GPU utilization across actors

CONFIGURATION:
Edit CONFIG dict below

NEXT: 05_nway_advanced.py to learn generalized N-way buffering
"""

import ray
from ray.util.queue import Queue
import torch
import torch.cuda.nvtx as nvtx
import time
import numpy as np

# Configuration
CONFIG = {
    'batch_size': 8,
    'total_samples': 320,  # Total samples to process
    'input_shape': (1, 512, 512),  # (C, H, W)
    'num_iterations': 10,  # Matmul loop iterations
    'num_gpu_actors': 2,  # Number of GPU workers (one per GPU)
    'pin_memory': True,
    'queue_size': 20,
}


def create_matmul_workload(num_iterations):
    """Same workload as previous files"""
    def workload(x):
        batch_size = x.shape[0]
        features = x.view(batch_size, -1)
        feat_dim = features.shape[1]
        weight = torch.randn(feat_dim, feat_dim, device=x.device, dtype=x.dtype)

        result = features
        for i in range(num_iterations):
            result = torch.matmul(result, weight)
            result = torch.relu(result)

        return result.view_as(x)
    return workload


class DoubleBufferedPipeline:
    """
    Double-buffered pipeline (same as file 02).

    Inlined here for self-contained GitHub gist.
    """

    def __init__(self, batch_size, input_shape, num_iterations, gpu_id=0, pin_memory=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')

        self.compute = create_matmul_workload(num_iterations)

        # 3 streams
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.d2h_stream = torch.cuda.Stream(device=self.device)

        # 2 sets of buffers
        self.gpu_input_buffers = [
            torch.empty(batch_size, *input_shape, device=self.device) for _ in range(2)
        ]
        self.gpu_output_buffers = [
            torch.empty(batch_size, *input_shape, device=self.device) for _ in range(2)
        ]
        self.cpu_output_buffers = [
            torch.empty(batch_size, *input_shape, pin_memory=pin_memory) for _ in range(2)
        ]

        # 6 events
        self.h2d_done_event = [torch.cuda.Event() for _ in range(2)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(2)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(2)]

        # Prime events
        for i in range(2):
            self.h2d_done_event[i].record(torch.cuda.current_stream())
            self.compute_done_event[i].record(torch.cuda.current_stream())
            self.d2h_done_event[i].record(torch.cuda.current_stream())

        self.current_idx = 0

    def swap(self):
        self.current_idx = 1 - self.current_idx

    def process_batch(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix=""):
        buffer_idx = self.current_idx

        with nvtx.range(f"{nvtx_prefix}Batch{batch_idx}[buf={buffer_idx}]"):
            self._h2d_transfer(cpu_batch, buffer_idx, current_batch_size)
            self._compute_workload(buffer_idx, current_batch_size)
            self._d2h_transfer(buffer_idx, current_batch_size)

    def _h2d_transfer(self, cpu_batch, buffer_idx, current_batch_size):
        with nvtx.range(f"H2D[buf={buffer_idx}]"):
            self.h2d_stream.wait_event(self.compute_done_event[buffer_idx])
            with torch.cuda.stream(self.h2d_stream):
                self.gpu_input_buffers[buffer_idx][:current_batch_size].copy_(
                    cpu_batch[:current_batch_size], non_blocking=True
                )
            self.h2d_done_event[buffer_idx].record(self.h2d_stream)

    def _compute_workload(self, buffer_idx, current_batch_size):
        with nvtx.range(f"Compute[buf={buffer_idx}]"):
            self.compute_stream.wait_event(self.h2d_done_event[buffer_idx])
            with torch.cuda.stream(self.compute_stream):
                input_slice = self.gpu_input_buffers[buffer_idx][:current_batch_size]
                output_slice = self.compute(input_slice)
                self.gpu_output_buffers[buffer_idx][:current_batch_size] = output_slice
            self.compute_done_event[buffer_idx].record(self.compute_stream)

    def _d2h_transfer(self, buffer_idx, current_batch_size):
        with nvtx.range(f"D2H[buf={buffer_idx}]"):
            self.d2h_stream.wait_event(self.compute_done_event[buffer_idx])
            with torch.cuda.stream(self.d2h_stream):
                self.cpu_output_buffers[buffer_idx][:current_batch_size].copy_(
                    self.gpu_output_buffers[buffer_idx][:current_batch_size], non_blocking=True
                )
            self.d2h_done_event[buffer_idx].record(self.d2h_stream)

    def wait_for_completion(self):
        self.h2d_stream.synchronize()
        self.compute_stream.synchronize()
        self.d2h_stream.synchronize()


@ray.remote(num_gpus=1)
class GPUWorkerActor:
    """
    GPU worker actor with double-buffered pipeline.

    Ray allocates one GPU to this actor via num_gpus=1.
    CUDA_VISIBLE_DEVICES is automatically set by Ray.
    """

    def __init__(self, actor_id, batch_size, input_shape, num_iterations, pin_memory):
        self.actor_id = actor_id
        self.batch_size = batch_size

        # Ray assigns GPU as cuda:0 (physical GPU set via CUDA_VISIBLE_DEVICES)
        import os
        physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        print(f"[Worker {actor_id}] Assigned to physical GPU {physical_gpu}")

        # Initialize double-buffered pipeline
        self.pipeline = DoubleBufferedPipeline(
            batch_size, input_shape, num_iterations, gpu_id=0, pin_memory=pin_memory
        )

        self.processed_count = 0
        print(f"[Worker {actor_id}] Initialized with double-buffered pipeline")

    def process_from_queue(self, queue, num_batches):
        """
        Pull batches from queue and process with double buffering.

        Args:
            queue: Ray Queue containing (batch_idx, batch_data)
            num_batches: Number of batches to process

        Returns:
            Number of batches processed
        """
        nvtx_prefix = f"Worker{self.actor_id}/"

        for _ in range(num_batches):
            try:
                # Get batch from queue
                batch_idx, cpu_batch_np = queue.get(timeout=10.0)

                # Convert to torch tensor
                cpu_batch = torch.from_numpy(cpu_batch_np)

                # Swap buffers (except for first batch)
                if self.processed_count > 0:
                    self.pipeline.swap()

                # Process with double buffering
                self.pipeline.process_batch(
                    cpu_batch, batch_idx, self.batch_size, nvtx_prefix
                )

                self.processed_count += 1

                if self.processed_count % 20 == 0:
                    print(f"[Worker {self.actor_id}] Processed {self.processed_count} batches")

            except Exception as e:
                print(f"[Worker {self.actor_id}] Error: {e}")
                break

        # Wait for all GPU work to finish
        self.pipeline.wait_for_completion()

        print(f"[Worker {self.actor_id}] Finished. Total processed: {self.processed_count}")
        return self.processed_count


@ray.remote
class DataProducer:
    """Producer actor (same as file 03, but with torch tensors)"""

    def __init__(self, batch_size, input_shape):
        self.batch_size = batch_size
        self.input_shape = input_shape

    def produce(self, queue, total_samples):
        num_batches = total_samples // self.batch_size
        print(f"[Producer] Starting production of {num_batches} batches")

        for batch_idx in range(num_batches):
            # Generate random batch as numpy (for Ray object store)
            batch = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)

            # Push to queue
            queue.put((batch_idx, batch))

            if batch_idx % 20 == 0:
                print(f"[Producer] Produced batch {batch_idx}/{num_batches}")

        print(f"[Producer] Finished producing {num_batches} batches")
        return num_batches


def ray_gpu_pipeline(config):
    """
    Complete pipeline: Producer -> GPU Workers (double-buffered) -> Results

    Architecture:
    - 1 Producer: Generates batches
    - N GPU Workers: Each with exclusive GPU and double-buffered pipeline
    - Ray Queue: Distributes work across workers
    """
    batch_size = config['batch_size']
    total_samples = config['total_samples']
    input_shape = config['input_shape']
    num_iterations = config['num_iterations']
    num_gpu_actors = config['num_gpu_actors']
    pin_memory = config['pin_memory']
    queue_size = config['queue_size']

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    print("Ray Cluster Resources:")
    resources = ray.cluster_resources()
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  GPUs: {resources.get('GPU', 0)}")
    print()

    if resources.get('GPU', 0) < num_gpu_actors:
        print(f"WARNING: Requested {num_gpu_actors} GPU actors but only {resources.get('GPU', 0)} GPUs available")
        print("Reducing num_gpu_actors to match available GPUs")
        num_gpu_actors = int(resources.get('GPU', 0))

    print(f"Configuration:")
    print(f"  Total samples: {total_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {input_shape}")
    print(f"  Matmul iterations: {num_iterations}")
    print(f"  Number of GPU actors: {num_gpu_actors}")
    print(f"  Pinned memory: {pin_memory}")
    print()

    num_batches = total_samples // batch_size

    # Create Ray Queue
    queue = Queue(maxsize=queue_size)

    # Create producer
    producer = DataProducer.remote(batch_size, input_shape)

    # Create GPU worker actors
    workers = [
        GPUWorkerActor.remote(i, batch_size, input_shape, num_iterations, pin_memory)
        for i in range(num_gpu_actors)
    ]

    print(f"Created {num_gpu_actors} GPU worker actors")
    print()

    start_time = time.time()

    # Start producer
    producer_task = producer.produce.remote(queue, total_samples)

    # Start workers
    batches_per_worker = num_batches // num_gpu_actors
    worker_tasks = [
        worker.process_from_queue.remote(queue, batches_per_worker)
        for worker in workers
    ]

    # Wait for completion
    total_produced = ray.get(producer_task)
    processed_counts = ray.get(worker_tasks)
    total_processed = sum(processed_counts)

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = total_processed * batch_size / elapsed

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Total batches processed: {total_processed}/{total_produced}")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Worker breakdown: {processed_counts}")
    print(f"{'='*60}")
    print()
    print("KEY CONCEPTS:")
    print("  1. Ray allocates one GPU per actor (num_gpus=1)")
    print("  2. Each actor runs independent double-buffered pipeline")
    print("  3. Queue distributes work across GPUs automatically")
    print("  4. NVTX annotations include actor_id for multi-GPU profiling")
    print("  5. Throughput scales with number of GPUs")
    print()
    print("NVTX PROFILING TIP:")
    print("  In nsys timeline, filter by 'Worker0/', 'Worker1/', etc.")
    print("  to see per-GPU pipeline behavior")

    ray.shutdown()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires GPU(s).")
        exit(1)

    ray_gpu_pipeline(CONFIG)
