#!/usr/bin/env python3
"""
06_ray_nway_pipeline.py - Production-Ready Ray + N-Way Pipeline

PURPOSE:
Complete production pattern combining Ray orchestration with N-way GPU buffering.

LEARNING GOALS:
- Independent control of horizontal (GPUs) and vertical (buffers) scaling
- Production-ready architecture for high-throughput ML inference
- Performance analysis and tuning strategies
- Apply all concepts from files 01-05

WHAT TO OBSERVE:
- Run with: python 06_ray_nway_pipeline.py
- Profile with: nsys profile -o ray_nway.nsys-rep --trace=cuda,nvtx python 06_ray_nway_pipeline.py
- Ray dashboard: GPU utilization, actor status, object store usage
- nsys timeline: N actors × M buffers = N×M concurrent operations
- Compare: Vary num_gpu_actors and num_buffers_per_actor to see scaling

CONFIGURATION:
Edit CONFIG dict below
- num_gpu_actors: Horizontal scaling (across GPUs)
- num_buffers_per_actor: Vertical scaling (concurrency per GPU)

PRODUCTION PATTERN:
Producer -> Ray Queue -> N GPU Actors (each with M-way pipeline) -> Results

TUNING GUIDE:
- Start with num_buffers_per_actor=2 (double buffering)
- Increase num_gpu_actors to use more GPUs
- If CPU preprocessing is slow, try num_buffers_per_actor=3
- Monitor GPU utilization to identify bottlenecks

This is the pattern used in production PeakNet pipeline!
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
    'total_samples': 640,  # Total samples to process
    'input_shape': (1, 512, 512),  # (C, H, W)
    'num_iterations': 10,  # Matmul loop iterations
    'num_gpu_actors': 2,  # Horizontal scaling: number of GPUs
    'num_buffers_per_actor': 3,  # Vertical scaling: concurrency per GPU (2=double, 3=triple)
    'pin_memory': True,
    'queue_size': 20,
    'cpu_delay_ms': 0,  # Simulated CPU preprocessing (0 = none)
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


class NWayBufferedPipeline:
    """
    N-way buffered pipeline (same as file 05).
    Inlined for self-contained GitHub gist.
    """

    def __init__(self, batch_size, input_shape, num_iterations, num_buffers=2,
                 gpu_id=0, pin_memory=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_buffers = num_buffers
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')

        self.compute = create_matmul_workload(num_iterations)

        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.d2h_stream = torch.cuda.Stream(device=self.device)

        self.gpu_input_buffers = [
            torch.empty(batch_size, *input_shape, device=self.device)
            for _ in range(num_buffers)
        ]
        self.gpu_output_buffers = [
            torch.empty(batch_size, *input_shape, device=self.device)
            for _ in range(num_buffers)
        ]
        self.cpu_output_buffers = [
            torch.empty(batch_size, *input_shape, pin_memory=pin_memory)
            for _ in range(num_buffers)
        ]

        self.h2d_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(num_buffers)]

        for i in range(num_buffers):
            self.h2d_done_event[i].record(torch.cuda.current_stream())
            self.compute_done_event[i].record(torch.cuda.current_stream())
            self.d2h_done_event[i].record(torch.cuda.current_stream())

        self.current_idx = 0

    def swap(self):
        self.current_idx = (self.current_idx + 1) % self.num_buffers

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
class NWayGPUWorkerActor:
    """
    GPU worker actor with configurable N-way buffering.

    Production-ready: Supports any N >= 2.
    """

    def __init__(self, actor_id, batch_size, input_shape, num_iterations,
                 num_buffers, pin_memory):
        self.actor_id = actor_id
        self.batch_size = batch_size
        self.num_buffers = num_buffers

        import os
        physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        print(f"[Worker {actor_id}] Assigned to physical GPU {physical_gpu} "
              f"with {num_buffers}-way buffering")

        # Initialize N-way buffered pipeline
        self.pipeline = NWayBufferedPipeline(
            batch_size, input_shape, num_iterations, num_buffers,
            gpu_id=0, pin_memory=pin_memory
        )

        self.processed_count = 0

    def process_from_queue(self, queue, num_batches, cpu_delay_ms=0):
        """
        Pull batches from queue and process with N-way buffering.

        Args:
            queue: Ray Queue containing (batch_idx, batch_data)
            num_batches: Number of batches to process
            cpu_delay_ms: Simulated CPU preprocessing delay

        Returns:
            Number of batches processed
        """
        nvtx_prefix = f"Worker{self.actor_id}/"

        for _ in range(num_batches):
            try:
                # Get batch from queue
                batch_idx, cpu_batch_np = queue.get(timeout=10.0)

                # Simulate CPU preprocessing (if enabled)
                if cpu_delay_ms > 0:
                    with nvtx.range(f"{nvtx_prefix}CPU_Preprocess"):
                        time.sleep(cpu_delay_ms / 1000.0)

                # Convert to torch tensor
                cpu_batch = torch.from_numpy(cpu_batch_np)

                # Swap buffers (except first batch)
                if self.processed_count > 0:
                    self.pipeline.swap()

                # Process with N-way buffering
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
    """Producer actor (same as file 04)"""

    def __init__(self, batch_size, input_shape):
        self.batch_size = batch_size
        self.input_shape = input_shape

    def produce(self, queue, total_samples):
        num_batches = total_samples // self.batch_size
        print(f"[Producer] Starting production of {num_batches} batches")

        for batch_idx in range(num_batches):
            batch = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)
            queue.put((batch_idx, batch))

            if batch_idx % 40 == 0 and batch_idx > 0:
                print(f"[Producer] Produced batch {batch_idx}/{num_batches}")

        print(f"[Producer] Finished producing {num_batches} batches")
        return num_batches


def ray_nway_pipeline(config):
    """
    Production pipeline: Producer -> Ray Queue -> N×M GPU Pipeline -> Results

    Architecture:
    - 1 Producer: Generates batches
    - N GPU Actors: Each with M-way buffering
    - Ray Queue: Load balances across actors
    - Total concurrency: N actors × M buffers

    Scaling strategies:
    - Horizontal (num_gpu_actors): Add more GPUs
    - Vertical (num_buffers_per_actor): Add more concurrency per GPU
    """
    batch_size = config['batch_size']
    total_samples = config['total_samples']
    input_shape = config['input_shape']
    num_iterations = config['num_iterations']
    num_gpu_actors = config['num_gpu_actors']
    num_buffers_per_actor = config['num_buffers_per_actor']
    pin_memory = config['pin_memory']
    queue_size = config['queue_size']
    cpu_delay_ms = config['cpu_delay_ms']

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    print("="*60)
    print("Ray Cluster Resources:")
    resources = ray.cluster_resources()
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  GPUs: {resources.get('GPU', 0)}")
    print(f"  Object Store Memory: {resources.get('object_store_memory', 0) / 1e9:.2f} GB")
    print("="*60)
    print()

    if resources.get('GPU', 0) < num_gpu_actors:
        print(f"WARNING: Requested {num_gpu_actors} GPU actors but only "
              f"{resources.get('GPU', 0)} GPUs available")
        num_gpu_actors = int(resources.get('GPU', 0))
        if num_gpu_actors == 0:
            print("ERROR: No GPUs available")
            ray.shutdown()
            return

    print(f"Configuration:")
    print(f"  Total samples: {total_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {input_shape}")
    print(f"  Matmul iterations: {num_iterations}")
    print(f"  Number of GPU actors: {num_gpu_actors}")
    print(f"  Buffers per actor: {num_buffers_per_actor}")
    print(f"  Total concurrency: {num_gpu_actors} × {num_buffers_per_actor} = "
          f"{num_gpu_actors * num_buffers_per_actor} concurrent batches")
    print(f"  CPU preprocessing delay: {cpu_delay_ms} ms")
    print(f"  Pinned memory: {pin_memory}")
    print()

    num_batches = total_samples // batch_size

    # Create Ray Queue
    queue = Queue(maxsize=queue_size)

    # Create producer
    producer = DataProducer.remote(batch_size, input_shape)

    # Create N-way GPU worker actors
    workers = [
        NWayGPUWorkerActor.remote(
            i, batch_size, input_shape, num_iterations, num_buffers_per_actor, pin_memory
        )
        for i in range(num_gpu_actors)
    ]

    print(f"Created {num_gpu_actors} GPU worker actors with {num_buffers_per_actor}-way buffering")
    print()

    start_time = time.time()

    # Start producer
    producer_task = producer.produce.remote(queue, total_samples)

    # Start workers
    batches_per_worker = num_batches // num_gpu_actors
    worker_tasks = [
        worker.process_from_queue.remote(queue, batches_per_worker, cpu_delay_ms)
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
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Total batches processed: {total_processed}/{total_produced}")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Per-GPU throughput: {throughput/num_gpu_actors:.1f} samples/second")
    print(f"  Worker breakdown: {processed_counts}")
    print(f"{'='*60}")
    print()
    print("PRODUCTION PATTERN SUMMARY:")
    print(f"  ✓ Horizontal scaling: {num_gpu_actors} GPUs")
    print(f"  ✓ Vertical scaling: {num_buffers_per_actor}-way buffering per GPU")
    print(f"  ✓ Ray orchestration: Automatic load balancing")
    print(f"  ✓ NVTX profiling: Worker{id}/Batch{n}[buf={b}]")
    print(f"  ✓ Total concurrency: {num_gpu_actors * num_buffers_per_actor} batches in flight")
    print()
    print("TUNING RECOMMENDATIONS:")
    print("  1. Profile with nsys to identify bottlenecks")
    print("  2. If GPU utilization < 80%: Check queue_size, increase if needed")
    print("  3. If CPU busy: Increase num_buffers_per_actor to 3")
    print("  4. If I/O bound: Add more producers or optimize data loading")
    print("  5. Scale horizontally (GPUs) before vertically (buffers)")
    print()
    print("NEXT STEPS:")
    print("  - Replace matmul workload with your ML model")
    print("  - Add real data producer (disk, network, stream)")
    print("  - Add result collector and output writer")
    print("  - Monitor with Ray dashboard and nsys profiling")
    print("  - This is the pattern used in production PeakNet pipeline!")

    ray.shutdown()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires GPU(s).")
        exit(1)

    ray_nway_pipeline(CONFIG)

    # Optional: Scaling experiment
    print("\n" + "="*60)
    print("OPTIONAL: Scaling Experiment")
    print("="*60)
    print("\nTry varying num_gpu_actors and num_buffers_per_actor")
    print("to see how throughput scales with your hardware.")
    print("\nExample:")
    print("  1 GPU × 2 buffers: Baseline")
    print("  1 GPU × 3 buffers: Better if CPU preprocessing is slow")
    print("  2 GPUs × 2 buffers: ~2x throughput (linear scaling)")
    print("  2 GPUs × 3 buffers: Best for CPU-heavy workloads")
