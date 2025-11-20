#!/usr/bin/env python3
"""
06_ray_nway_pipeline.py - Production-Ready Ray + N-Way Pipeline

PURPOSE:
Complete production pattern combining Ray orchestration with N-way GPU buffering.

PROFILING:
Set enable_profiling=True in CONFIG, then run: python 06_ray_nway_pipeline.py
Profiles: /tmp/ray/session_latest/logs/nsight/*.nsys-rep
View: nsys-ui /tmp/ray/session_latest/logs/nsight/*.nsys-rep

*** CRITICAL FOR PROFILING ***
The graceful_shutdown() method is REQUIRED for nsys profiling to capture GPU activities.
Without it, actors exit before GPU kernels complete and profiling data is flushed.
This is THE key lesson for Ray + GPU profiling.

CONFIGURATION:
Edit CONFIG dict below.
"""

import ray
from ray.util.queue import Queue
import torch
import torch.cuda.nvtx as nvtx
import time
import numpy as np
import logging

# Configuration
CONFIG = {
    'batch_size': 80,
    'total_samples': 640,  # Total samples to process
    'input_shape': (1, 256, 256),  # (C, H, W)
    'num_iterations': 50,  # Matmul loop iterations
    'num_gpu_actors': 1,  # Horizontal scaling: number of GPUs
    'num_buffers_per_actor': 2,  # Vertical scaling: concurrency per GPU (2=double, 3=triple)
    'pin_memory': True,
    'queue_size': 80,
    'cpu_delay_ms': 0,  # Simulated CPU preprocessing (0 = none)
    'enable_profiling': True,  # Set to True for nsys profiling via Ray runtime_env
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

    def __init__(self, batch_size, input_shape, output_shape, num_iterations, num_buffers=2,
                 gpu_id=0, pin_memory=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
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
            torch.empty(batch_size, *output_shape, device=self.device)
            for _ in range(num_buffers)
        ]
        self.cpu_output_buffers = [
            torch.empty(batch_size, *output_shape, pin_memory=pin_memory)
            for _ in range(num_buffers)
        ]

        self.h2d_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(num_buffers)]

        # Prime all events so wait_event() never deadlocks on first use
        for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
            for ev in events:
                ev.record()  # Record on default stream makes them signaled immediately

        self.current_idx = 0

    def swap(self):
        self.current_idx = (self.current_idx + 1) % self.num_buffers

    def process_batch(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix=""):
        buffer_idx = self.current_idx

        with nvtx.range(f"{nvtx_prefix}Batch{batch_idx}[buf={buffer_idx}]"):
            self._h2d_transfer(cpu_batch, buffer_idx, current_batch_size)
            self._compute_workload(buffer_idx, current_batch_size)
            self._d2h_transfer(buffer_idx, current_batch_size)

        return buffer_idx

    def _h2d_transfer(self, cpu_batch, buffer_idx, current_batch_size):
        with nvtx.range(f"H2D[buf={buffer_idx}]"):
            self.h2d_stream.wait_event(self.compute_done_event[buffer_idx])
            with torch.cuda.stream(self.h2d_stream):
                self.gpu_input_buffers[buffer_idx][:current_batch_size].copy_(
                    cpu_batch[:current_batch_size], non_blocking=True
                )
            self.h2d_stream.record_event(self.h2d_done_event[buffer_idx])

    def _compute_workload(self, buffer_idx, current_batch_size):
        with nvtx.range(f"Compute[buf={buffer_idx}]"):
            self.compute_stream.wait_event(self.h2d_done_event[buffer_idx])
            with torch.cuda.stream(self.compute_stream):
                input_slice = self.gpu_input_buffers[buffer_idx][:current_batch_size]
                output_slice = self.compute(input_slice)
                self.gpu_output_buffers[buffer_idx][:current_batch_size] = output_slice
            self.compute_stream.record_event(self.compute_done_event[buffer_idx])

    def _d2h_transfer(self, buffer_idx, current_batch_size):
        with nvtx.range(f"D2H[buf={buffer_idx}]"):
            self.d2h_stream.wait_event(self.compute_done_event[buffer_idx])
            with torch.cuda.stream(self.d2h_stream):
                self.cpu_output_buffers[buffer_idx][:current_batch_size].copy_(
                    self.gpu_output_buffers[buffer_idx][:current_batch_size], non_blocking=True
                )
            self.d2h_stream.record_event(self.d2h_done_event[buffer_idx])

    def wait_for_completion(self):
        self.h2d_stream.synchronize()
        self.compute_stream.synchronize()
        self.d2h_stream.synchronize()


class NWayGPUWorkerActorBase:
    """
    Base class for GPU worker actor with configurable N-way buffering.

    Production-ready: Supports any N >= 2.
    """

    def __init__(self, actor_id, batch_size, input_shape, output_shape, num_iterations,
                 num_buffers, pin_memory):
        self.actor_id = actor_id
        self.batch_size = batch_size
        self.num_buffers = num_buffers
        self._is_profiling_enabled = False  # Set to True in profiling subclass

        import os
        physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        print(f"[Worker {actor_id}] Assigned to physical GPU {physical_gpu} "
              f"with {num_buffers}-way buffering")

        # Initialize N-way buffered pipeline
        self.pipeline = NWayBufferedPipeline(
            batch_size, input_shape, output_shape, num_iterations, num_buffers,
            gpu_id=0, pin_memory=pin_memory
        )

        # Metadata tracking (matches num_buffers)
        self.metadata_buffers = [None] * num_buffers

        self.processed_count = 0

    def graceful_shutdown(self):
        """
        Gracefully shutdown the actor, ensuring all GPU work completes and profiling data is captured.

        This method is CRITICAL for nsys profiling to work correctly with Ray actors.
        Without it, the actor process may terminate before:
        1. GPU kernels finish executing
        2. CUDA driver flushes kernel trace data
        3. nsys captures the GPU timeline

        Returns:
            Number of batches processed by this actor
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Actor {self.actor_id}: Starting graceful shutdown...")

        # FINAL synchronization - wait for all pending operations to complete
        if self.processed_count > 0:
            with nvtx.range(f"Worker{self.actor_id}/final_streaming_sync"):
                self.pipeline.wait_for_completion()

                # CRITICAL: Additional GPU synchronization for profiling
                # Ensures all GPU kernels complete before process exit
                torch.cuda.synchronize(self.pipeline.device.index)

            logger.info(f"Actor {self.actor_id}: GPU synchronization completed")

        # Give nsys profiling time to flush data (important for profile data integrity)
        import os
        import json
        is_profiling = False

        # Check both attribute and environment variable for profiling mode
        if hasattr(self, '_is_profiling_enabled') and self._is_profiling_enabled:
            is_profiling = True

        runtime_env_str = os.environ.get('RAY_RUNTIME_ENV', '{}')
        try:
            runtime_env = json.loads(runtime_env_str)
            if 'nsight' in runtime_env:
                is_profiling = True
        except:
            pass

        if is_profiling:
            logger.info(f"Actor {self.actor_id}: Flushing profiling data...")
            time.sleep(0.5)  # Allow profiling data to flush

        logger.info(f"Actor {self.actor_id}: Graceful shutdown completed. Total processed: {self.processed_count}")

        # Exit the actor
        ray.actor.exit_actor()

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
        logger = logging.getLogger(__name__)
        nvtx_prefix = f"Worker{self.actor_id}/"

        # Warmup phase: Process initial batches to fill pipeline
        warmup_batches = self.num_buffers  # For N-way buffering
        logger.info(f"Actor {self.actor_id}: Starting warmup ({warmup_batches} batches with {self.num_buffers}-way buffering)...")

        for warmup_idx in range(min(warmup_batches, num_batches)):
            try:
                batch_idx, cpu_batch_np = queue.get(timeout=10.0)

                if cpu_delay_ms > 0:
                    with nvtx.range(f"{nvtx_prefix}CPU_Preprocess"):
                        time.sleep(cpu_delay_ms / 1000.0)

                cpu_batch = torch.from_numpy(cpu_batch_np.astype(np.float32))

                if self.processed_count > 0:
                    self.pipeline.swap()

                with nvtx.range(f"{nvtx_prefix}Warmup[batch={warmup_idx}]"):
                    buffer_idx = self.pipeline.process_batch(
                        cpu_batch, batch_idx, self.batch_size, nvtx_prefix
                    )

                self.metadata_buffers[buffer_idx] = {
                    'batch_idx': batch_idx,
                    'batch_size': self.batch_size,
                    'timestamp': time.time(),
                }
                self.processed_count += 1

            except Exception as e:
                logger.error(f"Actor {self.actor_id}: Warmup error: {e}")
                return self.processed_count

        # CRITICAL: Synchronize after warmup to create clear profiling boundary
        if self.processed_count > 0:
            with nvtx.range(f"{nvtx_prefix}WarmupSync"):
                self.pipeline.wait_for_completion()
                torch.cuda.synchronize(self.pipeline.device.index)
            logger.info(f"Actor {self.actor_id}: Warmup completed, {self.processed_count} batches processed")

        # Main processing loop
        remaining_batches = num_batches - self.processed_count
        for _ in range(remaining_batches):
            try:
                # Get batch from queue
                batch_idx, cpu_batch_np = queue.get(timeout=10.0)

                # Simulate CPU preprocessing (if enabled)
                if cpu_delay_ms > 0:
                    with nvtx.range(f"{nvtx_prefix}CPU_Preprocess"):
                        time.sleep(cpu_delay_ms / 1000.0)

                # Convert to torch tensor with explicit copy for writability
                # astype() creates a writable copy, enabling pin_memory() and copy_()
                cpu_batch = torch.from_numpy(cpu_batch_np.astype(np.float32))

                # Swap buffers (except first batch)
                if self.processed_count > 0:
                    self.pipeline.swap()

                # Process through pipeline - schedules async H2D/Compute/D2H
                buffer_idx = self.pipeline.process_batch(
                    cpu_batch, batch_idx, self.batch_size, nvtx_prefix
                )

                # CRITICAL PATTERN: Store metadata AFTER process_batch, BEFORE reading output
                # Timeline for N=3:
                #   Iteration 0: Store meta[0], process batch 0
                #   Iteration 1: Store meta[1], process batch 1
                #   Iteration 2: Store meta[2], process batch 2, READ output[0]
                #   Iteration 3: Store meta[0], process batch 3, READ output[1]
                #
                # Why (N-1)? With N buffers in flight:
                #   - Current buffer (idx=0): H2D happening now
                #   - Next buffer (idx=1): Compute happening now
                #   - Previous buffer (idx=2): D2H just finished, safe to read

                self.metadata_buffers[buffer_idx] = {
                    'batch_idx': batch_idx,
                    'batch_size': self.batch_size,
                    'timestamp': time.time(),
                }

                # Handle output from (N-1) iterations ago
                # Only start after pipeline has filled (processed >= N-1 batches)
                if self.processed_count >= self.num_buffers - 1:
                    # Calculate which buffer to read from (N-1 iterations old)
                    output_idx = (self.pipeline.current_idx - (self.num_buffers - 1)) % self.num_buffers

                    # Retrieve metadata for the batch we're outputting
                    output_meta_idx = (self.processed_count - (self.num_buffers - 1)) % self.num_buffers
                    output_meta = self.metadata_buffers[output_meta_idx]

                    # CRITICAL: Synchronize on D2H event for this specific buffer
                    # With N>=3, this completes instantly (D2H finished long ago)
                    with nvtx.range(f"{nvtx_prefix}SyncD2H[buf={output_idx}]"):
                        self.pipeline.d2h_done_event[output_idx].synchronize()

                    # Clone output (critical for async safety!)
                    with nvtx.range(f"{nvtx_prefix}CloneOutput"):
                        output = self.pipeline.cpu_output_buffers[output_idx][:output_meta['batch_size']].clone()

                    # Package and send to next stage (CPU-bound work, overlaps with GPU)
                    # In production: output_queue.put(...), save to file, etc.
                    if self.processed_count % 20 == 0:
                        print(f"[Worker {self.actor_id}] Collected batch {output_meta['batch_idx']}, shape: {output.shape}")

                self.processed_count += 1

            except Exception as e:
                logger.error(f"[Worker {self.actor_id}] Error: {e}")
                break

        logger.info(f"[Worker {self.actor_id}] Processing completed. Total processed: {self.processed_count}")
        return self.processed_count


@ray.remote(num_gpus=1)
class NWayGPUWorkerActor(NWayGPUWorkerActorBase):
    """Standard N-way GPU worker actor without profiling."""
    pass


@ray.remote(num_gpus=1, runtime_env={"nsight": {
    "t": "cuda,cudnn,cublas,nvtx,osrt",
    "cuda-graph-trace": "node",
    "cuda-memory-usage": "true",
    "stop-on-exit": "true",
}})
class NWayGPUWorkerActorWithProfiling(NWayGPUWorkerActorBase):
    """
    N-way GPU worker actor with nsys profiling enabled.

    Ray automatically wraps this actor's worker process with nsys profiling.
    Profiling files are generated in: /tmp/ray/session_*/logs/nsight/*.nsys-rep

    Runtime environment configuration:
    - "t": "cuda,cudnn,cublas,nvtx,osrt" - Trace CUDA, cuDNN, cuBLAS, NVTX, and OS runtime
    - "cuda-graph-trace": "node" - Enable CUDA graph tracing
    - "cuda-memory-usage": "true" - Track memory allocations
    - "stop-on-exit": "true" - Properly finalize profile on actor exit (CRITICAL)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_profiling_enabled = True  # Enable profiling flush in graceful_shutdown()


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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    batch_size = config['batch_size']
    total_samples = config['total_samples']
    input_shape = config['input_shape']
    num_iterations = config['num_iterations']
    num_gpu_actors = config['num_gpu_actors']
    num_buffers_per_actor = config['num_buffers_per_actor']
    pin_memory = config['pin_memory']
    queue_size = config['queue_size']
    cpu_delay_ms = config['cpu_delay_ms']
    enable_profiling = config.get('enable_profiling', False)

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
    print(f"  Profiling: {'ENABLED' if enable_profiling else 'DISABLED'}")
    if enable_profiling:
        print(f"  Profile location: /tmp/ray/session_latest/logs/nsight/")
    print()

    num_batches = total_samples // batch_size

    # For matmul workload, output shape = input shape
    # (In production, this would be determined from model output)
    output_shape = input_shape

    # Create Ray Queue
    queue = Queue(maxsize=queue_size)

    # Create producer
    producer = DataProducer.remote(batch_size, input_shape)

    # Select actor class based on profiling preference
    ActorClass = NWayGPUWorkerActorWithProfiling if enable_profiling else NWayGPUWorkerActor

    # Create N-way GPU worker actors
    workers = [
        ActorClass.remote(
            i, batch_size, input_shape, output_shape, num_iterations, num_buffers_per_actor, pin_memory
        )
        for i in range(num_gpu_actors)
    ]

    profiling_status = "WITH profiling" if enable_profiling else "WITHOUT profiling"
    print(f"Created {num_gpu_actors} GPU worker actors with {num_buffers_per_actor}-way buffering {profiling_status}")

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

    # Gracefully shutdown workers to ensure profiling data is captured
    logger.info("Initiating graceful shutdown of all workers...")
    shutdown_futures = [worker.graceful_shutdown.remote() for worker in workers]

    # Wait for all workers to complete shutdown (with timeout)
    # Note: Actors will exit via ray.actor.exit_actor(), so we expect RayActorError
    for i, future in enumerate(shutdown_futures):
        try:
            ray.get(future, timeout=30.0)
        except ray.exceptions.RayActorError as e:
            # Expected - actor exited gracefully
            logger.info(f"Worker {i}: Exited gracefully")
        except Exception as e:
            logger.warning(f"Worker {i}: Shutdown error: {e}")

    logger.info("All workers shut down gracefully")

    print(f"Processed {total_processed}/{total_produced} batches across {num_gpu_actors} workers with {num_buffers_per_actor}-way buffering: {processed_counts}")

    ray.shutdown()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires GPU(s).")
        exit(1)

    ray_nway_pipeline(CONFIG)
