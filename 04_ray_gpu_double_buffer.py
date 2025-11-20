#!/usr/bin/env python3
"""
04_ray_gpu_double_buffer.py - Ray + GPU Double Buffering Integration

PURPOSE:
Combine Ray orchestration with GPU double-buffering for multi-GPU scaling.

PROFILING:
Set enable_profiling=True in CONFIG, then run: python 04_ray_gpu_double_buffer.py
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
    'batch_size': 8,
    'total_samples': 320,  # Total samples to process
    'input_shape': (1, 128, 128),  # (C, H, W)
    'num_iterations': 10,  # Matmul loop iterations
    'num_gpu_actors': 2,  # Number of GPU workers (one per GPU)
    'pin_memory': True,
    'queue_size': 20,
    'enable_profiling': False,  # Set to True for nsys profiling via Ray runtime_env
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

    def __init__(self, batch_size, input_shape, output_shape, num_iterations, gpu_id=0, pin_memory=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
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
            torch.empty(batch_size, *output_shape, device=self.device) for _ in range(2)
        ]
        self.cpu_output_buffers = [
            torch.empty(batch_size, *output_shape, pin_memory=pin_memory) for _ in range(2)
        ]

        # 6 events
        self.h2d_done_event = [torch.cuda.Event() for _ in range(2)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(2)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(2)]

        # Prime all events so wait_event() never deadlocks on first use
        for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
            for ev in events:
                ev.record()  # Record on default stream makes them signaled immediately

        self.current_idx = 0

    def swap(self):
        self.current_idx = 1 - self.current_idx

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


class GPUWorkerActorBase:
    """
    Base class for GPU worker actor with double-buffered pipeline.

    Ray allocates one GPU to this actor via num_gpus=1.
    CUDA_VISIBLE_DEVICES is automatically set by Ray.
    """

    def __init__(self, actor_id, batch_size, input_shape, output_shape, num_iterations, pin_memory):
        self.actor_id = actor_id
        self.batch_size = batch_size
        self._is_profiling_enabled = False  # Set to True in profiling subclass

        # Ray assigns GPU as cuda:0 (physical GPU set via CUDA_VISIBLE_DEVICES)
        import os
        physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        print(f"[Worker {actor_id}] Assigned to physical GPU {physical_gpu}")

        # Initialize double-buffered pipeline
        self.pipeline = DoubleBufferedPipeline(
            batch_size, input_shape, output_shape, num_iterations, gpu_id=0, pin_memory=pin_memory
        )

        # Metadata tracking (matches pipeline's num_buffers=2)
        self.metadata_buffers = [None] * 2

        self.processed_count = 0
        print(f"[Worker {actor_id}] Initialized with double-buffered pipeline")

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

    def process_from_queue(self, queue, num_batches):
        """
        Pull batches from queue and process with double buffering.

        Args:
            queue: Ray Queue containing (batch_idx, batch_data)
            num_batches: Number of batches to process

        Returns:
            Number of batches processed
        """
        logger = logging.getLogger(__name__)
        nvtx_prefix = f"Worker{self.actor_id}/"

        # Warmup phase: Process initial batches to fill pipeline
        warmup_batches = 2  # For double buffering (N=2)
        logger.info(f"Actor {self.actor_id}: Starting warmup ({warmup_batches} batches)...")

        for warmup_idx in range(min(warmup_batches, num_batches)):
            try:
                batch_idx, cpu_batch_np = queue.get(timeout=10.0)
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

                # Convert to torch tensor with explicit copy for writability
                # astype() creates a writable copy, enabling pin_memory() and copy_()
                cpu_batch = torch.from_numpy(cpu_batch_np.astype(np.float32))

                # Swap buffers (except for first batch)
                if self.processed_count > 0:
                    self.pipeline.swap()

                # Process with double buffering
                buffer_idx = self.pipeline.process_batch(
                    cpu_batch, batch_idx, self.batch_size, nvtx_prefix
                )

                # Store metadata
                self.metadata_buffers[buffer_idx] = {
                    'batch_idx': batch_idx,
                    'batch_size': self.batch_size,
                }

                # Collect output from previous iteration
                # For double buffering (N=2), can read after 1 iteration (N-1=1)
                if self.processed_count >= 1:
                    output_idx = 1 - buffer_idx  # The other buffer

                    # Sync D2H for this specific buffer
                    self.pipeline.d2h_done_event[output_idx].synchronize()

                    # Clone output (critical for async safety!)
                    output = self.pipeline.cpu_output_buffers[output_idx].clone()

                    # In production: send to result queue, save to file, etc.
                    # Here we just verify the shape is correct
                    if self.processed_count % 20 == 0:
                        print(f"[Worker {self.actor_id}] Collected output shape: {output.shape}")

                self.processed_count += 1

            except Exception as e:
                logger.error(f"[Worker {self.actor_id}] Error: {e}")
                break

        logger.info(f"[Worker {self.actor_id}] Processing completed. Total processed: {self.processed_count}")
        return self.processed_count


@ray.remote(num_gpus=1)
class GPUWorkerActor(GPUWorkerActorBase):
    """Standard GPU worker actor without profiling."""
    pass


@ray.remote(num_gpus=1, runtime_env={"nsight": {
    "t": "cuda,cudnn,cublas,nvtx,osrt",
    "cuda-graph-trace": "node",
    "cuda-memory-usage": "true",
    "stop-on-exit": "true",
}})
class GPUWorkerActorWithProfiling(GPUWorkerActorBase):
    """
    GPU worker actor with nsys profiling enabled.

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
    pin_memory = config['pin_memory']
    queue_size = config['queue_size']
    enable_profiling = config.get('enable_profiling', False)

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
    ActorClass = GPUWorkerActorWithProfiling if enable_profiling else GPUWorkerActor

    # Create GPU worker actors
    workers = [
        ActorClass.remote(i, batch_size, input_shape, output_shape, num_iterations, pin_memory)
        for i in range(num_gpu_actors)
    ]

    profiling_status = "WITH profiling" if enable_profiling else "WITHOUT profiling"
    print(f"Created {num_gpu_actors} GPU worker actors {profiling_status}")
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

    print(f"Processed {total_processed}/{total_produced} batches across {num_gpu_actors} workers: {processed_counts}")

    ray.shutdown()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires GPU(s).")
        exit(1)

    ray_gpu_pipeline(CONFIG)
