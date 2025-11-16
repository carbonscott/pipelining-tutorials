#!/usr/bin/env python3
"""
02_double_buffering.py - Classic Double Buffering for GPU Pipeline

PURPOSE:
Eliminate GPU idle time by overlapping H2D, Compute, and D2H operations.

LEARNING GOALS:
- Understand the double-buffering pattern (2 buffers, 3 streams)
- Implement process_batch() interface matching production code
- Use CUDA events for fine-grained synchronization
- Learn when/where to place NVTX annotations for profiling

WHAT TO OBSERVE:
- Run with: python 02_double_buffering.py
- Profile with: nsys profile -o double.nsys-rep --trace=cuda,nvtx python 02_double_buffering.py
- In nsys timeline: H2D, Compute, D2H overlap (no gaps!)
- Compare with baseline: Open both baseline.nsys-rep and double.nsys-rep in nsys GUI

CONFIGURATION:
Edit CONFIG dict below (same parameters as file 01)

NEXT: 03_ray_producer_consumer.py to learn Ray orchestration
"""

import torch
import torch.cuda.nvtx as nvtx
import time

# Configuration
CONFIG = {
    'batch_size': 8,
    'num_batches': 40,
    'input_shape': (1, 512, 512),  # (C, H, W)
    'num_iterations': 10,  # Controls compute time
    'gpu_id': 0,
    'pin_memory': True,  # Use pinned CPU memory for faster async transfers
}


def create_matmul_workload(num_iterations):
    """Same workload as file 01"""
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
    Classic double-buffering pipeline with 3 concurrent stages.

    Architecture:
    - 2 buffers (alternating between Buffer 0 and Buffer 1)
    - 3 streams (H2D, Compute, D2H)
    - Events for synchronization

    At steady state:
    - Buffer 0: H2D transfer for batch N+1
    - Buffer 1: Compute for batch N, D2H transfer for batch N
    """

    def __init__(self, batch_size, input_shape, num_iterations, gpu_id=0, pin_memory=True):
        """
        Initialize double-buffered pipeline.

        Args:
            batch_size: Batch size for processing
            input_shape: Input tensor shape (C, H, W) without batch dimension
            num_iterations: Matmul loop iterations (controls compute time)
            gpu_id: GPU device ID
            pin_memory: Use pinned CPU memory for async transfers
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')

        # Create compute workload
        self.compute = create_matmul_workload(num_iterations)

        # Create 3 CUDA streams
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.d2h_stream = torch.cuda.Stream(device=self.device)

        # Allocate 2 sets of buffers
        self.gpu_input_buffers = [
            torch.empty(batch_size, *input_shape, device=self.device) for _ in range(2)
        ]
        self.gpu_output_buffers = [
            torch.empty(batch_size, *input_shape, device=self.device) for _ in range(2)
        ]
        self.cpu_output_buffers = [
            torch.empty(batch_size, *input_shape, pin_memory=pin_memory) for _ in range(2)
        ]

        # Create 6 CUDA events (3 per buffer: h2d_done, compute_done, d2h_done)
        self.h2d_done_event = [torch.cuda.Event() for _ in range(2)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(2)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(2)]

        # Prime events to prevent deadlock on first batch
        for i in range(2):
            self.h2d_done_event[i].record(torch.cuda.current_stream())
            self.compute_done_event[i].record(torch.cuda.current_stream())
            self.d2h_done_event[i].record(torch.cuda.current_stream())

        # Metadata tracking (matches number of buffers)
        # Stores information about which batch is in which buffer
        self.metadata_buffers = [None] * 2

        # Track current buffer
        self.current_idx = 0

    def swap(self):
        """Swap to the other buffer"""
        self.current_idx = 1 - self.current_idx

    def process_batch(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix=""):
        """
        Process a batch through the full H2D -> Compute -> D2H pipeline.

        This method schedules async GPU work and returns immediately (non-blocking).
        Use wait_for_completion() or sync events when you need results.

        Args:
            cpu_batch: CPU input tensor (batch_size, C, H, W)
            batch_idx: Sequential batch number (for profiling/logging)
            current_batch_size: Actual size of this batch (handles partial batches)
            nvtx_prefix: String prefix for NVTX annotations

        Returns:
            buffer_idx: Index of buffer used for this batch (for metadata tracking)
        """
        buffer_idx = self.current_idx

        # NVTX Annotation Strategy:
        # - Use buffer index to distinguish overlapping operations in timeline
        # - Helps visualize which buffer is being processed at each stage
        with nvtx.range(f"{nvtx_prefix}Batch{batch_idx}[buf={buffer_idx}]"):
            self._h2d_transfer(cpu_batch, buffer_idx, current_batch_size)
            self._compute_workload(buffer_idx, current_batch_size)
            self._d2h_transfer(buffer_idx, current_batch_size)

        return buffer_idx

    def _h2d_transfer(self, cpu_batch, buffer_idx, current_batch_size):
        """Host to Device transfer"""
        with nvtx.range(f"H2D[buf={buffer_idx}]"):
            # Wait for previous compute on this buffer to finish
            self.h2d_stream.wait_event(self.compute_done_event[buffer_idx])

            with torch.cuda.stream(self.h2d_stream):
                # Only transfer the valid slice if partial batch
                self.gpu_input_buffers[buffer_idx][:current_batch_size].copy_(
                    cpu_batch[:current_batch_size], non_blocking=True
                )

            # Record completion
            self.h2d_done_event[buffer_idx].record(self.h2d_stream)

    def _compute_workload(self, buffer_idx, current_batch_size):
        """GPU compute workload"""
        with nvtx.range(f"Compute[buf={buffer_idx}]"):
            # Wait for H2D to finish
            self.compute_stream.wait_event(self.h2d_done_event[buffer_idx])

            with torch.cuda.stream(self.compute_stream):
                input_slice = self.gpu_input_buffers[buffer_idx][:current_batch_size]
                output_slice = self.compute(input_slice)
                self.gpu_output_buffers[buffer_idx][:current_batch_size] = output_slice

            # Record completion
            self.compute_done_event[buffer_idx].record(self.compute_stream)

    def _d2h_transfer(self, buffer_idx, current_batch_size):
        """Device to Host transfer"""
        with nvtx.range(f"D2H[buf={buffer_idx}]"):
            # Wait for compute to finish
            self.d2h_stream.wait_event(self.compute_done_event[buffer_idx])

            with torch.cuda.stream(self.d2h_stream):
                self.cpu_output_buffers[buffer_idx][:current_batch_size].copy_(
                    self.gpu_output_buffers[buffer_idx][:current_batch_size], non_blocking=True
                )

            # Record completion
            self.d2h_done_event[buffer_idx].record(self.d2h_stream)

    def wait_for_completion(self):
        """Wait for all pipeline stages to complete"""
        self.h2d_stream.synchronize()
        self.compute_stream.synchronize()
        self.d2h_stream.synchronize()


def double_buffered_pipeline(config):
    """Run double-buffered pipeline"""
    batch_size = config['batch_size']
    num_batches = config['num_batches']
    input_shape = config['input_shape']
    num_iterations = config['num_iterations']
    gpu_id = config['gpu_id']
    pin_memory = config['pin_memory']

    # Create pipeline
    pipeline = DoubleBufferedPipeline(batch_size, input_shape, num_iterations, gpu_id, pin_memory)

    # Create input data
    cpu_input = torch.randn(batch_size, *input_shape, pin_memory=pin_memory)

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Input shape: {input_shape}")
    print(f"  Matmul iterations: {num_iterations}")
    print(f"  Pinned memory: {pin_memory}")
    print()

    # Warmup
    for i in range(3):
        if i > 0:
            pipeline.swap()
        pipeline.process_batch(cpu_input, i, batch_size, "Warmup/")
    pipeline.wait_for_completion()

    print("Starting double-buffered processing...")
    start_time = time.time()

    # Track collected outputs for demonstration
    collected_outputs = []

    with nvtx.range("Double-Buffered Processing"):
        for batch_idx in range(num_batches):
            if batch_idx > 0:
                pipeline.swap()  # Alternate buffers

            # Schedule async work (returns immediately!)
            buffer_idx = pipeline.process_batch(cpu_input, batch_idx, batch_size)

            # Store metadata for this batch
            # This allows us to track which batch is in which buffer
            pipeline.metadata_buffers[buffer_idx] = {
                'batch_idx': batch_idx,
                'batch_size': batch_size,
            }

            # Collect output from previous iteration
            # For double buffering (N=2), we can read output after 1 iteration (N-1=1)
            if batch_idx >= 1:
                # Which buffer has the previous batch? The other one!
                output_idx = 1 - buffer_idx  # Simple for N=2

                # CRITICAL: Synchronize D2H for this specific buffer
                pipeline.d2h_done_event[output_idx].synchronize()

                # CRITICAL: Clone output to safely use it
                # Without clone(), the buffer will be overwritten in next iteration
                output = pipeline.cpu_output_buffers[output_idx].clone()

                # Retrieve associated metadata
                output_meta = pipeline.metadata_buffers[output_idx]

                # Now safe to use output (save to file, send to queue, etc.)
                collected_outputs.append({'metadata': output_meta, 'output': output})

                if batch_idx % 10 == 0:
                    print(f"  Collected output for batch {output_meta['batch_idx']}")

    # Wait for all work to complete
    pipeline.wait_for_completion()

    # Collect final batch output (the last one)
    if num_batches > 0:
        final_idx = pipeline.current_idx
        pipeline.d2h_done_event[final_idx].synchronize()
        output = pipeline.cpu_output_buffers[final_idx].clone()
        output_meta = pipeline.metadata_buffers[final_idx]
        collected_outputs.append({'metadata': output_meta, 'output': output})
        print(f"  Collected final output for batch {output_meta['batch_idx']}")

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = num_batches * batch_size / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Time per batch: {elapsed/num_batches*1000:.2f} ms")
    print(f"  Outputs collected: {len(collected_outputs)}/{num_batches}")
    print()
    print("OBSERVATION: In nsys timeline, you'll see H2D, Compute, and D2H overlap!")
    print("The GPU is never idle - we've eliminated the gaps from file 01.")
    print()
    print("OUTPUT HANDLING PATTERN:")
    print("  1. Store metadata AFTER process_batch() call")
    print("  2. Read output from previous iteration (batch_idx >= 1)")
    print("  3. Synchronize D2H event for specific buffer")
    print("  4. Clone output tensor (critical for async safety!)")
    print("  5. Use output (save, send to queue, etc.)")
    print()
    print("NVTX PLACEMENT LESSON:")
    print("  - Annotate each stage (H2D/Compute/D2H) with buffer index")
    print("  - This helps visualize which buffer is where in the pipeline")
    print("  - In nsys, you can see Buffer 0 and Buffer 1 alternating through stages")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        exit(1)

    double_buffered_pipeline(CONFIG)
