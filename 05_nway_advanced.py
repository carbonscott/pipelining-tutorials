#!/usr/bin/env python3
"""
05_nway_advanced.py - Generalized N-Way Buffering (Advanced Topic)

PURPOSE:
Learn when and how to use N>2 buffers for improved performance.

PROFILING:
nsys profile -o nway.nsys-rep --trace=cuda,nvtx python 05_nway_advanced.py

CONFIGURATION:
Edit CONFIG dict below.
"""

import torch
import torch.cuda.nvtx as nvtx

# Configuration
CONFIG = {
    'batch_size': 8,
    'num_batches': 40,
    'input_shape': (1, 128, 128),  # (C, H, W)
    'num_iterations': 10,  # Matmul loop iterations
    'gpu_id': 0,
    'pin_memory': True,
    'num_buffers': 3,  # Try 2, 3, 4 to see the difference
    'cpu_delay_ms': 5,  # Simulated CPU preprocessing delay (0 = no delay)
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
    Generalized N-way buffered pipeline.

    Key difference from double buffering:
    - num_buffers is configurable (not hardcoded to 2)
    - Circular buffer rotation: current_idx = (current_idx + 1) % num_buffers
    - More parallelism when CPU has work to do

    Architecture:
    - N buffers (circular rotation)
    - 3 streams (H2D, Compute, D2H) - same as double buffering
    - N sets of events (one per buffer)

    At steady state with N=3:
    - Buffer 0: H2D for batch i+2
    - Buffer 1: Compute + D2H for batch i+1
    - Buffer 2: Being read by CPU (batch i)
    """

    def __init__(self, batch_size, input_shape, output_shape, num_iterations, num_buffers=2,
                 gpu_id=0, pin_memory=True):
        """
        Initialize N-way buffered pipeline.

        Args:
            batch_size: Batch size for processing
            input_shape: Input tensor shape (C, H, W)
            output_shape: Output tensor shape (C, H, W)
            num_iterations: Matmul loop iterations
            num_buffers: Number of concurrent buffers (2=double, 3=triple, etc.)
            gpu_id: GPU device ID
            pin_memory: Use pinned CPU memory
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_buffers = num_buffers
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')

        # Create compute workload
        self.compute = create_matmul_workload(num_iterations)

        # Create 3 CUDA streams (same as double buffering)
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.d2h_stream = torch.cuda.Stream(device=self.device)

        # Allocate N sets of buffers (generalized from 2)
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

        # Create 3N CUDA events (3 per buffer)
        self.h2d_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(num_buffers)]

        # Prime all events so wait_event() never deadlocks on first use
        for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
            for ev in events:
                ev.record()  # Record on default stream makes them signaled immediately

        # Metadata tracking (matches number of buffers)
        # Stores information about which batch is in which buffer
        self.metadata_buffers = [None] * num_buffers

        # Track current buffer
        self.current_idx = 0

        print(f"Initialized {num_buffers}-way buffered pipeline")
        print(f"  Memory per buffer: {batch_size * torch.tensor(input_shape).prod().item() * 4 / 1e6:.2f} MB")
        print(f"  Total GPU memory: {num_buffers * batch_size * torch.tensor(input_shape).prod().item() * 4 / 1e6:.2f} MB")

    def swap(self):
        """
        Advance to next buffer (circular rotation).

        For N=2: 0 -> 1 -> 0 -> 1 ...
        For N=3: 0 -> 1 -> 2 -> 0 -> 1 -> 2 ...
        For N=4: 0 -> 1 -> 2 -> 3 -> 0 -> 1 -> 2 -> 3 ...
        """
        self.current_idx = (self.current_idx + 1) % self.num_buffers

    def process_batch(self, cpu_batch, batch_idx, current_batch_size, nvtx_prefix=""):
        """
        Process a batch through H2D -> Compute -> D2H pipeline.

        Same interface as DoubleBufferedPipeline for consistency.

        Returns:
            buffer_idx: Index of buffer used for this batch (for metadata tracking)
        """
        buffer_idx = self.current_idx

        with nvtx.range(f"{nvtx_prefix}Batch{batch_idx}[buf={buffer_idx}]"):
            self._h2d_transfer(cpu_batch, buffer_idx, current_batch_size)
            self._compute_workload(buffer_idx, current_batch_size)
            self._d2h_transfer(buffer_idx, current_batch_size)

        return buffer_idx

    def _h2d_transfer(self, cpu_batch, buffer_idx, current_batch_size):
        with nvtx.range(f"H2D[buf={buffer_idx}]"):
            # Wait for previous compute on this buffer to finish
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
        """Wait for all pipeline stages to complete"""
        self.h2d_stream.synchronize()
        self.compute_stream.synchronize()
        self.d2h_stream.synchronize()

    def get_result_buffer_idx(self, current_batch_idx):
        """
        Get buffer index for reading results.

        With N buffers, results for batch i are in buffer:
        (current_idx - (N-1)) % N

        This ensures we only read buffers that have completed D2H.
        """
        result_idx = (self.current_idx - (self.num_buffers - 1)) % self.num_buffers
        return result_idx


def simulate_cpu_preprocessing(delay_ms):
    """
    Simulate CPU preprocessing time.

    In real pipelines, this might be:
    - Reading from disk/network
    - Decompression
    - Data augmentation
    - Format conversion
    """
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


def nway_pipeline(config):
    """
    Run N-way buffered pipeline and compare with different N values.
    """
    batch_size = config['batch_size']
    num_batches = config['num_batches']
    input_shape = config['input_shape']
    num_iterations = config['num_iterations']
    gpu_id = config['gpu_id']
    pin_memory = config['pin_memory']
    num_buffers = config['num_buffers']
    cpu_delay_ms = config['cpu_delay_ms']

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Input shape: {input_shape}")
    print(f"  Matmul iterations: {num_iterations}")
    print(f"  Number of buffers: {num_buffers}")
    print(f"  CPU preprocessing delay: {cpu_delay_ms} ms")
    print(f"  Pinned memory: {pin_memory}")
    print()

    # For matmul workload, output shape = input shape
    # (In production, this would be determined from model output)
    output_shape = input_shape

    # Create pipeline
    pipeline = NWayBufferedPipeline(
        batch_size, input_shape, output_shape, num_iterations, num_buffers, gpu_id, pin_memory
    )

    # Create input data
    cpu_input = torch.randn(batch_size, *input_shape, pin_memory=pin_memory)

    # Warmup
    for i in range(num_buffers):
        if i > 0:
            pipeline.swap()
        pipeline.process_batch(cpu_input, i, batch_size, "Warmup/")
    pipeline.wait_for_completion()

    # Track collected outputs for demonstration
    collected_outputs = []

    with nvtx.range(f"{num_buffers}-Way Buffered Processing"):
        for batch_idx in range(num_batches):
            # Simulate CPU preprocessing
            if cpu_delay_ms > 0:
                with nvtx.range(f"CPU_Preprocess[batch={batch_idx}]"):
                    simulate_cpu_preprocessing(cpu_delay_ms)

            # Swap buffers (except first batch)
            if batch_idx > 0:
                pipeline.swap()

            # Schedule async GPU work
            buffer_idx = pipeline.process_batch(cpu_input, batch_idx, batch_size)

            # Store metadata AFTER process_batch, BEFORE reading output
            pipeline.metadata_buffers[buffer_idx] = {
                'batch_idx': batch_idx,
                'batch_size': batch_size,
            }

            # Handle output from (N-1) iterations ago
            # For N>=3, we can safely read older results while GPU works
            if batch_idx >= num_buffers - 1:
                # Calculate which buffer to read (N-1 iterations old)
                result_idx = pipeline.get_result_buffer_idx(batch_idx)

                # Calculate metadata index for the batch we're reading
                # This is (N-1) batches earlier than current
                output_meta_idx = (batch_idx - (num_buffers - 1)) % num_buffers

                # Sync only the specific buffer we need
                pipeline.d2h_done_event[result_idx].synchronize()

                # CRITICAL: Clone output tensor (prevents async overwrites!)
                # WHY CLONE? This buffer will be reused in N iterations.
                # Without clone(), your "output" reference points to memory
                # that gets overwritten when this buffer cycles back around.
                output = pipeline.cpu_output_buffers[result_idx].clone()

                # Retrieve associated metadata
                output_meta = pipeline.metadata_buffers[output_meta_idx]

                # Now safe to use output (save to file, send to queue, etc.)
                collected_outputs.append({'metadata': output_meta, 'output': output})

                if batch_idx % 10 == 0:
                    print(f"  Collected output for batch {output_meta['batch_idx']}")

    # Wait for all work to complete
    pipeline.wait_for_completion()

    # Collect remaining (N-1) batches that are still in flight
    for i in range(1, num_buffers):
        remaining_batch_idx = num_batches - i
        if remaining_batch_idx >= 0:
            result_idx = (pipeline.current_idx - i) % num_buffers
            output_meta_idx = remaining_batch_idx % num_buffers

            pipeline.d2h_done_event[result_idx].synchronize()
            output = pipeline.cpu_output_buffers[result_idx].clone()
            output_meta = pipeline.metadata_buffers[output_meta_idx]

            collected_outputs.append({'metadata': output_meta, 'output': output})
            print(f"  Collected remaining output for batch {output_meta['batch_idx']}")

    print(f"Processed {num_batches} batches with {num_buffers}-way buffering, collected {len(collected_outputs)} outputs")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        exit(1)

    nway_pipeline(CONFIG)
