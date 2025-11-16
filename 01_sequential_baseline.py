#!/usr/bin/env python3
"""
01_sequential_baseline.py - Sequential GPU Processing (No Concurrency)

PURPOSE:
Establish baseline performance and demonstrate GPU idle time problem.

LEARNING GOALS:
- Understand H2D, Compute, D2H pipeline stages
- Measure GPU idle time during data transfers
- Learn basic NVTX profiling annotations

WHAT TO OBSERVE:
- Run with: python 01_sequential_baseline.py
- Profile with: nsys profile -o baseline.nsys-rep --trace=cuda,nvtx python 01_sequential_baseline.py
- In nsys timeline: Notice gaps between compute kernels (GPU idle during transfers)

CONFIGURATION:
Edit CONFIG dict below to adjust:
- batch_size: Number of samples per batch (affects I/O time)
- num_batches: Total batches to process
- input_shape: Tensor dimensions (C, H, W)
- num_iterations: Matmul loop count (affects compute time)

NEXT: 02_double_buffering.py to eliminate idle time
"""

import torch
import torch.cuda.nvtx as nvtx
import time

# Configuration
CONFIG = {
    'batch_size': 8,
    'num_batches': 40,
    'input_shape': (1, 512, 512),  # (C, H, W)
    'num_iterations': 10,  # Controls compute time (~5-10ms per iteration on V100)
    'gpu_id': 0,
}


def create_matmul_workload(num_iterations):
    """
    Creates a controllable GPU compute workload using matrix multiplication.

    Args:
        num_iterations: Number of matmul loops (controls compute time)

    Returns:
        Callable that takes (x: Tensor) -> Tensor
    """
    def workload(x):
        # x shape: (batch, channels, height, width)
        batch_size = x.shape[0]
        features = x.view(batch_size, -1)

        feat_dim = features.shape[1]
        # Create weight on same device as input
        weight = torch.randn(feat_dim, feat_dim, device=x.device, dtype=x.dtype)

        result = features
        for i in range(num_iterations):
            result = torch.matmul(result, weight)
            result = torch.relu(result)  # Prevent numerical issues

        return result.view_as(x)

    return workload


def sequential_pipeline(config):
    """
    Sequential processing: H2D -> Compute -> D2H for each batch.

    This is the naive approach that causes GPU idle time.
    """
    batch_size = config['batch_size']
    num_batches = config['num_batches']
    input_shape = config['input_shape']
    num_iterations = config['num_iterations']
    gpu_id = config['gpu_id']

    device = torch.device(f'cuda:{gpu_id}')

    # Create compute workload
    compute = create_matmul_workload(num_iterations)

    # Allocate buffers
    # CPU input buffer (not pinned for now - we'll show pinned memory in file 02)
    cpu_input = torch.randn(batch_size, *input_shape)
    # GPU buffers
    gpu_input = torch.empty(batch_size, *input_shape, device=device)
    gpu_output = torch.empty(batch_size, *input_shape, device=device)
    # CPU output buffer
    cpu_output = torch.empty(batch_size, *input_shape)

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Input shape: {input_shape}")
    print(f"  Matmul iterations: {num_iterations}")
    print(f"  GPU device: {gpu_id}")
    print()

    # Warmup
    for _ in range(3):
        gpu_input.copy_(cpu_input)
        gpu_output = compute(gpu_input)
        cpu_output.copy_(gpu_output)
    torch.cuda.synchronize()

    print("Starting sequential processing...")
    start_time = time.time()

    with nvtx.range("Sequential Processing"):
        for batch_idx in range(num_batches):
            with nvtx.range(f"Batch {batch_idx}"):
                # H2D Transfer
                with nvtx.range("H2D Transfer"):
                    gpu_input.copy_(cpu_input)

                # Compute
                with nvtx.range("Compute"):
                    gpu_output = compute(gpu_input)

                # D2H Transfer
                with nvtx.range("D2H Transfer"):
                    cpu_output.copy_(gpu_output)

    # Wait for all GPU work to complete
    torch.cuda.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = num_batches * batch_size / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Time per batch: {elapsed/num_batches*1000:.2f} ms")
    print()
    print("OBSERVATION: In nsys timeline, you'll see gaps between compute kernels.")
    print("These gaps are when the GPU is IDLE during H2D and D2H transfers.")
    print()
    print("NEXT: Run 02_double_buffering.py to see how we can eliminate this idle time")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        exit(1)

    sequential_pipeline(CONFIG)
