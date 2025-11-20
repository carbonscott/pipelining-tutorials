#!/usr/bin/env python3
"""
01_sequential_baseline.py - Sequential GPU Processing (No Concurrency)

PURPOSE:
Establish baseline performance and demonstrate GPU idle time problem.

PROFILING:
nsys profile -o baseline.nsys-rep --trace=cuda,nvtx python 01_sequential_baseline.py

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

    print(f"Processed {num_batches} batches")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        exit(1)

    sequential_pipeline(CONFIG)
