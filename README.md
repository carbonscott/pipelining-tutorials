# Minimal ML Inference Pipeline Tutorial

Standalone, minimal Python examples teaching concurrent GPU pipelining and Ray orchestration for high-throughput ML inference.

## Overview

This tutorial series teaches you how to build production-ready, high-performance GPU inference pipelines through 6 progressive examples. Each file is self-contained, GitHub gist-ready, and focuses on core concepts without excessive logging or complexity.

**Total learning time**: 2-3 hours
**Prerequisite knowledge**: Basic PyTorch and Python

## What You'll Learn

1. **GPU Concurrent Pipelining** (Files 01-02)
   - Why GPUs sit idle in naive implementations
   - Double-buffering pattern for maximum GPU utilization
   - CUDA streams and event-based synchronization
   - NVTX profiling for performance analysis

2. **Ray Distributed Computing** (File 03)
   - Ray object store and zero-copy data sharing
   - Producer-consumer patterns with actors
   - Graceful coordination and shutdown

3. **Integration** (File 04)
   - Combining Ray + GPU double buffering
   - Multi-GPU scaling strategies
   - Complete pipeline architecture

4. **Advanced Topics** (Files 05-06)
   - Generalized N-way buffering (N>2)
   - When to use triple/quad buffering
   - Production-ready pattern with full configurability

## Prerequisites

### Required
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Ray 2.0+
- At least one NVIDIA GPU

### Installation

```bash
# PyTorch (visit pytorch.org for your specific CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ray
pip install ray

# NumPy (usually included with PyTorch)
pip install numpy
```

### Verify Setup

```python
import torch
import ray

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Ray version: {ray.__version__}")
```

## Tutorial Structure

### Core Sequence (Must Complete in Order)

#### File 01: Sequential Baseline (~15 min)
**`01_sequential_baseline.py`**

Learn the problem we're solving: GPU idle time.

- Sequential H2D → Compute → D2H processing
- Timing measurements and performance baseline
- Basic NVTX annotations for profiling
- Configurable matmul workload

**Run**:
```bash
python 01_sequential_baseline.py
nsys profile -o baseline.nsys-rep --trace=cuda,nvtx python 01_sequential_baseline.py
```

**Key Takeaway**: "The GPU is idle during data transfers - we're wasting compute resources."

---

#### File 02: Double Buffering (~25 min)
**`02_double_buffering.py`**

Solve the idle time problem with double buffering.

- `DoubleBufferedPipeline` class with `process_batch()` interface
- 2 buffers, 3 streams (H2D, Compute, D2H)
- Event-based synchronization
- Pinned memory for async transfers
- Fine-grained NVTX: annotate each stage with buffer index

**Run**:
```bash
python 02_double_buffering.py
nsys profile -o double.nsys-rep --trace=cuda,nvtx python 02_double_buffering.py
```

**Key Takeaway**: "Two buffers keep the GPU continuously busy - H2D, Compute, D2H all overlap."

**Compare**: Open both `baseline.nsys-rep` and `double.nsys-rep` in Nsight Systems GUI to see the difference.

---

#### File 03: Ray Producer-Consumer (~20 min)
**`03_ray_producer_consumer.py`**

Learn Ray orchestration without GPU complexity.

- Ray initialization and resource discovery
- Producer and Consumer actors
- Ray Queue for communication
- Object store zero-copy pattern
- Graceful shutdown
- CPU-only workload (no CUDA)

**Run**:
```bash
python 03_ray_producer_consumer.py
# Watch Ray dashboard: http://127.0.0.1:8265
```

**Key Takeaway**: "Ray's object store enables zero-copy data sharing across distributed workers."

---

#### File 04: Ray + GPU Double Buffer (~30 min)
**`04_ray_gpu_double_buffer.py`**

Complete pattern: Ray orchestration + GPU pipelining.

- Combines Files 02 and 03
- `GPUWorkerActor` with double-buffered pipeline
- Multi-GPU support via `num_gpus=1`
- NVTX with actor_id for multi-GPU profiling
- Throughput scaling demonstration

**Run**:
```bash
python 04_ray_gpu_double_buffer.py
nsys profile -o ray_gpu.nsys-rep --trace=cuda,nvtx python 04_ray_gpu_double_buffer.py
```

**Key Takeaway**: "Ray scales across GPUs, double-buffering optimizes each GPU."

---

### Advanced Topics (Optional)

#### File 05: N-Way Buffering (~30 min)
**`05_nway_advanced.py`**

Generalized N-way buffering for N>2.

- `NWayBufferedPipeline` with configurable `num_buffers`
- Circular buffer rotation
- When N>2 helps (CPU preprocessing scenarios)
- Performance comparison across N values
- Simulated CPU delay to demonstrate benefits

**Run**:
```bash
python 05_nway_advanced.py
# Try different configurations in CONFIG dict
```

**Key Takeaway**: "Use N>2 when CPU has work to do during GPU processing."

**Experiment**:
- `num_buffers=2, cpu_delay_ms=0`: Baseline
- `num_buffers=3, cpu_delay_ms=10`: Triple buffering shines
- `num_buffers=4`: Diminishing returns

---

#### File 06: Production Pattern (~30 min)
**`06_ray_nway_pipeline.py`**

Production-ready: Ray + N-way pipeline.

- Complete architecture with full configurability
- Independent control: `num_gpu_actors` × `num_buffers_per_actor`
- Horizontal (GPUs) and vertical (buffers) scaling
- Performance analysis and tuning recommendations
- Pattern used in production PeakNet pipeline

**Run**:
```bash
python 06_ray_nway_pipeline.py
nsys profile -o ray_nway.nsys-rep --trace=cuda,nvtx python 06_ray_nway_pipeline.py
```

**Key Takeaway**: "This is the production pattern - scale horizontally first, then vertically if needed."

---

## Configuration

All files use a `CONFIG` dictionary at the top for easy modification:

```python
CONFIG = {
    'batch_size': 8,           # Samples per batch
    'num_batches': 40,         # Total batches to process
    'input_shape': (1, 512, 512),  # (C, H, W)
    'num_iterations': 10,      # Controls compute time
    'gpu_id': 0,               # GPU device ID
    'pin_memory': True,        # Use pinned CPU memory
}
```

**Tuning the workload**:
- Increase `num_iterations` for heavier compute (more GPU time)
- Increase `batch_size` or `input_shape` for more I/O (memory transfers)
- Balance to match your real workload characteristics

## Profiling with NVIDIA Nsight Systems

### Basic Workflow

1. **Run with profiling**:
   ```bash
   nsys profile -o output.nsys-rep --trace=cuda,nvtx python your_file.py
   ```

2. **View results**:
   ```bash
   nsight-sys output.nsys-rep  # GUI
   ```

### What to Look For

**Sequential (File 01)**:
- GPU timeline: Compute kernels with gaps between them
- Gaps = idle time during H2D and D2H transfers
- Low GPU utilization (~30-40%)

**Double Buffered (File 02)**:
- H2D, Compute, D2H overlap on timeline
- Buffer indices (buf=0, buf=1) alternate
- High GPU utilization (~80-90%)
- Minimal idle time

**Ray + GPU (File 04)**:
- Multiple GPU streams (one per actor)
- Each shows double-buffered pattern
- Filter by "Worker0/", "Worker1/" to isolate actors

**N-Way (Files 05-06)**:
- N buffers cycling through pipeline
- More concurrent operations visible
- Optimal N shows maximum utilization

### NVTX Annotations

Each file demonstrates NVTX annotation strategies:

**File 01**: Basic structure
```python
with nvtx.range("Process All Batches"):
    for batch_idx in range(num_batches):
        with nvtx.range(f"Batch {batch_idx}"):
            # processing
```

**File 02**: Fine-grained with buffer index
```python
with nvtx.range(f"H2D[buf={buffer_idx}]"):
    # H2D transfer
with nvtx.range(f"Compute[buf={buffer_idx}]"):
    # Compute
```

**Files 04-06**: Multi-actor with IDs
```python
nvtx_prefix = f"Worker{actor_id}/"
with nvtx.range(f"{nvtx_prefix}Batch{batch_idx}[buf={buffer_idx}]"):
    # processing
```

### Profiling Tips

- Start with small datasets for quick iteration
- Profile both sequential and concurrent versions
- Compare side-by-side in Nsight Systems
- Use NVTX to navigate complex timelines
- Look for unexpected synchronization or idle time

## Common Issues & Solutions

### GPU Out of Memory

Reduce memory usage:
```python
CONFIG = {
    'batch_size': 4,           # Reduce from 8
    'input_shape': (1, 256, 256),  # Reduce from 512x512
    'num_buffers': 2,          # Reduce from 3
}
```

### Ray Initialization Errors

Check GPU visibility:
```python
import torch
print(torch.cuda.device_count())  # Should match expected GPUs
```

Initialize Ray explicitly:
```python
ray.init(num_gpus=2)  # Specify GPU count
```

### Slow Performance

Verify GPU is being used:
```python
# Should print 'cuda', not 'cpu'
print(next(model.parameters()).device)
```

Check pinned memory is enabled:
```python
CONFIG = {
    'pin_memory': True,  # Faster async transfers
}
```

### Import Errors

All files are self-contained - no cross-file imports needed. If you see import errors, check:
- PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Ray installation: `python -c "import ray; print(ray.__version__)"`
- CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

## Performance Expectations

Approximate speedups (compared to sequential baseline):

| Configuration | Expected Speedup | Use Case |
|--------------|------------------|----------|
| Sequential (File 01) | 1.0× (baseline) | Reference only |
| Double Buffer (File 02) | 2.0-3.0× | Most common case |
| Ray 2 GPUs (File 04) | 4.0-5.0× | Linear scaling |
| N-way N=3 (File 05) | 2.5-3.5× | CPU-heavy preprocessing |
| Ray 2×3 (File 06) | 5.0-7.0× | Production workload |

Actual speedups depend on:
- GPU compute time vs transfer time ratio
- CPU preprocessing overhead
- Queue and memory bandwidth
- Model characteristics

## When to Use Which Pattern

### Double Buffering (File 02)
✅ **Use when**:
- Single GPU inference
- Minimal CPU preprocessing
- Batch processing workloads
- ~80% GPU utilization is sufficient

### Ray + Double Buffer (File 04)
✅ **Use when**:
- Multiple GPUs available
- Need horizontal scaling
- Batch processing with distribution
- Want automatic load balancing

### N-Way Buffering (File 05)
✅ **Use when**:
- CPU preprocessing takes significant time
- Memory permits extra buffers
- GPU is the bottleneck
- Need fine-tuned per-GPU optimization

### Ray + N-Way (File 06)
✅ **Use when**:
- Production deployment
- Multiple GPUs + CPU overhead
- Need both horizontal and vertical scaling
- Maximum throughput required
- This is the full production pattern

## Next Steps

After completing this tutorial:

1. **Replace dummy workload** with your ML model:
   ```python
   # Instead of create_matmul_workload()
   model = YourModel()
   model.eval()
   model.to(device)
   ```

2. **Add real data source**:
   ```python
   # Instead of random data
   dataset = YourDataset()
   dataloader = DataLoader(dataset)
   ```

3. **Add result processing**:
   ```python
   # After pipeline.process_batch()
   results = pipeline.cpu_output_buffers[result_idx]
   save_results(results)
   ```

4. **Monitor and tune**:
   - Profile with nsys regularly
   - Adjust `num_buffers` based on CPU overhead
   - Scale GPUs based on throughput needs
   - Monitor Ray dashboard for bottlenecks

## Reference Implementation

This pattern is used in production:
- **PeakNet Pipeline**: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray`
- Real-world crystallography ML inference at SLAC National Lab
- Processes streaming X-ray diffraction data in real-time
- Scales across multiple GPUs with N-way buffering

See the full implementation for advanced features:
- Streaming data integration
- Cheetah format integration
- CXI file output
- PSANA integration
- Production monitoring

## Resources

### Documentation
- PyTorch CUDA Streams: [pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- Ray Core: [docs.ray.io/en/latest/ray-core](https://docs.ray.io/en/latest/ray-core)
- NVIDIA Nsight Systems: [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)

### Related Tutorials
- Full interactive tutorials: `$PIPELINE_DEV_DIR/tutorials/`
- Ray patterns: `$PIPELINE_DEV_DIR/tutorials/ray/`
- Concurrent pipeline: `$PIPELINE_DEV_DIR/tutorials/concurrent-pipeline/`

## License

These teaching examples are part of the peaknet-pipeline-ray project and are provided for educational purposes.

## Contributing

Found a bug or have suggestions? These are teaching resources - feedback is welcome!

---

**Happy learning!** Start with `01_sequential_baseline.py` and work through the series in order.

```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference
python 01_sequential_baseline.py
```
