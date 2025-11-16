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

## Output Handling and Metadata Tracking

A critical production pattern: tracking which data is in which buffer and safely retrieving results.

### The Problem

With N-way buffering, multiple batches are "in flight" simultaneously:
- Buffer 0: Processing batch i+2 (H2D in progress)
- Buffer 1: Processing batch i+1 (Compute in progress)
- Buffer 2: Just finished batch i (D2H complete, ready to read)

**Challenge**: How do you know which batch is in which buffer? How do you safely retrieve results without corrupting ongoing GPU work?

### The Solution: Metadata Circular Buffer

Use a metadata buffer (same size as GPU buffers) to track what's where:

```python
# Allocate metadata buffer (matches number of GPU buffers)
self.metadata_buffers = [None] * num_buffers

# AFTER scheduling GPU work, store metadata
buffer_idx = pipeline.process_batch(...)
pipeline.metadata_buffers[buffer_idx] = {
    'batch_idx': batch_idx,
    'batch_size': actual_batch_size,
    'custom_data': your_metadata,
}

# BEFORE reading output, find which buffer is ready
if batch_idx >= num_buffers - 1:
    # Calculate buffer index for (N-1)-old batch
    output_idx = (current_idx - (num_buffers - 1)) % num_buffers

    # Sync D2H for this specific buffer
    pipeline.d2h_done_event[output_idx].synchronize()

    # Clone output tensor (critical!)
    output = pipeline.cpu_output_buffers[output_idx].clone()

    # Retrieve associated metadata
    output_meta_idx = (batch_idx - (num_buffers - 1)) % num_buffers
    output_meta = pipeline.metadata_buffers[output_meta_idx]

    # Now safe to use output
    save_to_file(output, output_meta)
```

### Why Clone?

**Critical**: Always clone output tensors:
```python
output = cpu_output_buffers[idx].clone()  # REQUIRED
```

Without `clone()`:
- Your "output" variable points directly to the buffer memory
- Buffer gets reused in N iterations
- Your "output" reference gets overwritten unexpectedly
- Results in silent data corruption

With `clone()`:
- Creates an independent copy in new memory
- Original buffer can be safely reused
- Your output remains intact

### The (N-1) Pattern

**Why read from (N-1) iterations ago?**

Timeline for N=3 buffers:
```
Iteration 0: Process batch 0 in buffer 0
Iteration 1: Process batch 1 in buffer 1
Iteration 2: Process batch 2 in buffer 2, READ batch 0 (from buffer 0)
Iteration 3: Process batch 3 in buffer 0, READ batch 1 (from buffer 1)
Iteration 4: Process batch 4 in buffer 1, READ batch 2 (from buffer 2)
```

Formula: `output_idx = (current_idx - (N - 1)) % N`

**Why (N-1)?** With N buffers in flight:
- Current buffer: H2D starting/in progress
- (N-2) older buffers: Various stages of H2D/Compute/D2H
- (N-1) oldest buffer: D2H complete, safe to read

### Which Files Demonstrate This?

All files with actual workloads now include metadata tracking:

- **File 02** (`02_double_buffering.py`): Basic pattern with N=2
  - Simple case: other buffer = `1 - current_idx`
  - Shows metadata storage and output collection
  - Demonstrates clone() necessity

- **File 05** (`05_nway_advanced.py`): Generalized pattern for any N
  - Uses `get_result_buffer_idx()` helper method
  - Handles remaining (N-1) batches after main loop
  - Shows complete output collection

- **File 06** (`06_ray_nway_pipeline.py`): Full production pattern
  - Complete pattern with Ray + N-way buffering
  - NVTX annotations for profiling
  - Detailed comments explaining timeline
  - Production-ready implementation

### Common Pitfalls

**❌ Forgetting to clone**:
```python
output = pipeline.cpu_output_buffers[idx]  # WRONG - no clone()
results.append(output)  # All results will show the same data!
```

**✓ Correct approach**:
```python
output = pipeline.cpu_output_buffers[idx].clone()  # RIGHT
results.append(output)  # Each result is independent
```

**❌ Wrong metadata index**:
```python
# Wrong: using buffer_idx for old batch
output_meta = metadata_buffers[output_idx]  # Might be wrong batch!
```

**✓ Correct approach**:
```python
# Right: calculate metadata index based on batch number
output_meta_idx = (batch_idx - (num_buffers - 1)) % num_buffers
output_meta = metadata_buffers[output_meta_idx]
```

**❌ Not synchronizing**:
```python
# Wrong: reading without sync
output = pipeline.cpu_output_buffers[idx].clone()  # Might be incomplete!
```

**✓ Correct approach**:
```python
# Right: sync D2H event first
pipeline.d2h_done_event[idx].synchronize()
output = pipeline.cpu_output_buffers[idx].clone()  # Now safe
```

### Pattern Summary

The complete pattern in order:
1. **Schedule GPU work**: `buffer_idx = pipeline.process_batch(...)`
2. **Store metadata**: `metadata_buffers[buffer_idx] = {...}`
3. **Check if output ready**: `if batch_idx >= num_buffers - 1:`
4. **Calculate output buffer**: `output_idx = (current_idx - (N-1)) % N`
5. **Calculate metadata index**: `meta_idx = (batch_idx - (N-1)) % N`
6. **Synchronize**: `d2h_done_event[output_idx].synchronize()`
7. **Clone output**: `output = cpu_output_buffers[output_idx].clone()`
8. **Retrieve metadata**: `meta = metadata_buffers[meta_idx]`
9. **Use result**: `process_output(output, meta)`

This pattern matches the production PeakNet pipeline implementation.

## Profiling with NVIDIA Nsight Systems

Profiling differs between standalone GPU files and Ray-based files.

### Standalone Files (01, 02, 05)

These files run directly without Ray, so use standard nsys profiling:

1. **Run with profiling**:
   ```bash
   nsys profile -o output.nsys-rep --trace=cuda,nvtx python your_file.py
   ```

2. **View results**:
   ```bash
   nsight-sys output.nsys-rep  # GUI
   ```

**Example**:
```bash
nsys profile -o baseline.nsys-rep --trace=cuda,nvtx python 01_sequential_baseline.py
nsys profile -o double.nsys-rep --trace=cuda,nvtx python 02_double_buffering.py
nsight-sys baseline.nsys-rep double.nsys-rep  # Compare both
```

### Ray Files (04, 06) - Different Approach!

**IMPORTANT**: Ray spawns worker processes for actors, so profiling works differently.

**DO NOT** use command-line nsys for Ray files:
```bash
# WRONG - only profiles the driver, not GPU workers!
nsys profile python 04_ray_gpu_double_buffer.py
```

**CORRECT** approach - use `runtime_env` configuration:

1. **Enable profiling in CONFIG**:
   ```python
   CONFIG = {
       'enable_profiling': True,  # Enables runtime_env profiling
       ...
   }
   ```

2. **Run normally** (no nsys wrapper):
   ```bash
   python 04_ray_gpu_double_buffer.py
   ```

3. **Profiling files auto-generated**:
   ```bash
   # Ray creates .nsys-rep files per worker
   ls /tmp/ray/session_*/logs/nsight/*.nsys-rep
   ```

4. **View results**:
   ```bash
   nsys-ui /tmp/ray/session_latest/logs/nsight/*.nsys-rep
   ```

**How it works**:
- Ray actors use `runtime_env={"nsight": {...}}` decorator
- Ray automatically wraps each worker process with nsys
- Each GPU actor gets its own profiling file
- NVTX annotations include actor IDs (Worker0/, Worker1/)

**Comparison Table**:

| File Type | Profiling Method | Command |
|-----------|------------------|---------|
| 01-02, 05 (standalone) | Command-line nsys | `nsys profile python file.py` |
| 03 (Ray CPU-only) | No profiling needed | N/A |
| 04, 06 (Ray + GPU) | `runtime_env` config | Set `enable_profiling=True` in CONFIG |

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
