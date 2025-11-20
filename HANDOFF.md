# GPU Pipelining Tutorial Repository - Developer Handoff Document

**Repository**: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference/`
**Production Codebase**: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/`
**Target Audience**: Advanced CUDA developer familiar with streams/events
**Last Updated**: 2025-11-19 (Commit 61c0903)

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Production Code Mapping](#2-production-code-mapping)
3. [Design Patterns and Conventions](#3-design-patterns-and-conventions)
4. [Recent Changes Documentation](#4-recent-changes-documentation)
5. [Testing and Validation](#5-testing-and-validation)
6. [Common Pitfalls](#6-common-pitfalls)
7. [Key Implementation Patterns](#7-key-implementation-patterns)
8. [Production Command Examples](#8-production-command-examples)
9. [Critical Knowledge for Advanced Developers](#9-critical-knowledge-for-advanced-developers)
10. [Testing Strategy](#10-testing-strategy)
11. [Future Maintenance Guidance](#11-future-maintenance-guidance)
12. [Quick Reference](#12-quick-reference)
13. [Appendix A: Complete Event Pattern Example](#appendix-a-complete-event-pattern-example)
14. [Appendix B: Alignment Commit Diff Summary](#appendix-b-alignment-commit-diff-summary)

---

## 1. Repository Structure

### 1.1 Tutorial Repository Overview

**Git Repository**:
- Remote: `git@github.com:carbonscott/pipelining-tutorials.git`
- Branch: `main`
- Location: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference/`

**Tutorial Files** (2,085 total lines):

| File | Lines | Purpose | Key Concepts |
|------|-------|---------|--------------|
| `01_sequential_baseline.py` | 153 | Baseline performance, demonstrates GPU idle time | H2D/Compute/D2H stages, NVTX basics |
| `02_double_buffering.py` | 328 | Double-buffering pattern (2 buffers, 3 streams) | Event synchronization, pinned memory, metadata tracking |
| `03_ray_producer_consumer.py` | 251 | Ray orchestration without GPU complexity | Ray Queue, actors, object store, zero-copy |
| `04_ray_gpu_double_buffer.py` | 428 | Ray + GPU double buffering integration | Multi-GPU scaling, runtime_env profiling |
| `05_nway_advanced.py` | 410 | Generalized N-way buffering (N>2) | Circular rotation, when to use N>2 |
| `06_ray_nway_pipeline.py` | 515 | Production pattern: Ray + N-way | Complete architecture, horizontal/vertical scaling |

**Support Files**:
- `README.md` (692 lines): Comprehensive tutorial guide with learning objectives, configuration examples, profiling instructions, and common pitfalls

**Recent Commit History**:
```
61c0903 Align tutorials with production patterns (2025-11-19)
087096d Add output handling with metadata tracking to tutorials
4f478f2 first commit
```

### 1.2 Production Repository Structure

**Git Repository**:
- Remote: `git@github.com:carbonscott/peaknet-pipeline-ray.git`
- Current Branch: `dev/min`
- Main Branch: (empty - use dev branches)
- Location: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/`

**Key Production Files**:
- **Core Pipeline**: `peaknet_pipeline_ray/core/peaknet_pipeline.py` (617 lines)
  - Class: `NWayBufferedPipeline` (lines 191-414)
  - Source of truth for all patterns
- **Ray Actor**: `peaknet_pipeline_ray/core/peaknet_ray_pipeline_actor.py` (715 lines)
  - Class: `PeakNetPipelineActorBase` (wraps NWayBufferedPipeline)
- **Orchestration**: `peaknet_pipeline_ray/pipeline.py`
- **Documentation**: `RAY_FEATURES_ANALYSIS.md`, `RAY_ANALYSIS_INDEX.md`, `RAY_FEATURES_QUICK_REFERENCE.md`

**Example Configurations**: `examples/configs/`
- `peaknet.yaml`, `peaknet-random.yaml`, `peaknet-socket.yaml`
- `peaknet-with-cheetah-integration.yaml`

---

## 2. Production Code Mapping

### 2.1 Primary Production Reference

**Production File**: `peaknet_pipeline_ray/core/peaknet_pipeline.py`

**Class**: `NWayBufferedPipeline` (lines 191-414)
- This is the canonical implementation that tutorials are based on
- Supports configurable N-way buffering (default N=2 for backward compatibility)
- Used in production PeakNet crystallography ML inference at SLAC

### 2.2 Tutorial to Production Mapping

| Tutorial File | Production Sections | Patterns Demonstrated |
|--------------|-------------------|---------------------|
| **01_sequential_baseline.py** | Baseline reference | Problem statement: GPU idle time |
| **02_double_buffering.py** | `NWayBufferedPipeline` (N=2) | Event API, event priming, output shape separation |
| **03_ray_producer_consumer.py** | Ray actors, Queue patterns | Ray fundamentals (no GPU) |
| **04_ray_gpu_double_buffer.py** | `NWayBufferedPipeline` + Ray actors | Integration pattern, runtime_env profiling |
| **05_nway_advanced.py** | `NWayBufferedPipeline` (N≥2) | Circular rotation, configurable N |
| **06_ray_nway_pipeline.py** | Complete production pattern | Full system: Producer → Queue → N actors × M buffers |

### 2.3 Key Production Methods

**Pipeline Initialization** (`__init__`, lines 204-273):
- Separate `input_shape` and `output_shape` parameters (line 204)
- Event priming pattern (lines 228-231)
- Buffer allocation using appropriate shapes (lines 233-249)

**Pipeline Stages**:
- `_h2d_transfer` (lines 288-309): Uses `stream.record_event(event)` API
- `_compute_workload` (lines 311-382): Model inference or no-op
- `_d2h_transfer` (lines 384-402): Device to host transfer
- `process_batch` (lines 404-408): Public API orchestrating all stages

---

## 3. Design Patterns and Conventions

### 3.1 Event Recording API Pattern

**CRITICAL**: The correct API is `stream.record_event(event)`, NOT `event.record(stream)`

**Production Implementation** (peaknet_pipeline.py):
```python
# Line 309 - H2D stage
self.h2d_stream.record_event(h2d_event)

# Line 382 - Compute stage
self.compute_stream.record_event(compute_event)

# Line 402 - D2H stage
self.d2h_stream.record_event(d2h_event)
```

**Why This Matters**:
- `stream.record_event(event)` is the official PyTorch `Stream.record_event()` method
- `event.record(stream)` is a legacy/alternative API that works but may have different semantics
- Production code uses the stream-centric API for clarity and consistency

**Alignment Changes** (commit 61c0903):
- Changed 24 occurrences across 4 tutorial files
- Pattern: `event.record(stream)` → `stream.record_event(event)`
- Locations: `_h2d_transfer`, `_compute_workload`, `_d2h_transfer` methods

### 3.2 Event Priming Pattern

**CRITICAL**: Events must be primed after creation to prevent first-use deadlock

**Production Implementation** (peaknet_pipeline.py, lines 228-231):
```python
# Prime all events so wait_event() never deadlocks on first use
for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
    for ev in events:
        ev.record()  # Record on default stream makes them signaled immediately
```

**Why This Is Critical**:
- Newly created events are in "unsignaled" state
- First `wait_event()` call will deadlock if event hasn't been recorded
- Priming with `ev.record()` on default stream signals them immediately
- Ensures first iteration doesn't block waiting for non-existent previous work

**Before Alignment** (incorrect pattern):
```python
# Old approach - more verbose, explicit stream
for i in range(2):
    self.h2d_done_event[i].record(torch.cuda.current_stream())
    self.compute_done_event[i].record(torch.cuda.current_stream())
    self.d2h_done_event[i].record(torch.cuda.current_stream())
```

**After Alignment** (production pattern):
```python
# Cleaner, more maintainable for any N
for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
    for ev in events:
        ev.record()  # Simpler, works for any num_buffers
```

**Alignment Changes** (commit 61c0903):
- Added to 4 tutorial files (02, 04, 05, 06)
- Each file: 1 priming section added after event creation

### 3.3 Output Shape Separation Pattern

**CRITICAL**: Always maintain separate `input_shape` and `output_shape` parameters

**Production Implementation** (peaknet_pipeline.py, lines 204-249):
```python
def __init__(self, model, batch_size, input_shape, output_shape, gpu_id, ...):
    self.input_shape = input_shape    # e.g., (1, 512, 512) - raw detector image
    self.output_shape = output_shape  # e.g., (3, 512, 512) - segmentation map

    # GPU input buffers use input_shape
    self.gpu_input_buffers = [
        torch.zeros(batch_size, *input_shape, device=f'cuda:{gpu_id}')
        for _ in range(num_buffers)
    ]

    # GPU output buffers use output_shape (may differ from input!)
    self.gpu_output_buffers = [
        torch.zeros(batch_size, *output_shape, device=f'cuda:{gpu_id}')
        for _ in range(num_buffers)
    ]

    # CPU output buffers also use output_shape
    self.cpu_output_buffers = [
        torch.empty((batch_size, *output_shape), pin_memory=pin_memory)
        for _ in range(num_buffers)
    ]
```

**Why This Matters**:
- In PeakNet: Input is (1, H, W) detector image, output is (3, H, W) segmentation map
- In matmul workload: Input and output happen to be the same shape
- Hardcoding assumption `output_shape = input_shape` limits generalizability
- Explicit separation makes code adaptable to any model architecture

**Tutorial Handling** (matmul workload):
```python
# For matmul workload, output shape = input shape
# (In production, this would be determined from model output)
output_shape = input_shape

pipeline = DoubleBufferedPipeline(batch_size, input_shape, output_shape, ...)
```

**Alignment Changes** (commit 61c0903):
- Added `output_shape` parameter to `__init__` in all 4 tutorial files
- Updated buffer allocations: 12 changes total (gpu_output_buffers, cpu_output_buffers)
- Added clarifying comments explaining matmul-specific assumption

### 3.4 NVTX Annotation Conventions

**Strategy**: Include buffer index in annotations to distinguish overlapping operations

**Production Pattern** (peaknet_pipeline.py):
```python
# Top-level batch annotation
with nvtx.range(f"{nvtx_prefix}_h2d_batch_{batch_idx}"):
    # H2D work...

with nvtx.range(f"{nvtx_prefix}_compute_batch_{batch_idx}"):
    # Compute work...
```

**Tutorial Pattern** (includes buffer index for clarity):
```python
# File 02, 04, 05, 06:
with nvtx.range(f"{nvtx_prefix}Batch{batch_idx}[buf={buffer_idx}]"):
    self._h2d_transfer(...)
    self._compute_workload(...)
    self._d2h_transfer(...)

# Stage-level annotations
with nvtx.range(f"H2D[buf={buffer_idx}]"):
    # H2D work for specific buffer
```

**Multi-Actor Pattern** (file 04, 06):
```python
nvtx_prefix = f"Worker{self.actor_id}/"
# Results in: "Worker0/Batch5[buf=1]", "Worker1/Batch7[buf=0]"
```

**Profiling Benefit**:
- In nsys timeline, you can see exactly which buffer is being processed
- For N=2: Alternating `buf=0` and `buf=1` annotations show double buffering
- For multi-GPU: `Worker0/` and `Worker1/` prefixes distinguish actors

### 3.5 Buffer Naming Conventions

**Consistent Naming** (from production):
```python
self.gpu_input_buffers   # List of GPU tensors for input (size: num_buffers)
self.gpu_output_buffers  # List of GPU tensors for output (size: num_buffers)
self.cpu_output_buffers  # List of pinned CPU tensors for output (size: num_buffers)

self.h2d_stream          # Host-to-device transfer stream
self.compute_stream      # GPU compute stream
self.d2h_stream          # Device-to-host transfer stream

self.h2d_done_event      # List of events signaling H2D completion (size: num_buffers)
self.compute_done_event  # List of events signaling compute completion
self.d2h_done_event      # List of events signaling D2H completion

self.current_idx         # Current buffer index (0 to num_buffers-1)
```

**Indexing Pattern**:
- All lists sized by `num_buffers` (2 for double, 3 for triple, etc.)
- Per-buffer resources: Each buffer has dedicated events in all 3 lists
- Current buffer accessed via `self.current_idx`

---

## 4. Recent Changes Documentation

### 4.1 Commit 61c0903 Summary

**Commit**: `61c0903ec411d19df700314650c7baf0f2c6a632`
**Date**: 2025-11-19 17:18:29 -0800
**Author**: Cong Wang
**Message**: "Align tutorials with production patterns"

**Files Modified**: 4 tutorial files (02, 04, 05, 06)
**Total Changes**: 40 changes (70 insertions, 51 deletions)

### 4.2 Change Categories

**1. Event Recording API** (24 changes):
- **What**: Convert `event.record(stream)` → `stream.record_event(event)`
- **Where**: `_h2d_transfer`, `_compute_workload`, `_d2h_transfer` methods in all 4 files
- **Why**: Match production PyTorch Stream.record_event() API
- **Example**:
  ```python
  # Before:
  self.h2d_done_event[buffer_idx].record(self.h2d_stream)

  # After:
  self.h2d_stream.record_event(self.h2d_done_event[buffer_idx])
  ```

**2. Event Priming** (4 additions):
- **What**: Added production priming pattern after event creation
- **Where**: Pipeline `__init__` in all 4 files
- **Why**: Prevent deadlocks on first wait_event() call
- **Example**:
  ```python
  # Added:
  # Prime all events so wait_event() never deadlocks on first use
  for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
      for ev in events:
          ev.record()  # Record on default stream makes them signaled immediately
  ```

**3. Output Shape Separation** (12 changes):
- **What**: Add `output_shape` parameter and use it for output buffers
- **Where**: Pipeline `__init__` signature and buffer allocations in all 4 files
- **Why**: Maintain separation for generalizability (production has different input/output shapes)
- **Example**:
  ```python
  # Before:
  def __init__(self, batch_size, input_shape, num_iterations, ...):
      self.gpu_output_buffers = [torch.empty(batch_size, *input_shape, ...) for _ in range(2)]

  # After:
  def __init__(self, batch_size, input_shape, output_shape, num_iterations, ...):
      self.gpu_output_buffers = [torch.empty(batch_size, *output_shape, ...) for _ in range(2)]
  ```

### 4.3 Per-File Breakdown

| File | Event API | Event Priming | Output Shape | Total Changes |
|------|-----------|---------------|--------------|---------------|
| 02_double_buffering.py | 3 | 1 section | 2 buffer + 3 params | 9 |
| 04_ray_gpu_double_buffer.py | 3 | 1 section | 2 buffer + 3 params + actor | 10 |
| 05_nway_advanced.py | 3 | 1 section | 2 buffer + 3 params | 9 |
| 06_ray_nway_pipeline.py | 3 | 1 section | 2 buffer + 3 params + actor | 10 |
| **Total** | **12** | **4** | **24** | **40** |

### 4.4 What Was Wrong Before vs. Correct Now

**Before Alignment**:

1. **Inconsistent Event API**: Tutorials used `event.record(stream)` while production used `stream.record_event(event)`
   - Problem: Different API style creates confusion when comparing to production
   - Impact: Developers might not recognize the canonical pattern

2. **Different Priming Style**: Tutorials used explicit loop with `torch.cuda.current_stream()`
   - Problem: More verbose, harder to maintain when `num_buffers` changes
   - Impact: Less clear intent, doesn't scale cleanly to N-way buffering

3. **Hardcoded Output Shape**: Tutorials assumed `output_shape = input_shape` implicitly
   - Problem: Works for matmul but misleading for actual PeakNet use case
   - Impact: Developer might not realize PeakNet has different input/output shapes

**After Alignment**:

1. **Consistent Event API**: All code uses `stream.record_event(event)`
   - Benefit: Matches production, clearer ownership (stream records event)
   - Easier transition: Copy-paste from tutorials to production works

2. **Production Priming Pattern**: Compact, scalable nested loop
   - Benefit: Works for any N, clear intent ("prime all events")
   - Maintainability: Adding new event types is one-line change

3. **Explicit Shape Separation**: Always pass both input_shape and output_shape
   - Benefit: Matches production signature, makes shape transformation explicit
   - Clarity: Comments explain when they're equal (matmul) vs. different (PeakNet)

---

## 5. Testing and Validation

### 5.1 Verifying Tutorials Work

**Standalone Files (01, 02, 05)** - No Ray dependency:
```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference

# Basic execution
python 01_sequential_baseline.py
python 02_double_buffering.py
python 05_nway_advanced.py

# With profiling
nsys profile -o baseline.nsys-rep --trace=cuda,nvtx python 01_sequential_baseline.py
nsys profile -o double.nsys-rep --trace=cuda,nvtx python 02_double_buffering.py
nsys profile -o nway.nsys-rep --trace=cuda,nvtx python 05_nway_advanced.py

# View profiles
nsight-sys baseline.nsys-rep double.nsys-rep  # Compare side-by-side
```

**Ray Files (03, 04, 06)** - Special profiling approach:
```bash
# File 03 - Ray basics (no profiling needed, CPU-only)
python 03_ray_producer_consumer.py

# Files 04, 06 - DO NOT use command-line nsys!
# Instead, edit CONFIG in the file:
# CONFIG = {
#     'enable_profiling': True,  # Enable runtime_env profiling
#     ...
# }

python 04_ray_gpu_double_buffer.py
python 06_ray_nway_pipeline.py

# Profiling files auto-generated in Ray logs:
ls /tmp/ray/session_*/logs/nsight/*.nsys-rep

# View multi-GPU profiles
nsys-ui /tmp/ray/session_latest/logs/nsight/*.nsys-rep
```

### 5.2 Profiling Approaches

**Standalone (Non-Ray) Files**:
- **Method**: Command-line `nsys profile`
- **Why**: Direct process profiling works
- **Command**: `nsys profile -o output.nsys-rep --trace=cuda,nvtx python file.py`
- **Files**: 01, 02, 05

**Ray Files - DIFFERENT APPROACH**:
- **Method**: `runtime_env` configuration in actor decorators
- **Why**: Ray spawns worker processes; command-line nsys only profiles driver
- **How**: Set `enable_profiling=True` in CONFIG dict
- **Mechanism**: Ray wraps each actor's worker process with nsys automatically
- **Output**: Per-actor .nsys-rep files in `/tmp/ray/session_*/logs/nsight/`
- **Files**: 04, 06

**Profiling Pattern in Code** (file 04, 06):
```python
@ray.remote(num_gpus=1, runtime_env={"nsight": {
    "t": "cuda,nvtx",
    "cuda-memory-usage": "true",
}})
class GPUWorkerActorWithProfiling(GPUWorkerActorBase):
    """GPU worker with nsys profiling enabled via Ray runtime_env"""
    pass
```

### 5.3 What to Check After Making Changes

**Correctness Checks**:
1. **Event Creation**: All events created with correct list size (`num_buffers`)
2. **Event Priming**: Priming loop immediately after event creation
3. **Event Recording**: Using `stream.record_event(event)` API
4. **Buffer Shapes**: Output buffers use `output_shape`, not `input_shape`
5. **NVTX Annotations**: Include buffer index `[buf={buffer_idx}]`

**Performance Validation** (via nsys):
1. **Overlap Visualization**: H2D, Compute, D2H should overlap on timeline
2. **No Deadlocks**: Profile completes without hanging
3. **Buffer Alternation**: For N=2, see `buf=0` and `buf=1` alternating
4. **Multi-GPU**: For Ray files, see separate streams per Worker

**Output Validation**:
1. **Sample Count**: `len(collected_outputs) == num_batches`
2. **Shape Correctness**: Each output has shape `(batch_size, *output_shape)`
3. **No Corruption**: Verify output values change between batches (check with clone())

**Throughput Check**:
```bash
# Expected speedups (vs. sequential baseline):
# Double buffering (file 02): 2.0-3.0x
# Ray 2 GPUs (file 04): 4.0-5.0x
# N-way N=3 (file 05): 2.5-3.5x
# Ray 2 GPUs x 3 buffers (file 06): 5.0-7.0x
```

---

## 6. Common Pitfalls

### 6.1 Event API Confusion

**MISTAKE 1**: Using wrong API direction
```python
# WRONG - event-centric API
self.h2d_done_event[idx].record(self.h2d_stream)

# CORRECT - stream-centric API (production pattern)
self.h2d_stream.record_event(self.h2d_done_event[idx])
```

**Why it matters**: While both may work, production uses stream-centric for consistency

**MISTAKE 2**: Recording on wrong stream
```python
# WRONG - recording H2D event on compute stream
self.compute_stream.record_event(self.h2d_done_event[idx])

# CORRECT - H2D event recorded on H2D stream
self.h2d_stream.record_event(self.h2d_done_event[idx])
```

**Detection**: Nsys shows unexpected synchronization points; throughput drops

### 6.2 Missing Event Priming Leading to Deadlocks

**MISTAKE**: Forgetting to prime events after creation
```python
# WRONG - events not primed
self.h2d_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
self.compute_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
self.d2h_done_event = [torch.cuda.Event() for _ in range(num_buffers)]

# First wait_event() call will deadlock!
```

**CORRECT** - Always prime:
```python
self.h2d_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
self.compute_done_event = [torch.cuda.Event() for _ in range(num_buffers)]
self.d2h_done_event = [torch.cuda.Event() for _ in range(num_buffers)]

# Prime immediately
for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
    for ev in events:
        ev.record()  # CRITICAL: Makes events signaled
```

**Symptom**: First batch hangs forever; nsys shows no activity after initialization
**Root cause**: `wait_event()` waits for unsignaled event that will never be signaled

### 6.3 Output Shape Assumptions

**MISTAKE 1**: Hardcoding output buffer shape to input shape
```python
# WRONG - assumes output_shape = input_shape
self.gpu_output_buffers = [
    torch.empty(batch_size, *input_shape, device=device) for _ in range(num_buffers)
]
```

**CORRECT**:
```python
# RIGHT - use explicit output_shape parameter
self.gpu_output_buffers = [
    torch.empty(batch_size, *output_shape, device=device) for _ in range(num_buffers)
]
```

**Impact**: Works for matmul workload, fails for PeakNet (input: 1 channel, output: 3 channels)

**MISTAKE 2**: Not passing output_shape to constructor
```python
# WRONG - missing output_shape
pipeline = DoubleBufferedPipeline(batch_size, input_shape, num_iterations, ...)

# RIGHT - explicit separation
pipeline = DoubleBufferedPipeline(batch_size, input_shape, output_shape, num_iterations, ...)
```

### 6.4 Metadata Tracking Errors

**MISTAKE**: Using wrong buffer index for metadata retrieval
```python
# WRONG - using output_idx for metadata
output_idx = (current_idx - (num_buffers - 1)) % num_buffers
output_meta = self.metadata_buffers[output_idx]  # May be wrong batch!
```

**CORRECT**:
```python
# RIGHT - calculate metadata index based on batch number
output_idx = (current_idx - (num_buffers - 1)) % num_buffers
output_meta_idx = (batch_idx - (num_buffers - 1)) % num_buffers
output_meta = self.metadata_buffers[output_meta_idx]
```

**Why**: Buffer indices rotate circularly, but batch indices are sequential

### 6.5 Forgetting to Clone Output

**MISTAKE**: Reading buffer directly without cloning
```python
# WRONG - no clone()
output = self.cpu_output_buffers[output_idx]
results.append(output)  # All results will show same (latest) data!
```

**CORRECT**:
```python
# RIGHT - clone creates independent copy
self.d2h_done_event[output_idx].synchronize()  # Wait for D2H
output = self.cpu_output_buffers[output_idx].clone()  # CRITICAL: clone!
results.append(output)  # Now safe
```

**Impact**: Silent data corruption - all collected outputs show data from last batch
**Root cause**: Buffer memory gets reused; without clone, all references point to same memory

### 6.6 Ray Profiling Pitfalls

**MISTAKE**: Using command-line nsys for Ray files
```bash
# WRONG - only profiles driver process, not GPU workers!
nsys profile python 04_ray_gpu_double_buffer.py
```

**CORRECT**: Use runtime_env configuration
```python
# In file, set:
CONFIG = {'enable_profiling': True, ...}

# Then run normally:
python 04_ray_gpu_double_buffer.py

# Profiles appear in:
# /tmp/ray/session_latest/logs/nsight/*.nsys-rep
```

**Detection**: Profile shows Ray initialization but no GPU kernels

---

## 7. Key Implementation Patterns

### 7.1 N-Way Buffer Rotation

**Pattern**: Circular rotation through N buffers
```python
def swap(self):
    """Advance to next buffer (circular rotation)"""
    self.current_idx = (self.current_idx + 1) % self.num_buffers

# For N=2: 0 → 1 → 0 → 1 ...
# For N=3: 0 → 1 → 2 → 0 → 1 → 2 ...
```

**Usage**:
```python
for batch_idx in range(num_batches):
    if batch_idx > 0:
        pipeline.swap()  # Rotate to next buffer

    buffer_idx = pipeline.process_batch(...)
```

### 7.2 (N-1) Output Reading Pattern

**When to Read**: With N buffers, read output from (N-1) batches ago

**Formula**: `output_idx = (current_idx - (N-1)) % num_buffers`

**Example Timeline** (N=3):
```
Iteration 0: Process batch 0 in buffer 0
Iteration 1: Process batch 1 in buffer 1
Iteration 2: Process batch 2 in buffer 2, READ batch 0 from buffer 0
Iteration 3: Process batch 3 in buffer 0, READ batch 1 from buffer 1
Iteration 4: Process batch 4 in buffer 1, READ batch 2 from buffer 2
```

**Implementation**:
```python
if batch_idx >= num_buffers - 1:
    output_idx = (buffer_idx - (num_buffers - 1)) % num_buffers

    # Sync specific buffer's D2H
    pipeline.d2h_done_event[output_idx].synchronize()

    # Clone for safety
    output = pipeline.cpu_output_buffers[output_idx].clone()
```

### 7.3 Metadata Circular Buffer

**Pattern**: Track which batch is in which buffer
```python
# During pipeline initialization
self.metadata_buffers = [None] * num_buffers

# After scheduling work
buffer_idx = pipeline.process_batch(...)
pipeline.metadata_buffers[buffer_idx] = {
    'batch_idx': batch_idx,
    'batch_size': current_batch_size,
    # ... custom fields
}

# When reading output
output_meta_idx = (batch_idx - (num_buffers - 1)) % num_buffers
output_meta = pipeline.metadata_buffers[output_meta_idx]
```

**Complete Pattern** (9-step process from tutorials):
```python
# 1. Schedule GPU work
buffer_idx = pipeline.process_batch(cpu_batch, batch_idx, batch_size, nvtx_prefix)

# 2. Store metadata AFTER process_batch, BEFORE reading output
pipeline.metadata_buffers[buffer_idx] = {'batch_idx': batch_idx, ...}

# 3. Check if output ready (need N-1 batches processed first)
if batch_idx >= num_buffers - 1:
    # 4. Calculate output buffer index
    output_idx = (buffer_idx - (num_buffers - 1)) % num_buffers

    # 5. Calculate metadata index (may differ from output_idx!)
    meta_idx = (batch_idx - (num_buffers - 1)) % num_buffers

    # 6. Synchronize on specific buffer's D2H event
    pipeline.d2h_done_event[output_idx].synchronize()

    # 7. Clone output (critical for async safety!)
    output = pipeline.cpu_output_buffers[output_idx].clone()

    # 8. Retrieve associated metadata
    meta = pipeline.metadata_buffers[meta_idx]

    # 9. Use result safely
    save_to_file(output, meta)
```

---

## 8. Production Command Examples

### 8.1 Running Production Pipeline

```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray

# Basic random data test
peaknet-pipeline --config examples/configs/peaknet-random.yaml \
                 --max-actors 4 \
                 --total-samples 10240 \
                 --verbose

# With torch.compile optimization
peaknet-pipeline --config examples/configs/peaknet-random.yaml \
                 --max-actors 2 \
                 --total-samples 10240 \
                 --verbose \
                 --compile-mode reduce-overhead

# With specific GPU selection
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline \
    --config examples/configs/peaknet-random.yaml \
    --max-actors 2 \
    --total-samples 20480 \
    --verbose
```

### 8.2 Profiling Production Code

**Small Dataset for Quick Profiling** (from production CLAUDE.md):
```bash
# Test with 1x512x512 data
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline \
    --config $TEST_DIR/peaknet-random-profile.yaml \
    --max-actors 2 \
    --total-samples 20480 \
    --verbose
```

### 8.3 Key Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Key directories (from production CLAUDE.md)
export TEST_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml"
export PIPELINE_DEV_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray"
export MIN_ML_INFERENCE_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference"
```

---

## 9. Critical Knowledge for Advanced Developers

### 9.1 When to Use N>2 Buffers

**Use N=2 (Double Buffering)** when:
- Minimal CPU preprocessing
- GPU is the primary bottleneck
- Memory is constrained
- Standard case for most workloads

**Use N=3 (Triple Buffering)** when:
- CPU has non-trivial preprocessing (>5ms)
- Memory permits additional buffer
- GPU utilization with N=2 is <90%
- CPU can work while GPU processes previous batch

**Use N≥4** rarely:
- Very heavy CPU preprocessing (>10ms)
- Diminishing returns (marginal improvement)
- Increased memory pressure
- More complex debugging

**Production Recommendation** (from file 06):
> "Scale horizontally first (more GPUs), then vertically (more buffers per GPU) if needed."

### 9.2 Synchronization Philosophy

**Fine-Grained Event-Based** (production pattern):
```python
# Wait only for specific buffer's previous stage
self.compute_stream.wait_event(self.h2d_done_event[buffer_idx])
```

**Not Stream-Based**:
```python
# AVOID - coarse synchronization
self.compute_stream.wait_stream(self.h2d_stream)  # Too broad!
```

**Why Events Are Better**:
- Wait only for THIS buffer's dependency, not ALL previous work
- Enables maximum parallelism across buffers
- Each buffer has independent event chain

**Event Chain Per Buffer**:
```
Buffer 0: H2D_event0 → Compute_event0 → D2H_event0
Buffer 1: H2D_event1 → Compute_event1 → D2H_event1
Buffer 2: H2D_event2 → Compute_event2 → D2H_event2
```

### 9.3 Memory Management Strategy

**Pinned Memory** (production uses this):
```python
# CPU buffers
cpu_output_buffers = [
    torch.empty(batch_size, *output_shape, pin_memory=True)
    for _ in range(num_buffers)
]
```

**Benefits**:
- Enables async H2D/D2H transfers (non_blocking=True)
- Faster transfers via DMA
- No paging overhead

**Caution**: Pinned memory is limited resource; don't over-allocate

**Lazy Pinning Pattern** (production, peaknet_pipeline.py:303-305):
```python
# Only pin if not already pinned (prevents exhaustion)
tensor = cpu_batch[i] if cpu_batch[i].is_pinned() else cpu_batch[i].pin_memory()
gpu_buffer[i].copy_(tensor, non_blocking=True)
# tensor freed immediately by GC (short lifetime = no exhaustion)
```

### 9.4 Ray-Specific Patterns

**GPU Allocation**:
```python
@ray.remote(num_gpus=1)  # Ray assigns one GPU per actor
class GPUWorkerActor:
    def __init__(self, ...):
        # Ray sets CUDA_VISIBLE_DEVICES automatically
        # Always use gpu_id=0 in actor (physical GPU mapped to logical 0)
        self.pipeline = Pipeline(..., gpu_id=0, ...)
```

**Object Store Zero-Copy**:
```python
# Producer: Put numpy array in object store
batch_ref = ray.put(numpy_array)

# Consumer: Get from object store (zero-copy if possible)
numpy_array = ray.get(batch_ref)
```

**Graceful Shutdown**:
```python
# In actor method
def shutdown(self):
    self.pipeline.wait_for_completion()
    ray.actor.exit_actor()

# From driver
ray.get([actor.shutdown.remote() for actor in actors])
```

---

## 10. Testing Strategy

### 10.1 Functional Testing

**Test Each Tutorial Independently**:
```bash
for file in 01_sequential_baseline.py 02_double_buffering.py 05_nway_advanced.py; do
    echo "Testing $file..."
    python $file || echo "FAILED: $file"
done

for file in 03_ray_producer_consumer.py 04_ray_gpu_double_buffer.py 06_ray_nway_pipeline.py; do
    echo "Testing $file..."
    python $file || echo "FAILED: $file"
    ray stop  # Clean up Ray between tests
done
```

**Verify Output Collection**:
```python
# Add at end of tutorial
assert len(collected_outputs) == num_batches, f"Expected {num_batches}, got {len(collected_outputs)}"
print(f"✓ All {num_batches} outputs collected successfully")
```

### 10.2 Performance Testing

**Baseline Measurement** (file 01):
```bash
python 01_sequential_baseline.py | grep "Throughput"
# Expected: ~100-200 samples/s (depends on GPU)
```

**Double Buffering** (file 02):
```bash
python 02_double_buffering.py | grep "Throughput"
# Expected: 2-3x baseline
```

**Multi-GPU Scaling** (file 04):
```bash
python 04_ray_gpu_double_buffer.py | grep "Throughput"
# Expected: ~4-5x baseline for 2 GPUs
```

**Verify Scaling**:
```python
# Approximately linear: throughput ≈ num_gpus × single_gpu_throughput
# With some overhead: expect 0.8-0.9 efficiency
```

### 10.3 Regression Testing

**After Making Changes**:
1. Run all 6 tutorial files without errors
2. Verify nsys profiles show overlap (files 02, 05)
3. Verify Ray profiles generated (files 04, 06 with enable_profiling=True)
4. Check throughput is within 10% of baseline
5. Verify output counts match input counts

**Automated Check Script**:
```bash
#!/bin/bash
# test_tutorials.sh

cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference

# Run each tutorial, capture exit code
for i in 01 02 03 04 05 06; do
    echo "=== Testing ${i}*.py ==="
    python ${i}*.py > /tmp/test_${i}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ ${i} PASSED"
    else
        echo "✗ ${i} FAILED - see /tmp/test_${i}.log"
    fi

    # Clean up Ray if needed
    ray stop 2>/dev/null
done
```

---

## 11. Future Maintenance Guidance

### 11.1 When Adding New Patterns to Production

**Process**:
1. Implement pattern in production (`peaknet_pipeline.py`)
2. Test thoroughly in production context
3. Create minimal reproduction in new or existing tutorial
4. Update tutorial README with new pattern
5. Update this handoff document

### 11.2 Keeping Tutorials Aligned

**Regular Alignment Checks**:
```bash
# Compare key patterns between tutorial and production
diff -u \
    <(grep -A3 "record_event\|\.record(" min-ml-inference/02_double_buffering.py) \
    <(grep -A3 "record_event\|\.record(" peaknet-pipeline-ray/peaknet_pipeline_ray/core/peaknet_pipeline.py)

# Should show no differences in pattern usage
```

**When to Realign**:
- Major production pattern change
- New PyTorch/CUDA API version
- Performance optimization discovered
- Bug fix in synchronization logic

### 11.3 Documentation Maintenance

**Update These Sections When**:
- Production file paths change → Update Section 2
- New pattern added → Update Sections 3, 7
- API changes → Update Sections 3.1, 6.1
- New pitfall discovered → Update Section 6

**Cross-References to Maintain**:
- Tutorial README.md (tutorial repo)
- Production CLAUDE.md (production repo)
- This handoff document
- Code comments in production files

---

## 12. Quick Reference

### 12.1 Critical Patterns Checklist

When implementing double buffering, ensure:
- [ ] Events created with correct list size
- [ ] Events primed immediately after creation
- [ ] Using `stream.record_event(event)` API
- [ ] Separate `input_shape` and `output_shape` parameters
- [ ] Output buffers allocated with `output_shape`
- [ ] NVTX annotations include buffer index
- [ ] Metadata tracking implemented
- [ ] Output cloning before use
- [ ] (N-1) pattern for output reading

### 12.2 File Hierarchy (Complexity)

**Learning Path** (recommended order):
1. `01_sequential_baseline.py` - Understand the problem
2. `02_double_buffering.py` - Learn core solution
3. `03_ray_producer_consumer.py` - Ray without GPU
4. `04_ray_gpu_double_buffer.py` - Integration
5. `05_nway_advanced.py` - Generalization
6. `06_ray_nway_pipeline.py` - Production pattern

**Production Reference Path**:
- For double buffering: `02_double_buffering.py` → `peaknet_pipeline.py` (N=2)
- For N-way buffering: `05_nway_advanced.py` → `peaknet_pipeline.py` (N≥2)
- For Ray integration: `06_ray_nway_pipeline.py` → `peaknet_ray_pipeline_actor.py`

### 12.3 Key File Locations

**Tutorials**:
- Repo: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/min-ml-inference/`
- GitHub: `git@github.com:carbonscott/pipelining-tutorials.git`

**Production**:
- Repo: `/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray/`
- GitHub: `git@github.com:carbonscott/peaknet-pipeline-ray.git`
- Core: `peaknet_pipeline_ray/core/peaknet_pipeline.py`
- Actor: `peaknet_pipeline_ray/core/peaknet_ray_pipeline_actor.py`

**Documentation**:
- Tutorial guide: `min-ml-inference/README.md`
- This handoff: `min-ml-inference/HANDOFF.md`
- Production guide: `peaknet-pipeline-ray/CLAUDE.md`
- Ray patterns: `peaknet-pipeline-ray/RAY_FEATURES_ANALYSIS.md`

### 12.4 Contact and Resources

**Documentation Hierarchy**:
1. Tutorial README.md - Start here for learning
2. This handoff document - Architecture and patterns
3. Production CLAUDE.md - Production deployment details
4. Code comments - Implementation specifics

**When in Doubt**:
- Consult production `peaknet_pipeline.py` as ground truth
- Check nsys profiles for performance validation
- Compare tutorial and production side-by-side for patterns

---

## Appendix A: Complete Event Pattern Example

```python
class DoubleBufferedPipeline:
    def __init__(self, batch_size, input_shape, output_shape, num_iterations,
                 gpu_id=0, pin_memory=True):
        # ... other initialization ...

        # Create events (one per buffer)
        self.h2d_done_event = [torch.cuda.Event() for _ in range(2)]
        self.compute_done_event = [torch.cuda.Event() for _ in range(2)]
        self.d2h_done_event = [torch.cuda.Event() for _ in range(2)]

        # CRITICAL: Prime all events
        for events in [self.h2d_done_event, self.compute_done_event, self.d2h_done_event]:
            for ev in events:
                ev.record()  # Makes them signaled immediately

    def _h2d_transfer(self, cpu_batch, buffer_idx, current_batch_size):
        with nvtx.range(f"H2D[buf={buffer_idx}]"):
            # Wait for this buffer's previous compute to finish
            self.h2d_stream.wait_event(self.compute_done_event[buffer_idx])

            with torch.cuda.stream(self.h2d_stream):
                # Transfer data
                self.gpu_input_buffers[buffer_idx][:current_batch_size].copy_(
                    cpu_batch[:current_batch_size], non_blocking=True
                )

            # Record H2D completion using STREAM-CENTRIC API
            self.h2d_stream.record_event(self.h2d_done_event[buffer_idx])

    def _compute_workload(self, buffer_idx, current_batch_size):
        with nvtx.range(f"Compute[buf={buffer_idx}]"):
            # Wait for this buffer's H2D to finish
            self.compute_stream.wait_event(self.h2d_done_event[buffer_idx])

            with torch.cuda.stream(self.compute_stream):
                # Compute
                input_slice = self.gpu_input_buffers[buffer_idx][:current_batch_size]
                output_slice = self.compute(input_slice)
                self.gpu_output_buffers[buffer_idx][:current_batch_size] = output_slice

            # Record compute completion
            self.compute_stream.record_event(self.compute_done_event[buffer_idx])

    def _d2h_transfer(self, buffer_idx, current_batch_size):
        with nvtx.range(f"D2H[buf={buffer_idx}]"):
            # Wait for this buffer's compute to finish
            self.d2h_stream.wait_event(self.compute_done_event[buffer_idx])

            with torch.cuda.stream(self.d2h_stream):
                # Transfer result back
                self.cpu_output_buffers[buffer_idx][:current_batch_size].copy_(
                    self.gpu_output_buffers[buffer_idx][:current_batch_size],
                    non_blocking=True
                )

            # Record D2H completion
            self.d2h_stream.record_event(self.d2h_done_event[buffer_idx])
```

---

## Appendix B: Alignment Commit Diff Summary

```
commit 61c0903ec411d19df700314650c7baf0f2c6a632
Date:   Wed Nov 19 17:18:29 2025 -0800

Files changed: 4 (02, 04, 05, 06)
Insertions: +70
Deletions: -51

Pattern 1: Event API (24 changes)
  - event.record(stream) → stream.record_event(event)

Pattern 2: Event Priming (4 sections)
  - Added nested loop priming pattern

Pattern 3: Output Shape (12 changes + 12 parameters)
  - Added output_shape parameter
  - Used output_shape for output buffers
```

---

**End of Handoff Document**
