# Emulator profile trace: impact + execution order summary

Trace analyzed: [`scripts/profile_analysis/emulator_profile_trace.json`](scripts/profile_analysis/emulator_profile_trace.json:1)

Captured by: [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319) using [`torch.profiler.profile(...)`](src/trainer.py:333) and [`export_chrome_trace(...)`](src/trainer.py:350).

Analysis script: [`scripts/profile_analysis/analyze_emulator_profile_trace.py`](scripts/profile_analysis/analyze_emulator_profile_trace.py:1)

## 0) Executive summary (most impactful)

1) **Host-side staging / tensor copies dominate early**
   - The trace begins with a large block of `aten::to → aten::_to_copy → aten::copy_ → aten::contiguous/clone`.
   - The timeline shows a **~95ms** `aten::to/_to_copy/copy_` block very early (around **+7ms** into the trace).
   - This is consistent with: *inputs not already in the right device/dtype/layout*, forcing expensive conversions/copies.

2) **Pinned host allocation is a huge overhead spike**
   - `cudaHostAlloc`: **~230.6ms total across 6 calls**.
   - In the chronological view you see repeated **~35–40ms** `cudaHostAlloc` chunks.
   - This often correlates with pageable→pinned staging for H2D copies (or DataLoader pinning behavior).

3) **Kernel launch overhead is large relative to GPU compute**
   - `cudaLaunchKernel`: **~228.6ms total** across **5,799 launches**.
   - Total kernel execution time (sum of `cat=kernel` durations): **~48.4ms** across **7,962 kernels**.
   - That ratio strongly suggests you’re **launch/overhead bound** (many tiny kernels) more than “GPU compute bound”.

4) **The backward pass is heavy and runs mainly on an autograd engine thread**
   - On the busiest CPU thread, the biggest exclusive/self-time contributors include:
     - `cudaLaunchKernel` (**~138.3ms exclusive** on that thread)
     - `aten::mm` (**~37.8ms exclusive**)
     - `autograd::engine::evaluate_function: AddmmBackward0` (**~14.2ms exclusive**; **~145.9ms inclusive** across the trace)

## 1) Trace context and caveats

- This trace covers a wall span of **~657.5ms**.
- Inclusive “total time per op” is **not additive** because:
  - many CPU ops are nested (`aten::to` contains `aten::_to_copy`, etc.)
  - multiple threads overlap
- The trace was produced from **`prof_batches=1`** in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319), so one-off costs (kernel attribute queries, allocator behavior, caching) can be overrepresented.

## 2) Key totals (inclusive)

### By category (inclusive sums of durations)

| Category | Total dur (µs) | Total (ms) | Count |
|---|---:|---:|---:|
| `cpu_op` | 1,758,733.944 | 1,758.7 | 34,051 |
| `cuda_runtime` | 518,620.819 | 518.6 | 10,383 |
| `kernel` | 48,389.036 | 48.4 | 7,962 |
| `gpu_memcpy` | 7,978.547 | 8.0 | 244 |

### Top names by inclusive time

| Name | Total (ms) | Count | Why it matters |
|---|---:|---:|---|
| `cudaHostAlloc` | 230.6 | 6 | pinned host allocation / staging overhead |
| `cudaLaunchKernel` | 228.6 | 5,799 | CPU launch overhead from many kernels |
| `aten::copy_` | 210.9 | 966 | heavy tensor copying (often avoidable) |
| `autograd::engine::evaluate_function: AddmmBackward0` | 145.9 | 717 | linear layer backward compute + overhead |
| `aten::to` | 101.1 | 250 | device/dtype/layout conversion |
| `aten::linear` | 87.1 | 720 | forward compute dominated by GEMMs |
| `aten::addmm` | 80.8 | 717 | GEMM-like op, main matmul payload |
| `aten::mm` | 52.3 | 1,442 | matmul (forward/backward) |
| `aten::relu` | 54.9 | 478 | elementwise activation (kernel launch heavy) |
| `aten::clamp_min` | 50.7 | 478 | elementwise clamp (kernel launch heavy) |

## 3) Order-of-operations: what happens first vs later

This is a *chronological* view from the earliest “significant” events (durations roughly ≥2ms, plus key CUDA-runtime events).

### Phase A — immediate input conversion/copy block (≈ +1ms to +95ms)

Observed early sequence (main thread `tid=557914`):
- `aten::to` (~5.8ms)
- `aten::_to_copy` (~5.6ms)
- `aten::copy_` (~5.5ms)
- `aten::contiguous`/`aten::clone` (~5.2ms)
- then a much larger repeat:
  - `aten::to` / `aten::_to_copy` / `aten::copy_` ~**95ms**
  - `aten::contiguous` / `aten::clone` ~**88ms**

Interpretation:
- Something about the first profiled batch’s inputs triggers **large format/device conversions**.
- The presence of GPU event `Memcpy HtoD (Pageable -> Device)` later suggests at least some host tensors are **pageable** (not pinned), which makes `non_blocking=True` less effective.

### Phase B — first visible H2D copy event (≈ +95ms)

Around **+95ms**:
- `cudaMemcpyAsync` ~**2.4ms**
- `gpu_memcpy` shows **HtoD (Pageable -> Device)** ~**2.37ms**

Interpretation:
- The memcpy itself is not the major cost; the surrounding *host-side staging/allocation/copies* are.

### Phase C — activation/clamp and pinned alloc spikes (≈ +182ms to +285ms)

Around **+182ms**:
- `aten::relu` ~**40.4ms**
- `aten::clamp_min` ~**40.4ms**
- `cudaHostAlloc` ~**40.3ms** (on a different thread)
- `cudaLaunchKernel` ~**40.3ms**

Then another similar “~40ms” chunk later (`aten::add` + `cudaHostAlloc` + `cudaLaunchKernel`).

Interpretation:
- Elementwise ops (relu/clamp/add) tend to generate **many small kernels**.
- Pinned allocation spikes (`cudaHostAlloc`) indicate expensive memory management during the step.

### Phase D — linear/addmm and kernel attribute queries (≈ +285ms to +340ms)

Around **+285ms**:
- `aten::linear` ~**35.5ms**
- `aten::addmm` ~**35.5ms**
- `cudaFuncGetAttributes` ~**35.4ms**

Interpretation:
- This is the “matmul core” of the forward path (or part of it).
- `cudaFuncGetAttributes` is frequently a “first time we see this kernel” overhead (can shrink after warmup / multi-step profiling).

### Phase E — burst of GPU kernels (≈ +340ms to +350ms)

You then see many sub-millisecond kernels: cutlass GEMMs + elementwise kernels + a couple fused kernels.

Interpretation:
- GPU compute is happening, but the trace indicates it’s chopped into lots of small pieces.

### Phase F — backward + engine thread activity (≈ +343ms onward)

The first major `autograd::engine::evaluate_function: AddmmBackward0` appears at about **+343ms**, and later there are very large blocks on the autograd engine thread (`tid=557963`) such as:
- `autograd::engine::evaluate_function: TBackward0` ~**42.8ms**
- `torch::autograd::CopySlices` family ~**36.8ms**

Interpretation:
- A lot of the “real work” of backprop is happening off the Python main thread.
- CopySlices/CopyBackwards are often signals of **views/slicing/indexing** in the graph causing gradient scatter/gathers.

## 4) GPU kernel profile: what the GPU is doing

- Kernels: **7,962 launches**, **59 unique** names.
- Top kernel families by time/count:
  - reductions (`reduce_kernel`) and elementwise (`vectorized_elementwise_kernel`)
  - cutlass / xmma GEMMs (from `addmm`/matmul)

Key interpretation:
- The GPU kernel time sum is only **~48ms**, while CPU runtime overhead (`cudaLaunchKernel` + other runtime calls) is much higher.
- This is typical for **too many small kernels** (launch-bound), especially when the model includes lots of elementwise ops.

## 5) Actionable “most likely wins” (ranked)

1) **Eliminate the huge early `aten::to/_to_copy/contiguous/copy_` block**
   - Goal: make input tensors already match the device/dtype/layout expected by the step.
   - Tie-in: your batch transfers happen right before compute in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319) and training loop in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:391).

2) **Fix pageable→pinned→device behavior to reduce `cudaHostAlloc`**
   - The trace explicitly shows `Memcpy HtoD (Pageable -> Device)`.
   - If using a `DataLoader`, typical levers are pinning and worker persistence.

3) **Reduce kernel launch count / fuse pointwise ops**
   - Because `cudaLaunchKernel` time is huge relative to `kernel` time.
   - The most impactful direction is usually graph compilation/fusion rather than micro-optimizing individual elementwise ops.

4) **Profile more than one “active” step for stability**
   - With `prof_batches=1`, you see large one-time costs like `cudaFuncGetAttributes`.
   - A multi-step profile (or a schedule) typically gives a clearer steady-state picture.

