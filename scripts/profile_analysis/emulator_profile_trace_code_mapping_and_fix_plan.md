# Emulator trace → code mapping + fix plan (no code changes yet)

This document links the biggest trace hot-spots in [`scripts/profile_analysis/emulator_profile_trace.json`](scripts/profile_analysis/emulator_profile_trace.json:1) to concrete code sites, then proposes the smallest upstream changes likely to reduce the cost.

It’s based on:
- The trace summary in [`scripts/profile_analysis/emulator_profile_trace_summary.md`](scripts/profile_analysis/emulator_profile_trace_summary.md:1)
- The profiler capture path in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319)
- The emulator training step in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:358)

## 1) Trace event → code mapping

### A) “Big early copy/conversion block”: `aten::to → aten::_to_copy → aten::copy_ → contiguous/clone`

**Where in code the transfers are triggered**

These are the explicit device transfers in the profiled step:
- Warmup transfers in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319) at the warmup call site ([`Tensor.to()`](src/trainer.py:326)–[`Tensor.to()`](src/trainer.py:329))
- Profiled step transfers in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319) ([`Tensor.to()`](src/trainer.py:337)–[`Tensor.to()`](src/trainer.py:340))

And in normal training (non-profile run), the same pattern appears in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:391) ([`Tensor.to()`](src/trainer.py:400)–[`Tensor.to()`](src/trainer.py:402)).

**Why those `.to()` calls can expand into `contiguous/clone/copy_`**

The dataset returns *sliced views* that are typically **non-contiguous**:
- Batch is created in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)
  - `rows = self.data_matrix[data_indices]` ([`Tensor.__getitem__()`](src/data_loading.py:203)) returns a (usually) contiguous gather result.
  - Then `phys/features/targets` are created via slicing:
    - `physical_parameters = rows[:, :-1, ...]` ([`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:205))
    - `features = rows[:, 0, ...]` ([`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:208))
    - `targets = rows[:, 1:, ...]` ([`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:209))

Those tensors are “views with strides”. When PyTorch later tries to do an H2D copy in [`Tensor.to()`](src/trainer.py:337), it often has to make them contiguous first (hence `aten::contiguous` / `aten::clone` / `aten::copy_`).

**How this connects to the trace**

In your trace, the first major wall-time block is exactly that pattern (large `aten::to/_to_copy/copy_` + `contiguous/clone`). The most direct code-level trigger is the trio of `.to(..., non_blocking=True)` calls in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319).

### B) `cudaMemcpyAsync` + `Memcpy HtoD (Pageable -> Device)`

This is the CUDA runtime work underneath the same `.to()` calls:
- The H2D copies are initiated by [`Tensor.to()`](src/trainer.py:337) / [`Tensor.to()`](src/trainer.py:338) / [`Tensor.to()`](src/trainer.py:339).

If the source CPU tensor is not pinned *and* contiguous (or becomes pageable during a pre-copy contiguous conversion), the profiler can show “Pageable → Device”, which is slower and can force synchrony.

### C) `cudaHostAlloc` spikes (pinned host allocation)

The most likely source is DataLoader pinned-memory behavior:
- DataLoader is created with `pin_memory=True` in [`tensor_to_dataloader()`](src/data_loading.py:224) ([`DataLoader(..., pin_memory=True)`](src/data_loading.py:232)).

The trace shows `cudaHostAlloc` on a non-main thread, consistent with PyTorch’s pin-memory thread allocating page-locked buffers.

Why it can be *surprisingly* expensive here:
- Large batches (see [`EMConfig.batch_size`](src/configs/emulator.py:42)) and long sequences (see [`EMConfig.window_size`](src/configs/emulator.py:32)) create large CPU batch tensors.
- If batch tensors are non-contiguous views, pinning may require extra copies/packing, which can trigger repeated pinned allocations.

### D) “Tons of tiny ops”: `aten::linear`, `aten::addmm`, `aten::relu`, `aten::clamp_min`, `aten::cat`, `aten::add`

These map cleanly to the emulator’s explicit per-time-step loop:

- The loop itself is in [`Emulator.forward()`](src/models/emulator.py:35):
  - `for t in range(T)` ([`Emulator.forward()`](src/models/emulator.py:41))
  - `input = torch.cat([current_phys, latents], dim=1)` → `aten::cat` ([`torch.cat()`](src/models/emulator.py:43))
  - `update = self.net(input)` → repeated linear + relu layers ([`nn.Sequential(...)`](src/models/emulator.py:24))
  - `latents = latents + update` → `aten::add` ([`Emulator.forward()`](src/models/emulator.py:47))

**Count-based evidence (strong signal)**

From configs and slicing:
- Sequence length in the emulator forward is `T = window_size - 1` because the dataset builds phys with `rows[:, :-1, ...]` ([`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:205)).
- With [`EMConfig.window_size`](src/configs/emulator.py:32) set to 240, that implies **T = 239**.

The emulator MLP has **2 ReLUs** ([`nn.ReLU()`](src/models/emulator.py:26) and [`nn.ReLU()`](src/models/emulator.py:29)), so you expect `2 * T = 478` ReLU-ish events. The trace shows `aten::relu` count ≈ 478, and `aten::clamp_min` count ≈ 478 (ReLU is implemented via clamp).

Similarly, the MLP has **3 linear layers** ([`nn.Linear()`](src/models/emulator.py:25), [`nn.Linear()`](src/models/emulator.py:28), [`nn.Linear()`](src/models/emulator.py:31)) so you expect `3 * T = 717` linear/addmm-like payloads just from the emulator loop.

### E) Big GEMMs: autoencoder decode (`aten::linear/addmm/mm`)

After the emulator predicts latents for all timesteps, the trainer decodes them with the (frozen) autoencoder:
- Decode call site: [`Autoencoder.decode()`](src/models/autoencoder.py:55) is invoked from [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:358) at [`self.ae.decode(outputs)`](src/trainer.py:368).

The decoder has **3** explicit linear layers via [`F.linear()`](src/models/autoencoder.py:55) at:
- [`F.linear(z, self.encoder_fc3.weight.t())`](src/models/autoencoder.py:57)
- [`F.linear(z, self.encoder_fc2.weight.t())`](src/models/autoencoder.py:60)
- [`F.linear(z, self.encoder_fc1.weight.t())`](src/models/autoencoder.py:64)

These three are likely responsible for the **few “very large”** `aten::linear/addmm` events (tens of ms) seen in the chronological view.

### F) Autograd engine hotspots: `AddmmBackward0`, `TBackward0`, `CopySlices`

All of these are consequences of calling backward:
- [`loss.backward()`](src/trainer.py:372) inside [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:358)

What they correspond to:
- `AddmmBackward0` / `TBackward0`: linear-layer backward (from emulator MLP + AE decoder)
- `CopySlices`: strongly associated with **in-place slice assignment** in the emulator forward:
  - `outputs[:, t, :] = latents` ([`Emulator.forward()`](src/models/emulator.py:49))

That single line creates a “scatter gradients back into a larger tensor” pattern in autograd, which shows up as `torch::autograd::CopySlices` / `CopyBackwards` in the trace.

### G) `Optimizer.zero_grad#AdamW.zero_grad` user annotation

This maps directly to:
- [`self.optimizer.zero_grad()`](src/trainer.py:362) in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:358)

## 2) What’s *likely* happening (root causes)

### Root cause 1 — Non-contiguous CPU batch tensors defeat fast pinned, non-blocking H2D copies

Current pipeline:
1) Worker builds a contiguous `rows` gather ([`Tensor.__getitem__()`](src/data_loading.py:203))
2) Worker returns non-contiguous views for phys/features/targets ([`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:205)–[`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:211))
3) Main thread calls `.to(device, non_blocking=True)` on those views ([`Tensor.to()`](src/trainer.py:337)–[`Tensor.to()`](src/trainer.py:339))
4) PyTorch performs a CPU-side contiguous pack/copy and then a GPU copy (trace shows `contiguous/clone/copy_` + `Memcpy HtoD (Pageable -> Device)`)

Net effect:
- You pay **extra CPU copies**
- You get **pageable** H2D copies (slower / less overlap)

### Root cause 2 — Extremely high kernel launch count from the per-timestep Python loop

[`Emulator.forward()`](src/models/emulator.py:35) runs a python loop over time. Each timestep triggers multiple GPU kernels (linear, relu, add, cat), so the total launches explode.

This matches the trace:
- very high `cudaLaunchKernel` time and count
- relatively low summed kernel execution time

### Root cause 3 — Autograd `CopySlices` overhead from slice assignment into `outputs`

The preallocation + assignment pattern in [`Emulator.forward()`](src/models/emulator.py:35) (specifically [`outputs[:, t, :] = latents`](src/models/emulator.py:49)) causes costly `CopySlices` backward work.

## 3) Fix plan (ranked, minimal-first)

No code changes are made in this step; these are proposed next actions.

### Tier 1 (high ROI, minimal conceptual change): make the batch contiguous *before* `.to()`

Goal: remove the expensive `aten::contiguous/clone` and make H2D copies truly non-blocking.

Options (choose one):

1) **Return contiguous tensors from the dataset**
   - Make `physical_parameters`, `features`, `targets` contiguous in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197) before returning.
   - Expected result: pin-memory thread pins contiguous buffers; main `.to(non_blocking=True)` can do fast async copies.
   - Trade-off: shifts some CPU copy into worker processes (often a win because it parallelizes).

2) **Return a single contiguous `rows` tensor, transfer once, slice on GPU**
   - Instead of transferring 3 strided views, transfer one contiguous block produced at [`rows = self.data_matrix[data_indices]`](src/data_loading.py:203).
   - Then slice into phys/features/targets on GPU (cheap views).
   - Expected result: fewer pinned allocations + fewer `.to()` calls + fewer “pageable → device” copies.
   - This is the cleanest “upstream” fix because it removes the source of strided host tensors.

### Tier 2 (high ROI, moderate change): reduce kernel launches from the time loop

1) **Compile/fuse**
   - Apply graph compilation to reduce kernel launch overhead and fuse pointwise ops.
   - Best target is the full step function that includes [`Emulator.forward()`](src/models/emulator.py:35) + decode + loss.

2) **Use a fused recurrent primitive**
   - The update equation in [`Emulator.forward()`](src/models/emulator.py:35) is effectively an RNN cell.
   - Replacing the python loop with a fused RNN (or a scan-like primitive) can collapse hundreds/thousands of launches.
   - This is more architecture-intrusive but can be the biggest throughput win.

### Tier 3 (targeted autograd cleanup): remove `CopySlices`

Goal: reduce `torch::autograd::CopySlices` overhead.

Options:
- Replace the in-place slice assignment [`outputs[:, t, :] = latents`](src/models/emulator.py:49) with an out-of-place accumulation pattern (collect + stack).
- If you keep preallocation, consider alternatives that avoid autograd slice-copy nodes.

### Tier 4 (profiling stability): profile steady-state, not one-off overhead

Current profiling uses only one active batch in [`EmulatorTrainerSequential._profile_epoch()`](src/trainer.py:319).

Plan:
- Profile multiple active steps and/or use a schedule so that first-use overhead (`cudaFuncGetAttributes`, allocator warmups) is amortized.
- Add high-level `record_function` ranges around: data transfer, emulator forward, AE decode, loss, backward, optimizer.

## 4) “Checkpoints” to verify improvements (before/after)

When you implement Tier 1 fixes, you should see in the trace:
- Fewer/shorter `aten::contiguous` / `aten::clone` under the `.to()` calls
- More “Pinned → Device” (or at least fewer “Pageable → Device”) memcpy events
- Reduced `cudaHostAlloc` spikes (because pinned allocations get reused / are smaller / are fewer)

When you implement Tier 2 fixes, you should see:
- Dramatically reduced `cudaLaunchKernel` count and total time
- Similar or slightly lower total kernel time, but much better wall time

