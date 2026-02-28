# Why emulator training can be slow on an H200 (and what to check)

This repo’s emulator training loop can become **CPU / Python / synchronization bound**. When that happens, upgrading the GPU (e.g. 3060 → H200) may not help and can even *look* slower because the “GPU part” is no longer the limiter.

This document lists the most likely causes **in this codebase**, with concrete pointers to the exact lines that introduce overhead.

## 0) First sanity check: are you actually training on the H200?

The runtime device is chosen in [`DatasetConfig.device`](src/configs/datasets.py:30) using `torch.cuda.is_available()`.

What to confirm:

- The program prints something like `Training Emulator on cuda` from [`handle_train()`](src/cli.py:17)
- `torch.cuda.get_device_name(0)` shows the H200
- GPU utilization increases during training (via `nvidia-smi`)

If it’s running on CPU (or a different GPU), nothing else matters.

## 1) Biggest “gotcha”: per-batch printing forces GPU synchronization

In the emulator hot path, the loss function prints **every batch** in [`Loss.training()`](src/loss.py:82).

The line:

- [`print(...)`](src/loss.py:100)

formats tensors like `{mean_loss.detach():.3e}`.

Why this is expensive:

- formatting a CUDA tensor for printing implicitly converts it to a Python scalar (equivalent to calling `.item()`), which **forces a CUDA synchronize**
- synchronizing every batch prevents overlap and can destroy throughput
- on a faster GPU, you hit this synchronization/stdio wall earlier, so it can appear *worse* than on a slower GPU

If you experienced “extremely fast” behavior previously, this is the first thing I would suspect when a run becomes unexpectedly slow.

**Fix applied:** the per-batch `print(...)` in [`Loss.training()`](src/loss.py:82) has been commented out to avoid the forced synchronization.

Related: the trainer also prints per epoch in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:347), but that is once per epoch and is typically not the main issue.

## 2) Autograd anomaly detection is enabled globally

This line is executed at import time in [`src/trainer.py`](src/trainer.py:28):

- [`torch.autograd.set_detect_anomaly(True)`](src/trainer.py:28)

Why it matters:

- anomaly detection adds extra bookkeeping and checks during backward
- it can slow training significantly
- it is useful for debugging NaNs/infs, but it’s usually disabled for performance runs

If anomaly detection is on in your H200 environment but was off in your old 3060 environment (or code revision), that alone can explain a large slowdown.

**Fix applied:** anomaly detection has been set to `False` in [`src/trainer.py`](src/trainer.py:28).

## 3) The emulator forward is a Python loop over timesteps

The emulator is implemented as an explicit loop in [`Emulator.forward()`](src/models/emulator.py:35):

- `for t in range(T): ...` with one MLP call per timestep

Consequences:

- many small kernel launches (and Python overhead) rather than a few large fused ops
- harder for the GPU to reach high utilization
- on a very fast GPU, the CPU/Python launch overhead can dominate

This can make a high-end GPU look underwhelming, especially if the rest of the step includes forced synchronizations (see §1).

## 4) The hot path decodes **B×T** latents through the autoencoder decoder

In emulator training, the model predicts latents, then decodes them back to species space before computing the loss:

- decode happens in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314)
- decoder implementation is [`Autoencoder.decode()`](src/models/autoencoder.py:55)

Key point:

- decode is performed on `outputs.reshape(-1, latent_dim)` (so `B*T` items), which can be large

This is not necessarily “wrong”, but it means emulator training cost is not just the emulator; it includes a significant AE decode + loss computation every step.

## 5) DataLoader and CPU indexing can become the bottleneck

The DataLoader is built by [`tensor_to_dataloader()`](src/data_loading.py:224) with:

- `num_workers=10`, `pin_memory=True`, `in_order=False`

The dataset fetch path for emulator sequences is:

- [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)

It constructs an indices tensor each time:

- `indices_tensor = torch.tensor(indices, ...)` in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)

and then gathers:

- `rows = self.data_matrix[data_indices]`

On a system where:

- CPU resources are constrained,
- storage is remote,
- NUMA placement is unfavorable,

…this can dominate wallclock time even if the GPU is extremely fast.

Note: the preprocessed HDF5 is loaded fully into CPU RAM up front in [`load_tensors_from_hdf5()`](src/data_loading.py:84), so *steady-state* training should not be streaming from disk—if it is, something else is going on (e.g. swapping / memory pressure).

## 6) Batch size and window size amplify everything

Two config values strongly scale per-step work:

- [`EMConfig.batch_size`](src/configs/emulator.py:42)
- [`EMConfig.window_size`](src/configs/emulator.py:31)

Given `T = window_size - 1`, the step effectively processes `B*T` decoded states.

If your H200 run used different config values than the 3060 run (even unintentionally), the speed comparison won’t be apples-to-apples.

## 7) How to localize the slowdown (no code changes required)

### A) Separate “data time” vs “compute time”

The epoch timer printed in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:347) lumps everything together. To separate:

- measure time spent waiting for the DataLoader iterator (CPU)
- measure time spent in forward/backward (GPU)

If the GPU is mostly idle and step time is dominated by Python/data, the H200 will not help.

### B) Check for synchronization symptoms

If you see:

- low GPU utilization,
- lots of short kernels,
- training speed proportional to how much logging you do,

…it usually indicates synchronization and/or Python overhead (see §1 and §3).

### C) Use a profiler

The most informative tools for this kind of issue:

- `torch.profiler` (CPU + CUDA timeline)
- Nsight Systems (`nsys`) to see kernel launch gaps and CPU-GPU sync points

The first places to look in the trace are exactly:

- [`Loss.training()`](src/loss.py:82) (printing + tensor→scalar conversions)
- [`Emulator.forward()`](src/models/emulator.py:35) (timestep loop)
- [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197) (CPU gather / overhead)

## Summary (most likely explanations in this repo)

If emulator training is slower on an H200 than on a 3060, the highest-probability causes in the current code are:

1. **Per-batch printing in** [`Loss.training()`](src/loss.py:82) **forcing CUDA synchronization**
2. **Anomaly detection enabled** via [`torch.autograd.set_detect_anomaly(True)`](src/trainer.py:28)
3. **Python timestep loop** in [`Emulator.forward()`](src/models/emulator.py:35) limiting GPU utilization
4. **CPU/DataLoader bottlenecks** in [`tensor_to_dataloader()`](src/data_loading.py:224) + [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)

These are the concrete, code-level reasons the pipeline can become non-GPU-bound, which is the typical scenario where “bigger GPU = not faster” (and sometimes “bigger GPU = looks slower”).
