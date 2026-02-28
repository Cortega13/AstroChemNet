# Emulator training pipeline (end-to-end)

This document explains **how emulator training works** in this repo: what data is used, how it is loaded/batched, what preprocessing happens, and what the training loop actually computes.

The goal is to make it easy to reason about (and later analyze) why training might be slow on a GPU like an H200, by mapping “what the code does” to concrete tensors, shapes, and steps.

## High-level flow

There are two learned components:

1. **Autoencoder (AE)**: compresses per-timestep species abundances into a latent vector and decodes latents back to abundances.
   - Training entrypoint: [`train_autoencoder.main()`](src/training/train_autoencoder.py:38)
2. **Emulator (EM)**: predicts *time evolution in latent space* conditioned on physical parameters.
   - Training entrypoint: [`train_emulator.main()`](src/training/train_emulator.py:16)

Emulator training depends on **preprocessing artifacts** and usually on a trained autoencoder:

- Dataset artifacts (species list, ranges, stoichiometry) are loaded by [`DatasetConfig.__post_init__()`](src/configs/datasets.py:52)
- AE latent scaling relies on `latents_minmax.npy` written by [`save_latents_minmax()`](src/training/train_autoencoder.py:17)
- Emulator sequence data is created by [`preprocess_emulator()`](src/preprocessing/emulator_preprocessing.py:14)

## CLI entrypoints (what you run)

The canonical CLI entrypoint is [`run.py`](run.py:1), which dispatches into [`handle_train()`](src/cli.py:17) and [`handle_preprocess()`](src/cli.py:38).

Typical order:

1. Preprocess the base dataset (creates `train.npy`, `val.npy`, and metadata under the preprocessing directory)
   - Handled by [`handle_preprocess()`](src/cli.py:38)
2. Train the autoencoder
   - Handled by [`handle_train()`](src/cli.py:17) → [`train_autoencoder.main()`](src/training/train_autoencoder.py:38)
3. Preprocess emulator sequences (creates HDF5 sequence tensors)
   - Handled by [`handle_preprocess()`](src/cli.py:38) → [`preprocess_emulator()`](src/preprocessing/emulator_preprocessing.py:14)
4. Train the emulator
   - Handled by [`handle_train()`](src/cli.py:17) → [`train_emulator.main()`](src/training/train_emulator.py:16)

## Dataset: what a “row” contains

The training data is stored as numpy arrays loaded by [`load_datasets()`](src/data_loading.py:17), which reads:

- `train.npy` via [`load_datasets()`](src/data_loading.py:41)
- `val.npy` via [`load_datasets()`](src/data_loading.py:42)

Column selection is driven by per-model configs:

- Autoencoder uses only species columns: [`AEConfig.__post_init__()`](src/configs/autoencoder.py:51)
- Emulator uses `metadata + phys + species`: [`EMConfig.__post_init__()`](src/configs/emulator.py:54)

### Metadata columns

The repo assumes three metadata columns, defined in [`DatasetConfig.metadata`](src/configs/datasets.py:37):

- `Index`
- `Model`
- `Time`

During emulator preprocessing, the `Index` column is overwritten to be `0..N-1` via [`preprocessing_emulator_dataset()`](src/data_processing.py:192).

### Physical parameters

Physical parameters are not hard-coded; they come from preprocessing output:

- ranges are loaded by [`DatasetConfig._load_physical_parameter_ranges()`](src/configs/datasets.py:96)
- the list of phys parameter names is derived in [`DatasetConfig.__post_init__()`](src/configs/datasets.py:52)

They are scaled with log10 + minmax scaling in-place by [`Processing.physical_parameter_scaling()`](src/data_processing.py:55).

### Species abundances

Species list + stoichiometry are loaded from preprocessing output by [`DatasetConfig.__post_init__()`](src/configs/datasets.py:52).

Abundances are clipped in [`load_datasets()`](src/data_loading.py:17) (see [`np.clip()`](src/data_loading.py:50)), then scaled with log10 + minmax scaling by [`Processing.abundances_scaling()`](src/data_processing.py:74).

## Autoencoder training (so emulator has latents)

Autoencoder training happens in [`train_autoencoder.main()`](src/training/train_autoencoder.py:38):

1. Load `train.npy`/`val.npy` with species columns: [`load_datasets()`](src/data_loading.py:17)
2. Scale abundances in numpy (in-place): [`Processing.abundances_scaling()`](src/data_processing.py:74)
3. Wrap into [`AutoencoderDataset`](src/data_loading.py:145) and build DataLoaders using [`tensor_to_dataloader()`](src/data_loading.py:224)
4. Train with [`AutoencoderTrainer`](src/trainer.py:198)
5. After training, compute min/max of latent components and save them via [`save_latents_minmax()`](src/training/train_autoencoder.py:17)

That min/max file is later loaded in [`Processing.__init__()`](src/data_processing.py:20) when `ae_config` is passed.

## Emulator preprocessing (build the sequence dataset)

Emulator preprocessing is in [`preprocess_emulator()`](src/preprocessing/emulator_preprocessing.py:14) and calls [`preprocessing_emulator_dataset()`](src/data_processing.py:192) for train/val.

### Step-by-step: `preprocessing_emulator_dataset`

Given `dataset_np` with columns `metadata + phys + species`:

1. **Assign a stable row index**
   - `dataset_np[:, 0] = arange(...)` in [`preprocessing_emulator_dataset()`](src/data_processing.py:192)

2. **Scale physical parameters** (in-place)
   - [`Processing.physical_parameter_scaling()`](src/data_processing.py:55)

3. **Scale species abundances** (in-place)
   - [`Processing.abundances_scaling()`](src/data_processing.py:74)

4. **Encode abundances into AE latents**
   - `latent_components = inference_functions.encode(...)` in [`preprocessing_emulator_dataset()`](src/data_processing.py:192)
   - This uses the autoencoder encoder path via [`Autoencoder.encode()`](src/models/autoencoder.py:47)

5. **Scale latents** to `[0, 1]` using min/max
   - [`Processing.latent_components_scaling()`](src/data_processing.py:84)

6. **Append latents to each row**
   - `encoded_dataset_np = hstack((dataset_np, latent_components))` in [`preprocessing_emulator_dataset()`](src/data_processing.py:192)

So the final per-row representation becomes:

- `[metadata | phys | species | latents]`

7. **Build rolling windows (sequence indices)**
   - sequence indices are created by [`calculate_emulator_indices()`](src/data_processing.py:165)
   - grouping is based on changes in the `Model` column (`dataset_np[:, 1]`) in [`calculate_emulator_indices()`](src/data_processing.py:165)
   - each sequence is a length-`window_size` list of **row indices**

8. **Shuffle sequences** and convert to torch tensors
   - `perm = np.random.permutation(...)` in [`preprocessing_emulator_dataset()`](src/data_processing.py:192)

9. **Save `(dataset, indices)` to HDF5**
   - written by [`save_tensors_to_hdf5()`](src/data_loading.py:68)
   - saved under a dataset-specific preprocessing directory chosen in [`DatasetConfig.__post_init__()`](src/configs/datasets.py:52)

The preprocessing step writes two files:

- `training_seq.h5` (category `training_seq`)
- `validation_seq.h5` (category `validation_seq`)

as determined by the `category` argument passed in [`preprocess_emulator()`](src/preprocessing/emulator_preprocessing.py:14) and the filename logic in [`save_tensors_to_hdf5()`](src/data_loading.py:68).

## Emulator training: what is loaded and how it is batched

### Loading the preprocessed tensors

Emulator training loads the preprocessed HDF5 tensors in [`train_emulator.main()`](src/training/train_emulator.py:16) using [`load_tensors_from_hdf5()`](src/data_loading.py:84):

- `dataset`: float32 tensor of shape `[N_rows, D_row]`
- `indices`: int32 tensor of shape `[N_sequences, window_size]`

Important: [`load_tensors_from_hdf5()`](src/data_loading.py:84) reads `[:]` from HDF5 into memory (no streaming); it then converts to torch CPU tensors.

### Building the `Dataset`

The training dataset is [`EmulatorSequenceDataset`](src/data_loading.py:165). For a *batch* of sequence IDs, [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197) does:

- `data_indices = self.data_indices[indices_tensor]` → `[B, window_size]`
- `rows = self.data_matrix[data_indices]` → `[B, window_size, D_row]`

Then it slices out three tensors:

1. **Physical parameters**
   - `physical_parameters = rows[:, :-1, metadata : metadata+phys]` in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)
   - shape: `[B, window_size-1, num_phys]`

2. **Features (initial latents at t0)**
   - `features = rows[:, 0, -latent_dim:]` in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)
   - shape: `[B, latent_dim]`

3. **Targets (species abundances for t=1..W-1)**
   - `targets = rows[:, 1:, metadata+phys : -latent_dim]` in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)
   - shape: `[B, window_size-1, num_species]`

All of these are still on CPU at this point.

### DataLoader configuration

The DataLoader is created by [`tensor_to_dataloader()`](src/data_loading.py:224) with:

- `batch_size = em_config.batch_size` from [`EMConfig.batch_size`](src/configs/emulator.py:42)
- `num_workers=10`, `pin_memory=True`, `in_order=False` in [`tensor_to_dataloader()`](src/data_loading.py:224)
- a sampler: [`ChunkedShuffleSampler`](src/data_loading.py:99)

Shuffling:

- the sampler calls `set_epoch(epoch)` each epoch via [`Trainer._run_epoch()`](src/trainer.py:172) (it is cast as a `DistributedSampler`, but only `set_epoch()` is used)
- the sampler builds a permutation inside [`ChunkedShuffleSampler.__iter__()`](src/data_loading.py:122)

## Emulator model: what it computes

The emulator model is a simple MLP applied in an *explicit python loop over time*:

- forward pass: [`Emulator.forward()`](src/models/emulator.py:35)

Given:

- `phys`: `[B, T, P]` where `T = window_size-1`
- `latents` (features): `[B, L]`

The code loops `for t in range(T)`:

- concatenate `phys[:, t, :]` and current latents
- predict an update `Δlatent = net([phys_t, latent_t])`
- update state `latent_{t+1} = latent_t + Δlatent`
- store latents into `outputs[:, t, :]`

Outputs returned are:

- `outputs`: `[B, T, L]`

## Emulator training loop: forward → decode → loss → backward

All of emulator training is in [`EmulatorTrainerSequential`](src/trainer.py:276).

### Training batch

Inside [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314):

1. **Emulator forward in latent space**
   - `outputs = self.model(phys, features)`

2. **Flatten time into batch**
   - `outputs.reshape(-1, latent_dim)` in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314)

3. **Unscale latents from [0,1] back to raw latent range**
   - `inverse_latent_components_scaling()` from [`Processing.inverse_latent_components_scaling()`](src/data_processing.py:156)

4. **Decode latents into species abundances**
   - `self.ae.decode(outputs)` in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314)
   - decode implementation: [`Autoencoder.decode()`](src/models/autoencoder.py:55)

5. **Flatten targets to match decoded outputs**
   - `targets.reshape(-1, num_species)` in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314)

6. **Compute loss**
   - training loss: [`Loss.training()`](src/loss.py:82)
   - includes:
     - an elementwise exponential error via [`Loss.elementwise_loss()`](src/loss.py:42)
     - an elemental conservation penalty via [`Loss.elemental_conservation()`](src/loss.py:59)
   - `Loss.training()` prints per batch in [`print(...)`](src/loss.py:100)

7. **Backward + optimizer step**
   - gradient clipping in [`torch.nn.utils.clip_grad_norm_()`](src/trainer.py:314)
   - optimizer is AdamW with `fused=True` in [`load_objects()`](src/trainer.py:375)

### Validation batch

Validation is similar in [`EmulatorTrainerSequential._run_validation_batch()`](src/trainer.py:332), except:

- decoded outputs are reshaped back to `[B, T, S]`
- validation loss uses relative error per species via [`Loss.validation()`](src/loss.py:106)

### Epoch structure + stopping

Each epoch in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:347) does:

- training loop over batches
- validation loop over batches
- prints wallclock training/validation time for the epoch

The outer loop is in [`Trainer.train()`](src/trainer.py:177), which runs until early stopping in [`Trainer._check_early_stopping()`](src/trainer.py:96) based on stagnant validation metric.

## Tensor shape summary (emulator)

Let:

- `B = batch_size`
- `W = window_size` from [`EMConfig.window_size`](src/configs/emulator.py:31)
- `T = W - 1`
- `P = num_phys` from [`DatasetConfig.num_phys`](src/configs/datasets.py:48)
- `S = num_species` from [`DatasetConfig.num_species`](src/configs/datasets.py:50)
- `L = latent_dim` from [`AEConfig.latent_dim`](src/configs/autoencoder.py:29)

From [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197):

- `phys`: `[B, T, P]`
- `features` (initial latents): `[B, L]`
- `targets`: `[B, T, S]` (scaled abundances)

From [`Emulator.forward()`](src/models/emulator.py:35):

- `outputs_latent`: `[B, T, L]`

In [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314):

- decode input: `outputs_latent.reshape(-1, L)` → `[B*T, L]`
- decoded abundances: `[B*T, S]`
- targets reshaped: `targets.reshape(-1, S)` → `[B*T, S]`

## Pipeline facts that affect performance (for later analysis)

This section does **not** diagnose your specific slowdown; it just records “what’s in the hot path” so you can inspect later.

1. **The emulator forward is a python loop over timesteps**
   - [`Emulator.forward()`](src/models/emulator.py:35)
   - cost scales roughly with `T = window_size-1`

2. **Every training batch decodes `B*T` latents back to species space**
   - decode call is in [`EmulatorTrainerSequential._run_training_batch()`](src/trainer.py:314)
   - decode implementation is non-trivial (BatchNorms + matmuls) in [`Autoencoder.decode()`](src/models/autoencoder.py:55)

3. **Loss does extra math and prints per batch**
   - exponential loss and conservation matmul in [`Loss.training()`](src/loss.py:82)
   - per-batch printing is in [`print(...)`](src/loss.py:100)

4. **Dataloader uses 10 workers and builds batches via tensor indexing**
   - [`tensor_to_dataloader()`](src/data_loading.py:224)
   - dataset fetch does `rows = data_matrix[data_indices]` in [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)

5. **Autograd anomaly detection is enabled globally**
   - [`torch.autograd.set_detect_anomaly(True)`](src/trainer.py:28)

6. **Preprocessed HDF5 is loaded fully into CPU RAM before training starts**
   - [`load_tensors_from_hdf5()`](src/data_loading.py:84)

7. **HPC stdout/logging can become a bottleneck (especially with redirected output)**
   - In HPC runs, stdout is often written to a scheduler-managed log file on a shared filesystem.
   - Even modest per-step/per-batch logging can become *orders of magnitude* slower than local terminal output.

8. **CPU allocation vs. DataLoader worker count mismatch**
   - The emulator uses `num_workers=10` in [`tensor_to_dataloader()`](src/data_loading.py:224).
   - If the job only allocates a small number of CPUs (or the node is oversubscribed), worker processes can thrash and slow training dramatically.

9. **NUMA / CPU affinity / PCIe locality issues can slow host→device transfer**
   - Batches are prepared on CPU and then moved to GPU each step via `.to(self.device, non_blocking=True)` in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:347).
   - On HPC, binding the job to a CPU socket that is “far” from the GPU (NUMA mismatch) can reduce effective transfer bandwidth and increase latency.

10. **GPU clocks / power caps / partitioning (MIG/MPS) can make a top-end GPU look slow**
   - HPC systems sometimes run with power limits, lower application clocks, or GPU partitioning.
   - If the H200 is not running at expected clocks (or is shared), it may underperform drastically compared to a local, exclusive 3060.

11. **Different software stack (driver/CUDA/PyTorch) can change kernel quality and fallbacks**
   - HPC environments may pin older CUDA/PyTorch builds.
   - That can change whether certain kernels are fused/optimized, affecting performance even for the same model and batch shapes.

12. **Host CPU may be the real limiter (HPC CPU can be “worse” than a local desktop CPU for this workload)**
   - This training loop can be Python + launch + indexing heavy (see [`Emulator.forward()`](src/models/emulator.py:35) and [`EmulatorSequenceDataset.__getitems__()`](src/data_loading.py:197)).
   - If your local machine had a strong desktop CPU (e.g. a Ryzen 5 3600-class part) but the HPC node provides fewer cores per GPU, lower per-core clocks, or tighter CPU quotas, the H200 will sit idle more often.

13. **cgroup / scheduler limits on CPU time and memory bandwidth**
   - On HPC, SLURM (or similar) can enforce CPU quotas, memory limits, and cpusets.
   - If the job is allocated fewer CPU cores than you expect, or if other system daemons share the same cores, the DataLoader and Python overhead can balloon.

14. **`/dev/shm` (shared memory) constraints can cripple DataLoader performance**
   - PyTorch DataLoader workers rely heavily on shared memory for inter-process transfer.
   - Some HPC environments have small `tmpfs` for `/dev/shm`, causing slowdowns or worker issues when `num_workers` is high (see [`tensor_to_dataloader()`](src/data_loading.py:224)).

15. **Pinned-memory / locked-memory limits can make host→device transfer effectively synchronous**
   - The code uses pinned memory (`pin_memory=True`) plus non-blocking GPU transfers (e.g. `.to(..., non_blocking=True)` in [`EmulatorTrainerSequential._run_epoch()`](src/trainer.py:347)).
   - If the environment restricts locked memory (common on shared systems), pinning may be ineffective and transfers can serialize.

16. **Networked home / output paths can make checkpointing and metrics I/O unexpectedly expensive**
   - Even if the dataset is preloaded, training still writes checkpoints and loss JSONs via [`Trainer._save_checkpoint()`](src/trainer.py:82) and [`Trainer.save_loss_per_epoch()`](src/trainer.py:69).
   - On HPC, those paths are often on a shared filesystem; periodic writes can stall training or introduce long tail latency.

These are the main places where “time can go” in the emulator training pipeline, based on the current code structure.
