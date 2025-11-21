# Missing Components for Autoencoder Training

Based on a review of the codebase, the following components or configurations appear to be missing or require attention to successfully train the autoencoder model as described in the architecture documentation.

## 1. Configuration Inconsistencies

### `configs/components/autoencoder_grav.yaml`
- **Missing `latents_minmax_path`**: The `Processing` class in `src/data_processing.py` expects `autoencoder_cfg.latents_minmax_path` to be defined when `autoencoder_cfg` is provided (lines 57, 180). This path is used to save/load the min/max values of the latent space for normalization. It is currently missing from the component config.
- **Missing `dropout_decay_patience` and `dropout_reduction_factor`**: The `AutoencoderTrainer` in `src/trainers/autoencoder_trainer.py` (lines 204-210) attempts to access these parameters from `cfg.component`. While defaults are provided in the code (10 and 0.1 respectively), it is better practice to make them explicit in the configuration file.

### `configs/data/grav.yaml`
- **Missing `dataset_path`**: The `dataset_path` is set to `null` (line 42). While the `AbundancesOnlyPreprocessor` saves `.pt` files directly, the `load_dataset` function in `src/data_loading.py` (lines 24-37) uses `dataset_cfg.dataset_path` to load HDF5 splits. The preprocessor should likely update this path or the loader should be aware of the `.pt` file locations.
- **Inconsistent `phys` vs `physical_parameters`**: The `InitialPreprocessor` in `src/preprocessors/initial.py` (line 83) accesses `self.dataset_cfg.phys`. However, `configs/data/grav.yaml` defines `physical_parameters`. The config should likely include an alias or the code should be updated to use `physical_parameters`.

## 2. Preprocessing Logic Gaps

### `src/preprocessors/abundances_only.py`
- **Hardcoded Output Paths**: The preprocessor saves files as `autoencoder_train_preprocessed.pt` and `autoencoder_val_preprocessed.pt` (lines 41-42). The `AutoencoderTrainer` expects these exact names (lines 40-41). This tight coupling is fragile.
- **Missing `dataset_path` Update**: The preprocessor does not seem to update the `dataset_path` in the configuration or provide a mechanism for the trainer to know where the *original* HDF5 data is if it needed it (though `AutoencoderTrainer` seems to rely solely on the `.pt` files).

## 3. Trainer Implementation Details

### `src/trainers/autoencoder_trainer.py`
- **`_save_latents_minmax` Dependency**: The `save_weights` method calls `_save_latents_minmax` (line 107). This method (lines 109-125) calculates min/max latents on the validation set and saves them to `latents_minmax.npy`. It relies on `self.output_dir`, which is correctly set in `BaseTrainer`. However, it doesn't update the config to point to this new file for future use (e.g., by the emulator).
- **Metric Logging**: The `metrics` list is updated with a dictionary (lines 234-242), but `BaseTrainer` also appends to `metrics` (lines 37-44). This might lead to duplicate or conflicting entries if not managed carefully. `AutoencoderTrainer` seems to be accessing `self.metrics[-1]` which implies it expects `BaseTrainer` to have already appended an entry, but `BaseTrainer.train` calls `train_epoch` and `validate_epoch` *before* appending to metrics. This will likely cause an `IndexError`.

## 4. Data Loading

### `src/data_loading.py`
- **`AutoencoderDataset` Memory Usage**: The `__init__` method (line 114) prints memory usage. Ensure that for large datasets, this doesn't become a bottleneck. The current implementation loads the entire dataset into memory (RAM) via `torch.load` in the trainer. For the full dataset (~3M samples x 333 floats), this is ~4GB, which is manageable, but could be an issue if the dataset grows.

## 5. Missing Files/Paths

- **`outputs/utils/grav/species.txt`**: Referenced in `configs/data/grav.yaml`. Ensure this file exists or is generated.
- **`outputs/utils/grav/stoichiometric_matrix.npy`**: Referenced in `configs/data/grav.yaml`.
- **`outputs/utils/grav/initial_abundances.npy`**: Referenced in `configs/data/grav.yaml`.

## Recommendations

1.  **Update `configs/components/autoencoder_grav.yaml`**:
    *   Add `latents_minmax_path: ${paths.weights}/${name}/latents_minmax.npy` (or similar).
    *   Add `dropout_decay_patience` and `dropout_reduction_factor`.

2.  **Fix `AutoencoderTrainer.train_epoch` / `BaseTrainer` Interaction**:
    *   The `AutoencoderTrainer` tries to update `self.metrics[-1]` inside `_run_epoch` (called by `train_epoch`?), but `BaseTrainer` appends to `metrics` *after* `train_epoch` returns.
    *   *Correction*: `AutoencoderTrainer` does *not* use `_run_epoch` in the provided code snippet. It implements `train_epoch` and `validate_epoch` directly.
    *   *Issue*: `AutoencoderTrainer.validate_epoch` (lines 234-242) tries to update `self.metrics[-1]`. At this point in `BaseTrainer.train` loop, `self.metrics` has *not* yet been appended to for the current epoch. This will raise an `IndexError` on the first epoch.
    *   *Fix*: Return the metrics dict from `validate_epoch` or `train_epoch` and let `BaseTrainer` handle the appending, or override `train` completely if the flow is too different.

3.  **Verify `phys` vs `physical_parameters`**:
    *   Check `src/preprocessors/initial.py` line 83: `physical = tdf[self.dataset_cfg.phys].shift(-1)`.
    *   Check `configs/data/grav.yaml`: It defines `physical_parameters` but not `phys`.
    *   *Fix*: Add `phys: ${physical_parameters.names}` to `configs/data/grav.yaml` or update the code.

4.  **Ensure `latents_minmax_path` is defined**:
    *   The `Processing` class needs it. It should probably be defined in the `autoencoder` config or passed dynamically.
