# Preprocessing Restructuring Plan

## Overview

This plan outlines the restructuring of dataset preprocessing to create a dedicated `outputs/preprocessing/<dataset_name>/` folder structure. All preprocessing artifacts will be generated during preprocessing and stored in this folder, rather than being computed during training or loaded from scattered locations.

## Current State Analysis

### Current File Locations
- `outputs/utils/initial_abundances.npy` - Initial abundances array
- `outputs/utils/species.txt` - Species list
- `outputs/utils/stoichiometric_matrix.npy` - Stoichiometric matrix

### Current Issues
1. **Hardcoded paths** in `GeneralConfig` that assume specific file locations
2. **Stoichiometric matrix** is built in `data_processing.py` but loaded from config
3. **Physical parameter ranges** are hardcoded in `GeneralConfig` rather than computed from data
4. **Species list** is loaded separately but could be generated during preprocessing
5. **No dataset versioning** - all datasets share the same `outputs/utils/` folder

## Proposed Structure

### New Directory Layout
```
outputs/
├── preprocessing/
│   └── grav_collapse/           # Dataset-specific folder
│       ├── metadata.json        # Dataset metadata (version, creation date, etc.)
│       ├── species.json         # Species list with metadata
│       ├── initial_abundances.npy
│       ├── stoichiometric_matrix.npy
│       ├── physical_parameter_ranges.json  # Min/max for each physical parameter
│       └── train_val_split.json # Train/validation split indices
└── utils/                       # Deprecated - will be removed after migration
```

### File Descriptions

#### `metadata.json`
```json
{
  "dataset_name": "grav_collapse",
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "source_file": "data/gravitational_collapse.h5",
  "num_species": 333,
  "num_physical_params": 4,
  "num_train_samples": 1500000,
  "num_val_samples": 500000,
  "train_val_split_ratio": 0.75
}
```

#### `species.json`
```json
{
  "species": ["H2", "H", "HE", "..."],
  "num_species": 333,
  "elements": ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]
}
```

#### `physical_parameter_ranges.json`
```json
{
  "Density": {"min": 68481.0, "max": 1284211415.0, "unit": "H nuclei per cm^3"},
  "Radfield": {"min": 1e-4, "max": 26.0, "unit": "Habing field"},
  "Av": {"min": 0.1, "max": 6914.0, "unit": "Magnitudes"},
  "gasTemp": {"min": 13.0, "max": 133.0, "unit": "Kelvin"}
}
```

#### `train_val_split.json`
```json
{
  "train_models": ["model_001", "model_002", "..."],
  "val_models": ["model_003", "model_005", "..."],
  "seed": 42
}
```

## Implementation Plan

### Phase 1: Create Preprocessing Output Manager

#### Step 1.1: Create `src/preprocessing/preprocessing_output.py`
- Create `PreprocessingOutput` class to manage preprocessing artifacts
- Methods:
  - `save_metadata()` - Save dataset metadata
  - `save_species()` - Save species list
  - `save_stoichiometric_matrix()` - Build and save stoichiometric matrix
  - `save_physical_parameter_ranges()` - Compute and save min/max values
  - `save_initial_abundances()` - Save initial abundances
  - `save_train_val_split()` - Save train/validation split info
  - `load_all()` - Class method to load all artifacts
  - `get_output_path()` - Get path for a specific artifact

#### Step 1.2: Update `GeneralConfig` class
- Add `dataset_name` parameter
- Load all settings from `outputs/preprocessing/<dataset_name>/`
- Properties:
  - `dataset_name`
  - `preprocessing_path`
  - `species`
  - `num_species`
  - `physical_parameter_ranges`
  - `stoichiometric_matrix`
  - `initial_abundances`

### Phase 2: Update Preprocessing Functions

#### Step 2.1: Modify `preprocessing.py`
- Update `preprocess_gravitational_collapse()` to:
  1. Create output directory `outputs/preprocessing/grav_collapse/`
  2. Compute physical parameter ranges from actual data
  3. Extract and save species list
  4. Build and save stoichiometric matrix
  5. Save initial abundances
  6. Save train/validation split information
  7. Save metadata

#### Step 2.2: Add CLI support
- Add `--dataset-name` argument to specify output folder
- Add `--force` flag to overwrite existing preprocessing

### Phase 3: Update Configuration Classes

#### Step 3.1: Refactor `GeneralConfig`
- Remove hardcoded paths and values
- Add `dataset_name` parameter
- Load all settings from preprocessing output folder

#### Step 3.2: Update dependent configs
- Update `AEConfig` to use `GeneralConfig` preprocessing paths
- Update `EMConfig` to use `GeneralConfig` preprocessing paths

### Phase 4: Update Data Processing

#### Step 4.1: Modify `data_processing.py`
- Remove `build_stoichiometric_matrix()` method (moved to preprocessing)
- Update `Processing` class to accept `GeneralConfig`
- Load stoichiometric matrix from preprocessing output

#### Step 4.2: Update `data_loading.py`
- Update to use `GeneralConfig` preprocessing paths
- Load species from preprocessing output

### Phase 5: Update Training Scripts

#### Step 5.1: Update `train_autoencoder.py`
- Load config from preprocessing output
- Remove stoichiometric matrix generation

#### Step 5.2: Update `train_emulator.py`
- Load config from preprocessing output

### Phase 6: Migration and Cleanup

#### Step 6.1: Create migration script
- Script to move existing `outputs/utils/` files to new structure
- Generate metadata for existing datasets

#### Step 6.2: Delete old paths
- Delete `outputs/utils/` folder entirely
- Update documentation

## Code Changes Summary

### New Files
1. `src/preprocessing/preprocessing_output.py` - Preprocessing output manager

### Modified Files
1. `src/configs/general.py` - Refactored to load from preprocessing
2. `src/preprocessing/preprocessing.py` - Generate and save all artifacts
3. `src/data_processing.py` - Remove stoichiometric matrix building
4. `src/data_loading.py` - Use preprocessing paths
5. `src/training/train_autoencoder.py` - Use new config loading
6. `src/training/train_emulator.py` - Use new config loading

### Deleted Files
1. `outputs/utils/` - Entire folder will be deleted after migration to new structure

## Benefits

1. **Dataset Versioning**: Each dataset has its own preprocessing folder
2. **Reproducibility**: All preprocessing artifacts are in one place
3. **Flexibility**: Easy to add new datasets with different parameters
4. **Separation of Concerns**: Preprocessing generates artifacts, training consumes them
5. **Data Provenance**: Metadata tracks source and creation time

## No Backward Compatibility

- **Full commitment to new structure** - No deprecation warnings or compatibility layers
- Clean break from old `outputs/utils/` folder
- `GeneralConfig` will be fully refactored to load from preprocessing output
- Old files in `outputs/utils/` will be deleted after migration
- Simpler, cleaner codebase with no legacy support

## Testing Plan

1. Run preprocessing on existing dataset
2. Verify all artifacts are saved correctly
3. Load config from preprocessing output
4. Run training with new config
5. Compare results with old workflow

## Timeline

- Phase 1: Preprocessing output manager (1-2 hours)
- Phase 2: Update preprocessing functions (1-2 hours)
- Phase 3: Update configuration classes (1 hour)
- Phase 4: Update data processing (30 minutes)
- Phase 5: Update training scripts (30 minutes)
- Phase 6: Migration and cleanup (1 hour)

**Total Estimated Time: 5-7 hours**
