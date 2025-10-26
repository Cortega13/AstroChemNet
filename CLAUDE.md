# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AstroChemNet is a deep learning framework for training surrogate models that emulate astrochemical networks. The project uses a two-stage architecture:
1. **Autoencoder** - Compresses chemical species abundances (333 species) into a lower-dimensional latent space (typically 12-14 dimensions)
2. **Emulator** - A sequential model that predicts evolution of latent representations over time given physical parameters

The models are trained on data from the UCLCHEM astrochemical network, with datasets stored in HDF5 format.

## Development Commands

### Setup
```bash
pip install -e .
pip install -r requirements.txt
```

### Training Models

Train the autoencoder first (required before training emulator):
```bash
python scripts/train/train_autoencoder.py
```

Train the emulator (requires pretrained autoencoder):
```bash
python scripts/train/train_emulator.py model=emulator
```

Override config parameters from command line:
```bash
python scripts/train/train_autoencoder.py model.lr=1e-4 model.batch_size=32768
python scripts/train/train_emulator.py dataset=grav model=emulator model.hidden_dim=256
```

### Code Quality

Lint and auto-fix code:
```bash
ruff check --fix
```

Format notebooks and Python files:
```bash
ruff format
```

## Architecture Details

### Two-Stage Model Pipeline

The system uses a decoupled approach where the autoencoder learns to compress/decompress chemical abundances, and the emulator learns temporal evolution in the compressed latent space:

1. **Autoencoder** (`nn_architectures/autoencoder.py`):
   - Encoder: 333 → 320 → 160 → latent_dim (typically 12-14)
   - Decoder: Tied weights (transposed encoder weights)
   - Batch normalization after each layer
   - GELU activation, Sigmoid output activation
   - Training noise injection in latent space for regularization

2. **Emulator** (`nn_architectures/emulator.py`):
   - Sequential predictor that takes physical parameters + current latent state
   - Predicts next latent state via residual updates
   - Input: `[phys_params, latent_state]` → Output: `latent_update`
   - Autoregressive: `latent[t+1] = latent[t] + emulator(phys[t], latent[t])`

### Configuration System

**Uses Hydra for configuration management** with structured YAML configs and CLI overrides.

#### Configuration Structure

```
configs/
├── config.yaml              # Main config with defaults
├── datasets/
│   └── grav.yaml           # Dataset config (physical params, paths, species)
└── models/
    ├── autoencoder.yaml    # Autoencoder hyperparameters
    └── emulator.yaml       # Emulator hyperparameters
```

#### Structured Config Schemas

Defined in `src/AstroChemNet/config_schemas.py`:

- **DatasetConfig** - Dataset paths, physical parameter ranges, device settings, species lists (replaces old `GeneralConfig`)
- **ModelsConfig** - Single reusable model schema for both autoencoder and emulator; uses `Optional` fields for model-specific parameters
- **Config** - Top-level composition of `dataset` and `model` configs

#### Config Access Pattern

Training scripts receive OmegaConf DictConfig objects:
```python
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = cfg.dataset.device
    lr = cfg.model.lr
    batch_size = cfg.model.batch_size
```

Computed fields (e.g., `num_species`, loaded numpy arrays) are added via `setup_config()` helper in training scripts.

#### CLI Overrides

Override any config value from command line:
```bash
python train_autoencoder.py model.lr=5e-4 model.dropout=0.2
python train_emulator.py dataset=grav model=emulator model.window_size=128
```

#### Legacy Config Classes (Deprecated)

Old Python class-based configs in `configs/{general,autoencoder,emulator}.py` are deprecated. They emit warnings pointing to Hydra configs. Will be removed in future version.

### Data Flow

1. Raw data loaded from HDF5 (`data_loading.load_datasets()`)
2. Abundances clipped to [1e-20, 1.0] range
3. Scaled via `Processing.abundances_scaling()` (log10 transformation)
4. Autoencoder trained on scaled abundances
5. Latent representations computed and scaled via `Processing.save_latents_minmax()`
6. Emulator trained on sequences in latent space

### Custom Loss Functions

Located in `src/AstroChemNet/loss.py`:

- **Training Loss**: Combines reconstruction error, worst-species penalty, and elemental conservation
  - Exponential weighting emphasizes larger errors
  - Conservation loss: ensures elemental abundances are preserved (via stoichiometric matrix)

- **Validation Loss**: Species-wise mean relative error
  - Returns per-species loss vector (333 elements)
  - Used for early stopping and learning rate scheduling

### Training Infrastructure

`src/AstroChemNet/trainer.py` contains base `Trainer` class with:
- Automatic early stopping based on validation stagnation
- Adaptive dropout rate reduction when training stagnates
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping
- Best model checkpoint saving
- Detailed per-epoch metrics logging to JSON

Specialized trainers:
- `AutoencoderTrainer` - Standard reconstruction training
- `EmulatorTrainerSequential` - Sequential prediction with autoregressive rollout

### Data Loading Optimizations

`src/AstroChemNet/data_loading.py` implements efficient batching:

- `ChunkedShuffleSampler`: Shuffles data in chunks to manage large datasets
- `__getitems__` batching: Loads entire batch in single call (10^3x faster than individual loads)
- `EmulatorSequenceDataset`: Memory-efficient sequence dataset using index arrays
- Pin memory enabled for GPU transfer

## Key Implementation Details

### Physical Parameters

The model tracks 4 physical parameters (defined in `configs/datasets/grav.yaml`):
- Density: H nuclei per cm³
- Radfield: Habing field (radiation field strength)
- Av: Visual extinction in magnitudes
- gasTemp: Gas temperature in Kelvin

All physical parameters are sampled and stored in log space.

### Species Abundances

- 333 chemical species tracked (list in `utils/species.txt`)
- All abundances relative to H nuclei (max = 1.0)
- Clipped to [1e-20, 1.0] before processing
- Log10 scaling applied for training

### Stoichiometric Matrix

Matrix stored in `utils/stoichiometric_matrix.npy` maps species abundances to elemental abundances. Used in conservation loss to ensure physical validity (atoms can't be created/destroyed).

### Pretrained Model Loading

Both model classes have `load_autoencoder()` and `load_emulator()` functions:
- Check for pretrained weights at configured path
- Set `inference=True` to freeze weights and set eval mode
- Models default to CPU unless CUDA available

### Weight Tying

Autoencoder uses tied weights (decoder reuses transposed encoder weights) to:
- Reduce parameter count
- Improve generalization
- Enforce symmetry in latent space

## File Organization

### Core Package (`src/AstroChemNet/`)
- `trainer.py` - Training loops and optimization logic
- `loss.py` - Custom loss functions with conservation constraints
- `data_loading.py` - HDF5 loading, Dataset/DataLoader implementations
- `data_processing.py` - Scaling, normalization, preprocessing utilities
- `inference.py` - Inference pipelines and evaluation
- `analysis.py` - Plotting and metrics helpers
- `utils.py` - Shared helper functions

### Model Definitions (`nn_architectures/`)
- `autoencoder.py` - Tied-weight autoencoder with batch norm
- `emulator.py` - Sequential latent dynamics predictor

### Experiments
- `vibecode/` - ViBe baseline experiments and comparison scripts
- `research/` - Exploratory Jupyter notebooks
- `scripts/analysis/` - Post-training analysis notebooks

### Data Storage
- `data/` - Local HDF5 datasets (not in repo, download separately)
- `weights/` - Model checkpoints (archived_original/ contains reference models)
- `utils/` - Cached arrays (species list, stoichiometric matrix, initial abundances)

## Notes on Dependencies

The `requirements.txt` file appears to have encoding issues. Expected dependencies:
- PyTorch (with CUDA 12.6 support)
- NumPy, Pandas, SciPy
- H5py, tables (HDF5 support)
- Matplotlib, Seaborn (visualization)
- tqdm (progress bars)
- Numba (JIT compilation)

## Common Workflows

### Training a New Model

1. Ensure dataset exists at path specified in `configs/datasets/grav.yaml`
2. Configure hyperparameters by editing YAML files in `configs/models/` or via CLI overrides
3. Train autoencoder first: `python scripts/train/train_autoencoder.py`
4. After autoencoder converges, train emulator: `python scripts/train/train_emulator.py model=emulator`

Example with hyperparameter tuning:
```bash
python scripts/train/train_autoencoder.py model.lr=5e-4 model.dropout=0.25 model.batch_size=32768
```

### Evaluating Models

Use `src/AstroChemNet/inference.py` which provides the `Inference` class for:
- Encoding abundances to latent space
- Decoding latent representations to abundances
- Running emulator predictions

### Adding New Datasets

1. Create new YAML file in `configs/datasets/` (e.g., `turbulent.yaml`)
2. Define dataset path, physical parameter ranges, and species list
3. Train with: `python scripts/train/train_autoencoder.py dataset=turbulent`

### Adding New Model Variants

1. Create new YAML file in `configs/models/` (e.g., `autoencoder_large.yaml`)
2. Define architecture and hyperparameters (reuses `ModelsConfig` schema)
3. Train with: `python scripts/train/train_autoencoder.py model=autoencoder_large`

### Adding New Species or Physical Parameters

1. Update dataset YAML in `configs/datasets/` with new parameter ranges/species path
2. Regenerate `utils/stoichiometric_matrix.npy` if species changed
3. Update `input_dim` in model YAML to match new dimensions
4. Hydra will handle config composition automatically