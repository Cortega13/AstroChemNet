# System Architecture

## Framework Overview

AstroChemNet is a Python package that provides a complete framework for training, evaluating, and deploying neural network surrogate models for astrochemical simulations. The system follows a modular, configuration-driven architecture that separates concerns and enables flexible experimentation.

## High-Level Design

```
CLI Commands → Hydra Config → Data Pipeline → Model Training → Inference API
     ↓              ↓              ↓                ↓              ↓
Entry Points   YAML Files    HDF5 Loading     Trainer Classes   Production Use
```

## Core Components

### 1. CLI System (`src/AstroChemNet/cli/`)

**Entry Points:**
- `train_autoencoder.py` - Autoencoder training workflow
- `train_emulator.py` - Emulator training workflow
- `preprocess.py` - Data preprocessing utilities

**Pattern:** Hydra decorators load configs, delegate to trainer classes.

**Registration:** `pyproject.toml` entry points → `astrochemnet-train-*` commands

### 2. Configuration System

**Technology:** Hydra with structured configs (dataclasses)

**Schemas** (`config_schemas.py`): `DatasetConfig`, `ModelsConfig`, `Config`

**Structure:**
```
configs/
├── config.yaml / config_emulator.yaml
├── datasets/grav.yaml
└── models/autoencoder.yaml / emulator.yaml
```

**Pattern:** Composition via Hydra groups, CLI overrides

### 3. Data Pipeline

**Loading** (`data_loading.py`):
- `load_datasets()` - HDF5 chunked access
- `ChunkedShuffleSampler` - Memory-efficient shuffling
- `EmulatorSequenceDataset` - Time-series indexing
- `__getitems__` batching (10³× speedup)

**Processing** (`data_processing.py`):
- `Processing` - Scaling/normalization static methods
- `abundances_scaling()` - Log10 transform + clipping
- `save_latents_minmax()` - Latent normalization + caching

**Caching:** `.pt` (autoencoder) / `.h5` (emulator), mode flags control preprocess/train/both

### 4. Training Infrastructure (`trainer.py`)

**Base `Trainer`:**
- Early stopping (patience=20), LR scheduling, gradient clipping
- Adaptive dropout reduction, JSON metrics, checkpointing

**Specialized:** `AutoencoderTrainer`, `EmulatorTrainerSequential` (autoregressive rollout)

**Pattern:** Template method - base workflow, subclass hooks

### 5. Model Definitions (`models/`)

- `autoencoder.py` - Tied-weight architecture
- `emulator.py` - Residual prediction network

**Design:** Pure PyTorch modules, device-agnostic, pretrained loading

### 6. Loss Functions (`loss.py`)

**Training:** Multi-objective (reconstruction + worst-species penalty + conservation)

**Validation:** Per-species relative error (333-D) for early stopping

**Pattern:** Pure functions for easy experimentation

### 7. Inference API (`inference.py`)

**`Inference` Class:** Production interface with `encode()`, `decode()`, `latent_emulate()`, `emulate()`

```python
inf = Inference(dataset_cfg, processing, autoencoder, emulator)
predictions = inf.emulate(physical_params, initial_abundances)
```

### 8. Analysis Tools

- `analysis.py` - Plotting/metrics
- `scripts/models/` - Error analysis notebooks
- `scripts/datasets/` - Dataset exploration

## Critical Implementation Paths

**Autoencoder:** CLI → `@hydra.main` → `setup_config()` → `load_datasets()` → `Processing.abundances_scaling()` → `AutoencoderTrainer.train()` → Save `weights/autoencoder.pth`, `utils/latents_minmax.npy`

**Emulator:** CLI → Load pretrained → Encode datasets → Save sequences → `EmulatorTrainerSequential.train()` → Save `weights/mlp.pth`

**Inference:** `Inference(cfg, processing, models)` → `inf.emulate()` → encode → autoregressive rollout → decode

## File Organization

```
src/AstroChemNet/
├── cli/                      # train_autoencoder.py, train_emulator.py, preprocess.py
├── models/                   # autoencoder.py, emulator.py
├── config_schemas.py         # Structured configs
├── trainer.py                # Base + specialized trainers
├── loss.py, data_loading.py, data_processing.py, inference.py, analysis.py, utils.py

data/                         # HDF5 + preprocessed caches
weights/                      # Model checkpoints
utils/                        # Static arrays (species, matrices)
configs/                      # Hydra YAMLs
outputs/                      # Training logs (timestamped)
scripts/                      # Notebooks (models/, datasets/)
```

## Key Design Patterns

1. **Configuration Composition** - Hydra combines dataset + model configs via groups
2. **Template Method Pattern** - Base `Trainer` class defines workflow, subclasses customize steps
3. **Factory Pattern** - CLI scripts instantiate models/trainers based on config
4. **Caching Layer** - Expensive preprocessing saved to disk for reuse
5. **Mode-based Execution** - `mode` flag controls preprocess/train/both workflows
6. **Device Agnostic** - Automatic CUDA/CPU detection throughout
7. **Modular Losses** - Loss functions as composable pure functions

## Extension Points

**Adding New Datasets:**
1. Create YAML in `configs/datasets/`
2. Update `input_dim` if species count changes
3. Override via CLI: `dataset=new_dataset`

**Adding New Models:**
1. Create YAML in `configs/models/`
2. Optionally extend model classes in `models/`
3. Override via CLI: `model=new_model`

**Adding New Trainers:**
1. Subclass `Trainer` in `trainer.py`
2. Implement `train_epoch()` and `validate_epoch()` hooks
3. Update CLI scripts to use new trainer class

**Adding Analysis Tools:**
1. Add helper functions to `analysis.py`
2. Create notebooks in `scripts/` for visualization
3. Use `Inference` class for model predictions
