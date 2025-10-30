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

**Purpose:** Command-line interface for training and preprocessing

**Entry Points:**
- `train_autoencoder.py` - Autoencoder training workflow
- `train_emulator.py` - Emulator training workflow
- `preprocess.py` - Data preprocessing utilities

**Pattern:** Each CLI script uses Hydra decorators to load configurations, orchestrates setup via helper functions, and delegates training to specialized trainer classes.

**Registration:** Automatic via `pyproject.toml` entry points:
```toml
[project.scripts]
astrochemnet-train-autoencoder = "AstroChemNet.cli.train_autoencoder:main"
astrochemnet-train-emulator = "AstroChemNet.cli.train_emulator:main"
```

### 2. Configuration System

**Technology:** Hydra with structured configs (dataclasses)

**Schema Definitions** (`config_schemas.py`):
- `DatasetConfig` - Dataset paths, physical parameters, species lists, device settings
- `ModelsConfig` - Unified schema for both models using Optional fields
- `Config` - Top-level composition

**File Structure:**
```
configs/
├── config.yaml              # Autoencoder defaults
├── config_emulator.yaml     # Emulator defaults
├── datasets/
│   └── grav.yaml           # Dataset-specific settings
└── models/
    ├── autoencoder.yaml    # Model hyperparameters
    └── emulator.yaml       # Model hyperparameters
```

**Design Pattern:** Composition over inheritance - combine dataset + model configs via Hydra groups, override via CLI.

### 3. Data Pipeline

**Loading** (`data_loading.py`):
- `load_datasets()` - HDF5 file reading with chunked access
- `ChunkedShuffleSampler` - Memory-efficient shuffling for large datasets
- `EmulatorSequenceDataset` - Sequential time-series indexing
- `__getitems__` batching for 10³× speedup

**Processing** (`data_processing.py`):
- `Processing` class - Static methods for scaling and normalization
- `abundances_scaling()` - Log10 transformation with clipping
- `save_latents_minmax()` - Latent space normalization and caching

**Caching Strategy:**
- Preprocessed data saved to disk (`.pt` for autoencoder, `.h5` for emulator)
- Mode flags control preprocess/train/both workflow
- Avoids redundant computation across experiments

### 4. Training Infrastructure (`trainer.py`)

**Base `Trainer` Class:**
- Abstract training loop with hooks for customization
- Early stopping via validation plateau detection (patience=20)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for stability
- Adaptive dropout reduction on training plateau
- JSON metrics logging per epoch
- Best model checkpointing

**Specialized Trainers:**
- `AutoencoderTrainer` - Standard reconstruction training
- `EmulatorTrainerSequential` - Autoregressive sequence prediction with rollout

**Pattern:** Template method - base class defines workflow structure, subclasses implement model-specific logic in hooks.

### 5. Model Definitions (`models/`)

**Organization:**
- `autoencoder.py` - Tied-weight autoencoder architecture
- `emulator.py` - Residual prediction network

**Design Principles:**
- Models are pure PyTorch modules (nn.Module subclasses)
- Pretrained weight loading via static methods
- Device-agnostic initialization
- Inference mode support (frozen weights, eval mode)

### 6. Loss Functions (`loss.py`)

**Training Loss:**
- Multi-objective: reconstruction + worst-species penalty + conservation
- Exponential weighting emphasizes large errors
- Stoichiometric matrix enforces elemental conservation

**Validation Loss:**
- Per-species relative error (333-D vector)
- Used for early stopping decisions

**Pattern:** Loss functions are pure functions taking model outputs and targets, enabling easy experimentation.

### 7. Inference API (`inference.py`)

**`Inference` Class:**
- High-level interface for production use
- Methods: `encode()`, `decode()`, `latent_emulate()`, `emulate()`
- Automatic preprocessing/postprocessing
- Pretrained model loading

**Usage Pattern:**
```python
inf = Inference(dataset_cfg, processing, autoencoder, emulator)
predictions = inf.emulate(physical_params, initial_abundances)
```

### 8. Analysis Tools (`analysis.py`, `scripts/`)

**Components:**
- `analysis.py` - Plotting helpers and metric calculations
- `scripts/models/` - Jupyter notebooks for error analysis
- `scripts/datasets/` - Dataset exploration notebooks

## Critical Implementation Paths

### Training Workflow

**Autoencoder Pipeline:**
```
astrochemnet-train-autoencoder (CLI)
  ↓
train_autoencoder.py:main() [@hydra.main decorator]
  ↓
setup_config() [load species, stoichiometric matrix, add to cfg]
  ↓
load_datasets() [HDF5 → PyTorch tensors]
  ↓
Processing.abundances_scaling() [clip + log10 + normalize]
  ↓
AutoencoderTrainer.train() [training loop with early stopping]
  ↓
Save: weights/autoencoder.pth, utils/latents_minmax.npy
```

**Emulator Pipeline:**
```
astrochemnet-train-emulator (CLI)
  ↓
train_emulator.py:main() [@hydra.main decorator]
  ↓
Load pretrained autoencoder
  ↓
Encode datasets to latent space
  ↓
Save encoded sequences to HDF5
  ↓
EmulatorTrainerSequential.train() [autoregressive rollout]
  ↓
Save: weights/mlp.pth
```

### Inference Workflow

```
User code
  ↓
Inference(cfg, processing, models)
  ↓
inf.emulate(phys_params, abundances)
  ↓
  encode(abundances) → latent
  ↓
  for timestep: latent = latent + emulator(phys, latent)
  ↓
  decode(latent) → abundances
```

## File Organization

**Core Package:**
```
src/AstroChemNet/
├── cli/                      # Command-line entry points
│   ├── train_autoencoder.py
│   ├── train_emulator.py
│   └── preprocess.py
├── models/                   # Neural network architectures
│   ├── autoencoder.py
│   └── emulator.py
├── config_schemas.py         # Structured config dataclasses
├── trainer.py                # Base trainer + specialized trainers
├── loss.py                   # Loss function implementations
├── data_loading.py           # HDF5 loading + Dataset classes
├── data_processing.py        # Preprocessing utilities
├── inference.py              # Production inference API
├── analysis.py               # Plotting and metrics
└── utils.py                  # Helper functions
```

**Data Storage:**
```
data/                         # HDF5 datasets + preprocessed caches
weights/                      # Model checkpoints
utils/                        # Static arrays (species, matrices)
configs/                      # Hydra YAML configurations
outputs/                      # Training logs (timestamped)
```

**Analysis:**
```
scripts/models/               # Model error analysis notebooks
scripts/datasets/             # Dataset visualization notebooks
research/                     # Exploratory experiments
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
