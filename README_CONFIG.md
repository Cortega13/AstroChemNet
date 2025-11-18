# AstroChemNet Configuration Guide

## Overview

AstroChemNet now uses a simplified, single-file configuration system with one root config defining general settings like paths, device, and output directories.

## Configuration Structure

### Root Config: `config.yaml`

Located in the project root, this file defines:
- **Device**: `cuda` or `cpu` (default: `cuda`)
- **Working directory**: Auto-resolved from environment
- **Random seed**: For reproducibility (default: `42`)
- **Path structure**: All output directories

```yaml
device: cuda
working_dir: ${oc.env:PWD}
seed: 42

paths:
  data: data
  outputs: outputs
  weights: ${paths.outputs}/weights
  preprocessed: ${paths.outputs}/preprocessed
  runs: ${paths.outputs}/runs
  utils: ${paths.outputs}/utils
```

### Component Configs: `configs/components/`

Define specific model architectures and hyperparameters:
- `autoencoder_grav.yaml` - Autoencoder configuration
- `emulator_grav.yaml` - Emulator configuration

Each component specifies:
- Dataset dependency (`dataset: grav`)
- Preprocessing method (`preprocessing_method: timeless` or `sequential`)
- Model architecture (hidden dims, latent dim)
- Training hyperparameters (lr, batch size, epochs)
- Loss weights

### Dataset Configs: `configs/data/`

Define dataset-specific settings:
- `grav.yaml` - Gravitational collapse dataset

Specifies:
- Raw data path
- Species information
- Physical parameter ranges
- Train/validation split

### Preprocessing Configs: `configs/preprocessing/`

Define data preprocessing methods:
- `abundances_only.yaml` - Simple abundance preprocessing
- `initial.yaml` - Initial state preprocessing
- `markovianautoregressive.yaml` - Sequential preprocessing

## Usage

### Training

```bash
# Train autoencoder
python train.py component=autoencoder_grav

# Train emulator
python train.py component=emulator_grav

# Override device
python train.py component=autoencoder_grav device=cpu

# Override hyperparameters
python train.py component=autoencoder_grav component.lr=1e-4 component.batch_size=32768
```

### Preprocessing

```bash
# Preprocess dataset for autoencoder
python preprocess.py grav abundances_only

# Preprocess dataset for emulator (sequential)
python preprocess.py grav markovianautoregressive
```

## Migration from Old Config System

**Old system** had multiple config files:
- `configs/config.yaml` - Autoencoder-specific
- `configs/config_emulator.yaml` - Emulator-specific
- `configs/train.yaml` - Training orchestration

**New system** consolidates general settings into `config.yaml` in the root directory, while component-specific settings remain in `configs/components/`.

### Key Changes

1. **Centralized paths**: All directory paths defined in root `config.yaml`
2. **Explicit component selection**: Must specify component via CLI
3. **Dynamic config loading**: `train.py` loads and merges configs at runtime
4. **Device config**: Specified in root config, can be overridden per run

### Path Resolution

The system uses Hydra's variable interpolation:
- `${oc.env:PWD}` - Current working directory
- `${paths.weights}` - References `paths.weights` from config
- `${now:%Y-%m-%d}` - Timestamped output directories

## File Structure

```
AstroChemNet/
├── config.yaml                    # Root config (device, paths, seed)
├── train.py                       # Training entrypoint
├── preprocess.py                  # Preprocessing entrypoint
├── configs/
│   ├── components/                # Model architectures
│   │   ├── autoencoder_grav.yaml
│   │   └── emulator_grav.yaml
│   ├── data/                      # Dataset definitions
│   │   └── grav.yaml
│   └── preprocessing/             # Preprocessing methods
│       ├── abundances_only.yaml
│       ├── initial.yaml
│       └── markovianautoregressive.yaml
├── outputs/
│   ├── weights/                   # Trained model weights
│   ├── preprocessed/              # Cached preprocessed data
│   └── runs/                      # Training logs (timestamped)
└── data/                          # Raw datasets
```

## Examples

### Train autoencoder with custom settings

```bash
python train.py \
  component=autoencoder_grav \
  device=cuda \
  component.lr=5e-4 \
  component.batch_size=65536
```

### Train emulator after preprocessing

```bash
# 1. Preprocess data
python preprocess.py grav markovianautoregressive

# 2. Train emulator
python train.py component=emulator_grav
```

### Override output directories

```bash
python train.py \
  component=autoencoder_grav \
  paths.weights=custom_weights \
  paths.runs=custom_runs
