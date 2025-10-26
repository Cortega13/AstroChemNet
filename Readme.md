### Note: This package is currently under development. Any and all suggestions for improvements are welcome.

## Surrogate Modeling for Astrochemical Networks.
This package contains training/testing procedures for training deep neural network surrogate models for astrochemical networks. We use the [UCLCHEM](https://github.com/uclchem/UCLCHEM) chemical network for our datasets. Datasets are available upon request.

## Project Structure
```
AstroChemNet/
├── configs/                    # Training and evaluation configuration objects.
├── data/                       # Drop-in location for local .h5 datasets.
├── nn_architectures/           # PyTorch module definitions for the surrogate models.
├── plots/
│   ├── assets/                 # Static images used across documentation/notebooks.
│   └── trajectories/           # Generated trajectory visualisations.
├── research/                   # Exploratory notebooks testing alternative approaches.
├── scripts/
│   ├── analysis/               # Notebooks for post-training investigation.
│   ├── preprocessing/          # Data-prep notebooks and utilities.
│   └── train/                  # CLI entry points and helpers for training jobs.
├── src/
│   └── AstroChemNet/           # Installable package code.
│       ├── analysis.py         # Plotting and metrics helpers.
│       ├── data_loading.py     # Dataset loaders and batching helpers.
│       ├── data_processing.py  # Pre-/post-processing utilities.
│       ├── inference.py        # Inference pipelines.
│       ├── loss.py             # Loss definitions used during training.
│       ├── trainer.py          # High-level training loop.
│       └── utils.py            # Shared helper functions.
├── utils/                      # Cached numpy arrays and lookup tables.
├── vibecode/                   # ViBe baseline experiments and scripts.
│   └── archive/                # Historical utilities for ViBe experiments.
├── weights/
│   └── archived_original/      # Reference checkpoints and experiment artefacts.
├── .gitignore                  # Git ignore rules.
├── README.md                   # Project documentation (this file).
├── pyproject.toml              # Project configuration for packaging.
└── requirements.txt            # Python dependency lockfile for development.
```
## Setup Guide
Clone the repository.

```sh
git clone https://github.com/Cortega13/AstroChemNet.git
```

Enter the project directory.

```sh
cd AstroChemNet
```

Install the package using pip. Note: No necessary package dependencies are defined yet, so you must download them separately.

```sh
pip install -e .
```

## Configuration System

AstroChemNet uses [Hydra](https://hydra.cc/) for hierarchical configuration management, allowing flexible training setups with YAML files and CLI overrides.

### Configuration Structure

```
configs/
├── config.yaml              # Autoencoder training config
├── config_emulator.yaml     # Emulator training config
├── preprocess.yaml          # Data preprocessing config
├── datasets/
│   └── grav.yaml           # Dataset paths and physical parameters
└── models/
    ├── autoencoder.yaml    # Autoencoder hyperparameters
    └── emulator.yaml       # Emulator hyperparameters
```

### Config Files Explained

#### Dataset Config (`configs/datasets/grav.yaml`)
Defines data paths, physical parameter ranges, and species information:
- `dataset_path`: Path to HDF5 dataset
- `physical_parameter_ranges`: Min/max values for density, radiation field, visual extinction, temperature
- `species_path`: Path to species list file
- `stoichiometric_matrix_path`: Path to element conservation matrix
- `metadata`: Column names for trajectory metadata (Index, Model, Time)
- `phys`: Physical parameter column names

#### Model Configs (`configs/models/`)
**Autoencoder** (`autoencoder.yaml`):
- Architecture: `input_dim`, `hidden_dims`, `latent_dim`
- Training: `lr`, `batch_size`, `dropout`, `noise`
- Paths: `pretrained_model_path`, `save_model_path`, `latents_minmax_path`

**Emulator** (`emulator.yaml`):
- Architecture: `input_dim`, `output_dim`, `hidden_dim`, `window_size`
- Training: `lr`, `batch_size`, `dropout`
- Paths: `pretrained_model_path`, `save_model_path`

#### Top-Level Configs
**Autoencoder Training** (`config.yaml`):
```yaml
defaults:
  - datasets: grav
  - models: autoencoder

device: cuda
mode: both  # Options: preprocess, train, both
```

**Emulator Training** (`config_emulator.yaml`):
```yaml
defaults:
  - datasets: grav
  - models@autoencoder: autoencoder
  - models@model: emulator

device: cuda
mode: both
```

### Training Commands

After installation, use the CLI commands:

```bash
# Train autoencoder (required first)
astrochemnet-train-autoencoder

# Train emulator (requires pretrained autoencoder)
astrochemnet-train-emulator

# Preprocess raw UCLCHEM data
astrochemnet-preprocess
```

### Training Modes

All training commands support three execution modes via the `mode` parameter:

- **`mode=preprocess`**: Only preprocess data and save to disk
- **`mode=train`**: Only train (loads preprocessed data from disk)
- **`mode=both`**: Preprocess and train in one run (default)

This allows you to preprocess once and run multiple training experiments:

```bash
# Preprocess data once
astrochemnet-train-autoencoder mode=preprocess

# Run multiple training experiments with different hyperparameters
astrochemnet-train-autoencoder mode=train model.lr=1e-3
astrochemnet-train-autoencoder mode=train model.lr=5e-4 model.dropout=0.2
```

### Overriding Config Parameters

Override any parameter from the command line using Hydra syntax:

```bash
# Override learning rate and batch size
astrochemnet-train-autoencoder model.lr=5e-4 model.batch_size=32768

# Change dataset
astrochemnet-train-autoencoder dataset=turbulent

# Use different model variant
astrochemnet-train-autoencoder model=autoencoder_large

# Multiple overrides
astrochemnet-train-emulator model.lr=1e-3 model.window_size=128 device=cpu
```

### Config Schemas

Configuration schemas are defined in `src/AstroChemNet/config_schemas.py` using Python dataclasses:

- **`DatasetConfig`**: Dataset paths, physical parameters, species information
- **`ModelsConfig`**: Reusable model configuration for both autoencoder and emulator
- **`Config`**: Top-level composition of dataset and model configs

These schemas provide:
- Type validation at runtime via Hydra
- Clear documentation of expected config structure
- Automatic field computation (e.g., `num_species` from species list)

### Preprocessed Data Locations

Training commands save preprocessed data to avoid re-processing:

**Autoencoder:**
- `data/autoencoder_train_preprocessed.pt`
- `data/autoencoder_val_preprocessed.pt`

**Emulator:**
- `data/training_seq.h5`
- `data/validation_seq.h5`

### Example Workflow

```bash
# 1. Install package
pip install -e .

# 2. Train autoencoder with custom hyperparameters
astrochemnet-train-autoencoder model.lr=1e-3 model.latent_dim=14

# 3. Train emulator using the pretrained autoencoder
astrochemnet-train-emulator model.window_size=240

# 4. Override device if needed
astrochemnet-train-emulator device=cpu
```

## Gravitational Collapse Benchmark

## Turbulent Gas Benchmark
