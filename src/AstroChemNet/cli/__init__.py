"""AstroChemNet Command Line Interface.

This package provides CLI commands for training and preprocessing AstroChemNet
surrogate models. The framework uses a two-stage architecture:

1. **Autoencoder** - Compresses chemical species abundances (333 species) into
   a lower-dimensional latent space (typically 12-14 dimensions)

2. **Emulator** - Predicts temporal evolution of latent representations given
   physical parameters (density, radiation field, visual extinction, temperature)

Architecture Overview
--------------------

The training pipeline follows this sequence:

1. Raw Data → Preprocessing → Cleaned HDF5
   ├── Command: astrochemnet-preprocess
   └── Config: configs/preprocess.yaml

2. Cleaned HDF5 → Autoencoder Training → Latent Space Model
   ├── Command: astrochemnet-train-autoencoder
   ├── Config: configs/config.yaml (dataset + autoencoder)
   └── Outputs:
       ├── weights/autoencoder.pth
       └── utils/latents_minmax.npy

3. Latent Space → Emulator Training → Temporal Dynamics Model
   ├── Command: astrochemnet-train-emulator
   ├── Config: configs/config_emulator.yaml (dataset + autoencoder + emulator)
   └── Output: weights/emulator.pth

Available Commands
-----------------

**Data Preprocessing**
    astrochemnet-preprocess
        Converts raw UCLCHEM outputs to cleaned HDF5 format.
        Handles column filtering, train/validation splitting, and basic cleaning.

**Model Training**
    astrochemnet-train-autoencoder
        Trains the autoencoder to compress species abundances into latent space.
        Must be run before training the emulator.

    astrochemnet-train-emulator
        Trains the emulator to predict temporal evolution in latent space.
        Requires a pretrained autoencoder.

Configuration System
-------------------

Uses Hydra for hierarchical configuration management:

- configs/
  ├── config.yaml              # Autoencoder training config
  ├── config_emulator.yaml     # Emulator training config
  ├── preprocess.yaml          # Data preprocessing config
  ├── datasets/
  │   └── grav.yaml           # Dataset paths and parameters
  └── models/
      ├── autoencoder.yaml    # Autoencoder hyperparameters
      └── emulator.yaml       # Emulator hyperparameters

Override any parameter from the command line:
    astrochemnet-train-autoencoder model.lr=5e-4 model.batch_size=32768

Training Modes
-------------

All training commands support three modes via the `mode` parameter:

- mode=preprocess : Only preprocess data (save to disk)
- mode=train      : Only train (load preprocessed data from disk)
- mode=both       : Preprocess and train (default)

Examples:
    # Preprocess once, then run multiple training experiments
    astrochemnet-train-autoencoder mode=preprocess
    astrochemnet-train-autoencoder mode=train model.lr=1e-3
    astrochemnet-train-autoencoder mode=train model.lr=5e-4

Module Structure
---------------

cli/
├── __init__.py              # This file - high-level overview
├── preprocess.py            # Data preprocessing command
├── train_autoencoder.py     # Autoencoder training command
└── train_emulator.py        # Emulator training command

Each module is self-contained with:
- Comprehensive docstrings and usage examples
- Hydra configuration management
- Preprocessing and training functions
- Entry point for the CLI command

Import Structure
---------------

Commands are exposed through their respective modules:

    from AstroChemNet.cli.preprocess import main as preprocess_dataset
    from AstroChemNet.cli.train_autoencoder import main as train_autoencoder
    from AstroChemNet.cli.train_emulator import main as train_emulator

These entry points are registered in pyproject.toml [project.scripts] section.
"""

# Import main functions for programmatic access
from .preprocess import main as preprocess_dataset
from .train_autoencoder import main as train_autoencoder
from .train_emulator import main as train_emulator

__all__ = [
    "preprocess_dataset",
    "train_autoencoder",
    "train_emulator",
]

# CLI version information
__version__ = "0.1.0"
__author__ = "AstroChemNet Team"
__description__ = "Command-line interface for AstroChemNet surrogate model training"
