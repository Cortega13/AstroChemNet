"""CLI command for training the emulator model.

This module provides the entry point for training the emulator component
of the AstroChemNet architecture. The emulator learns to predict temporal
evolution of chemical abundances in the compressed latent space learned
by the autoencoder.

The emulator requires a pretrained autoencoder to encode abundances into
latent representations before training.

Usage:
    astrochemnet-train-emulator [OPTIONS]

Examples:
    # Train with default config
    astrochemnet-train-emulator

    # Override hyperparameters
    astrochemnet-train-emulator model.lr=1e-3 model.window_size=128

    # Change dataset
    astrochemnet-train-emulator dataset=turbulent

    # Only preprocess sequences
    astrochemnet-train-emulator mode=preprocess

    # Only train (requires preprocessed sequences)
    astrochemnet-train-emulator mode=train
"""

import gc
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .. import data_loading as dl
from .. import data_processing as dp
from ..inference import Inference
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..models.emulator import Emulator, load_emulator
from ..trainer import EmulatorTrainerSequential, load_objects


def setup_config(cfg: DictConfig):
    """Add computed fields to config based on dataset."""
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.device}")
    cfg.working_path = os.getcwd()
    return cfg


def preprocess_emulator_data(cfg: DictConfig):
    """Load autoencoder, preprocess emulator sequences, and save to HDF5."""
    print("=" * 80)
    print("Preprocessing Emulator Sequences")
    print("=" * 80)

    # Load pretrained autoencoder for inference
    print("\nLoading pretrained autoencoder...")
    if not os.path.exists(cfg.autoencoder.pretrained_model_path):
        raise FileNotFoundError(
            f"Pretrained autoencoder not found at {cfg.autoencoder.pretrained_model_path}. "
            "Please train the autoencoder first using: astrochemnet-train-autoencoder"
        )

    processing_functions = dp.Processing(cfg.dataset, cfg.autoencoder)
    autoencoder = load_autoencoder(
        Autoencoder, cfg.dataset, cfg.autoencoder, inference=True
    )
    inference_functions = Inference(cfg.dataset, processing_functions, autoencoder)

    # Load and preprocess datasets
    print("\nLoading and preprocessing training dataset...")
    training_np, validation_np = dl.load_datasets(cfg.dataset, cfg.model.columns)

    print("Preprocessing training sequences...")
    training_dataset = dp.preprocessing_emulator_dataset(
        cfg.dataset, cfg.model, training_np, processing_functions, inference_functions
    )

    print("Preprocessing validation sequences...")
    validation_dataset = dp.preprocessing_emulator_dataset(
        cfg.dataset, cfg.model, validation_np, processing_functions, inference_functions
    )

    # Save to HDF5
    print("\nSaving preprocessed sequences to HDF5...")
    dl.save_tensors_to_hdf5(cfg.dataset, training_dataset, category="training_seq")
    dl.save_tensors_to_hdf5(cfg.dataset, validation_dataset, category="validation_seq")

    print("\nPreprocessing complete!")
    del training_np, validation_np, training_dataset, validation_dataset
    gc.collect()


def load_data_and_create_dataloaders(cfg: DictConfig):
    """Load preprocessed sequences and create emulator dataloaders.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (training_dataloader, validation_dataloader).
    """
    # Load preprocessed sequences from HDF5
    print("\nLoading preprocessed sequences from HDF5...")
    training_dataset, training_indices = dl.load_tensors_from_hdf5(
        cfg.dataset, category="training_seq"
    )
    validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
        cfg.dataset, category="validation_seq"
    )

    print(f"Training sequences: {len(training_indices)}")
    print(f"Validation sequences: {len(validation_indices)}")

    # Create sequence datasets
    print("Creating sequence datasets and dataloaders...")
    training_Dataset = dl.EmulatorSequenceDataset(
        cfg.dataset, cfg.autoencoder, training_dataset, training_indices
    )
    validation_Dataset = dl.EmulatorSequenceDataset(
        cfg.dataset, cfg.autoencoder, validation_dataset, validation_indices
    )

    # Clean up large arrays
    del training_dataset, validation_dataset, training_indices, validation_indices
    gc.collect()

    # Create dataloaders
    training_dataloader = dl.tensor_to_dataloader(cfg.model, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(cfg.model, validation_Dataset)

    return training_dataloader, validation_dataloader


def setup_training_components(cfg: DictConfig):
    """Initialize emulator, autoencoder, optimizer, scheduler, processing, and loss.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (autoencoder, emulator, optimizer, scheduler, processing, loss_functions).
    """
    print("\nInitializing training components...")

    # Initialize processing and load autoencoder for validation
    processing_functions = dp.Processing(cfg.dataset, cfg.autoencoder)
    autoencoder = load_autoencoder(
        Autoencoder, cfg.dataset, cfg.autoencoder, inference=True
    )

    # Initialize emulator
    emulator = load_emulator(Emulator, cfg.dataset, cfg.model)
    print(f"Emulator architecture:\n{emulator}")

    # Initialize optimizer and scheduler
    optimizer, scheduler = load_objects(emulator, cfg.model)

    # Initialize loss functions
    loss_functions = Loss(processing_functions, cfg.dataset, ModelConfig=cfg.model)

    return (
        autoencoder,
        emulator,
        optimizer,
        scheduler,
        processing_functions,
        loss_functions,
    )


def train_emulator_model(cfg: DictConfig):
    """Train the emulator model."""
    print("=" * 80)
    print("Training Emulator")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Load data and create dataloaders
    training_dataloader, validation_dataloader = load_data_and_create_dataloaders(cfg)

    # Setup training components
    (
        autoencoder,
        emulator,
        optimizer,
        scheduler,
        processing_functions,
        loss_functions,
    ) = setup_training_components(cfg)

    # Initialize and run trainer
    print("\nInitializing trainer...")
    trainer = EmulatorTrainerSequential(
        cfg.dataset,
        cfg.autoencoder,
        cfg.model,
        loss_functions,
        processing_functions,
        autoencoder,
        emulator,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
        cfg.device,
    )

    print("\nStarting training...")
    print("=" * 80)
    trainer.train()

    print("\nTraining complete!")


@hydra.main(
    config_path="../../../configs", config_name="config_emulator", version_base=None
)
def main(cfg: DictConfig):
    """Main entry point for emulator training CLI command.

    Args:
        cfg: Hydra configuration object composed from YAML files and CLI overrides.
            Uses config_emulator.yaml which includes both autoencoder and emulator configs.
    """
    # Setup config
    cfg = setup_config(cfg)

    # Execute based on mode
    mode = cfg.get("mode", "both")

    if mode == "preprocess":
        preprocess_emulator_data(cfg)
    elif mode == "train":
        train_emulator_model(cfg)
    elif mode == "both":
        preprocess_emulator_data(cfg)
        train_emulator_model(cfg)
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'preprocess', 'train', or 'both'."
        )


if __name__ == "__main__":
    main()
