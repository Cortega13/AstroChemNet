"""CLI command for training the autoencoder model.

This module provides the entry point for training the autoencoder component
of the AstroChemNet architecture. The autoencoder learns to compress chemical
species abundances (333 dimensions) into a lower-dimensional latent space
(typically 12-14 dimensions).

Usage:
    astrochemnet-train-autoencoder [OPTIONS]

Examples:
    # Train with default config
    astrochemnet-train-autoencoder

    # Override hyperparameters
    astrochemnet-train-autoencoder model.lr=5e-4 model.batch_size=32768

    # Change dataset or model variant
    astrochemnet-train-autoencoder dataset=grav model=autoencoder_large

    # Only preprocess data
    astrochemnet-train-autoencoder mode=preprocess

    # Only train (requires preprocessed data)
    astrochemnet-train-autoencoder mode=train
"""

import gc
import os
from typing import TYPE_CHECKING

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .. import data_loading as dl
from .. import data_processing as dp
from ..inference import Inference
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..trainer import AutoencoderTrainer, load_objects

# Import for type hints only (doesn't affect runtime)
if TYPE_CHECKING:
    from ..config_schemas import DatasetConfig, ModelsConfig


def setup_config(cfg: DictConfig) -> DictConfig:
    """Add computed fields to config based on dataset.

    Args:
        cfg: Hydra DictConfig with nested 'dataset' and 'model' configs.
            - cfg.dataset follows DatasetConfig schema
            - cfg.model follows ModelsConfig schema

    Returns:
        Modified DictConfig with device and working_path fields.
    """
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.device}")
    cfg.working_path = os.getcwd()
    return cfg


def preprocess_autoencoder_data(cfg: DictConfig):
    """Load, scale, and save preprocessed autoencoder training data."""
    print("=" * 80)
    print("Preprocessing Autoencoder Data")
    print("=" * 80)

    # Load raw data
    print("\nLoading datasets...")
    training_np, validation_np = dl.load_datasets(cfg.dataset, cfg.model.columns)

    # Scale abundances
    print("Scaling abundances...")
    processing = dp.Processing(cfg.dataset, cfg.model)
    processing.abundances_scaling(training_np)
    processing.abundances_scaling(validation_np)

    # Convert to tensors
    training_tensor = torch.from_numpy(training_np).float()
    validation_tensor = torch.from_numpy(validation_np).float()

    # Save to disk
    preprocessed_dir = os.path.join(cfg.working_path, "data")
    os.makedirs(preprocessed_dir, exist_ok=True)
    train_path = os.path.join(preprocessed_dir, "autoencoder_train_preprocessed.pt")
    val_path = os.path.join(preprocessed_dir, "autoencoder_val_preprocessed.pt")

    print(f"Saving preprocessed data to {preprocessed_dir}")
    torch.save(training_tensor, train_path)
    torch.save(validation_tensor, val_path)

    print("\nPreprocessing complete!")
    del training_np, validation_np, training_tensor, validation_tensor
    gc.collect()


def load_data_and_create_dataloaders(cfg: DictConfig):
    """Load preprocessed data and create dataloaders.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (training_tensor, validation_tensor, training_dataloader, validation_dataloader).
    """
    # Load preprocessed data
    preprocessed_dir = os.path.join(cfg.working_path, "data")
    train_path = os.path.join(preprocessed_dir, "autoencoder_train_preprocessed.pt")
    val_path = os.path.join(preprocessed_dir, "autoencoder_val_preprocessed.pt")

    print(f"\nLoading preprocessed data from {train_path}")
    training_tensor = torch.load(train_path)
    validation_tensor = torch.load(val_path)

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    training_dataset = dl.AutoencoderDataset(training_tensor)
    validation_dataset = dl.AutoencoderDataset(validation_tensor)
    training_dataloader = dl.tensor_to_dataloader(cfg.model, training_dataset)
    validation_dataloader = dl.tensor_to_dataloader(cfg.model, validation_dataset)

    return (
        training_tensor,
        validation_tensor,
        training_dataloader,
        validation_dataloader,
    )


def setup_training_components(cfg: DictConfig):
    """Initialize model, optimizer, scheduler, processing, and loss functions.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (autoencoder, optimizer, scheduler, processing, loss_functions).
    """
    print("\nInitializing training components...")

    # Initialize model
    autoencoder = load_autoencoder(Autoencoder, cfg.dataset, cfg.model)
    print(f"Model architecture:\n{autoencoder}")

    # Initialize optimizer and scheduler
    optimizer, scheduler = load_objects(autoencoder, cfg.model)

    # Initialize processing and loss functions
    processing = dp.Processing(cfg.dataset, cfg.model)
    loss_functions = Loss(processing, cfg.dataset, ModelConfig=cfg.model)

    return autoencoder, optimizer, scheduler, processing, loss_functions


def save_latent_statistics(
    cfg: DictConfig,
    autoencoder: Autoencoder,
    training_tensor: torch.Tensor,
    validation_tensor: torch.Tensor,
    processing: dp.Processing,
):
    """Compute and save latent space min/max statistics.

    Args:
        cfg: Configuration object.
        autoencoder: Trained autoencoder model.
        training_tensor: Training data tensor.
        validation_tensor: Validation data tensor.
        processing: Processing object with scaling functions.
    """
    if cfg.model.save_model:
        print("\nComputing and saving latent space statistics...")
        total_dataset = torch.vstack((training_tensor, validation_tensor))
        inference_functions = Inference(cfg.dataset, processing, autoencoder)
        processing.save_latents_minmax(cfg.model, total_dataset, inference_functions)


def train_autoencoder_model(cfg: DictConfig):
    """Train the autoencoder model."""
    print("=" * 80)
    print("Training Autoencoder")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Load data and create dataloaders
    training_tensor, validation_tensor, training_dataloader, validation_dataloader = (
        load_data_and_create_dataloaders(cfg)
    )

    # Setup training components
    autoencoder, optimizer, scheduler, processing, loss_functions = (
        setup_training_components(cfg)
    )

    # Initialize and run trainer
    print("\nInitializing trainer...")
    trainer = AutoencoderTrainer(
        cfg.dataset,
        cfg.model,
        loss_functions,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
        cfg.device,
    )

    print("\nStarting training...")
    print("=" * 80)
    trainer.train()

    # Save latent space statistics for emulator training
    save_latent_statistics(
        cfg, autoencoder, training_tensor, validation_tensor, processing
    )

    print("\nTraining complete!")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for autoencoder training CLI command.

    Args:
        cfg: Hydra configuration object composed from YAML files and CLI overrides.
    """
    # Setup config
    cfg = setup_config(cfg)

    # Execute based on mode
    mode = cfg.get("mode", "both")

    if mode == "preprocess":
        preprocess_autoencoder_data(cfg)
    elif mode == "train":
        train_autoencoder_model(cfg)
    elif mode == "both":
        preprocess_autoencoder_data(cfg)
        train_autoencoder_model(cfg)
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'preprocess', 'train', or 'both'."
        )


if __name__ == "__main__":
    main()
