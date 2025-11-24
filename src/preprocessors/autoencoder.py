"""Autoencoder preprocessor for autoencoder training data."""

import gc
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.data_processing import Processing, load_3d_tensors


class AutoencoderPreprocessor:
    """Preprocessor for autoencoder training data (abundances only)."""

    def __init__(self, cfg: DictConfig):
        """Initialize the preprocessor."""
        self.cfg = cfg

    def run(self, output_dir: Path):
        """Load, scale, and save preprocessed autoencoder training data."""
        print("=" * 80)
        print("Preprocessing Autoencoder Data")
        print("=" * 80)

        # Load 3D tensors from initial preprocessing
        print("\nLoading datasets...")
        training_tensor, validation_tensor = load_3d_tensors(self.cfg)

        # Get dimensions from dataset config
        num_phys = self.cfg.num_phys

        # Reshape to 2D: (N_tracers * N_timesteps, N_features)
        train_shape = training_tensor.shape
        val_shape = validation_tensor.shape

        training_flat = training_tensor.reshape(-1, train_shape[-1])
        validation_flat = validation_tensor.reshape(-1, val_shape[-1])

        # Scale abundances (in-place modification on species columns only)
        print("Scaling abundances...")
        processing = Processing(self.cfg, "cpu")
        processing.abundances_scaling(training_flat[:, num_phys:])
        processing.abundances_scaling(validation_flat[:, num_phys:])

        # Extract species only (autoencoder trains on abundances)
        training_species = training_flat[:, num_phys:]
        validation_species = validation_flat[:, num_phys:]

        # Save to disk
        train_filename = getattr(
            self.cfg.output, "train_tensor", "autoencoder_train_preprocessed.pt"
        )
        val_filename = getattr(
            self.cfg.output, "val_tensor", "autoencoder_val_preprocessed.pt"
        )

        train_path = output_dir / train_filename
        val_path = output_dir / val_filename

        print("\nSaving preprocessed data:")
        print(f"  Train: {train_path} - Shape: {training_species.shape}")
        print(f"  Val: {val_path} - Shape: {validation_species.shape}")

        torch.save(training_species, train_path)
        torch.save(validation_species, val_path)

        print("\nPreprocessing complete!")
        del training_tensor, validation_tensor, training_species, validation_species
        gc.collect()
