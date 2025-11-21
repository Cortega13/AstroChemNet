"""Abundances-only preprocessor for autoencoder training data."""

import gc
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from ..data_processing import Processing


class AbundancesOnlyPreprocessor:
    """Preprocessor for autoencoder training data (abundances only)."""

    def __init__(self, cfg: DictConfig):
        """Initialize the preprocessor."""
        self.cfg = cfg

    def run(self, output_dir: Path):
        """Load, scale, and save preprocessed autoencoder training data."""
        print("=" * 80)
        print("Preprocessing Autoencoder Data")
        print("=" * 80)

        # Load raw data
        print("\nLoading datasets...")
        training_np, validation_np = load_datasets(self.cfg)

        # Scale abundances
        print("Scaling abundances...")
        processing = Processing(self.cfg, "cpu")

        # Data is now (N_tracers, N_timesteps, N_features)
        # N_features = N_phys + N_species
        # We need to scale only the species part

        num_phys = self.cfg.physical_parameters.n_params

        # Reshape to 2D for scaling: (N_tracers * N_timesteps, N_features)
        train_shape = training_np.shape
        val_shape = validation_np.shape

        training_flat = training_np.reshape(-1, train_shape[-1])
        validation_flat = validation_np.reshape(-1, val_shape[-1])

        # Scale species (last N_species columns)
        processing.abundances_scaling(training_flat[:, num_phys:])
        processing.abundances_scaling(validation_flat[:, num_phys:])

        # Reshape back to 3D
        training_np = training_flat.reshape(train_shape)
        validation_np = validation_flat.reshape(val_shape)

        # Convert to tensors
        # For autoencoder, we might want to flatten the time dimension if we treat each point independently
        # But keeping 3D structure allows for sequence models later
        # The AutoencoderDataset currently expects 2D (N_samples, N_features) or just N_species?
        # Let's check AutoencoderDataset. It takes data_matrix.
        # If we want to train on individual points, we should flatten.

        # Flattening for Autoencoder training (treating each timestep as independent sample)
        # We only need the species part for the autoencoder
        training_species = training_flat[:, num_phys:]
        validation_species = validation_flat[:, num_phys:]

        training_tensor = torch.from_numpy(training_species).float()
        validation_tensor = torch.from_numpy(validation_species).float()

        # Save to disk
        train_path = output_dir / "autoencoder_train_preprocessed.pt"
        val_path = output_dir / "autoencoder_val_preprocessed.pt"

        print(f"Saving preprocessed data to {output_dir}")
        torch.save(training_tensor, train_path)
        torch.save(validation_tensor, val_path)

        print("\nPreprocessing complete!")
        del training_np, validation_np, training_tensor, validation_tensor
        gc.collect()
