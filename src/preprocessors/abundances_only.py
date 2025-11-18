"""Abundances-only preprocessor for autoencoder training data."""

import gc
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from ..data_loading import load_dataset as load_datasets
from ..data_processing import Processing


class AbundancesOnlyPreprocessor:
    def __init__(self, dataset_cfg: DictConfig, method_cfg: DictConfig, root: Path):
        self.dataset_cfg = dataset_cfg
        self.method_cfg = method_cfg
        self.root = root

    def run(self, output_dir: Path):
        """Load, scale, and save preprocessed autoencoder training data."""
        print("=" * 80)
        print("Preprocessing Autoencoder Data")
        print("=" * 80)

        # Load raw data
        print("\nLoading datasets...")
        training_np, validation_np = load_datasets(self.dataset_cfg)

        # Scale abundances
        print("Scaling abundances...")
        processing = Processing(self.dataset_cfg, "cpu")
        processing.abundances_scaling(training_np)
        processing.abundances_scaling(validation_np)

        # Convert to tensors
        training_tensor = torch.from_numpy(training_np).float()
        validation_tensor = torch.from_numpy(validation_np).float()

        # Save to disk
        train_path = output_dir / "autoencoder_train_preprocessed.pt"
        val_path = output_dir / "autoencoder_val_preprocessed.pt"

        print(f"Saving preprocessed data to {output_dir}")
        torch.save(training_tensor, train_path)
        torch.save(validation_tensor, val_path)

        print("\nPreprocessing complete!")
        del training_np, validation_np, training_tensor, validation_tensor
        gc.collect()
