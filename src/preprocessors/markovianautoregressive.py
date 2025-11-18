"""Markovian autoregressive preprocessor for emulator training data."""

import gc
import os
from pathlib import Path

import numpy as np
import torch
from numba import njit
from omegaconf import DictConfig

from ..data_loading import load_dataset as load_datasets
from ..data_processing import Processing
from ..inference import Inference
from ..models.autoencoder import Autoencoder, load_autoencoder


@njit
def generate_sequence_indices(
    dataset_np: np.ndarray,
    window_size: int = 16,
) -> np.ndarray:
    """Generate sliding window indices for sequence training."""
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)

    total_seqs = 0
    for group in model_groups:
        n = len(group)
        total_seqs += n - window_size + 1

    sequences = np.full((total_seqs, window_size), -1, dtype=np.int32)

    seq_idx = 0
    for group in model_groups:
        indices = group[:, 0]
        n = len(indices)
        for start_idx in range(n - window_size + 1):
            sequences[seq_idx, :] = indices[start_idx : start_idx + window_size]
            seq_idx += 1

    return sequences


def preprocess_sequences(
    dataset_cfg: DictConfig,
    emulator_cfg: DictConfig,
    dataset_np: np.ndarray,
    processing_functions: Processing,
    inference_functions: Inference,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess dataset: scale abundances, encode with autoencoder, and generate sequence indices."""
    num_species = dataset_cfg.num_species
    num_phys = dataset_cfg.num_phys
    num_metadata = dataset_cfg.num_metadata

    dataset_np[:, 0] = np.arange(len(dataset_np))

    processing_functions.physical_parameter_scaling(
        dataset_np[:, num_metadata : num_metadata + num_phys]
    )
    processing_functions.abundances_scaling(dataset_np[:, -num_species:])

    latent_components = inference_functions.encode(
        dataset_np[:, num_metadata + num_phys :]
    )
    latent_components = (
        processing_functions.latent_components_scaling(latent_components).cpu().numpy()
    )
    encoded_dataset_np = np.hstack((dataset_np, latent_components), dtype=np.float32)

    index_pairs_np = generate_sequence_indices(
        encoded_dataset_np, emulator_cfg.window_size
    )

    perm = np.random.permutation(len(index_pairs_np))
    index_pairs_shuffled_np = index_pairs_np[perm]

    encoded_t = torch.from_numpy(encoded_dataset_np).float()
    index_pairs_shuffled_t = torch.from_numpy(index_pairs_shuffled_np).int()

    gc.collect()
    torch.cuda.empty_cache()

    return encoded_t, index_pairs_shuffled_t


class MarkovianautoregressivePreprocessor:
    def __init__(self, dataset_cfg: DictConfig, method_cfg: DictConfig, root: Path):
        self.dataset_cfg = dataset_cfg
        self.method_cfg = method_cfg
        self.root = root

    def run(self, output_dir: Path):
        """Load autoencoder, preprocess emulator sequences, and save to HDF5."""
        print("=" * 80)
        print("Preprocessing Emulator Sequences")
        print("=" * 80)

        # Load pretrained autoencoder for inference
        print("\nLoading pretrained autoencoder...")
        pretrained_path = self.root / self.method_cfg.pretrained_model_path
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained autoencoder not found at {pretrained_path}. "
                "Please train the autoencoder first using: astrochemnet-train-autoencoder"
            )

        processing_functions = Processing(self.dataset_cfg, "cpu", self.method_cfg)
        autoencoder = load_autoencoder(
            Autoencoder, self.dataset_cfg, self.method_cfg, inference=True
        )
        inference_functions = Inference(
            self.dataset_cfg, processing_functions, autoencoder
        )

        # Load and preprocess datasets
        print("\nLoading and preprocessing training dataset...")
        training_np, validation_np = load_datasets(self.dataset_cfg)

        print("Preprocessing training sequences...")
        training_dataset = preprocess_sequences(
            self.dataset_cfg,
            self.method_cfg,
            training_np,
            processing_functions,
            inference_functions,
        )

        print("Preprocessing validation sequences...")
        validation_dataset = preprocess_sequences(
            self.dataset_cfg,
            self.method_cfg,
            validation_np,
            processing_functions,
            inference_functions,
        )

        # Save to PyTorch tensors
        print("\nSaving preprocessed sequences to .pt files...")
        torch.save(training_dataset, output_dir / "training_seq.pt")
        torch.save(validation_dataset, output_dir / "validation_seq.pt")

        print("\nPreprocessing complete!")
        del training_np, validation_np, training_dataset, validation_dataset
        gc.collect()
