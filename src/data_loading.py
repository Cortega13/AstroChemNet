"""Shared data loading helpers."""

import gc
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.datasets import DatasetConfig


def load_datasets(
    general_config: DatasetConfig,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load training and validation datasets from .npy files as numpy arrays.

    Args:
        general_config: Configuration containing dataset paths and settings
        columns: List of column names to select from the dataset

    Returns:
        Tuple of (training_data, validation_data) as numpy arrays
    """
    # Load column mapping to get column indices
    with open(general_config.columns_mapping_path) as f:
        columns_mapping = json.load(f)

    # Reverse the mapping from {index: name} to {name: index}
    columns_mapping = {name: int(idx) for idx, name in columns_mapping.items()}

    # Get indices for requested columns
    col_indices = [columns_mapping[col] for col in columns]

    # Load full datasets from .npy files
    train_full = np.load(os.path.join(general_config.dataset_artifacts_dir, "train.npy"))
    val_full = np.load(os.path.join(general_config.dataset_artifacts_dir, "val.npy"))

    # Select only requested columns
    training_np = train_full[:, col_indices].astype(np.float32)
    validation_np = val_full[:, col_indices].astype(np.float32)

    species_slice = slice(-general_config.num_species, None)

    np.clip(
        training_np[:, species_slice],
        general_config.abundances_lower_clipping,
        general_config.abundances_upper_clipping,
        out=training_np[:, species_slice],
    )

    np.clip(
        validation_np[:, species_slice],
        general_config.abundances_lower_clipping,
        general_config.abundances_upper_clipping,
        out=validation_np[:, species_slice],
    )

    gc.collect()
    return training_np, validation_np


def save_tensors(
    general_config: DatasetConfig,
    model_name: str,
    tensors: dict[str, torch.Tensor],
    category: str,
) -> None:
    """Save tensors to a .pt file."""
    dataset_path = general_config.model_path(model_name, f"{category}.pt")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    torch.save({name: tensor.cpu() for name, tensor in tensors.items()}, dataset_path)


def load_tensors(
    general_config: DatasetConfig,
    model_name: str,
    category: str,
) -> dict[str, torch.Tensor]:
    """Load tensors from a .pt file."""
    dataset_path = general_config.model_path(model_name, f"{category}.pt")
    return torch.load(dataset_path, map_location="cpu", weights_only=True)


class ChunkedShuffleSampler(Sampler):
    """Sampler that shuffles data in chunks for memory-efficient training."""

    def __init__(self, data_size: int, chunk_size: int, seed: int = 13) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.chunk_size = int(chunk_size)
        self.base_seed = seed
        self.epoch = 0

        self.chunks = []
        start = 0
        while start < self.data_size:
            end = min(start + self.chunk_size, self.data_size)
            self.chunks.append((start, end))
            start = end

        self.generator = torch.Generator()

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)

        chunk_indices = torch.randperm(len(self.chunks), generator=g)

        for i, chunk_idx in enumerate(chunk_indices):
            chunk_seed = self.base_seed + self.epoch * 10000 + i
            g.manual_seed(chunk_seed)

            start, end = self.chunks[chunk_idx]
            length = end - start

            chunk_perm = torch.randperm(length, generator=g)
            chunk_perm += start

            yield from chunk_perm.tolist()

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.data_size


def collate_function(batch):
    """Collate function that handles features, targets, and optional physical parameters."""
    if len(batch) == 4:
        delta_t, physical_parameters, features, targets = batch
        return delta_t, physical_parameters, features, targets
    if len(batch) == 2:
        features, targets = batch
        return features, targets

    physical_parameters, features, targets = batch
    return physical_parameters, features, targets


def tensor_to_dataloader(
    training_config,
    torch_dataset: Dataset,
) -> DataLoader:
    """Create a DataLoader with chunked shuffling for the given dataset."""
    data_size = len(torch_dataset)  # type: ignore
    multiplier = training_config.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=multiplier * data_size)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=training_config.batch_size,
        pin_memory=True,
        num_workers=10,
        in_order=False,
        sampler=sampler,
        collate_fn=collate_function,
    )
    return dataloader
