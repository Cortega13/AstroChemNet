"""Data loading utilities for training and inference."""

import gc
import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler


def _load_hdf5_split(path: str, key: str, max_len: Optional[int] = None) -> np.ndarray:
    """Load single HDF5 split and convert to numpy array."""
    return (
        pd.read_hdf(path, key, start=0, stop=max_len)
        .astype(np.float32)
        .to_numpy(copy=False)
    )


def load_dataset(
    dataset_cfg: DictConfig,
    max_len: Optional[int] = None,
    total: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load datasets from HDF5 files and convert to numpy arrays."""
    training_np = _load_hdf5_split(dataset_cfg.dataset_path, "train", max_len)
    validation_np = _load_hdf5_split(dataset_cfg.dataset_path, "val", max_len)

    gc.collect()
    if total:
        return np.concatenate((training_np, validation_np), axis=0)

    return training_np, validation_np


def save_tensors_to_hdf5(
    working_path: str, tensors: tuple[torch.Tensor, torch.Tensor], category: str
) -> None:
    """Save the dataset along with the encoded species and indices in tensor format."""
    dataset, indices = tensors
    dataset_path = os.path.join(working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "w") as f:
        f.create_dataset("dataset", data=dataset.numpy(), dtype=np.float32)
        f.create_dataset("indices", data=indices.numpy(), dtype=np.int32)


def load_tensors_from_hdf5(
    working_path: str, category: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load saved tensors to quickly run an emulator training session."""
    dataset_path = os.path.join(working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "r") as f:
        dataset = f["dataset"][:]  # type: ignore[index]
        indices = f["indices"][:]  # type: ignore[index]
    return torch.from_numpy(dataset).float(), torch.from_numpy(indices).int()


class ChunkedShuffleSampler(Sampler):
    """Shuffle data in chunks for memory efficiency."""

    def __init__(self, data_size: int, chunk_size: int, seed: int = 13):
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

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for reproducible shuffling."""
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
        return self.data_size


class AutoencoderDataset(Dataset):
    """Tensor Dataset for autoencoder training.

    Uses __getitems__ for efficient batch loading (~10^3x speedup).
    """

    def __init__(self, data_matrix: torch.Tensor):
        self.data_matrix = data_matrix
        self.size = len(data_matrix)
        print(f"Data_matrix Memory usage: {data_matrix.nbytes / (1024**2):.3f} MB")

    def __len__(self) -> int:
        return self.size

    def __getitems__(self, indices: list[int]) -> torch.Tensor:
        """Load multiple samples at once for efficient batching."""
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        return self.data_matrix[tensor_indices]


class EmulatorSequenceDataset(Dataset):
    """Tensor Dataset for emulator training with sequence indexing.

    Uses __getitems__ to reuse rows across training elements, reducing memory overhead.
    """

    def __init__(
        self,
        dataset_cfg: DictConfig,
        autoencoder_cfg: DictConfig,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
    ):
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = dataset_cfg.num_metadata
        self.num_phys = dataset_cfg.num_phys
        self.num_species = dataset_cfg.num_species
        self.num_latents = autoencoder_cfg.latent_dim

        print(f"Data_matrix Memory usage: {data_matrix.nbytes / (1024**2):.3f} MB")
        print(f"Indices_matrix Memory usage: {data_indices.nbytes / (1024**2):.3f} MB")
        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        return self.num_datapoints

    def __getitems__(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load sequence data for multiple samples efficiently."""
        data_indices = self.data_indices[indices]
        rows = self.data_matrix[data_indices]

        physical_parameters = rows[
            :, :-1, self.num_metadata : self.num_metadata + self.num_phys
        ]
        features = rows[:, 0, -self.num_latents :]
        targets = rows[:, 1:, self.num_metadata + self.num_phys : -self.num_latents]

        return physical_parameters, features, targets


def tensor_to_dataloader(
    model_cfg: DictConfig,
    torchDataset: AutoencoderDataset | EmulatorSequenceDataset,
) -> DataLoader:
    """Create a DataLoader with chunked shuffling for memory efficiency."""
    data_size = len(torchDataset)
    multiplier = model_cfg.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=int(multiplier * data_size))

    return DataLoader(
        torchDataset,
        batch_size=model_cfg.batch_size,
        pin_memory=True,
        num_workers=getattr(model_cfg, "num_workers", 10),
        in_order=False,
        sampler=sampler,
    )
