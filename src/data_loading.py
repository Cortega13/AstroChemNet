"""Data loading methods for loading datasets onto CPU memory and batching for training."""

import gc
import os
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from configs.autoencoder import AEConfig
from configs.general import GeneralConfig


def load_datasets(
    GeneralConfig: GeneralConfig,
    columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training and validation datasets from HDF5 files as numpy arrays."""
    training_dataset = pd.read_hdf(
        GeneralConfig.dataset_path,
        "train",
        start=0,
        # stop=5000,
        # stop=1500000
    ).astype(np.float32)
    validation_dataset = pd.read_hdf(
        GeneralConfig.dataset_path,
        "val",
        start=0,
        # stop=5000,
        # stop=1500000
    ).astype(np.float32)

    training_np = training_dataset[columns].to_numpy(copy=False)
    validation_np = validation_dataset[columns].to_numpy(copy=False)

    np.clip(
        training_np[:, -GeneralConfig.num_species :],
        GeneralConfig.abundances_lower_clipping,
        GeneralConfig.abundances_upper_clipping,
        out=training_np[:, -GeneralConfig.num_species :],
    )

    np.clip(
        validation_np[:, -GeneralConfig.num_species :],
        GeneralConfig.abundances_lower_clipping,
        GeneralConfig.abundances_upper_clipping,
        out=validation_np[:, -GeneralConfig.num_species :],
    )

    del training_dataset, validation_dataset
    gc.collect()
    return training_np, validation_np


def save_tensors_to_hdf5(
    GeneralConfig: GeneralConfig,
    tensors: Tuple[torch.Tensor, torch.Tensor],
    category: str,
) -> None:
    """Save dataset and indices tensors to HDF5 file for quick loading."""
    dataset, indices = tensors
    dataset_path = os.path.join(GeneralConfig.working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "w") as f:
        f.create_dataset("dataset", data=dataset.numpy(), dtype=np.float32)
        f.create_dataset("indices", data=indices.numpy(), dtype=np.int32)


def load_tensors_from_hdf5(
    GeneralConfig: GeneralConfig, category: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load saved dataset and indices tensors from HDF5 file."""
    dataset_path = os.path.join(GeneralConfig.working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "r") as f:
        dataset = f["dataset"][:]  # type:ignore
        indices = f["indices"][:]  # type:ignore
    dataset = torch.from_numpy(dataset).float()
    indices = torch.from_numpy(indices).int()
    return dataset, indices


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


class AutoencoderDataset(Dataset):
    """Tensor Dataset for autoencoder training with batch-efficient __getitems__ method."""

    def __init__(self, data_matrix: torch.Tensor) -> None:
        """Initialize the dataset with a data matrix tensor."""
        self.data_matrix = data_matrix
        data_matrix_size = self.data_matrix.nbytes / (1024**2)
        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_matrix)

    def __getitems__(self, indices: List[int]) -> Tuple[torch.Tensor, int]:
        """Load multiple samples at once for efficient batch retrieval."""
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        features = self.data_matrix[indices_tensor]
        return features, 1


class EmulatorSequenceDataset(Dataset):
    """Tensor Dataset for emulator training with memory-efficient sequence handling."""

    def __init__(
        self,
        GeneralConfig: GeneralConfig,
        AEConfig: AEConfig,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
    ) -> None:
        """Initialize the emulator dataset with data matrix and indices."""
        self.device = GeneralConfig.device
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = GeneralConfig.num_metadata
        self.num_phys = GeneralConfig.num_phys
        self.num_species = GeneralConfig.num_species
        self.num_latents = AEConfig.latent_dim

        data_matrix_size = self.data_matrix.nbytes / (1024**2)
        indices_matrix_size = self.data_indices.nbytes / (1024**2)

        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")
        print(f"Indices_matrix Memory usage: {indices_matrix_size:.3f} MB")

        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return self.num_datapoints

    def __getitems__(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load multiple sequence samples at once for batch retrieval."""
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        data_indices = self.data_indices[indices_tensor]
        rows = self.data_matrix[data_indices]

        physical_parameters = rows[
            :, :-1, self.num_metadata : self.num_metadata + self.num_phys
        ]
        features = rows[:, 0, -self.num_latents :]
        targets = rows[:, 1:, self.num_metadata + self.num_phys : -self.num_latents]

        return physical_parameters, features, targets


def collate_function(batch):
    """Collate function that handles features, targets, and optional physical parameters."""
    if len(batch) == 2:
        features, targets = batch
        return features, targets

    physical_parameters, features, targets = batch
    return physical_parameters, features, targets


def tensor_to_dataloader(
    training_config,
    torchDataset: Dataset,
) -> DataLoader:
    """Create a DataLoader with chunked shuffling for the given dataset."""
    data_size = len(torchDataset)  # type: ignore
    multiplier = training_config.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=multiplier * data_size)
    dataloader = DataLoader(
        torchDataset,
        batch_size=training_config.batch_size,
        pin_memory=True,
        num_workers=10,
        in_order=False,
        sampler=sampler,
        collate_fn=collate_function,
    )
    return dataloader
