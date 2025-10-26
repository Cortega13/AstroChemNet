"""Here we define several data loading methods needed for loading the datasets onto cpu memory and then loading them in batches for the Trainer class."""

import gc
import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler


def load_dataset(
    dataset_cfg: DictConfig,
    max_len: Optional[int] = None,
    total: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Datasets are loaded from hdf5 files, filtered to only contain the columns of interest, and converted to np arrays for speed."""
    training_dataset = pd.read_hdf(
        dataset_cfg.dataset_path, "train", start=0, stop=max_len
    ).astype(np.float32)
    validation_dataset = pd.read_hdf(
        dataset_cfg.dataset_path, "val", start=0, stop=max_len
    ).astype(np.float32)

    training_np = training_dataset.to_numpy(copy=False)
    validation_np = validation_dataset.to_numpy(copy=False)

    gc.collect()
    if total:
        combined = np.concatenate((training_np, validation_np), axis=0)
        return combined

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
        dataset = f["dataset"][:]  # type: ignore
        indices = f["indices"][:]  # type: ignore
    dataset = torch.from_numpy(dataset).float()
    indices = torch.from_numpy(indices).int()
    return dataset, indices


class ChunkedShuffleSampler(Sampler):
    """During training, we want to shuffle the data in chunks for efficiency. This is especially important when our dataset size is similar to our RAM limits."""

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

        self.generator = torch.Generator()

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

    Uses the "__getitems__" method which can load all elements in a batch at once.
    Since our batch sizes are ~10^3, instead of having ~10^3 calls to this function we only have 1.
    """

    def __init__(
        self,
        data_matrix: torch.Tensor,
    ):
        self.data_matrix = data_matrix
        data_matrix_size = self.data_matrix.nbytes / (1024**2)
        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")

    def __len__(self) -> int:
        return len(self.data_matrix)

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, int]:
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        features = self.data_matrix[tensor_indices]
        return features, 1


class EmulatorSequenceDataset(Dataset):
    """Tensor Dataset for emulator training.

    Uses the "__getitems__" method which can load multiple rows of data at once.
    Since we reuse rows of data for elements of the training dataset, we can recall the rows on the fly for batches.
    This reduces memory overhead significantly.
    """

    def __init__(
        self,
        dataset_cfg: DictConfig,
        autoencoder_cfg: DictConfig,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
    ):
        # self.data_matrix = data_matrix.to(self.device).contiguous()
        # self.data_indices = data_indices.to(self.device).contiguous()
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = dataset_cfg.num_metadata
        self.num_phys = dataset_cfg.num_phys
        self.num_species = dataset_cfg.num_species
        self.num_latents = autoencoder_cfg.latent_dim

        data_matrix_size = self.data_matrix.nbytes / (1024**2)
        indices_matrix_size = self.data_indices.nbytes / (1024**2)

        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")
        print(f"Indices_matrix Memory usage: {indices_matrix_size:.3f} MB")

        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        return self.num_datapoints

    def __getitems__(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        data_indices = self.data_indices[indices]

        rows = self.data_matrix[data_indices]

        physical_parameters = rows[
            :, :-1, self.num_metadata : self.num_metadata + self.num_phys
        ]
        features = rows[:, 0, -self.num_latents :]
        targets = rows[:, 1:, self.num_metadata + self.num_phys : -self.num_latents]

        return physical_parameters, features, targets


def collate_function(
    batch: tuple[torch.Tensor, ...] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """The collate_function usually pulls from the Tensor Dataset in (features, targets) format. Here we account for the physical parameters as well."""
    if len(batch) == 2:
        features, targets = batch
        return features, targets

    physical_parameters, features, targets = batch
    return physical_parameters, features, targets


def tensor_to_dataloader(
    model_cfg: DictConfig,
    torchDataset: AutoencoderDataset | EmulatorSequenceDataset,
) -> DataLoader:
    """Create a DataLoader for the given Torch Dataset."""
    data_size = len(torchDataset)
    multiplier = model_cfg.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=multiplier * data_size)
    dataloader = DataLoader(
        torchDataset,
        batch_size=model_cfg.batch_size,
        pin_memory=True,
        num_workers=10,
        in_order=False,
        sampler=sampler,
        collate_fn=collate_function,  # type:ignore
    )
    return dataloader
