"""
Here we define several data loading methods needed for loading the datasets onto cpu memory and then loading them in batches for the Trainer class.
"""

import torch
import numpy as np
import pandas as pd
import os
import gc
import h5py
from torch.utils.data import Dataset, DataLoader, Sampler


def load_datasets(
    GeneralConfig,
    columns: list,
):
    """
    Datasets are loaded from hdf5 files, filtered to only contain the columns of interest, and converted to np arrays for speed.
    """
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


def save_tensors_to_hdf5(GeneralConfig, tensors: torch.Tensor, category: str):
    """
    For convenience, we save the dataset along with the encoded species and indices in tensor format.
    This allows us to quickly load everything needed to train the emulator.
    """
    dataset, indices = tensors
    dataset_path = os.path.join(GeneralConfig.working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "w") as f:
        f.create_dataset("dataset", data=dataset.numpy(), dtype=np.float32)
        f.create_dataset("indices", data=indices.numpy(), dtype=np.int32)


def load_tensors_from_hdf5(GeneralConfig, category: str):
    """
    Load saved tensors to quickly run an emulator training session.
    """
    dataset_path = os.path.join(GeneralConfig.working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "r") as f:
        dataset = f["dataset"][:]
        indices = f["indices"][:]
    dataset = torch.from_numpy(dataset).float()
    indices = torch.from_numpy(indices).int()
    return dataset, indices


class ChunkedShuffleSampler(Sampler):
    """
    During training, we want to shuffle the data in chunks for efficiency. This is especially important when our dataset size is similar to our RAM limits.
    """

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

    def set_epoch(self, epoch: int):
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

    def __len__(self):
        return self.data_size


class AutoencoderDataset(Dataset):
    """
    This defines the Tensor Dataset for the autoencoder training. We use the new "__getitems__" method which can load all elements in a batch at once.
    Since our batch sizes are ~10^3, instead of having ~10^3 calls to this function we only have 1.
    """

    def __init__(
        self,
        data_matrix: torch.Tensor,
    ):
        self.data_matrix = data_matrix
        data_matrix_size = self.data_matrix.nbytes / (1024**2)
        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")

    def __len__(self):
        return len(self.data_matrix)

    def __getitems__(self, indices: list):
        indices = torch.tensor(indices, dtype=torch.long)
        features = self.data_matrix[indices]
        return features, 1


class EmulatorSequenceDataset(Dataset):
    """
    This defines the Tensor Dataset for the emulator training. We use the new "__getitems__" method which can load multiple rows of data at once.
    Since we reuse rows of data for elements of the training dataset, we can recall the rows on the fly for batches. This reduces memory overhead significantly.
    """

    def __init__(
        self,
        GeneralConfig,
        AEConfig,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
    ):
        self.device = GeneralConfig.device
        # self.data_matrix = data_matrix.to(self.device).contiguous()
        # self.data_indices = data_indices.to(self.device).contiguous()
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

    def __len__(self):
        return self.num_datapoints

    def __getitems__(self, indices: list):
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        data_indices = self.data_indices[indices]

        rows = self.data_matrix[data_indices]

        physical_parameters = rows[
            :, :-1, self.num_metadata : self.num_metadata + self.num_phys
        ]
        features = rows[:, 0, -self.num_latents :]
        targets = rows[:, 1:, self.num_metadata + self.num_phys : -self.num_latents]

        return physical_parameters, features, targets


def collate_function(batch):
    """
    The collate_function usually pulls from the Tensor Dataset in (features, targets) format. Here we account for the physical parameters as well.
    """
    if len(batch) == 2:
        features, targets = batch
        return features, targets

    physical_parameters, features, targets = batch
    return physical_parameters, features, targets


def tensor_to_dataloader(
    training_config,
    torchDataset: Dataset,
):
    """
    Create a DataLoader for the given Torch Dataset.
    """

    data_size = len(torchDataset)
    multiplier = training_config.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=multiplier * data_size)
    dataloader = DataLoader(
        torchDataset,
        batch_size=training_config.batch_size,
        pin_memory=True,
        num_workers=12,
        in_order=False,
        sampler=sampler,
        collate_fn=collate_function,
    )
    return dataloader
