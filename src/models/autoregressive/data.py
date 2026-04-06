"""Autoregressive data helpers."""

import gc
import os

import numpy as np
import torch
from numba import njit
from torch.utils.data import Dataset

from src import data_processing as dp
from src.data_loading import load_datasets, save_tensors
from src.models.autoregressive.config import build_config


@njit
def calculate_indices(dataset_np: np.ndarray, window_size: int = 16) -> np.ndarray:
    """Generate indices for autoregressive training sequences."""
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)

    total_seqs = 0
    for group in model_groups:
        total_seqs += len(group) - window_size + 1

    sequences = np.full((total_seqs, window_size), -1, dtype=np.int32)

    seq_idx = 0
    for group in model_groups:
        indices = group[:, 0]
        for start_idx in range(len(indices) - window_size + 1):
            sequences[seq_idx, :] = indices[start_idx : start_idx + window_size]
            seq_idx += 1

    return sequences


def preprocess_dataset(
    general_config,
    ar_config,
    dataset_np: np.ndarray,
    processing_functions,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess dataset for abundance autoregressive training."""
    num_species = general_config.num_species
    num_phys = general_config.num_phys
    num_metadata = general_config.num_metadata

    dataset_np[:, 0] = np.arange(len(dataset_np))
    processing_functions.physical_parameter_scaling(
        dataset_np[:, num_metadata : num_metadata + num_phys]
    )
    processing_functions.abundances_scaling(dataset_np[:, -num_species:])

    indices_np = calculate_indices(dataset_np, ar_config.window_size)
    indices_np = indices_np[np.random.permutation(len(indices_np))]

    dataset_t = torch.from_numpy(dataset_np).float()
    indices_t = torch.from_numpy(indices_np).int()

    gc.collect()
    torch.cuda.empty_cache()
    return dataset_t, indices_t


def preprocess_autoregressive(dataset_config, model_config) -> None:
    """Preprocess dataset for abundance autoregressive training."""
    processing_functions = dp.Processing(dataset_config)
    training_np, validation_np = load_datasets(dataset_config, model_config.columns)
    training_dataset = preprocess_dataset(
        dataset_config,
        model_config,
        training_np,
        processing_functions,
    )
    validation_dataset = preprocess_dataset(
        dataset_config,
        model_config,
        validation_np,
        processing_functions,
    )
    save_tensors(
        dataset_config,
        "autoregressive",
        {"dataset": training_dataset[0], "indices": training_dataset[1]},
        category="training_seq",
    )
    save_tensors(
        dataset_config,
        "autoregressive",
        {"dataset": validation_dataset[0], "indices": validation_dataset[1]},
        category="validation_seq",
    )


def ensure_preprocessed(dataset_config, force: bool = False) -> None:
    """Build autoregressive caches when missing or forced."""
    training_path = dataset_config.model_path("autoregressive", "training_seq.pt")
    validation_path = dataset_config.model_path("autoregressive", "validation_seq.pt")
    if not force and os.path.exists(training_path) and os.path.exists(validation_path):
        return
    preprocess_autoregressive(dataset_config, build_config(dataset_config))


class AutoregressiveSequenceDataset(Dataset):
    """Tensor dataset for abundance autoregressive training."""

    def __init__(self, general_config, data_matrix: torch.Tensor, data_indices: torch.Tensor) -> None:
        """Initialize abundance autoregressive dataset with sequence indices."""
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = general_config.num_metadata
        self.num_phys = general_config.num_phys

        print(f"Data_matrix Memory usage: {self.data_matrix.nbytes / (1024**2):.3f} MB")
        print(f"Indices_matrix Memory usage: {self.data_indices.nbytes / (1024**2):.3f} MB")
        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return self.num_datapoints

    def __getitems__(self, indices):
        """Load multiple abundance sequence samples at once for batch retrieval."""
        rows = self.data_matrix[self.data_indices[torch.tensor(indices, dtype=torch.long)]]
        physical_parameters = rows[:, :-1, self.num_metadata : self.num_metadata + self.num_phys].contiguous()
        features = rows[:, 0, self.num_metadata + self.num_phys :].contiguous()
        targets = rows[:, 1:, self.num_metadata + self.num_phys :].contiguous()
        return physical_parameters, features, targets
