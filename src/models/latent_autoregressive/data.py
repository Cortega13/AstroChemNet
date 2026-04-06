"""Latent sequence data helpers."""

import gc
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src import data_processing as dp
from src.data_loading import load_datasets, save_tensors
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.autoregressive.data import calculate_indices
from src.models.latent_autoregressive.config import build_config


def preprocess_dataset(
    general_config,
    model_config,
    dataset_np: np.ndarray,
    processing_functions: dp.Processing,
    inference_functions: Inference,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess dataset for latent sequence training."""
    num_species = general_config.num_species
    num_phys = general_config.num_phys
    num_metadata = general_config.num_metadata

    dataset_np[:, 0] = np.arange(len(dataset_np))
    processing_functions.physical_parameter_scaling(
        dataset_np[:, num_metadata : num_metadata + num_phys]
    )
    processing_functions.abundances_scaling(dataset_np[:, -num_species:])

    latent_components = inference_functions.encode(dataset_np[:, num_metadata + num_phys :])
    latent_components = processing_functions.latent_components_scaling(latent_components).cpu().numpy()
    encoded_dataset_np = np.hstack((dataset_np, latent_components), dtype=np.float32)

    indices_np = calculate_indices(encoded_dataset_np, model_config.window_size)
    indices_np = indices_np[np.random.permutation(len(indices_np))]

    encoded_t = torch.from_numpy(encoded_dataset_np).float()
    indices_t = torch.from_numpy(indices_np).int()

    gc.collect()
    torch.cuda.empty_cache()
    return encoded_t, indices_t


def preprocess_latent_autoregressive(
    dataset_config,
    ae_config,
    model_config,
    autoencoder_class: type[Autoencoder],
) -> None:
    """Preprocess latent autoregressive dataset."""
    processing_functions = dp.Processing(dataset_config, ae_config)
    autoencoder = load_autoencoder(
        autoencoder_class,
        dataset_config,
        ae_config,
        inference=True,
    )
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)
    training_np, validation_np = load_datasets(dataset_config, model_config.columns)
    training_dataset = preprocess_dataset(
        dataset_config,
        model_config,
        training_np,
        processing_functions,
        inference_functions,
    )
    validation_dataset = preprocess_dataset(
        dataset_config,
        model_config,
        validation_np,
        processing_functions,
        inference_functions,
    )
    save_tensors(
        dataset_config,
        "latent_autoregressive",
        {"dataset": training_dataset[0], "indices": training_dataset[1]},
        category="training_seq",
    )
    save_tensors(
        dataset_config,
        "latent_autoregressive",
        {"dataset": validation_dataset[0], "indices": validation_dataset[1]},
        category="validation_seq",
    )


def ensure_preprocessed(dataset_config, force: bool = False) -> None:
    """Build latent autoregressive caches when missing or forced."""
    training_path = dataset_config.model_path("latent_autoregressive", "training_seq.pt")
    validation_path = dataset_config.model_path("latent_autoregressive", "validation_seq.pt")
    if not force and os.path.exists(training_path) and os.path.exists(validation_path):
        return
    ae_config = build_ae_config(dataset_config)
    preprocess_latent_autoregressive(
        dataset_config,
        ae_config,
        build_config(dataset_config, ae_config),
        Autoencoder,
    )


class LatentSequenceDataset(Dataset):
    """Tensor dataset for latent sequence training."""

    def __init__(self, general_config, data_matrix: torch.Tensor, data_indices: torch.Tensor, num_latents: int) -> None:
        """Initialize latent sequence dataset with data matrix and indices."""
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = general_config.num_metadata
        self.num_phys = general_config.num_phys
        self.num_latents = num_latents

        print(f"Data_matrix Memory usage: {self.data_matrix.nbytes / (1024**2):.3f} MB")
        print(f"Indices_matrix Memory usage: {self.data_indices.nbytes / (1024**2):.3f} MB")
        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return self.num_datapoints

    def __getitems__(self, indices):
        """Load multiple latent sequence samples at once for batch retrieval."""
        rows = self.data_matrix[self.data_indices[torch.tensor(indices, dtype=torch.long)]]
        physical_parameters = rows[:, :-1, self.num_metadata : self.num_metadata + self.num_phys].contiguous()
        features = rows[:, 0, -self.num_latents :].contiguous()
        targets = rows[:, 1:, self.num_metadata + self.num_phys : -self.num_latents].contiguous()
        return physical_parameters, features, targets
