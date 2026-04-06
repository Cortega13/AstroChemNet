"""Latent ODE data helpers."""

import gc
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src import data_processing as dp
from src.data_loading import load_datasets, save_tensors
from src.datasets import DatasetConfig
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.autoregressive.data import calculate_indices
from src.models.latent_ode.config import LatentODEConfig, build_config


def compute_base_dt(dataset_np: np.ndarray) -> float:
    """Compute the dataset base observation interval from model time deltas."""
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)
    deltas = []
    for group in model_groups:
        if len(group) < 2:
            continue
        positive = np.diff(group[:, 2])
        positive = positive[positive > 0]
        if positive.size:
            deltas.append(positive)
    if not deltas:
        raise ValueError(
            "Unable to compute a positive base time interval from dataset."
        )
    return float(np.median(np.concatenate(deltas)))


def save_base_dt(path: str, base_dt: float) -> None:
    """Persist the dataset base time interval for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"base_dt": float(base_dt)}, f, indent=2)


def preprocess_dataset(
    general_config: DatasetConfig,
    ode_config: LatentODEConfig,
    dataset_np: np.ndarray,
    processing_functions: dp.Processing,
    inference_functions: Inference,
    base_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess dataset for latent ODE training with interval lengths."""
    num_species = general_config.num_species
    num_phys = general_config.num_phys
    num_metadata = general_config.num_metadata

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

    indices_np = calculate_indices(encoded_dataset_np, ode_config.window_size)
    times_np = encoded_dataset_np[indices_np, 2]
    delta_t_np = np.diff(times_np, axis=1).astype(np.float32) / np.float32(base_dt)
    perm = np.random.permutation(len(indices_np))

    encoded_t = torch.from_numpy(encoded_dataset_np).float()
    indices_t = torch.from_numpy(indices_np[perm]).int()
    delta_t_t = torch.from_numpy(delta_t_np[perm]).float()

    gc.collect()
    torch.cuda.empty_cache()
    return encoded_t, indices_t, delta_t_t


def preprocess_latent_ode(
    dataset_config,
    ae_config,
    model_config,
    autoencoder_class: type[Autoencoder],
) -> None:
    """Preprocess latent ODE dataset with latent states and intervals."""
    processing_functions = dp.Processing(dataset_config, ae_config)
    autoencoder = load_autoencoder(
        autoencoder_class,
        dataset_config,
        ae_config,
        inference=True,
    )
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)
    training_np, validation_np = load_datasets(dataset_config, model_config.columns)
    base_dt = compute_base_dt(training_np)
    training_dataset = preprocess_dataset(
        dataset_config,
        model_config,
        training_np,
        processing_functions,
        inference_functions,
        base_dt,
    )
    validation_dataset = preprocess_dataset(
        dataset_config,
        model_config,
        validation_np,
        processing_functions,
        inference_functions,
        base_dt,
    )
    save_tensors(
        dataset_config,
        "latent_ode",
        {
            "dataset": training_dataset[0],
            "indices": training_dataset[1],
            "delta_t": training_dataset[2],
        },
        category="training_seq",
    )
    save_tensors(
        dataset_config,
        "latent_ode",
        {
            "dataset": validation_dataset[0],
            "indices": validation_dataset[1],
            "delta_t": validation_dataset[2],
        },
        category="validation_seq",
    )
    save_base_dt(model_config.base_dt_path, base_dt)


def ensure_preprocessed(dataset_config, force: bool = False) -> None:
    """Build latent ODE caches when missing or forced."""
    training_path = dataset_config.model_path("latent_ode", "training_seq.pt")
    validation_path = dataset_config.model_path("latent_ode", "validation_seq.pt")
    if not force and os.path.exists(training_path) and os.path.exists(validation_path):
        return
    ae_config = build_ae_config(dataset_config)
    preprocess_latent_ode(
        dataset_config,
        ae_config,
        build_config(dataset_config, ae_config),
        Autoencoder,
    )


class LatentODESequenceDataset(Dataset):
    """Tensor dataset for latent ODE training with interval durations."""

    def __init__(
        self,
        general_config,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
        delta_t: torch.Tensor,
        num_latents: int,
    ) -> None:
        """Initialize the latent ODE dataset with latent windows and time deltas."""
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.delta_t = delta_t.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = general_config.num_metadata
        self.num_phys = general_config.num_phys
        self.num_latents = num_latents

        print(f"Data_matrix Memory usage: {self.data_matrix.nbytes / (1024**2):.3f} MB")
        print(
            f"Indices_matrix Memory usage: {self.data_indices.nbytes / (1024**2):.3f} MB"
        )
        print(f"Delta_t Memory usage: {self.delta_t.nbytes / (1024**2):.3f} MB")
        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return self.num_datapoints

    def __getitems__(self, indices):
        """Load multiple latent ODE windows for batch retrieval."""
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        rows = self.data_matrix[self.data_indices[indices_tensor]]
        delta_t = self.delta_t[indices_tensor].contiguous()
        physical_parameters = rows[
            :, :-1, self.num_metadata : self.num_metadata + self.num_phys
        ].contiguous()
        features = rows[:, 0, -self.num_latents :].contiguous()
        targets = rows[
            :, 1:, self.num_metadata + self.num_phys : -self.num_latents
        ].contiguous()
        return delta_t, physical_parameters, features, targets
