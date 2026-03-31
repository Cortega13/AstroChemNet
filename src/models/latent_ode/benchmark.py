"""Latent ODE benchmark entrypoint."""

import torch

from src.data_loading import load_datasets
from src.data_processing import Processing
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.latent_ode.config import build_config
from src.models.latent_ode.data import (
    LatentODESequenceDataset,
    compute_base_dt,
    preprocess_dataset,
)
from src.models.latent_ode.model import LatentODE, load_latent_ode


def benchmark(dataset_config):
    """Benchmark latent ODE model for a dataset."""
    ae_config = build_ae_config(dataset_config)
    ode_config = build_config(dataset_config, ae_config)
    processing_functions = Processing(dataset_config, ae_config)
    autoencoder = load_autoencoder(Autoencoder, dataset_config, ae_config, inference=True)
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)
    latent_ode = load_latent_ode(LatentODE, dataset_config, ode_config, inference=True)

    training_np, validation_np = load_datasets(dataset_config, ode_config.columns)
    base_dt = compute_base_dt(training_np)
    validation_dataset = preprocess_dataset(
        dataset_config,
        ode_config,
        validation_np,
        processing_functions,
        inference_functions,
        base_dt,
    )
    validation_sequence_dataset = LatentODESequenceDataset(
        dataset_config,
        validation_dataset[0],
        validation_dataset[1],
        validation_dataset[2],
        ae_config.latent_dim,
    )

    delta_t, physical_parameters, features, targets = validation_sequence_dataset.__getitems__(
        list(range(min(512, len(validation_sequence_dataset))))
    )
    delta_t = delta_t.to(dataset_config.device)
    physical_parameters = physical_parameters.to(dataset_config.device)
    features = features.to(dataset_config.device)
    targets = targets.to(dataset_config.device)

    with torch.no_grad():
        predicted_latents = latent_ode(delta_t, physical_parameters, features)
        predicted_latents = processing_functions.inverse_latent_components_scaling(
            predicted_latents.reshape(-1, ae_config.latent_dim)
        )
        predicted_abundances = autoencoder.decode(predicted_latents).reshape(
            targets.shape[0],
            targets.shape[1],
            -1,
        )

    unscaled_predicted = processing_functions.inverse_abundances_scaling(
        predicted_abundances.reshape(-1, dataset_config.num_species)
    )
    unscaled_targets = processing_functions.inverse_abundances_scaling(
        targets.reshape(-1, dataset_config.num_species)
    )

    return {
        "mse": torch.mean((unscaled_predicted - unscaled_targets) ** 2).item(),
        "mae": torch.mean(torch.abs(unscaled_predicted - unscaled_targets)).item(),
        "max_error": torch.max(torch.abs(unscaled_predicted - unscaled_targets)).item(),
        "num_samples": predicted_abundances.shape[0],
    }
