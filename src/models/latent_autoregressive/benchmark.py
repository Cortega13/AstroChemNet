"""Latent autoregressive benchmark entrypoint."""

import torch

from src.data_loading import load_datasets
from src.data_processing import Processing
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.latent_autoregressive.config import build_config
from src.models.latent_autoregressive.data import (
    LatentSequenceDataset,
    preprocess_dataset,
)
from src.models.latent_autoregressive.model import LatentAR, load_latent_autoregressive


def benchmark(dataset_config):
    """Benchmark latent autoregressive model for a dataset."""
    ae_config = build_ae_config(dataset_config)
    ar_config = build_config(dataset_config, ae_config)
    processing_functions = Processing(dataset_config, ae_config)
    autoencoder = load_autoencoder(Autoencoder, dataset_config, ae_config, inference=True)
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)
    latent_ar = load_latent_autoregressive(LatentAR, dataset_config, ar_config, inference=True)

    _, validation_np = load_datasets(dataset_config, ar_config.columns)
    validation_dataset = preprocess_dataset(
        dataset_config,
        ar_config,
        validation_np,
        processing_functions,
        inference_functions,
    )
    validation_sequence_dataset = LatentSequenceDataset(
        dataset_config,
        validation_dataset[0],
        validation_dataset[1],
        ae_config.latent_dim,
    )

    physical_parameters, features, targets = validation_sequence_dataset.__getitems__(
        list(range(min(1000, len(validation_sequence_dataset))))
    )
    physical_parameters = physical_parameters.to(dataset_config.device)
    features = features.to(dataset_config.device)
    targets = targets.to(dataset_config.device)

    with torch.no_grad():
        predicted_latents = latent_ar(physical_parameters, features)

    return {
        "mse": torch.mean((predicted_latents - targets) ** 2).item(),
        "mae": torch.mean(torch.abs(predicted_latents - targets)).item(),
        "num_samples": predicted_latents.shape[0],
    }


def benchmark_combined_pipeline(dataset_config):
    """Benchmark the combined latent autoregressive pipeline."""
    ae_config = build_ae_config(dataset_config)
    ar_config = build_config(dataset_config, ae_config)
    processing_functions = Processing(dataset_config, ae_config)
    autoencoder = load_autoencoder(Autoencoder, dataset_config, ae_config, inference=True)
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)
    latent_ar = load_latent_autoregressive(LatentAR, dataset_config, ar_config, inference=True)

    _, validation_np = load_datasets(dataset_config, ar_config.columns)
    validation_dataset = preprocess_dataset(
        dataset_config,
        ar_config,
        validation_np,
        processing_functions,
        inference_functions,
    )
    validation_sequence_dataset = LatentSequenceDataset(
        dataset_config,
        validation_dataset[0],
        validation_dataset[1],
        ae_config.latent_dim,
    )

    physical_parameters, features, targets = validation_sequence_dataset.__getitems__(
        list(range(min(1000, len(validation_sequence_dataset))))
    )
    physical_parameters = physical_parameters.to(dataset_config.device)
    features = features.to(dataset_config.device)
    targets = targets.to(dataset_config.device)

    with torch.no_grad():
        predicted_latents = latent_ar(physical_parameters, features)
        predicted_latents_scaled = processing_functions.inverse_latent_components_scaling(
            predicted_latents.reshape(-1, ae_config.latent_dim)
        )
        predicted_abundances = autoencoder.decode(predicted_latents_scaled).reshape(
            targets.shape[0],
            targets.shape[1],
            -1,
        )

    targets_scaled = processing_functions.inverse_latent_components_scaling(
        targets.reshape(-1, ae_config.latent_dim)
    )
    target_abundances = autoencoder.decode(targets_scaled).reshape(
        targets.shape[0],
        targets.shape[1],
        -1,
    )

    return {
        "mse": torch.mean((predicted_abundances - target_abundances) ** 2).item(),
        "mae": torch.mean(torch.abs(predicted_abundances - target_abundances)).item(),
        "max_error": torch.max(torch.abs(predicted_abundances - target_abundances)).item(),
        "num_samples": predicted_abundances.shape[0],
    }
