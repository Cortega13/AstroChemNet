"""Benchmarking functions for model evaluation."""

from typing import Any, Dict

import torch

from src.configs.autoencoder import AEConfig
from src.configs.autoregressive import AutoregressiveConfig
from src.configs.datasets import DatasetConfig
from src.configs.latent_autoregressive import ARConfig
from src.configs.latent_ode import LatentODEConfig
from src.models.autoencoder import Autoencoder, load_autoencoder
from src.models.autoregressive import Autoregressive, load_autoregressive
from src.models.latent_autoregressive import LatentAR, load_latent_autoregressive
from src.models.latent_ode import LatentODE, load_latent_ode

from . import data_loading as dl
from . import data_processing as dp
from .inference import Inference


def benchmark_autoencoder(
    general_config: DatasetConfig,
    ae_config: AEConfig,
) -> Dict[str, Any]:
    """Benchmark autoencoder reconstruction accuracy.

    Args:
        general_config: General configuration for runtime and dataset
        ae_config: Autoencoder model configuration

    Returns:
        Dictionary containing benchmark metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - max_error: Maximum reconstruction error
    """
    processing_functions = dp.Processing(general_config, ae_config)

    # Load validation data
    training_np, validation_np = dl.load_datasets(general_config, ae_config.columns)
    processing_functions.abundances_scaling(training_np)
    processing_functions.abundances_scaling(validation_np)

    validation_tensor = torch.from_numpy(validation_np).to(general_config.device)

    # Load autoencoder
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)

    # Run inference: encode then decode
    with torch.no_grad():
        latents = inference_functions.encode(validation_tensor)
        reconstructed = autoencoder.decode(latents)

    # Calculate metrics
    mse = torch.mean((validation_tensor - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(validation_tensor - reconstructed)).item()
    max_error = torch.max(torch.abs(validation_tensor - reconstructed)).item()

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "num_samples": validation_tensor.shape[0],
    }


def benchmark_latent_autoregressive(
    general_config: DatasetConfig,
    ae_config: AEConfig,
    ar_config: ARConfig,
) -> Dict[str, Any]:
    """Benchmark latent autoregressive prediction accuracy in latent space.

    Args:
        general_config: General configuration for runtime and dataset
        ae_config: Autoencoder model configuration
        ar_config: LatentAR model configuration

    Returns:
        Dictionary containing benchmark metrics:
            - mse: Mean squared error in latent space
            - mae: Mean absolute error in latent space
    """
    processing_functions = dp.Processing(general_config, ae_config)

    # Load autoencoder
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)

    # Load latent autoregressive
    latent_ar = load_latent_autoregressive(
        LatentAR, general_config, ar_config, inference=True
    )

    # Load validation data
    training_np, validation_np = dl.load_datasets(general_config, ar_config.columns)

    # Preprocess for latent autoregressive
    validation_dataset = dp.preprocessing_latent_autoregressive_dataset(
        general_config,
        ar_config,
        validation_np,
        processing_functions,
        inference_functions,
    )

    # Get sequences
    validation_Dataset = dl.ARSequenceDataset(
        general_config, validation_dataset[0], validation_dataset[1], ae_config.latent_dim
    )

    # Sample a batch for benchmarking
    physical_parameters, features, targets = validation_Dataset.__getitems__(
        list(range(min(1000, len(validation_Dataset))))
    )
    physical_parameters = physical_parameters.to(general_config.device)
    features = features.to(general_config.device)
    targets = targets.to(general_config.device)

    # Run inference
    with torch.no_grad():
        predicted_latents = latent_ar(physical_parameters, features)

    # Calculate metrics in latent space
    mse = torch.mean((predicted_latents - targets) ** 2).item()
    mae = torch.mean(torch.abs(predicted_latents - targets)).item()

    return {
        "mse": mse,
        "mae": mae,
        "num_samples": predicted_latents.shape[0],
    }


def benchmark_combined(
    general_config: DatasetConfig,
    ae_config: AEConfig,
    ar_config: ARConfig,
) -> Dict[str, Any]:
    """Benchmark full pipeline: latent autoregressive prediction + autoencoder decode.

    This evaluates the end-to-end accuracy of predicting chemical abundances
    by running the latent autoregressive in latent space and decoding back to abundance space.

    Args:
        general_config: General configuration for runtime and dataset
        ae_config: Autoencoder model configuration
        ar_config: LatentAR model configuration

    Returns:
        Dictionary containing benchmark metrics:
            - mse: Mean squared error in abundance space
            - mae: Mean absolute error in abundance space
            - max_error: Maximum prediction error
    """
    processing_functions = dp.Processing(general_config, ae_config)

    # Load autoencoder
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)

    # Load latent autoregressive
    latent_ar = load_latent_autoregressive(
        LatentAR, general_config, ar_config, inference=True
    )

    # Load validation data
    training_np, validation_np = dl.load_datasets(general_config, ar_config.columns)

    # Preprocess for latent autoregressive
    validation_dataset = dp.preprocessing_latent_autoregressive_dataset(
        general_config,
        ar_config,
        validation_np,
        processing_functions,
        inference_functions,
    )

    # Get sequences
    validation_Dataset = dl.ARSequenceDataset(
        general_config, validation_dataset[0], validation_dataset[1], ae_config.latent_dim
    )

    # Sample a batch for benchmarking
    physical_parameters, features, targets = validation_Dataset.__getitems__(
        list(range(min(1000, len(validation_Dataset))))
    )
    physical_parameters = physical_parameters.to(general_config.device)
    features = features.to(general_config.device)
    targets = targets.to(general_config.device)

    # Run inference: latent autoregressive predicts latent, then decode to abundance
    with torch.no_grad():
        predicted_latents = latent_ar(physical_parameters, features)
        predicted_latents_scaled = (
            processing_functions.inverse_latent_components_scaling(
                predicted_latents.reshape(-1, ae_config.latent_dim)
            )
        )
        predicted_abundances = autoencoder.decode(predicted_latents_scaled)
        predicted_abundances = predicted_abundances.reshape(
            targets.shape[0], targets.shape[1], -1
        )

    # Decode target latents to abundances for comparison
    targets_scaled = processing_functions.inverse_latent_components_scaling(
        targets.reshape(-1, ae_config.latent_dim)
    )
    target_abundances = autoencoder.decode(targets_scaled)
    target_abundances = target_abundances.reshape(
        targets.shape[0], targets.shape[1], -1
    )

    # Calculate metrics in abundance space
    mse = torch.mean((predicted_abundances - target_abundances) ** 2).item()
    mae = torch.mean(torch.abs(predicted_abundances - target_abundances)).item()
    max_error = torch.max(torch.abs(predicted_abundances - target_abundances)).item()

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "num_samples": predicted_abundances.shape[0],
    }


def benchmark_autoregressive(
    general_config: DatasetConfig,
    ar_config: AutoregressiveConfig,
) -> Dict[str, Any]:
    """Benchmark abundance autoregressive prediction accuracy in abundance space."""
    processing_functions = dp.Processing(general_config)
    autoregressive = load_autoregressive(
        Autoregressive, general_config, ar_config, inference=True
    )

    _, validation_np = dl.load_datasets(general_config, ar_config.columns)
    validation_dataset = dp.preprocessing_autoregressive_dataset(
        general_config,
        ar_config,
        validation_np,
        processing_functions,
    )
    validation_sequence_dataset = dl.AutoregressiveSequenceDataset(
        general_config,
        validation_dataset[0],
        validation_dataset[1],
    )

    physical_parameters, features, targets = validation_sequence_dataset.__getitems__(
        list(range(min(1000, len(validation_sequence_dataset))))
    )
    physical_parameters = physical_parameters.to(general_config.device)
    features = features.to(general_config.device)
    targets = targets.to(general_config.device)

    with torch.no_grad():
        predicted = autoregressive(physical_parameters, features)

    unscaled_predicted = processing_functions.inverse_abundances_scaling(
        predicted.reshape(-1, general_config.num_species)
    )
    unscaled_targets = processing_functions.inverse_abundances_scaling(
        targets.reshape(-1, general_config.num_species)
    )

    mse = torch.mean((unscaled_predicted - unscaled_targets) ** 2).item()
    mae = torch.mean(torch.abs(unscaled_predicted - unscaled_targets)).item()
    max_error = torch.max(torch.abs(unscaled_predicted - unscaled_targets)).item()

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "num_samples": predicted.shape[0],
    }


def benchmark_latent_ode(
    general_config: DatasetConfig,
    ae_config: AEConfig,
    ode_config: LatentODEConfig,
) -> Dict[str, Any]:
    """Benchmark latent ODE prediction accuracy in abundance space."""
    processing_functions = dp.Processing(general_config, ae_config)
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)
    latent_ode = load_latent_ode(LatentODE, general_config, ode_config, inference=True)

    training_np, validation_np = dl.load_datasets(general_config, ode_config.columns)
    base_dt = dp.compute_base_dt(training_np)
    validation_dataset = dp.preprocessing_latent_ode_dataset(
        general_config,
        ode_config,
        validation_np,
        processing_functions,
        inference_functions,
        base_dt,
    )
    validation_sequence_dataset = dl.LatentODESequenceDataset(
        general_config,
        validation_dataset[0],
        validation_dataset[1],
        validation_dataset[2],
        ae_config.latent_dim,
    )

    delta_t, physical_parameters, features, targets = validation_sequence_dataset.__getitems__(
        list(range(min(512, len(validation_sequence_dataset))))
    )
    delta_t = delta_t.to(general_config.device)
    physical_parameters = physical_parameters.to(general_config.device)
    features = features.to(general_config.device)
    targets = targets.to(general_config.device)

    with torch.no_grad():
        predicted_latents = latent_ode(delta_t, physical_parameters, features)
        predicted_latents = processing_functions.inverse_latent_components_scaling(
            predicted_latents.reshape(-1, ae_config.latent_dim)
        )
        predicted_abundances = autoencoder.decode(predicted_latents).reshape(
            targets.shape[0], targets.shape[1], -1
        )

    unscaled_predicted = processing_functions.inverse_abundances_scaling(
        predicted_abundances.reshape(-1, general_config.num_species)
    )
    unscaled_targets = processing_functions.inverse_abundances_scaling(
        targets.reshape(-1, general_config.num_species)
    )

    mse = torch.mean((unscaled_predicted - unscaled_targets) ** 2).item()
    mae = torch.mean(torch.abs(unscaled_predicted - unscaled_targets)).item()
    max_error = torch.max(torch.abs(unscaled_predicted - unscaled_targets)).item()

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "num_samples": predicted_abundances.shape[0],
    }
