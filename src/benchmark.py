"""Benchmarking functions for model evaluation."""

from typing import Any, Dict

import torch

from src.configs.autoencoder import AEConfig
from src.configs.emulator import EMConfig
from src.configs.general import GeneralConfig
from src.models.autoencoder import Autoencoder, load_autoencoder
from src.models.emulator import Emulator, load_emulator

from . import data_loading as dl
from . import data_processing as dp
from .inference import Inference


def benchmark_autoencoder(
    general_config: GeneralConfig,
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


def benchmark_emulator(
    general_config: GeneralConfig,
    ae_config: AEConfig,
    em_config: EMConfig,
) -> Dict[str, Any]:
    """Benchmark emulator prediction accuracy in latent space.

    Args:
        general_config: General configuration for runtime and dataset
        ae_config: Autoencoder model configuration
        em_config: Emulator model configuration

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

    # Load emulator
    emulator = load_emulator(Emulator, general_config, em_config, inference=True)

    # Load validation data
    training_np, validation_np = dl.load_datasets(general_config, em_config.columns)

    # Preprocess for emulator
    validation_dataset = dp.preprocessing_emulator_dataset(
        general_config,
        em_config,
        validation_np,
        processing_functions,
        inference_functions,
    )

    # Get sequences
    validation_Dataset = dl.EmulatorSequenceDataset(
        general_config, ae_config, validation_dataset[0], validation_dataset[1]
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
        predicted_latents = emulator(physical_parameters, features)

    # Calculate metrics in latent space
    mse = torch.mean((predicted_latents - targets) ** 2).item()
    mae = torch.mean(torch.abs(predicted_latents - targets)).item()

    return {
        "mse": mse,
        "mae": mae,
        "num_samples": predicted_latents.shape[0],
    }


def benchmark_combined(
    general_config: GeneralConfig,
    ae_config: AEConfig,
    em_config: EMConfig,
) -> Dict[str, Any]:
    """Benchmark full pipeline: emulator prediction + autoencoder decode.

    This evaluates the end-to-end accuracy of predicting chemical abundances
    by running the emulator in latent space and decoding back to abundance space.

    Args:
        general_config: General configuration for runtime and dataset
        ae_config: Autoencoder model configuration
        em_config: Emulator model configuration

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

    # Load emulator
    emulator = load_emulator(Emulator, general_config, em_config, inference=True)

    # Load validation data
    training_np, validation_np = dl.load_datasets(general_config, em_config.columns)

    # Preprocess for emulator
    validation_dataset = dp.preprocessing_emulator_dataset(
        general_config,
        em_config,
        validation_np,
        processing_functions,
        inference_functions,
    )

    # Get sequences
    validation_Dataset = dl.EmulatorSequenceDataset(
        general_config, ae_config, validation_dataset[0], validation_dataset[1]
    )

    # Sample a batch for benchmarking
    physical_parameters, features, targets = validation_Dataset.__getitems__(
        list(range(min(1000, len(validation_Dataset))))
    )
    physical_parameters = physical_parameters.to(general_config.device)
    features = features.to(general_config.device)
    targets = targets.to(general_config.device)

    # Run inference: emulator predicts latent, then decode to abundance
    with torch.no_grad():
        predicted_latents = emulator(physical_parameters, features)
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
