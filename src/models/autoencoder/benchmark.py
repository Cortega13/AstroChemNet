"""Autoencoder benchmark entrypoint."""

import torch

from src.data_loading import load_datasets
from src.data_processing import Processing
from src.models.autoencoder.config import build_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder


def benchmark(dataset_config):
    """Benchmark autoencoder model for a dataset."""
    ae_config = build_config(dataset_config)
    processing_functions = Processing(dataset_config, ae_config)

    training_np, validation_np = load_datasets(dataset_config, ae_config.columns)
    processing_functions.abundances_scaling(training_np)
    processing_functions.abundances_scaling(validation_np)
    validation_tensor = torch.from_numpy(validation_np).to(dataset_config.device)

    autoencoder = load_autoencoder(Autoencoder, dataset_config, ae_config, inference=True)
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)

    with torch.no_grad():
        latents = inference_functions.encode(validation_tensor)
        reconstructed = autoencoder.decode(latents)

    return {
        "mse": torch.mean((validation_tensor - reconstructed) ** 2).item(),
        "mae": torch.mean(torch.abs(validation_tensor - reconstructed)).item(),
        "max_error": torch.max(torch.abs(validation_tensor - reconstructed)).item(),
        "num_samples": validation_tensor.shape[0],
    }
