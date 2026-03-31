"""Autoregressive benchmark entrypoint."""

import torch

from src.data_loading import load_datasets
from src.data_processing import Processing
from src.models.autoregressive.config import build_config
from src.models.autoregressive.data import (
    AutoregressiveSequenceDataset,
    preprocess_dataset,
)
from src.models.autoregressive.model import Autoregressive, load_autoregressive


def benchmark(dataset_config):
    """Benchmark autoregressive model for a dataset."""
    ar_config = build_config(dataset_config)
    processing_functions = Processing(dataset_config)
    autoregressive = load_autoregressive(
        Autoregressive,
        dataset_config,
        ar_config,
        inference=True,
    )

    _, validation_np = load_datasets(dataset_config, ar_config.columns)
    validation_dataset = preprocess_dataset(
        dataset_config,
        ar_config,
        validation_np,
        processing_functions,
    )
    validation_sequence_dataset = AutoregressiveSequenceDataset(
        dataset_config,
        validation_dataset[0],
        validation_dataset[1],
    )

    physical_parameters, features, targets = validation_sequence_dataset.__getitems__(
        list(range(min(1000, len(validation_sequence_dataset))))
    )
    physical_parameters = physical_parameters.to(dataset_config.device)
    features = features.to(dataset_config.device)
    targets = targets.to(dataset_config.device)

    with torch.no_grad():
        predicted = autoregressive(physical_parameters, features)

    unscaled_predicted = processing_functions.inverse_abundances_scaling(
        predicted.reshape(-1, dataset_config.num_species)
    )
    unscaled_targets = processing_functions.inverse_abundances_scaling(
        targets.reshape(-1, dataset_config.num_species)
    )

    return {
        "mse": torch.mean((unscaled_predicted - unscaled_targets) ** 2).item(),
        "mae": torch.mean(torch.abs(unscaled_predicted - unscaled_targets)).item(),
        "max_error": torch.max(torch.abs(unscaled_predicted - unscaled_targets)).item(),
        "num_samples": predicted.shape[0],
    }
