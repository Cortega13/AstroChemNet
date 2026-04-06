"""Latent neural operator data helpers."""

import os

from src import data_processing as dp
from src.data_loading import load_datasets, save_tensors
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.latent_autoregressive.data import (
    LatentSequenceDataset,
    preprocess_dataset,
)
from src.models.latent_neural_operator.config import build_config


def preprocess_latent_neural_operator(
    dataset_config,
    ae_config,
    model_config,
    autoencoder_class: type[Autoencoder],
) -> None:
    """Preprocess dataset for latent neural operator training."""
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
        "latent_neural_operator",
        {"dataset": training_dataset[0], "indices": training_dataset[1]},
        category="training_seq",
    )
    save_tensors(
        dataset_config,
        "latent_neural_operator",
        {"dataset": validation_dataset[0], "indices": validation_dataset[1]},
        category="validation_seq",
    )


def ensure_preprocessed(dataset_config, force: bool = False) -> None:
    """Build latent neural operator caches when missing or forced."""
    training_path = dataset_config.model_path("latent_neural_operator", "training_seq.pt")
    validation_path = dataset_config.model_path("latent_neural_operator", "validation_seq.pt")
    if not force and os.path.exists(training_path) and os.path.exists(validation_path):
        return
    ae_config = build_ae_config(dataset_config)
    preprocess_latent_neural_operator(
        dataset_config,
        ae_config,
        build_config(dataset_config, ae_config),
        Autoencoder,
    )


__all__ = ["LatentSequenceDataset", "ensure_preprocessed", "preprocess_dataset"]
