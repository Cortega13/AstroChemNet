"""Latent RNN data helpers."""

from pathlib import Path

from src import data_processing as dp
from src.data_loading import load_datasets, save_tensors
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.latent_autoregressive.data import (
    LatentSequenceDataset,
    preprocess_dataset,
)
from src.models.latent_rnn.config import build_config


def preprocess_latent_rnn(
    dataset_config,
    ae_config,
    model_config,
    autoencoder_class: type[Autoencoder],
) -> None:
    """Preprocess latent RNN dataset using the latent sequence schema."""
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
        {"dataset": training_dataset[0], "indices": training_dataset[1]},
        category="training_seq",
        artifact_dir="latent_rnn",
    )
    save_tensors(
        dataset_config,
        {"dataset": validation_dataset[0], "indices": validation_dataset[1]},
        category="validation_seq",
        artifact_dir="latent_rnn",
    )


def artifact_paths(dataset_config) -> tuple[Path, Path]:
    """Return latent RNN cache paths."""
    artifact_dir = Path(dataset_config.preprocessing_dir) / "latent_rnn"
    return artifact_dir / "training_seq.pt", artifact_dir / "validation_seq.pt"


def ensure_preprocessed(dataset_config, force: bool = False) -> None:
    """Build latent RNN caches when missing or forced."""
    training_path, validation_path = artifact_paths(dataset_config)
    if not force and training_path.exists() and validation_path.exists():
        return
    ae_config = build_ae_config(dataset_config)
    preprocess_latent_rnn(
        dataset_config,
        ae_config,
        build_config(dataset_config, ae_config),
        Autoencoder,
    )


__all__ = ["LatentSequenceDataset", "artifact_paths", "ensure_preprocessed", "preprocess_dataset"]
