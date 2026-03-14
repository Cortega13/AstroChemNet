"""Preprocessing script for latent ODE dataset."""

from src.configs.autoencoder import AEConfig
from src.configs.datasets import DatasetConfig
from src.configs.latent_ode import LatentODEConfig
from src.inference import Inference
from src.models.autoencoder import Autoencoder
from src.models.latent_ode import save_base_dt

from .. import data_loading as dl
from .. import data_processing as dp
from ..models.autoencoder import load_autoencoder


def preprocess_latent_ode(
    general_config: DatasetConfig,
    ae_config: AEConfig,
    ode_config: LatentODEConfig,
    Autoencoder: type[Autoencoder],
) -> None:
    """Preprocess latent ODE dataset with latent states and interval lengths."""
    processing_functions = dp.Processing(general_config, ae_config)
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)

    training_np, validation_np = dl.load_datasets(general_config, ode_config.columns)
    base_dt = dp.compute_base_dt(training_np)

    training_dataset = dp.preprocessing_latent_ode_dataset(
        general_config,
        ode_config,
        training_np,
        processing_functions,
        inference_functions,
        base_dt,
    )
    validation_dataset = dp.preprocessing_latent_ode_dataset(
        general_config,
        ode_config,
        validation_np,
        processing_functions,
        inference_functions,
        base_dt,
    )

    dl.save_named_tensors_to_hdf5(
        general_config,
        {
            "dataset": training_dataset[0],
            "indices": training_dataset[1],
            "delta_t": training_dataset[2],
        },
        category="training_seq",
        artifact_dir="latent_ode",
    )
    dl.save_named_tensors_to_hdf5(
        general_config,
        {
            "dataset": validation_dataset[0],
            "indices": validation_dataset[1],
            "delta_t": validation_dataset[2],
        },
        category="validation_seq",
        artifact_dir="latent_ode",
    )
    save_base_dt(ode_config.base_dt_path, base_dt)
