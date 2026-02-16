"""Preprocessing script for emulator dataset."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.configs.autoencoder import AEConfig
    from src.configs.emulator import EMConfig
    from src.configs.general import GeneralConfig
    from src.inference import Inference
    from src.models.autoencoder import Autoencoder

from .. import data_loading as dl
from .. import data_processing as dp
from ..models.autoencoder import load_autoencoder


def preprocess_emulator(
    general_config: "GeneralConfig",
    ae_config: "AEConfig",
    em_config: "EMConfig",
    Autoencoder: type["Autoencoder"],
) -> None:
    """Preprocess emulator dataset.

    Creates training and validation datasets by processing raw data through
    the autoencoder latent space and saves them to HDF5 files.

    Args:
        general_config: General configuration for the project
        ae_config: Autoencoder configuration
        em_config: Emulator configuration
        Autoencoder: Autoencoder model class to use for inference
    """
    processing_functions = dp.Processing(
        general_config,
        ae_config,
    )
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)

    training_np, validation_np = dl.load_datasets(general_config, em_config.columns)
    training_dataset = dp.preprocessing_emulator_dataset(
        general_config,
        em_config,
        training_np,
        processing_functions,
        inference_functions,
    )
    validation_dataset = dp.preprocessing_emulator_dataset(
        general_config,
        em_config,
        validation_np,
        processing_functions,
        inference_functions,
    )

    dl.save_tensors_to_hdf5(general_config, training_dataset, category="training_seq")
    dl.save_tensors_to_hdf5(
        general_config, validation_dataset, category="validation_seq"
    )
