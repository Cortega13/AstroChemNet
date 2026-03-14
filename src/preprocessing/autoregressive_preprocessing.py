"""Preprocessing script for abundance autoregressive dataset."""

from src.configs.autoregressive import AutoregressiveConfig
from src.configs.datasets import DatasetConfig

from .. import data_loading as dl
from .. import data_processing as dp


def preprocess_autoregressive(
    general_config: DatasetConfig,
    ar_config: AutoregressiveConfig,
) -> None:
    """Preprocess abundance autoregressive dataset."""
    processing_functions = dp.Processing(general_config)

    training_np, validation_np = dl.load_datasets(general_config, ar_config.columns)
    training_dataset = dp.preprocessing_autoregressive_dataset(
        general_config,
        ar_config,
        training_np,
        processing_functions,
    )
    validation_dataset = dp.preprocessing_autoregressive_dataset(
        general_config,
        ar_config,
        validation_np,
        processing_functions,
    )

    dl.save_tensors_to_hdf5(
        general_config,
        training_dataset,
        category="training_seq",
        artifact_dir="autoregressive",
    )
    dl.save_tensors_to_hdf5(
        general_config,
        validation_dataset,
        category="validation_seq",
        artifact_dir="autoregressive",
    )
