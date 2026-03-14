"""Training script for abundance autoregressive model."""

from src.configs.autoregressive import AutoregressiveConfig
from src.configs.datasets import DatasetConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..loss import Loss
from ..models.autoregressive import Autoregressive, load_autoregressive
from ..trainer import AutoregressiveTrainerSequential, load_objects


def main(
    model_class: type[Autoregressive],
    general_config: DatasetConfig,
    ar_config: AutoregressiveConfig,
) -> None:
    """Train abundance autoregressive model with given configuration."""
    training_dataset, training_indices = dl.load_tensors_from_hdf5(
        general_config,
        category="training_seq",
        artifact_dir="autoregressive",
    )
    validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
        general_config,
        category="validation_seq",
        artifact_dir="autoregressive",
    )

    training_sequence_dataset = dl.AutoregressiveSequenceDataset(
        general_config, training_dataset, training_indices
    )
    validation_sequence_dataset = dl.AutoregressiveSequenceDataset(
        general_config, validation_dataset, validation_indices
    )
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dl.tensor_to_dataloader(ar_config, training_sequence_dataset)
    validation_dataloader = dl.tensor_to_dataloader(
        ar_config, validation_sequence_dataset
    )

    autoregressive = load_autoregressive(model_class, general_config, ar_config)
    optimizer, scheduler = load_objects(autoregressive, ar_config)
    processing_functions = dp.Processing(general_config)
    loss_functions = Loss(
        processing_functions,
        general_config,
        ModelConfig=ar_config,
    )

    ar_trainer = AutoregressiveTrainerSequential(
        general_config,
        ar_config,
        loss_functions,
        autoregressive,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    ar_trainer.train()


if __name__ == "__main__":
    from src.configs.factory import build_autoregressive_config, build_dataset_config

    general_config = build_dataset_config("uclchem_grav")
    ar_config = build_autoregressive_config(general_config)

    main(Autoregressive, general_config, ar_config)
