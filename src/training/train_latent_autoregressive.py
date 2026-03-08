"""Training script for latent autoregressive model."""

from src.configs.autoencoder import AEConfig
from src.configs.datasets import DatasetConfig
from src.configs.latent_autoregressive import ARConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..models.latent_autoregressive import LatentAR, load_latent_autoregressive
from ..trainer import LatentARTrainerSequential, load_objects


def main(
    Autoencoder: type[Autoencoder],
    LatentAR: type[LatentAR],
    general_config: DatasetConfig,
    ae_config: AEConfig,
    ar_config: ARConfig,
) -> None:
    """Train latent autoregressive model with given configuration.

    Note: Run 'python run.py preprocess latent_autoregressive' before training to prepare
    the training and validation datasets.
    """
    # Load preprocessed datasets (must run preprocessing first)
    training_dataset, training_indices = dl.load_tensors_from_hdf5(
        general_config, category="training_seq"
    )
    validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
        general_config, category="validation_seq"
    )

    training_Dataset = dl.ARSequenceDataset(
        general_config, ae_config, training_dataset, training_indices
    )
    validation_Dataset = dl.ARSequenceDataset(
        general_config, ae_config, validation_dataset, validation_indices
    )
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dl.tensor_to_dataloader(ar_config, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(ar_config, validation_Dataset)

    latent_ar = load_latent_autoregressive(LatentAR, general_config, ar_config)
    optimizer, scheduler = load_objects(latent_ar, ar_config)

    processing_functions = dp.Processing(general_config, ae_config)
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )

    loss_functions = Loss(
        processing_functions,
        general_config,
        ModelConfig=ar_config,
    )
    ar_trainer = LatentARTrainerSequential(
        general_config,
        ae_config,
        ar_config,
        loss_functions,
        processing_functions,
        autoencoder,
        latent_ar,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    ar_trainer.train()


if __name__ == "__main__":
    # Instantiate configs - GeneralConfig loads from preprocessing output
    from src.configs.factory import (
        build_ae_config,
        build_dataset_config,
        build_ar_config,
    )

    general_config = build_dataset_config("uclchem_grav")
    ae_config = build_ae_config(general_config)
    ar_config = build_ar_config(general_config, ae_config)

    # Run main script.
    main(Autoencoder, LatentAR, general_config, ae_config, ar_config)
